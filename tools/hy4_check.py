#!/usr/bin/env python3
"""Hy4-flash on a device: the checks and the measurements of one pack, one JSON line each.

    python tools/hy4_check.py --pack DIR [--max-len N] [--prompt N] [--steps N] [--ref] [--dfly] [--fit N]

  self_check   a recorded decode replayed as chain blocks and trees: every row bit for bit (always)
  ref          the engine's prompt and decode rows against the reference forward on the same weights
               (--ref; the reference dequantises every projection it touches, so small packs or slices only)
  decode       plain decode tokens a second after a prompt of --prompt tokens, --steps steps
  dfly         the same continuation in drafted rounds (the DFly2 head, the best token a depth as a chain
               block): equal tokens, tokens a second, kept tokens a round (--dfly)
  fit          a prompt of --fit tokens: seconds, the state's bytes, the device's memory before and after

The prompt is synthetic unless --text names a file the pack's tokenizer reads. For a tiny rehearsal pack whose
config names no draft targets, --dfly-layers and --mask-id give them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402


def mem() -> dict:
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            if k in ("MemAvailable", "MemFree", "Cached"):
                out[k] = round(int(v.split()[0]) / (1 << 20), 2)
    if torch.cuda.is_available():
        out["cuda_allocated_gib"] = round(torch.cuda.memory_allocated() / (1 << 30), 2)
        out["cuda_reserved_gib"] = round(torch.cuda.memory_reserved() / (1 << 30), 2)
    return out


def sync(dev) -> None:
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--text")
    ap.add_argument("--ref", action="store_true")
    ap.add_argument("--dfly", action="store_true")
    ap.add_argument("--dfly-layers")
    ap.add_argument("--mask-id", type=int)
    ap.add_argument("--fit", type=int, default=0)
    ap.add_argument("--table", default="cold")
    a = ap.parse_args()

    from engine.models.hy4_flash.config import DFlyConfig
    from engine.models.hy4_flash.engine import Hy4FlashEngine
    from engine.models.hy4_flash.load import load
    from engine.models.hy4_flash.serving import Hy4FlashServing

    def say(kind: str, **kw) -> None:
        print(json.dumps({"check": kind, **kw}), flush=True)

    say("start", pack=a.pack, device=a.device, max_len=a.max_len, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), mem=mem())
    pack, W, cuts, tables, banks, timing = load(a.pack, a.device, log=lambda s: print(s, flush=True), table_policy=a.table)
    cfg = pack.cfg
    if cfg.dfly is None and a.dfly_layers and pack.has_dfly is False and "dfly_layers.final_norm.weight" in pack.tensors:
        layers = [int(x) for x in a.dfly_layers.split(",")]
        cfg.dfly = DFlyConfig(layers=len(layers), mask_token_id=int(a.mask_id), index_layers=layers, kv_layers=layers,
                              index_topk=cfg.index_topk, draft_tokens=5)
        pack = type(pack).open(a.pack)
        pack.cfg.dfly = cfg.dfly
        pack.has_dfly, pack.dfly_mlp = True, int(pack.tensors["dfly_layers.0.mlp.gate_proj.svh"].shape[0])
        from engine.models.hy4_flash import load as L
        orig = L.Hy4Pack.open
        L.Hy4Pack.open = classmethod(lambda cls, p: pack)
        pack, W, cuts, tables, banks, timing = load(a.pack, a.device, log=lambda s: None, table_policy=a.table)
        L.Hy4Pack.open = orig
        cfg = pack.cfg
    eng = Hy4FlashEngine(cfg, W, tables, device=a.device, max_len=a.max_len, banks=banks, cuts=cuts)
    say("load", timing=timing, weights_gib=round(pack.resident_bytes() / (1 << 30), 3), banks=len(banks), mem=mem())

    g = torch.Generator().manual_seed(7)
    if a.text:
        from transformers import AutoTokenizer
        ids = AutoTokenizer.from_pretrained(a.pack)(open(a.text).read(), add_special_tokens=False)["input_ids"][:a.prompt]
    else:
        ids = torch.randint(3, min(cfg.vocab_size, cfg.engram_lookup_table_size), (a.prompt,), generator=g).tolist()

    # the gate's self-check: a prompt past the group selection's threshold with the rings wrapped (synthetic rows
    # beyond --prompt), blocks that close groups inside them
    deep = eng.check_rows()
    if deep + 64 > a.max_len:
        say("self_check_refused", why=f"--max-len {a.max_len} cannot hold the {deep} prompt rows the check needs")
        return 1
    long = (ids * (deep // max(len(ids), 1) + 1))[:deep]
    res = eng.self_check(long, steps=12, widths=(2, 4, 6, 12), log=lambda s: print(s, flush=True))
    say("self_check", **res)
    if not (res["covers"]["selection"] and res["covers"]["rings_wrapped"] and res["covers"]["groups_closed_in_blocks"] > 0):
        say("self_check_refused", why="the replay did not reach every read path", covers=res["covers"])
        return 1

    if a.ref:
        from engine.models.hy4_flash.ref import Hy4FlashRef, Hy4State
        ref = Hy4FlashRef(cfg, W, tables, dtype=eng.dtype, device=a.device, cuts=cuts)
        n = min(len(ids), 160)
        want = ref.forward(torch.tensor(ids[:n]), Hy4State())
        st = eng.new_state(n + 8)
        cut = n - 16
        got = torch.cat((eng.prefill(torch.tensor(ids[:cut]), st, head="all"), torch.cat([eng.step(t, st) for t in ids[cut:n]])))
        d = (got - want).abs()
        say("ref", rows=n, max_abs=round(d.max().item(), 4), mean_abs=round(d.mean().item(), 5), logit_scale=round(want.abs().mean().item(), 3),
            agree_prompt=round((got[:cut].argmax(-1) == want[:cut].argmax(-1)).float().mean().item(), 4),
            agree_steps=round((got[cut:].argmax(-1) == want[cut:].argmax(-1)).float().mean().item(), 4))

    srv = Hy4FlashServing(eng, a.max_len)
    t0 = time.perf_counter()
    lg = srv.forward(torch.tensor(ids), start=0)
    sync(a.device)
    t_prompt = time.perf_counter() - t0
    hidden0 = srv.last_streams().clone()
    snap = srv.cache_state().clone()
    first = int(lg[0, 0].argmax())
    toks, nxt = [], first
    t0 = time.perf_counter()
    for i in range(a.steps):
        toks.append(nxt)
        lg = srv.forward(torch.tensor([nxt]), start=len(ids) + i)
        nxt = int(lg[0, 0].argmax())
    sync(a.device)
    dt = time.perf_counter() - t0
    say("decode", prompt_rows=len(ids), prompt_s=round(t_prompt, 2), prompt_tok_s=round(len(ids) / t_prompt, 1), steps=a.steps,
        tok_s=round(a.steps / dt, 2), ms_step=round(1e3 * dt / a.steps, 2), graphs=False, engram_rows=eng.stats["engram_rows"],
        tables=[t.report() for t in tables.values()] if a.table == "cold" else None)

    if a.dfly and cfg.dfly is not None:
        from engine.models.hy4_flash.dfly import DFlyHead
        srv.cache_state().restore(snap)
        srv.kv.length = len(ids)
        srv.st.picks_at = -1
        # the picks of the prompt's last row are gone with the restore: one plain step, as the server does
        out, nxt = [first], None
        lg = srv.forward(torch.tensor([first]), start=len(ids))
        nxt = int(lg[0, 0].argmax())
        head = DFlyHead(srv)
        rounds = kept = 0
        draft_s = 0.0
        t0 = time.perf_counter()
        while len(out) < a.steps:
            td = time.perf_counter()
            head.launch(srv.last_streams(), [nxt])
            tok, _ = head.read()
            sync(a.device)
            draft_s += time.perf_counter() - td
            block = [nxt] + [t[0] for t in tok]
            lg = srv.forward_tree(torch.tensor(block), list(range(-1, len(block) - 1)), start=srv.kv.length)
            picks = srv.picks.tolist()
            n = 1
            while n < len(block) and block[n] == picks[n - 1]:
                n += 1
            srv.commit_tree(list(range(n)))
            out += block[:n]
            nxt, kept, rounds = picks[n - 1], kept + n - 1, rounds + 1
        sync(a.device)
        dt2 = time.perf_counter() - t0
        n_out = len(out) - 1
        say("dfly", equal=out[:a.steps] == toks[:len(out[:a.steps])], rounds=rounds, tokens=n_out, kept_per_round=round(kept / max(rounds, 1), 2),
            tok_s=round(n_out / dt2, 2), draft_ms_round=round(1e3 * draft_s / max(rounds, 1), 2), draft_tokens=cfg.dfly.draft_tokens,
            stale=head.stats["stale"])

    if a.fit:
        del srv
        if str(a.device).startswith("cuda"):
            torch.cuda.empty_cache()
        before = mem()
        srv = Hy4FlashServing(eng, max(a.max_len, a.fit + 64))
        long = torch.randint(3, min(cfg.vocab_size, cfg.engram_lookup_table_size), (a.fit,), generator=g)
        t0 = time.perf_counter()
        low = before.get("MemAvailable", 0.0)
        for i in range(0, a.fit, 4096):
            srv.forward(long[i:i + 4096], start=i)
            low = min(low, mem().get("MemAvailable", 0.0))
        sync(a.device)
        dt = time.perf_counter() - t0
        t1 = time.perf_counter()
        nxt = 5
        for i in range(8):
            lg = srv.forward(torch.tensor([nxt]), start=a.fit + i)
            nxt = int(lg[0, 0].argmax())
        sync(a.device)
        say("fit", rows=a.fit, prefill_s=round(dt, 1), prefill_tok_s=round(a.fit / dt, 1), state_gib=round(srv.st.nbytes() / (1 << 30), 3),
            bytes_per_token=srv.cost().view_bytes_per_token, decode_tok_s_at_end=round(8 / (time.perf_counter() - t1), 2),
            mem_before=before, mem_after=mem(), mem_available_low_gib=low)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
