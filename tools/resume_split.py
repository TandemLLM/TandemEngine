#!/usr/bin/env python3
"""A prompt read cold against the same prompt read in two pieces on one state (R83, the rowinv design's G2).

    python tools/resume_split.py --pack DIR --out DIR [--ids FILE] [--text FILE --rows N] [--ks auto|K,K..]
                                 [--arms v2|all|A,B] [--via prefill,forward] [--steps 64] [--ops]

For each prompt (`--ids`: a JSON list of token ids or {"l2_ids": [...]}; `--text`: the first --rows ids of a text
file in the pack's tokenizer), each arm (the prompt pass's knobs), each path and each k: the prompt in one call
against ids[:k] then ids[k:] on one state. The paths:

  prefill   the engine's `prefill` (a one-row piece is a one-row prompt chunk)
  forward   the server's verb, `Hy4FlashServing.forward`, as the server's prompt read calls it: with
            `prompt=True` where the family sets `Capabilities.prompt_rows` (a one-row piece is then a one-row prompt
            chunk), else a one-row piece at a start past 0 is a decode step through the captured graphs: what a
            request does

Compared: every section and view of the state by sha256 (the first layer whose latents differ), the last row's
logits (sha, largest difference, argmax) and --steps greedy tokens from each state (the first that differs). With
--ops (prefill only) every recorded op's output on rows k, k + 1, k + 2, n - 2 and n - 1 in layer order, and the first
op that differs. Each line names the kernel class each piece's products take on the tree's `lin`.

`--ks auto`: the design's list, chosen so both pieces hit each class: 1, 31, 32, 33, 127, 128, 129, 211, 349, n - 129,
n - 128, n - 33, n - 32, n - 1, and 4,095, 4,096, 4,097 for a prompt past them. On the 423-row run A prompt that holds
the diagnosis's cuts, [349, 74] and [390, 33].

One JSON line a case in OUT/split.jsonl; OUT/record.json is the record in tools/family_check.py's shape (pin "g2":
the cases by name, each its verdict and what differs), for a compare of the tree before with the tree after.

Arms: v2 (the served pass: HY4_PREFILL_V2=1, experts "group", grouped), and the passes before (v1-group, v1-grouped,
v1-pair). The family is Hy4-flash; a later family adds its own loader here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

#: arm -> (prefill_v2, prefill_experts, prefill_grouped)
ARMS = {"v2": (True, "group", True), "v1-group": (False, "group", True), "v1-grouped": (False, "rows", True),
        "v1-pair": (False, "rows", False)}
VIAS = ("prefill", "forward")
#: the ops --ops records (the engine's methods, wrapped on the instance; the V2 pass's packed products are
#: `lin_prompt` and `_dense_v2` since R83)
OPS = ("lin", "lin_prompt", "_dense_v2", "engram_r", "engram_apply", "_hc_v2", "hc", "_attention_chunk_v2", "attention_chunk", "_mla",
       "attend_chunk", "_index_chunk_v2", "_index_chunk", "_sparse_positions_chunk", "_mlp_chunk_v2", "mlp_chunk",
       "_experts_chunk", "_experts_grouped", "_final_v2", "final", "_inject_v2", "inject", "_norm_v2", "_heads_mm",
       "group_keys_r", "rms")


def sha(t: torch.Tensor) -> str:
    t = t.detach().contiguous().cpu()
    raw = t.numpy().tobytes() if t.dtype == torch.bool else t.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(repr(tuple(t.shape)).encode() + raw).hexdigest()[:16]


def ks_auto(n: int) -> list[int]:
    ks = {1, 31, 32, 33, 127, 128, 129, 211, 349, n - 129, n - 128, n - 33, n - 32, n - 1}
    if n > 4097:
        ks |= {4095, 4096, 4097}
    return sorted(k for k in ks if 1 <= k <= n - 1)


def piece_class(rows: int, start: int, via: str, chunk: int = 4096, block: int = 32, until: int | None = None,
                prompt_rows: bool = False) -> str:
    """The kernel class a piece's trunk products take on this tree's `lin` (by its chunks' rows), or on the prompt
    order (`prompt_rows`: the family's prompt pass on tools/exl3_rows.py)."""
    if prompt_rows:
        return "prompt order (S1 w8)"
    if via == "forward" and rows == 1 and start > 0 and not prompt_rows:
        return "decode step (row pass)"
    if until is None:
        from tools.exl3_linear import CHUNK_UNTIL as until
    last = rows % chunk or chunk
    name = lambda m: "row kernel" if m <= block else ("skinny" if m <= until else "slab")  # noqa: E731
    return name(last) if rows <= chunk else f"slab x{rows // chunk}, last chunk {name(last)}"


# ------------------------------------------------------------------------------------------ the state
def engine_state(eng, st) -> dict:
    """The engine state's tensors by sha, each layer's latents apart (`prefill` path)."""
    L, rr = st.length, eng.cfg.index_kpool
    out = {"lat_all": sha(st.lat_all[:, :L]), "ik_all": sha(st.ik_all[:, :L]), "win_all": sha(st.win_all),
           "hist": sha(st.hist), "gk_all": sha(st.gk_all[:, :L // rr])}
    out.update({f"layer:lat.{i}": sha(st.lat_all[i, :L]) for i in range(st.lat_all.shape[0])})
    return out


def serving_state(srv) -> dict:
    """The server's view of the state by sha (`forward` path): its sections and its views up to the length."""
    L = srv.st.length
    out = {k: sha(v) for k, v in srv.state.sections().items()}
    for name, t, axis in srv.state.views():
        out[f"view.{name}"] = sha(t.narrow(axis, 0, L))
        if axis == 1:
            out.update({f"layer:{name}.{i}": sha(t[i, :L]) for i in range(t.shape[0])})
    return out


def first_layer(a: dict, b: dict, prefix: str = "layer:lat.") -> int | None:
    """The first layer whose latents differ (the per-layer keys `layer:lat.<i>`)."""
    bad = sorted(int(k[len(prefix):]) for k in a if k.startswith(prefix) and a[k] != b.get(k))
    return bad[0] if bad else None


# ------------------------------------------------------------------------------------------ the runs
class Ops:
    """`--ops`: every OPS method's output rows `sel` (absolute rows) during a prefill, keyed by (op, layer, call)."""

    def __init__(self, eng):
        self.eng, self.on, self.sel, self.store, self.order = eng, False, set(), {}, {}
        self.L0 = self.T = 0
        self.li, self.cnt = -1, {}
        real_fc = eng.forward_chunk

        def fc(tokens, st, _r=real_fc):
            # a chunk starts at layer 0: an op without a layer argument (lin) belongs to the last layer an op named,
            # so the cold run's keys and the resumed run's are the same
            self.L0, self.T, self.cnt, self.li = int(st.length), int(tokens.shape[0]), {}, 0
            return _r(tokens, st)
        eng.forward_chunk = fc
        for name in OPS:
            if hasattr(eng, name):
                setattr(eng, name, self._wrap(name, getattr(eng, name)))

    def _wrap(self, name, real):
        def w(*a, **k):
            o = real(*a, **k)
            if not self.on:
                return o
            li = next((x for x in a if isinstance(x, int) and not isinstance(x, bool)), None)
            if li is None:
                s = next((x for x in a if isinstance(x, str) and x.startswith("layers.")), None)
                li = int(s.split(".")[1]) if s else self.li
            self.li = li
            t = o[0] if isinstance(o, tuple) else o
            if not torch.is_tensor(t) or t.dim() == 0 or t.shape[0] != self.T:
                return o
            key = (name, li, self.cnt.get((name, li), 0))
            self.cnt[(name, li)] = key[2] + 1
            for r in self.sel:
                j = r - self.L0
                if 0 <= j < self.T:
                    self.store[(key, r)] = t[j].detach().float().cpu().clone()
                    self.order.setdefault(key, len(self.order))
            return o
        return w

    def start(self, sel) -> None:
        self.on, self.sel, self.store, self.order = True, set(sel), {}, {}

    def stop(self) -> tuple:
        self.on = False
        return self.store, self.order


def first_op(cold: tuple, res: tuple) -> tuple:
    (cs, order), (rs, _) = cold, res
    first, n = None, 0
    for key in sorted(order, key=order.get):
        rows = sorted({r for (k, r) in cs if k == key})
        for r in rows:
            a, b = cs.get((key, r)), rs.get((key, r))
            if a is None or b is None or a.shape != b.shape:
                continue
            if not torch.equal(a, b):
                n += 1
                if first is None:
                    first = {"op": key[0], "layer": key[1], "call": key[2], "row": r,
                             "max_abs": float((a - b).abs().max()), "rel": float((a - b).abs().max() / max(float(a.abs().max()), 1e-30))}
    return first, n


def run_prefill(eng, ids: list, parts: list, cap: int) -> tuple:
    st = eng.new_state(cap)
    lg = None
    for a, b in parts:
        lg = eng.prefill(torch.tensor(ids[a:b]), st, head="last")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return st, lg.float().reshape(1, -1).clone(), engine_state(eng, st)


def greedy_prefill(eng, st, lg, steps: int) -> list:
    toks = []
    for _ in range(steps):
        t = int(lg[0].argmax())
        toks.append(t)
        lg = eng.step(t, st).float().reshape(1, -1)
    return toks


def run_forward(srv, ids: list, parts: list) -> tuple:
    """The prompt as the server reads it (engine/cache.py `_prefill`): `forward` piece by piece, with `prompt=True`
    where the family sets `Capabilities.prompt_rows` (a one-row piece then a prompt chunk, else a decode step)."""
    srv.restart()
    kw = {"prompt": True} if getattr(getattr(srv, "caps", None), "prompt_rows", False) else {}
    lg = None
    for a, b in parts:
        lg = srv.forward(torch.tensor(ids[a:b]), start=a, **kw)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # the logits are the row buffers' (the next step writes them): a copy
    return lg.float().reshape(1, -1).clone(), serving_state(srv)


def greedy_forward(srv, lg, steps: int) -> list:
    toks = []
    for _ in range(steps):
        t = int(lg[0].argmax())
        toks.append(t)
        lg = srv.forward(torch.tensor([t]), start=srv.st.length).float().reshape(1, -1)
    return toks


def case(name, prompt, n, arm, via, k, cold, res, extra=None) -> dict:
    (lg_c, s_c, d_c), (lg_r, s_r, d_r) = cold, res
    differing = sorted(x for x in s_c if s_c[x] != s_r.get(x) and not x.startswith("layer:"))
    fd = next((i for i, (p, q) in enumerate(zip(d_c, d_r)) if p != q), None)
    pr = bool((extra or {}).pop("prompt_rows", False))
    out = {"case": name, "prompt": prompt, "n": n, "arm": arm, "via": via, "k": k, "pieces": [k, n - k],
           "piece_classes": [piece_class(k, 0, via, prompt_rows=pr), piece_class(n - k, k, via, prompt_rows=pr)],
           "logits_equal": sha(lg_c) == sha(lg_r), "logits_max_abs": float((lg_c - lg_r).abs().max()),
           "argmax_equal": int(lg_c.argmax()) == int(lg_r.argmax()), "state_differing": differing,
           "first_lat_layer": first_layer(s_c, s_r),
           "decode_first_diff": fd, "decode_steps": len(d_c), "decode_cold": d_c[:12], "decode_resumed": d_r[:12]}
    out["equal"] = out["logits_equal"] and not differing and fd is None
    out.update(extra or {})
    return out


def run(eng, srv, prompts: dict, arms: list, vias: list, ks_of, steps: int, say, ops: bool = False) -> dict:
    """Every (prompt, arm, path, k) case -> {name: line}."""
    rec = Ops(eng) if ops else None
    cases = {}
    for pname, ids in prompts.items():
        n = len(ids)
        cap = n + steps + 64
        ks = ks_of(n)
        for arm in arms:
            eng.prefill_v2, eng.prefill_experts, eng.prefill_grouped = ARMS[arm]
            for via in vias:
                t0 = time.perf_counter()
                if via == "prefill":
                    if rec:
                        rec.start({kk + d for kk in ks for d in (0, 1, 2) if kk + d < n} | {n - 2, n - 1})
                    st, lg, s = run_prefill(eng, ids, [(0, n)], cap)
                    cold_ops = rec.stop() if rec else None
                    cold = (lg, s, greedy_prefill(eng, st, lg, steps))
                    del st
                else:
                    lg, s = run_forward(srv, ids, [(0, n)])
                    cold = (lg, s, greedy_forward(srv, lg, steps))
                for k in ks:
                    extra = {}
                    if via == "prefill":
                        if rec:
                            rec.start({k, k + 1, k + 2, n - 2, n - 1})
                        st, lg, s = run_prefill(eng, ids, [(0, k), (k, n)], cap)
                        if rec:
                            fo, nd = first_op(cold_ops, rec.stop())
                            extra = {"first_op": fo, "op_rows_differing": nd}
                        res = (lg, s, greedy_prefill(eng, st, lg, steps))
                        del st
                    else:
                        lg, s = run_forward(srv, ids, [(0, k), (k, n)])
                        res = (lg, s, greedy_forward(srv, lg, steps))
                    name = f"{pname}.{arm}.{via}.k{k}"
                    extra["prompt_rows"] = bool(getattr(getattr(srv, "caps", None), "prompt_rows", False)) and arm == "v2"
                    line = case(name, pname, n, arm, via, k, cold, res, extra)
                    cases[name] = line
                    say(line)
                say({"case": f"{pname}.{arm}.{via}", "seconds": round(time.perf_counter() - t0, 1), "ks": ks,
                     "equal": sum(cases[f"{pname}.{arm}.{via}.k{k}"]["equal"] for k in ks), "of": len(ks)})
    return cases


def record(cases: dict, family: str, source: str) -> dict:
    """The run in tools/family_check.py's record shape (pin "g2"): a case's verdict and what differed."""
    rev = None
    try:
        rev = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    keep = ("equal", "logits_equal", "argmax_equal", "state_differing", "first_lat_layer", "decode_first_diff")
    return {"pin": "g2", "family": family, "source": source, "commit": rev, "host": f"{platform.machine()}-{sys.platform}",
            "torch": torch.__version__, "cases": {k: {x: v[x] for x in keep} for k, v in sorted(cases.items())}}


# ------------------------------------------------------------------------------------------ the command
def _ids(path: str) -> list:
    with open(path) as f:
        d = json.load(f)
    return [int(x) for x in (d["l2_ids"] if isinstance(d, dict) else d)]


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", help="a JSON list of token ids, or {'l2_ids': [...]}")
    ap.add_argument("--text", help="a text file: its first --rows ids in the pack's tokenizer")
    ap.add_argument("--rows", type=int, default=4500)
    ap.add_argument("--ks", default="auto")
    ap.add_argument("--arms", default="v2", help="v2, all, or a comma list of " + ",".join(ARMS))
    ap.add_argument("--via", default="prefill,forward")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--ops", action="store_true", help="the first op that differs (the prefill path)")
    ap.add_argument("--table", default="cold")
    a = ap.parse_args(argv)
    arms = list(ARMS) if a.arms == "all" else a.arms.split(",")
    vias = a.via.split(",")
    if set(arms) - set(ARMS) or set(vias) - set(VIAS):
        raise SystemExit(f"arms {arms}, via {vias}: arms from {list(ARMS)}, via from {list(VIAS)}")
    prompts = {}
    if a.ids:
        prompts["ids"] = _ids(a.ids)
    if a.text:
        from tokenizers import Tokenizer
        ids = Tokenizer.from_file(os.path.join(a.pack, "tokenizer.json")).encode(open(a.text).read(), add_special_tokens=False).ids
        if len(ids) < a.rows:
            raise SystemExit(f"--text holds {len(ids)} tokens, --rows asks for {a.rows}")
        prompts["text"] = ids[:a.rows]
    if not prompts:
        raise SystemExit("--ids or --text")
    ks_of = ks_auto if a.ks == "auto" else (lambda n, ks=[int(x) for x in a.ks.split(",")]: [k for k in ks if 1 <= k < n])
    os.makedirs(a.out, exist_ok=True)
    sink = open(os.path.join(a.out, "split.jsonl"), "a")

    def say(line: dict) -> None:
        s = json.dumps(line)
        print(s, flush=True)
        sink.write(s + "\n")
        sink.flush()

    from engine.models.hy4_flash.engine import Hy4FlashEngine
    from engine.models.hy4_flash.load import load
    from engine.models.hy4_flash.serving import Hy4FlashServing
    from tools import exl3_linear as X
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pack, W, cuts, tables, banks, timing = load(a.pack, dev, log=lambda s: None, table_policy=a.table)
    cap = max(len(v) for v in prompts.values()) + a.steps + 64
    eng = Hy4FlashEngine(pack.cfg, W, tables, device=dev, max_len=cap, banks=banks, cuts=cuts)
    srv = Hy4FlashServing(eng, cap)
    say({"case": "load", "pack": a.pack, "prompts": {k: len(v) for k, v in prompts.items()}, "arms": arms, "via": vias,
         "steps": a.steps, "graphs": srv.graphs is not None, "fused": bool(eng.fused), "experts_min": eng.experts_min,
         "chunk_until": X.CHUNK_UNTIL, "rowgrid": X.ROWGRID, "torch": torch.__version__, "load": timing.get("total_s")})
    cases = run(eng, srv, prompts, arms, vias, ks_of, a.steps, say, ops=a.ops)
    rec = record(cases, "hy4_flash", os.path.basename(os.path.normpath(a.pack)))
    with open(os.path.join(a.out, "record.json"), "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    eq = sum(c["equal"] for c in cases.values())
    say({"case": "done", "equal": eq, "of": len(cases), "record": os.path.join(a.out, "record.json")})
    return 0 if eq == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
