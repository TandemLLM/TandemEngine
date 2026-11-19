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


def chat_ids(pack_dir: str, text: str, ask: str, effort: str = "no_think") -> tuple:
    """A one-turn chat request rendered by the pack's own template (jinja, as transformers renders it) and its ids."""
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda v, ensure_ascii=False, indent=None: json.dumps(v, ensure_ascii=ensure_ascii, indent=indent)
    env.globals["raise_exception"] = raise_exception
    tpl = env.from_string(open(os.path.join(pack_dir, "chat_template.jinja")).read())
    prompt = tpl.render(messages=[{"role": "user", "content": text + "\n\n" + ask}], add_generation_prompt=True, reasoning_effort=effort, tools=None)
    tok = Tokenizer.from_file(os.path.join(pack_dir, "tokenizer.json"))
    return tok.encode(prompt, add_special_tokens=False).ids, tok


def _kl(ref: torch.Tensor, got: torch.Tensor) -> dict:
    """KL(reference || engine) a position, top-1 agreement and the reference's top-1 inside the engine's top 5."""
    lr, lg = torch.log_softmax(ref.float(), -1), torch.log_softmax(got.float(), -1)
    kl = (lr.exp() * (lr - lg)).sum(-1)
    top = ref.argmax(-1)
    return {"kl": kl.cpu(), "top1": (got.argmax(-1) == top).cpu(), "top5": (got.topk(5, -1).indices == top[:, None]).any(-1).cpu()}


def _kl_say(say, tag: str, parts: list, **more) -> None:
    kl = torch.cat([p["kl"] for p in parts]).double()
    q = lambda f: round(float(kl.quantile(f)), 5)
    say(tag, positions=int(kl.numel()), kl_mean=round(float(kl.mean()), 5), kl_median=q(0.5), kl_p90=q(0.9), kl_p99=q(0.99),
        kl_max=round(float(kl.max()), 4), top1=round(float(torch.cat([p["top1"] for p in parts]).float().mean()), 5),
        ref_top1_in_top5=round(float(torch.cat([p["top5"] for p in parts]).float().mean()), 5), **more)


def quality(a, eng, cfg, say) -> None:
    """The engine against stored reference logits of the unquantized checkpoint (`--quality name=rows.pt=refdir=N`):
    the prompt pass on N rows; then (`--quality-steps R:L`) the decode path on the first R rows, one step a
    position on the kernels through the graphs over the whole row, and over the first L positions the same steps on
    plain torch sums and on the kernels with each row's experts forced to the torch pass's."""
    from engine.models.hy4_flash import engine as E_
    dev = a.device
    R, L = (int(x) for x in a.quality_steps.split(":")) if a.quality_steps else (0, 0)
    for spec in a.quality.split(","):
        name, rows_file, ref_dir, n = spec.split("=")
        ids_all = torch.load(rows_file)["ids"]
        n = min(int(n), len(ids_all))
        ref_of = lambda r: torch.load(os.path.join(ref_dir, f"row{r:04d}.pt")).to(dev)
        parts, t0 = [], time.perf_counter()
        for r in range(n):
            ids = ids_all[r].tolist()
            st = eng.new_state(len(ids) + 8)
            got = eng.prefill(torch.tensor(ids), st, head="all")
            parts.append(_kl(ref_of(r), got))
            if r < R and L and a.quality_no_steps:
                # the decode steps over the first L positions against the prompt pass's rows of the same positions
                st2 = eng.new_state(len(ids) + 8)
                gs, P0_ = {}, 32
                stp = torch.cat([eng.prefill(torch.tensor(ids[:P0_]), st2, head="all")] + [eng.step(ids[i], st2, graphs=gs).clone() for i in range(P0_, L)])
                d = (stp[P0_:] - got[P0_:L]).abs()
                both = _kl(got[P0_:L], stp[P0_:])
                say("quality_steps_vs_prompt_pass", set=name, row=r, positions=L - P0_, logits_mean_abs=round(float(d.mean()), 5),
                    logits_max_abs=round(float(d.max()), 4), logit_scale=round(float(got[P0_:L].abs().mean()), 3),
                    top1_equal=round(float(both["top1"].float().mean()), 5), kl_prompt_to_steps=round(float(both["kl"].mean()), 5),
                    kl_ref_steps=round(float(_kl(ref_of(r)[P0_:L], stp[P0_:])["kl"].mean()), 5),
                    kl_ref_prompt=round(float(parts[-1]["kl"][P0_:L].mean()), 5))
                del st2, stp
            del got, st
        _kl_say(say, "quality_prompt_pass", parts, set=name, rows=n, row_len=len(ids_all[0]), seconds=round(time.perf_counter() - t0, 1),
                kl_mean_a_row=[round(float(p["kl"].mean()), 4) for p in parts])
        if R:
            # the prompt pass on the rows and positions the decode steps are scored on
            P0_ = 32
            _kl_say(say, "quality_prompt_pass_step_rows", [{k: v[P0_:] for k, v in p.items()} for p in parts[:R]], set=name, rows=min(R, n), from_position=P0_)
            if L:
                _kl_say(say, "quality_prompt_pass_step_rows_first", [{k: v[:L] for k, v in p.items()} for p in parts[:R]], set=name, rows=min(R, n), first=L)
        if not (R and name == a.quality.split(",")[0].split("=")[0]) or a.quality_no_steps:
            continue
        P0 = 32                                                      # the rows before the first step come from the prompt pass
        real_router, real_route_r = E_.K_.router, eng.route_r
        k_parts, t_parts, f_parts, kt, ft, flips, pairs = [], [], [], [], [], 0, 0
        for r in range(min(R, n)):
            ids = ids_all[r].tolist()
            ref = ref_of(r)

            def run(upto, graphs):
                st = eng.new_state(len(ids) + 8)
                out = [eng.prefill(torch.tensor(ids[:P0]), st, head="all")]
                for i in range(P0, upto):
                    out.append(eng.step(ids[i], st, graphs=graphs).clone())
                return torch.cat(out)
            t0 = time.perf_counter()
            full = run(len(ids), {})
            say("quality_progress", set=name, row=r, kernel_graph_s=round(time.perf_counter() - t0, 1))
            k_parts.append(_kl(ref, full))
            if not L:
                continue
            # the same steps on plain torch sums, its experts recorded a (step, layer)
            seen = []

            def spy_route_r(x, li, _s=seen):
                idx, wts = real_route_r(x, li)
                _s.append(idx.clone())
                return idx, wts
            eng.row_torch, eng.route_r = True, spy_route_r
            tor = run(L, None)
            eng.row_torch, eng.route_r = False, real_route_r
            # the kernels again (no graphs), each row's experts the torch pass's; weights from the kernel's own scores
            pos = [0]

            def forced(x, w, bias, k, scale, renorm, _s=seen, _p=pos):
                own, _ = real_router(x, w, bias, k, scale, renorm)
                idx = _s[_p[0]].to(torch.int64)
                _p[0] += 1
                nonlocal flips, pairs
                flips += int((own.to(torch.int64).sort(-1).values != idx.sort(-1).values).any())
                pairs += 1
                wts = E_.K_.gemv(x, w, sigmoid=True).gather(1, idx)
                if renorm:
                    wts = wts / (wts.sum(-1, keepdim=True) + 1e-20)
                return idx.to(torch.int32), (wts * scale).float()
            E_.K_.router = forced
            frc = run(L, None)
            E_.K_.router = real_router
            assert pos[0] == len(seen), (pos[0], len(seen))
            t_parts.append(_kl(ref[:L], tor))
            f_parts.append(_kl(ref[:L], frc))
            kt.append((full[P0:L] - tor[P0:L]).abs().mean().item())
            ft.append((frc[P0:L] - tor[P0:L]).abs().mean().item())
            say("quality_progress", set=name, row=r, all_s=round(time.perf_counter() - t0, 1))
        _kl_say(say, "quality_steps_kernel_graphs", [{k: v[P0:] for k, v in p.items()} for p in k_parts], set=name, rows=len(k_parts), from_position=P0)
        if L and t_parts:
            cut = lambda ps: [{k: v[:L] for k, v in p.items()} for p in ps]
            _kl_say(say, "quality_steps_kernel_first", cut(k_parts), set=name, rows=len(k_parts), first=L)
            _kl_say(say, "quality_steps_torch_first", t_parts, set=name, rows=len(t_parts), first=L)
            _kl_say(say, "quality_steps_kernel_forced_routing_first", f_parts, set=name, rows=len(f_parts), first=L)
            say("quality_routing", row_layers=pairs, forced_run_pairs_where_the_kernels_own_pick_differed=flips,
                logits_mean_abs_kernel_vs_torch=round(sum(kt) / len(kt), 5), logits_mean_abs_forced_vs_torch=round(sum(ft) / len(ft), 5))


def _misses(eng) -> int:
    n = 0
    for t in eng.tables.values():
        r = t.report() if hasattr(t, "report") else {}
        n += int(r.get("misses", 0) or 0)
    return n


def prose(a, eng, cfg, say) -> None:
    """One request as a user sees it: real text, a prose answer of --answer tokens, greedy. Tokens a second from
    token 50 on, plain and in drafted rounds (the same text), and the draft's kept tokens a round."""
    from engine.models.hy4_flash.dfly import DFlyHead
    from engine.models.hy4_flash.serving import Hy4FlashServing
    text = open(a.prose).read()
    ids, tok = chat_ids(a.pack, text, "Write a detailed essay of at least 700 words that explains the text above to a newcomer. "
                        "Use plain prose paragraphs only: no lists, no tables, no headings, no code.")
    srv = Hy4FlashServing(eng, a.max_len)
    if a.no_graphs:
        srv.graphs = None
    t0 = time.perf_counter()
    lg = srv.forward(torch.tensor(ids), start=0)
    sync(a.device)
    t_prompt = time.perf_counter() - t0
    snap = srv.cache_state().clone()
    first = int(lg[0, 0].argmax())
    srv.forward(torch.tensor([first]), start=len(ids))            # captures the step's graph: undone
    srv.cache_state().restore(snap)
    srv.kv.length = len(ids)
    toks, nxt, stamps = [], first, []
    host = {"stage": 0.0, "commit": 0.0}
    real_sc = (eng.stage, eng.commit)

    def hosted(fn, key):                                # host time only (no device sync): what the step waits on the CPU
        def f(*x, **k):
            t = time.perf_counter()
            r = fn(*x, **k)
            host[key] += time.perf_counter() - t
            return r
        return f
    eng.stage, eng.commit = hosted(real_sc[0], "stage"), hosted(real_sc[1], "commit")
    miss0 = _misses(eng)
    for i in range(a.answer):
        if nxt in cfg.eos_token_ids:
            break
        toks.append(nxt)
        lg = srv.forward(torch.tensor([nxt]), start=len(ids) + i)
        nxt = int(lg[0, 0].argmax())
        stamps.append(time.perf_counter())
    eng.stage, eng.commit = real_sc
    gaps = sorted(1e3 * (b - a_) for a_, b in zip(stamps, stamps[1:]))
    host_ms = {k: round(1e3 * v / max(len(toks), 1), 3) for k, v in host.items()}
    host_ms.update(table_misses_a_step=round((_misses(eng) - miss0) / max(len(toks), 1), 1),
                   step_ms_median=round(gaps[len(gaps) // 2], 2) if gaps else None, step_ms_p90=round(gaps[int(len(gaps) * 0.9)], 2) if gaps else None,
                   step_ms_max=round(gaps[-1], 1) if gaps else None)
    skip = min(50, max(len(toks) - 2, 0))
    plain = (len(toks) - skip - 1) / max(stamps[-1] - stamps[skip], 1e-9) if len(toks) > skip + 1 else 0.0
    answer = tok.decode(toks, skip_special_tokens=False)
    say("prose_plain", prompt_tokens=len(ids), prompt_s=round(t_prompt, 1), prompt_tok_s=round(len(ids) / t_prompt, 1), answer_tokens=len(toks),
        ended=nxt in cfg.eos_token_ids, tok_s_from_50=round(plain, 2), ms_step=round(1e3 / plain, 2) if plain else None,
        graphs=srv.graphs is not None, host_ms=host_ms, answer_head=answer[:400], answer_tail=answer[-200:])
    if a.step_parts and toks:
        # where a served step's time goes: the same steps again, each part closed by a device sync
        srv.cache_state().restore(snap)
        srv.kv.length = len(ids)
        acc = {"stage": 0.0, "pass": 0.0, "commit": 0.0, "forward_rest": 0.0, "argmax": 0.0}
        real_stage, real_commit, real_rows = eng.stage, eng.commit, eng.rows

        def timed(fn, key):
            def f(*x, **k):
                sync(a.device)
                t = time.perf_counter()
                r = fn(*x, **k)
                sync(a.device)
                acc[key] += time.perf_counter() - t
                return r
            return f
        eng.stage, eng.commit, eng.rows = timed(real_stage, "stage"), timed(real_commit, "commit"), timed(real_rows, "pass")
        n = min(len(toks), a.step_parts)
        for i in range(n):
            sync(a.device)
            t = time.perf_counter()
            lg = srv.forward(torch.tensor([toks[i]]), start=len(ids) + i)
            sync(a.device)
            t1 = time.perf_counter()
            int(lg[0, 0].argmax())
            acc["argmax"] += time.perf_counter() - t1
            acc["forward_rest"] += t1 - t
        eng.stage, eng.commit, eng.rows = real_stage, real_commit, real_rows
        acc["pass"] -= acc["stage"]
        acc["forward_rest"] -= acc["pass"] + acc["stage"] + acc["commit"]
        say("prose_step_parts", steps=n, rows_at_start=len(ids), ms={k: round(1e3 * v / n, 3) for k, v in acc.items()},
            ms_sum=round(1e3 * sum(acc.values()) / n, 2))
    if cfg.dfly is None or not a.dfly:
        return

    def drafted(tag: str) -> None:
        srv.cache_state().restore(snap)
        srv.kv.length = len(ids)
        lg = srv.forward(torch.tensor([first]), start=len(ids))       # the picks are gone with the restore: one plain step
        out, nxt = [first], int(lg[0, 0].argmax())
        head = DFlyHead(srv, graphs=srv.graphs is not None)
        head.launch(srv.last_streams(), [nxt])                        # one round first, undone: it captures its graphs
        tk, _ = head.read()
        if tk:
            srv.forward_tree(torch.tensor([nxt] + [t[0] for t in tk]), list(range(-1, len(tk))), start=srv.kv.length)
            srv.discard_pending()
            srv.st.picks_at = srv.st.length
        rounds = kept = 0
        draft_s, hist, marks = 0.0, [0] * (cfg.dfly.draft_tokens + 1), []
        while len(out) < len(toks):
            td = time.perf_counter()
            head.launch(srv.last_streams(), [nxt])
            tk, _ = head.read()
            draft_s += time.perf_counter() - td
            block = [nxt] + [t[0] for t in tk]
            srv.forward_tree(torch.tensor(block), list(range(-1, len(block) - 1)), start=srv.kv.length)
            picks = srv.picks.tolist()
            n = 1
            while n < len(block) and block[n] == picks[n - 1]:
                n += 1
            srv.commit_tree(list(range(n)))
            out += block[:n]
            nxt, kept, rounds = picks[n - 1], kept + n - 1, rounds + 1
            hist[n - 1] += 1
            marks.append((len(out), time.perf_counter()))
        after = [(n, t) for n, t in marks if n >= 50]
        rate = (after[-1][0] - after[0][0]) / max(after[-1][1] - after[0][1], 1e-9) if len(after) > 1 else 0.0
        say(tag, equal=out[:len(toks)] == toks, rounds=rounds, tokens=len(out), tok_s_from_50=round(rate, 2),
            tokens_per_round=round(len(out) / max(rounds, 1), 3), kept_per_round=round(kept / max(rounds, 1), 3),
            accept_rate=round(kept / max(rounds * cfg.dfly.draft_tokens, 1), 3), kept_histogram=hist,
            ms_round=round(1e3 * (marks[-1][1] - marks[0][1]) / max(len(marks) - 1, 1), 2), draft_ms_round=round(1e3 * draft_s / max(rounds, 1), 2),
            draft_tokens=cfg.dfly.draft_tokens, graphs=srv.graphs is not None)
    drafted("prose_dfly")
    if a.dfly_src:
        n = dfly_source(eng.W, a.dfly_src, a.device)
        say("dfly_source", dir=a.dfly_src, tensors_replaced=n)
        drafted("prose_dfly_unquantized")


def dfly_source(W: dict, src: str, device) -> int:
    """The draft layers' projections from the source checkpoint's own BF16 tensors in place of the pack's (an A/B of
    the drafter's precision; the target's weights are untouched, so the text cannot change)."""
    import struct
    import numpy as np
    wm = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    n, heads = 0, {}
    for key, fn in wm.items():
        if not key.startswith("model.dfly_layers.") or not key.endswith(".weight"):
            continue
        name = key[len("model."):]
        if name not in W or (torch.is_tensor(W[name]) and W[name].dim() < 2):
            continue
        fp = os.path.join(src, fn)
        if fp not in heads:
            with open(fp, "rb") as f:
                hl = struct.unpack("<Q", f.read(8))[0]
                heads[fp] = (8 + hl, json.loads(f.read(hl)))
        start, hdr = heads[fp]
        t = hdr[key]
        b0, b1 = t["data_offsets"]
        arr = np.fromfile(fp, dtype=np.uint8, count=b1 - b0, offset=start + b0)
        W[name] = torch.from_numpy(arr).view(torch.bfloat16).reshape(t["shape"]).to(device)
        n += 1
    return n


def profile(a, eng, cfg, ids, say) -> None:
    """One decode step and one 6-row block taken apart: exclusive wall time a part, launched one by one with a
    device sync around every part (so the parts add up; a graph's replay of the same pass is given beside them)."""
    from engine.models.hy4_flash.dfly import DFlyHead
    from engine.models.hy4_flash.serving import Hy4FlashServing
    parts = {"lin1": "packed projections (EXL3)", "expert_pairs": "experts", "combine": "experts", "route_r": "router",
             "attend_r": "attention", "_gather": "attention (row gather)", "_select": "indexer (scores, top groups)",
             "group_keys_r": "indexer (group keys)", "engram_r": "Engram (arithmetic)", "_engram_staged": "Engram (row decode)",
             "hc_r": "hyper-connections", "inject": "hyper-connections", "final_r": "final norm", "head_rows": "head",
             "rms_r": "norms", "stage": "host: stage and table reads", "commit": "host: commit"}
    acc, stack = {}, []
    saved = {}

    def wrap(name):
        fn = getattr(eng, name)
        saved[name] = fn

        def timed(*args, **kw):
            sync(a.device)
            t0 = time.perf_counter()
            stack.append(0.0)
            try:
                return fn(*args, **kw)
            finally:
                sync(a.device)
                dt = time.perf_counter() - t0
                inner = stack.pop()
                acc[parts[name]] = acc.get(parts[name], 0.0) + dt - inner
                if stack:
                    stack[-1] += dt
        setattr(eng, name, timed)
    srv = Hy4FlashServing(eng, a.max_len)
    srv.forward(torch.tensor(ids), start=0)
    st = srv.st
    graphs = {} if str(a.device).startswith("cuda") else None
    out = {}
    for M in (1, 6):
        toks, par = [int(t) for t in ids[:M]], list(range(-1, M - 1))
        if graphs is not None:
            eng.rows(toks, par, st, graphs=graphs)
            best = 1e9
            for _ in range(5):                                  # the best of 5 trials of 8 replays
                sync(a.device)
                t0 = time.perf_counter()
                for _ in range(8):
                    eng.stage(toks, par, st)
                    graphs[(M, st.length // cfg.index_kpool >= cfg.group_topk)].replay()
                sync(a.device)
                best = min(best, (time.perf_counter() - t0) / 8)
            out[f"replay_ms_{M}"] = round(1e3 * best, 2)
        eng.rows(toks, par, st)
        for name in parts:
            wrap(name)
        acc.clear()
        N = 4
        sync(a.device)
        t0 = time.perf_counter()
        for _ in range(N):
            eng.rows(toks, par, st)
        sync(a.device)
        total = 1e3 * (time.perf_counter() - t0) / N
        for name, fn in saved.items():
            setattr(eng, name, fn)
        split = {k: round(1e3 * v / N, 2) for k, v in sorted(acc.items(), key=lambda kv: -kv[1])}
        split["other (element-wise, embedding)"] = round(total - sum(split.values()), 2)
        out[f"eager_ms_{M}"] = round(total, 2)
        out[f"parts_ms_{M}"] = split
    if graphs is not None:
        # the 6-row block again with its experts on the pair kernels (no grouping by expert): what grouping saves
        was, g2 = eng.block_grouped, {}
        eng.block_grouped = not was
        toks, par = [int(t) for t in ids[:6]], list(range(-1, 5))
        eng.rows(toks, par, st, graphs=g2)
        best = 1e9
        for _ in range(5):
            sync(a.device)
            t0 = time.perf_counter()
            for _ in range(8):
                next(iter(g2.values())).replay()
            sync(a.device)
            best = min(best, (time.perf_counter() - t0) / 8)
        out["replay_ms_6_grouped_off" if was else "replay_ms_6_grouped_on"] = round(1e3 * best, 2)
        eng.block_grouped = was
    if cfg.dfly is not None:
        eng.rows([int(ids[0])], [-1], st)
        eng.commit(st, [0])
        head = DFlyHead(srv)
        hid = st.rb[1]["hidden"][:1].clone()
        head.stage(7, hid, st)
        head.run(st)
        sync(a.device)
        t0 = time.perf_counter()
        for _ in range(5):
            head.run(st)
        sync(a.device)
        out["draft_eager_ms"] = round(1e3 * (time.perf_counter() - t0) / 5, 2)
        if graphs is not None:
            g = head._capture(st)
            sync(a.device)
            t0 = time.perf_counter()
            for _ in range(5):
                g.replay()
            sync(a.device)
            out["draft_replay_ms"] = round(1e3 * (time.perf_counter() - t0) / 5, 2)
    say("profile", context_rows=st.length, **out)


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
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--caches", action="store_true")
    ap.add_argument("--step-parts", type=int, default=0, help="with --prose: this many served steps again, timed by part")
    ap.add_argument("--quality", default="", help="name=rows.pt=refdir=N[,...]: the engine against stored reference logits")
    ap.add_argument("--quality-no-steps", action="store_true", help="score the prompt pass only (the step rows' figures still print)")
    ap.add_argument("--quality-steps", default="", help="R:L: the decode path on the first set's first R rows; torch sums and forced routing on L positions")
    ap.add_argument("--dfly-src", help="the source checkpoint's directory: the prose run again with its unquantized draft layers")
    ap.add_argument("--no-splits", action="store_true", help="the device's rule for every dense split (not the family's table)")
    ap.add_argument("--grouped", action="store_true", help="the prompt's experts on the grouped kernels")
    ap.add_argument("--prose", help="a text file: a chat request to answer it in prose, decode rates from token 50 on")
    ap.add_argument("--answer", type=int, default=600)
    ap.add_argument("--profile", action="store_true", help="one decode step and one drafted round taken apart by part")
    ap.add_argument("--only", help="comma list of checks to run after the load (self,experts,tolerance,decode,dfly,caches,prose,profile,fit)")
    ap.add_argument("--need-gib", type=float, default=0.0, help="refuse to load below this MemAvailable")
    ap.add_argument("--cache-chunk", type=int, default=512)
    ap.add_argument("--skip-tolerance", action="store_true")
    a = ap.parse_args()

    from engine.models.hy4_flash.config import DFlyConfig
    from engine.models.hy4_flash.engine import Hy4FlashEngine
    from engine.models.hy4_flash.load import load
    from engine.models.hy4_flash.serving import Hy4FlashServing

    def say(kind: str, **kw) -> None:
        print(json.dumps({"check": kind, **kw}), flush=True)

    say("start", pack=a.pack, device=a.device, max_len=a.max_len, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), mem=mem())
    if a.need_gib and mem().get("MemAvailable", 0.0) < a.need_gib:
        say("refused", why=f"MemAvailable {mem().get('MemAvailable')} GiB is below the {a.need_gib} GiB this run needs")
        return 3
    pack, W, cuts, tables, banks, timing = load(a.pack, a.device, log=lambda s: print(s, flush=True), table_policy=a.table,
                                                splits=not a.no_splits)
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
    eng.prefill_grouped = a.grouped
    say("load", timing=timing, weights_gib=round(pack.resident_bytes() / (1 << 30), 3), banks=len(banks), mem=mem())

    g = torch.Generator().manual_seed(7)
    if a.text:
        from tokenizers import Tokenizer
        ids = Tokenizer.from_file(os.path.join(a.pack, "tokenizer.json")).encode(open(a.text).read(), add_special_tokens=False).ids[:a.prompt]
        if len(ids) < a.prompt:
            raise SystemExit(f"--text holds {len(ids)} tokens, --prompt asks for {a.prompt}")
    else:
        ids = torch.randint(3, min(cfg.vocab_size, cfg.engram_lookup_table_size), (a.prompt,), generator=g).tolist()

    # the gate's self-check: a prompt past the group selection's threshold with the rings wrapped (synthetic rows
    # beyond --prompt), blocks that close groups inside them
    deep = eng.check_rows()
    if deep + 64 > a.max_len:
        say("self_check_refused", why=f"--max-len {a.max_len} cannot hold the {deep} prompt rows the check needs")
        return 1
    long = (ids * (deep // max(len(ids), 1) + 1))[:deep]
    only = set(a.only.split(",")) if a.only else None

    def want(name):
        return only is None or name in only
    res = {"ok": True}
    if want("self"):
        res = eng.self_check(long, steps=12, widths=(2, 4, 6, 12), log=lambda s: print(s, flush=True))
        say("self_check", **res)
    if want("self") and str(a.device).startswith("cuda") and not a.no_graphs:
        # the same replay through the captured graphs against the plain pass's recorded rows
        rg = eng.self_check(long, steps=12, widths=(1, 2, 4, 6, 12), log=lambda s: print(s, flush=True), graphs=True)
        say("self_check_graphs", **rg)
        res["ok"] = res["ok"] and rg["ok"]
    if want("self") and not (res["covers"]["selection"] and res["covers"]["rings_wrapped"] and res["covers"]["groups_closed_in_blocks"] > 0):
        say("self_check_refused", why="the replay did not reach every read path", covers=res["covers"])
        return 1

    if banks and want("experts"):
        # the prompt's experts: 32-row batches of the pair kernels, one launch of 256 rows, the grouped kernels
        li = min(banks)
        x = (torch.randn(256, cfg.hidden_size, generator=g) * 0.5).to(a.device, eng.dtype)
        idx, wts = eng.route(x, li)
        z = eng.lin(x, W[f"layers.{li}.mlp.routed_expert_down_proj.weight"])

        def experts(B, grouped):
            for _ in range(2):
                sync(a.device)
                t0 = time.perf_counter()
                out = torch.cat([eng.combine(eng.expert_pairs(z[i:i + B], idx[i:i + B], li, grouped=grouped), idx[i:i + B], wts[i:i + B])
                                 for i in range(0, 256, B)])
                sync(a.device)
            return out, round(1e3 * (time.perf_counter() - t0), 2)
        (b32, t32), (b256, t256), (bg, tg) = experts(32, False), experts(256, False), experts(256, True)
        say("prefill_experts", layer=li, rows=256, equal_256=torch.equal(b32, b256), equal_grouped=torch.equal(b32, bg),
            ms_32=t32, ms_256=t256, ms_grouped=tg)
        res_ok_experts = torch.equal(b32, b256)
    if str(a.device).startswith("cuda") and not a.skip_tolerance and want("tolerance"):
        # the kernel row pass against the same pass on plain torch sums (HY4_ROW_TORCH), teacher-forced on the same
        # rows: how far the kernel's summation order moves a logit (a tolerance, not an equality)
        n0 = min(len(ids), deep)
        outs = {}
        picks, real_pairs = {}, eng.expert_pairs
        from engine.models.hy4_flash import engine as E_
        import torch.nn.functional as F_
        scores, real_router, real_route_r = {}, E_.K_.router, eng.route_r
        gate_dtype = None

        def scored(x, w, bias):
            # the router's biased scores of these rows in one fixed torch order (fp32), whichever path asked
            return (F_.linear(x.float(), w.float()).sigmoid() + bias.float()).cpu()
        for name, flag in (("kernel", False), ("torch", True)):
            eng.row_torch = flag
            seen = picks[name] = []
            sc_seen = scores[name] = []

            def spy_router(x, w, bias, k, scale, renorm, _s=sc_seen):
                _s.append(scored(x, w, bias))
                return real_router(x, w, bias, k, scale, renorm)

            def spy_route_r(x, li, _s=sc_seen):
                _s.append(scored(x, eng.w(f"layers.{li}.mlp.gate.weight"), eng.w(f"layers.{li}.mlp.gate.e_score_correction_bias")))
                return real_route_r(x, li)
            E_.K_.router, eng.route_r = spy_router, spy_route_r

            def spy(z, idx, li, grouped=False, _seen=seen):
                _seen.append(idx.to(torch.int64).sort(dim=-1).values.clone())
                return real_pairs(z, idx, li, grouped=grouped)
            eng.expert_pairs = spy
            st = eng.new_state(n0 + 40)
            eng.prefill(torch.tensor(ids[:n0]), st, head=None)
            del seen[:]
            del sc_seen[:]
            lgs = [eng.rows(ids[i:i + 4] if len(ids) >= i + 4 else [5, 6, 7, 8], [-1, 0, 1, 2], st)[0] for i in (0, 4, 8)]
            outs[name] = torch.cat(lgs)
        eng.row_torch = False
        eng.expert_pairs = real_pairs
        E_.K_.router, eng.route_r = real_router, real_route_r
        # how often the two paths route a row to another expert: a near-tie at the 16th place flips on the last bit
        pk, pt = torch.cat(picks["kernel"]), torch.cat(picks["torch"])      # rows in (block, layer, row) order in both
        diff = (pk != pt).any(dim=-1)
        say("rows_routing", row_layers=int(diff.numel()), same_experts=int((~diff).sum()), differing=int(diff.sum()),
            first_differing=int(diff.nonzero()[0]) if bool(diff.any()) else None, moe_layers=len(banks))
        if sum(len(x) for x in scores["kernel"]) == sum(len(x) for x in scores["torch"]) == diff.numel() and bool(diff.any()):
            # at each (row, layer): the gap between the 16th and 17th score against how far the two paths' scores
            # of that row are apart (their inputs differ by the layers before). A gap below that distance flips.
            sk, stc = torch.cat(scores["kernel"]), torch.cat(scores["torch"])
            kk = cfg.num_experts_per_tok
            top = sk.sort(dim=-1, descending=True).values
            gap = top[:, kk - 1] - top[:, kk]
            dist = (sk - stc).abs().amax(dim=-1)
            own = torch.equal(sk.topk(kk, dim=-1).indices.sort(dim=-1).values, pk.cpu())
            f = int(diff.nonzero()[0])
            dc = diff.cpu()
            say("rows_routing_margin", first=f, first_gap=float(gap[f]), first_score_distance=float(dist[f]),
                first_before=int(dc[:f].sum()), kernel_picks_equal_torch_topk_on_its_own_input=own,
                differing_with_gap_below_distance=int((gap[dc] <= dist[dc]).sum()),
                same_with_gap_below_distance=int((gap[~dc] <= dist[~dc]).sum()),
                median_gap=float(gap.median()), median_distance=float(dist.median()),
                median_distance_before_first=float(dist[:f].median()) if f else None,
                gate_weight_dtype=str(eng.w("layers.1.mlp.gate.weight").dtype))
        else:
            say("rows_routing_margin", skipped=[sum(len(x) for x in scores[n]) for n in ("kernel", "torch")], rows=int(diff.numel()))
        d = (outs["kernel"] - outs["torch"]).abs()
        say("rows_tolerance", rows=int(d.shape[0]), max_abs=round(d.max().item(), 5), mean_abs=round(d.mean().item(), 6),
            logit_scale=round(outs["torch"].abs().mean().item(), 3),
            agree=round((outs["kernel"].argmax(-1) == outs["torch"].argmax(-1)).float().mean().item(), 4))
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

    if a.quality and want("quality"):
        quality(a, eng, cfg, say)
    if a.prose and want("prose"):
        prose(a, eng, cfg, say)
    if a.profile and want("profile"):
        profile(a, eng, cfg, ids, say)
    if not (want("decode") or want("dfly") or want("caches") or want("fit")):
        return 0 if res["ok"] else 1
    srv = Hy4FlashServing(eng, a.max_len)
    if a.no_graphs:
        srv.graphs = None
    t0 = time.perf_counter()
    lg = srv.forward(torch.tensor(ids), start=0)
    sync(a.device)
    t_prompt = time.perf_counter() - t0
    hidden0 = srv.last_streams().clone()
    snap = srv.cache_state().clone()
    first = int(lg[0, 0].argmax())
    # the first step captures the decode graph: outside the timed steps
    srv.forward(torch.tensor([first]), start=len(ids))
    srv.cache_state().restore(snap)
    srv.kv.length = len(ids)
    toks, nxt = [], first
    stage_s, t0 = 0.0, time.perf_counter()
    for i in range(a.steps):
        toks.append(nxt)
        lg = srv.forward(torch.tensor([nxt]), start=len(ids) + i)
        nxt = int(lg[0, 0].argmax())
    sync(a.device)
    dt = time.perf_counter() - t0
    say("decode", prompt_rows=len(ids), prompt_s=round(t_prompt, 2), prompt_tok_s=round(len(ids) / t_prompt, 1), steps=a.steps,
        tok_s=round(a.steps / dt, 2), ms_step=round(1e3 * dt / a.steps, 2), ms_layer=round(1e3 * dt / a.steps / cfg.num_hidden_layers, 3),
        graphs=srv.graphs is not None, engram_rows=eng.stats["engram_rows"],
        tables=[t.report() for t in tables.values()] if a.table == "cold" else None)

    if a.dfly and cfg.dfly is not None and want("dfly"):
        from engine.models.hy4_flash.dfly import DFlyHead
        srv.cache_state().restore(snap)
        srv.kv.length = len(ids)
        srv.st.picks_at = -1
        # the picks of the prompt's last row are gone with the restore: one plain step, as the server does
        out, nxt = [first], None
        lg = srv.forward(torch.tensor([first]), start=len(ids))
        nxt = int(lg[0, 0].argmax())
        head = DFlyHead(srv, graphs=srv.graphs is not None)
        # one round first, undone: it captures the draft graph and the block's graph
        head.launch(srv.last_streams(), [nxt])
        tok, _ = head.read()
        if tok:
            srv.forward_tree(torch.tensor([nxt] + [t[0] for t in tok]), list(range(-1, len(tok))), start=srv.kv.length)
            srv.discard_pending()
            srv.st.picks_at = srv.st.length
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
            accept_rate=round(kept / max(rounds * cfg.dfly.draft_tokens, 1), 3), tokens_per_round=round(n_out / max(rounds, 1), 2),
            tok_s=round(n_out / dt2, 2), ms_round=round(1e3 * dt2 / max(rounds, 1), 2), draft_ms_round=round(1e3 * draft_s / max(rounds, 1), 2),
            draft_tokens=cfg.dfly.draft_tokens, graphs=srv.graphs is not None,
            stale=head.stats["stale"])

    if a.caches and want("caches"):
        # the cache tiers with this family's state on the device: each gives back a cold prefill's bits
        import shutil
        import tempfile
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
        from disk_tier_util import waiting
        from engine import cache, disk_cache, spill
        CH, BIG = a.cache_chunk, 1 << 34
        n1 = (len(ids) // CH) * CH + CH // 3
        turn1 = (ids * (n1 // max(len(ids), 1) + 1))[:n1]
        more = [int(t) for t in torch.randint(3, min(cfg.vocab_size, cfg.engram_lookup_table_size), (CH // 2,), generator=g)]

        def fresh():
            s2 = Hy4FlashServing(eng, a.max_len)
            if a.no_graphs:
                s2.graphs = None
            return s2

        def same(x, y, n):
            ok = x.kv.length == y.kv.length == n
            for (k, u), v in zip(x.cache_state().sections().items(), y.cache_state().sections().values()):
                ok = ok and torch.equal(u, v)
            for (k, u, ax), (_, v, _) in zip(x.cache_state().views(), y.cache_state().views()):
                ok = ok and torch.equal(u.narrow(ax, 0, n), v.narrow(ax, 0, n))
            return bool(ok)
        t0 = time.perf_counter()
        cold, res_ = fresh(), {}
        lg_cold, _, _ = cache.prefill(cold, None, turn1 + more, a.device, chunk=CH)
        step_cold = cold.forward(torch.tensor([7]), start=len(turn1) + len(more)).clone()
        # a snapshot, the rows after it overwritten, restored
        s1 = fresh()
        cache.prefill(s1, None, turn1, a.device, chunk=CH)
        snap = cache.capture(s1)
        s1.forward(torch.tensor(more), start=len(turn1))
        cache.restore(s1, snap)
        lg = s1.forward(torch.tensor(more[:1]), start=len(turn1))
        ref1 = fresh()
        cache.prefill(ref1, None, turn1, a.device, chunk=CH)
        res_["snapshot"] = bool(torch.equal(lg, ref1.forward(torch.tensor(more[:1]), start=len(turn1))))
        # the resident prefix: the second turn resumes at the last anchor
        s2, resd = fresh(), cache.ResidentPrefix(BIG, CH, stash_bytes=BIG)
        cache.prefill(s2, None, turn1, a.device, chunk=CH, resident=resd)
        lg2, reused, _ = cache.prefill(s2, None, turn1 + more, a.device, chunk=CH, resident=resd)
        res_["resident"] = bool(reused == (len(turn1) // CH) * CH and torch.equal(lg2, lg_cold))
        res_["resident_step"] = bool(torch.equal(s2.forward(torch.tensor([7]), start=len(turn1) + len(more)), step_cold))
        d = tempfile.mkdtemp(prefix="hy4-cache-")
        try:
            def tier():
                return waiting(disk_cache.DiskTier(d, chunk=CH, fingerprint="hy4-check", settings={"admit_after": 1, "budget_gb": 8},
                                                   free_bytes=lambda p_: 10 ** 15, head_anchors=2, min_tokens=CH).open())
            t = tier()
            cache.prefill(fresh(), None, turn1, a.device, chunk=CH, disk=t)
            res_["disk_flush"] = bool(t.flush())
            t.close()
            t2, info, s3 = tier(), {}, fresh()
            lg3, reused3, _ = cache.prefill(s3, None, turn1 + more, a.device, chunk=CH, disk=t2, info=info)
            res_["disk"] = bool(info.get("kind") == "disk" and reused3 == (len(turn1) // CH) * CH and torch.equal(lg3, lg_cold))
            res_["disk_step"] = bool(torch.equal(s3.forward(torch.tensor([7]), start=len(turn1) + len(more)), step_cold))
            sp = spill.SessionSpill(os.path.join(d, "sessions"), chunk=CH, fingerprint="hy4-check", tier=t2,
                                    settings={"budget_gb": 8.0, "min_tokens": 64, "drain_min_tokens": CH, "codec": "off"},
                                    free_bytes=lambda p_: 10 ** 15, floor_bytes=0).open()
            s4, r4 = fresh(), cache.ResidentPrefix(BIG, CH, stash_bytes=BIG, end=True)
            cache.prefill(s4, None, turn1, a.device, chunk=CH, resident=r4)
            saved = sp.save_resident(r4, s4, None, "evict")
            ok4 = saved is not None
            if ok4:
                r4.clear()
                for _, v, _ in s4.cache_state().views():
                    v.zero_()
                ok4 = sp.restore(r4, s4, None, sp.entries[saved["eid"]]) is not None
                ok4 = ok4 and all(torch.equal(u.narrow(ax, 0, len(turn1)), v.narrow(ax, 0, len(turn1)))
                                  for (_, u, ax), (_, v, _) in zip(s4.cache_state().views(), ref1.cache_state().views()))
            res_["spill"] = bool(ok4)
            t2.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)
        sync(a.device)
        say("caches", chunk=CH, rows=len(turn1), more=len(more), seconds=round(time.perf_counter() - t0, 1), **res_)
        del cold, s1, s2, s3, s4, ref1
    if a.fit and want("fit"):
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
            if (i // 4096) % 8 == 7:
                sync(a.device)
                say("fit_progress", rows=i + 4096, seconds=round(time.perf_counter() - t0, 1), tok_s=round((i + 4096) / (time.perf_counter() - t0), 1),
                    mem_available_gib=mem().get("MemAvailable"))
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
