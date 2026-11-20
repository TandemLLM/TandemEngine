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
  prefill-digest  (--only) the prompt pass's record: digests of the last row's logits, the last chunk's head inputs
               and the state, for every expert path of --digest-arms (all equal), against a recorded file
  experts-equal  (--only) every MoE layer's routed experts on the pair, grouped and shared prefill kernels, on real
               inputs of --digest-text, cut row counts: equal pair outputs and combined rows (SPD-321)
  experts-crossover  (--only) the three paths' ms at 16 to 1,024 rows and the EXPERTS_MIN they give (SPD-321)
  table-rows   (--only) the last chunk's table rows: the sliced reader against the threads it replaces, the prompt
               pass's reader job against `engram_rows`, its reads a second (SPD-319)
  v2-micro     (--only) the prompt pass on fixed-order kernels: `mla_rows` on a window and a sparse layer's real inputs
               at --micro-ctx (ms, against its reference, a row alone), one hyper-connection of 4,096 rows (SPD-323/325)
  v2-cuts      (--only) the same pass with the trunk pinned row-invariant: one chunk, a cut, chunks of 257 and a
               one-row tail give equal bits (SPD-323/325; --prefill-v2 turns the pass on for every other check)

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
                # the design's next-token rule (docs/kernels.md: Flash-Next's): the prompt pass agrees with the decode
                # steps at least as well as the steps agree with the reference, less one point, and never disagrees
                # where both are sure (top-1 margin above 5 on both)
                steps_ref = float(_kl(ref_of(r)[P0_:L], stp[P0_:])["top1"].float().mean())
                margin = lambda x: (lambda v: v[:, 0] - v[:, 1])(x.float().topk(2, -1).values)  # noqa: E731
                sure = int(((got[P0_:L].argmax(-1) != stp[P0_:].argmax(-1)) & (margin(got[P0_:L]) > 5) & (margin(stp[P0_:]) > 5)).sum())
                prompt_steps = float(both["top1"].float().mean())
                say("quality_next_token_rule", set=name, row=r, positions=L - P0_, prompt_vs_steps=round(prompt_steps, 5),
                    steps_vs_reference=round(steps_ref, 5), sure_disagreements=sure,
                    passed=prompt_steps >= steps_ref - 0.01 and sure == 0, prefill_v2=bool(getattr(eng, "prefill_v2", False)))
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


def prices(a, eng, cfg, ids, say) -> None:
    """The verify price of a block by rows at a few context lengths, for the cut (ops/prices/hy4-flash/): at each
    context the served block of 1..N rows (a chain of the text's next tokens) through its graph, staged and replayed,
    the median of 7 after its capture; and a plain step with its commit."""
    from engine.models.hy4_flash.serving import Hy4FlashServing
    srv = Hy4FlashServing(eng, a.max_len)
    ctxs = sorted(int(x) for x in a.prices_at.split(","))
    N = a.prices_rows
    text = list(ids)
    while len(text) < ctxs[-1] + N + 8:
        text += ids
    doc, at = {}, 0
    for C in ctxs:
        srv.forward(torch.tensor(text[at:C]), start=at)
        at = C
        cls = srv.price_class(C)
        chain = {}
        for M in range(1, N + 1):
            blk = torch.tensor(text[C:C + M])
            ts = []
            for i in range(9):
                sync(a.device)
                t0 = time.perf_counter()
                srv.forward_tree(blk, list(range(-1, M - 1)), start=C)
                sync(a.device)
                ts.append(1e3 * (time.perf_counter() - t0))
                srv.discard_pending()
            chain[M] = sorted(ts[2:])[3]
        # a plain step and its commit: the next token for real (the context grows by one row)
        sync(a.device)
        t0 = time.perf_counter()
        srv.forward(torch.tensor(text[C:C + 1]), start=C)
        sync(a.device)
        step = 1e3 * (time.perf_counter() - t0)
        at = C + 1
        commit = max(step - chain[1], 0.0)
        # the file's prices: a block plus 0.3 ms for its commit (0.17 measured; the one step timed here carries a
        # first read of its table rows), never falling with more rows; the plain step the 1-row block plus 1.0 ms
        # (the commit and a step's cold table rows); the priors a frozen cut needs, when given
        mono, last = {}, 0.0
        for M in range(1, N + 1):
            last = max(last, chain[M] + 0.3)
            mono[str(M)] = round(last, 2)
        doc[str(cls)] = {"chain": mono, "plain": {"1": round(chain[1] + 1.0, 2)}}
        if a.prices_spec:
            doc[str(cls)].update(spec={"1": float(a.prices_spec)}, prime={"1": 0.001}, draft={"1": float(a.prices_draft)})
        say("prices", context=C, price_class=cls, chain_ms={k: round(v, 2) for k, v in chain.items()}, plain_step_ms=round(step, 2),
            commit_ms=round(commit, 2))
    step_ms = None
    c0 = doc[str(srv.price_class(ctxs[0]))]["chain"]
    if N > 1:
        step_ms = round((c0[str(N)] - c0["1"]) / (N - 1), 2)
    doc["step_ms"] = 7.0 if step_ms is None else max(step_ms, 1.0)
    doc["measured"] = a.prices_note
    if a.prices_out:
        with open(a.prices_out, "w") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")
    say("prices_file", path=a.prices_out, classes=[k for k in doc if k.isdigit()], step_ms=step_ms)


def dfly_accept(a, eng, cfg, say) -> None:
    """Teacher-forced on reference rows: after the row of token p the head drafts from the row's true next token;
    its first drafted token (position p + 2) against the pack's own greedy token there and against the unquantized
    reference's (stored logits), and its chain against the text."""
    from engine.models.hy4_flash.dfly import DFlyHead
    from engine.models.hy4_flash.serving import Hy4FlashServing
    rows_file, ref_dir, R, every = a.dfly_accept.split("=")
    ids_all = torch.load(rows_file)["ids"]
    srv = Hy4FlashServing(eng, a.max_len)
    head = DFlyHead(srv)
    n = pack1 = ref1 = both = text1 = pr = 0
    textk = [0] * head.depth
    conf = []
    for r in range(int(R)):
        ids = ids_all[r].tolist()
        ref_top = torch.load(os.path.join(ref_dir, f"row{r:04d}.pt")).to(a.device).argmax(-1).tolist()
        srv.kv.length = 0
        P0 = 64
        srv.forward(torch.tensor(ids[:P0]), start=0)
        pend = None                                        # (first drafted token, chain, position it predicts)
        for p in range(P0, len(ids) - head.depth - 2):
            lg = srv.forward(torch.tensor([ids[p]]), start=p)          # row p committed; lg predicts position p + 1
            pack_next = int(lg[0, 0].argmax())
            if pend is not None:                           # the draft made after row p - 1 predicted position p + 1
                d1, chain, prob = pend
                n += 1
                pack1 += d1 == pack_next
                ref1 += d1 == ref_top[p]
                both += pack_next == ref_top[p]
                text1 += d1 == ids[p + 1]
                pr += pack_next == ids[p + 1]
                k = 0
                while k < head.depth and chain[k] == ids[p + 1 + k]:
                    k += 1
                for j in range(k):
                    textk[j] += 1
                conf.append((prob, d1 == pack_next))
                pend = None
            if (p - P0) % int(every) == 0:
                tk, pb = head.draft(ids[p + 1], srv.last_streams(), srv.st)   # anchor: the text's token at p + 1
                pend = (tk[0][0], [t[0] for t in tk], pb[0][0])
    hi = [c for c in conf if c[0] >= 0.9]
    lo = [c for c in conf if c[0] < 0.5]
    say("dfly_accept_teacher_forced", rows=int(R), drafts=n, first_equals_pack_greedy=round(pack1 / max(n, 1), 4),
        first_equals_reference_greedy=round(ref1 / max(n, 1), 4), pack_greedy_equals_reference_greedy=round(both / max(n, 1), 4),
        first_equals_text=round(text1 / max(n, 1), 4), pack_greedy_equals_text=round(pr / max(n, 1), 4),
        chain_equals_text_through=[round(x / max(n, 1), 4) for x in textk],
        first_when_draft_prob_ge_0_9=[len(hi), round(sum(c[1] for c in hi) / max(len(hi), 1), 4)],
        first_when_draft_prob_lt_0_5=[len(lo), round(sum(c[1] for c in lo) / max(len(lo), 1), 4)])


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

    def drafted(tag: str, cut: float | None = None) -> None:
        srv.cache_state().restore(snap)
        srv.kv.length = len(ids)
        lg = srv.forward(torch.tensor([first]), start=len(ids))       # the picks are gone with the restore: one plain step
        out, nxt = [first], int(lg[0, 0].argmax())
        head = DFlyHead(srv, graphs=srv.graphs is not None)
        head.launch(srv.last_streams(), [nxt])                        # one round first, undone: it captures its graphs
        tk, _ = head.read()
        if tk:
            # every block size the round may verify, once and undone: its graph is captured before the clock runs
            for m in (range(1, len(tk) + 1) if cut is not None else (len(tk),)):
                srv.forward_tree(torch.tensor([nxt] + [t[0] for t in tk[:m]]), list(range(-1, m)), start=srv.kv.length)
                srv.discard_pending()
            srv.st.picks_at = srv.st.length
        rounds = kept = 0
        rows_total = [0]
        draft_s, hist, marks = 0.0, [0] * (cfg.dfly.draft_tokens + 1), []
        while len(out) < len(toks):
            td = time.perf_counter()
            head.launch(srv.last_streams(), [nxt])
            tk, pb = head.read()
            draft_s += time.perf_counter() - td
            if cut is not None:
                # the chain while the drafter's own path probability stays at `cut` or above
                keep, path = 0, 1.0
                while keep < len(tk) and path * pb[keep][0] >= cut:
                    path *= pb[keep][0]
                    keep += 1
                tk = tk[:keep]
            block = [nxt] + [t[0] for t in tk]
            rows_total[0] += len(block)
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
            draft_tokens=cfg.dfly.draft_tokens, graphs=srv.graphs is not None, path_prob_cut=cut,
            rows_per_round=round(rows_total[0] / max(rounds, 1), 3))
    drafted("prose_dfly")
    for c in [float(x) for x in a.dfly_cut.split(",") if x]:
        drafted("prose_dfly_cut", cut=c)
    if a.dfly_src:
        n = dfly_source(eng.W, a.dfly_src, a.device)
        say("dfly_source", dir=a.dfly_src, tensors_replaced=n)
        drafted("prose_dfly_unquantized")
    if a.dfly_dump:
        # a few rounds' inputs and this head's outputs, for a run of another implementation on the same inputs
        # (with --dfly-src the head's projections are the source checkpoint's by now)
        os.makedirs(a.dfly_dump, exist_ok=True)
        srv.cache_state().restore(snap)
        srv.kv.length = len(ids)
        head = DFlyHead(srv)
        d, st, fn = cfg.dfly, srv.st, eng
        marks = sorted({int(x) for x in a.dfly_dump_at.split(",")})
        rounds = []
        for i in range(min(len(toks) - 1, max(marks) + 1)):
            srv.forward(torch.tensor([toks[i]]), start=len(ids) + i)        # row of toks[i] committed
            if i not in marks:
                continue
            anchor = toks[i + 1]
            th = srv.last_streams().clone()
            hid = head.hidden(anchor, th, st).clone()                        # [depth + 1, H] after the final norm
            prev, chain, corr, pe = [anchor], [], [], []
            for j in range(head.depth):
                e = fn.embed(torch.tensor(prev[-1:]))
                c = head.correct(hid[j + 1:j + 2], e)
                tk = int(fn.head(c)[0].argmax())
                chain.append(tk)
                corr.append(c[0].clone())
                pe.append(e[0].clone())
                prev.append(tk)
            Lt = st.length
            layers = []
            for k in range(d.layers):
                pi = st.pick_layers.index(d.index_layers[k])
                pq = st.pick_q[pi].clone()
                ok = (st.pick_ok[pi] & (pq < Lt)).clone()
                lat = st.lat[d.kv_layers[k]]
                layers.append({"pq": pq.cpu(), "ok": ok.cpu(), "ctx": lat[pq.clamp(0, lat.shape[0] - 1)].clone().cpu()})
            e_in = fn.embed(torch.tensor([anchor, d.mask_token_id]))
            rounds.append({"step": i, "length": int(Lt), "delta": int(st.delta), "anchor": int(anchor), "target_hidden": th.cpu(),
                           "embed_anchor": e_in[0].cpu(), "embed_mask": e_in[1].cpu(), "layers": layers, "hidden": hid.cpu(),
                           "chain": chain, "corrected": torch.stack(corr).cpu(), "prev_embeds": torch.stack(pe).cpu(),
                           "truth": toks[i + 2:i + 2 + head.depth]})
        torch.save({"rounds": rounds, "depth": head.depth, "index_layers": list(d.index_layers), "kv_layers": list(d.kv_layers),
                    "mask_token_id": d.mask_token_id, "source_weights": bool(a.dfly_src)}, os.path.join(a.dfly_dump, "rounds.pt"))
        say("dfly_dump", dir=a.dfly_dump, rounds=len(rounds), steps=[r["step"] for r in rounds], source_weights=bool(a.dfly_src),
            kept=[next((j for j in range(head.depth) if r["chain"][j] != r["truth"][j]), head.depth) for r in rounds])


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


def _sha(t: torch.Tensor) -> str:
    """sha256 of a tensor's bytes as stored, its shape included."""
    import hashlib
    t = t.detach().contiguous().cpu()
    return hashlib.sha256(repr(tuple(t.shape)).encode() + t.view(torch.uint8).numpy().tobytes()).hexdigest()


ESSAY_ASK = ("Write a detailed essay of at least 700 words that explains the text above to a newcomer. "
             "Use plain prose paragraphs only: no lists, no tables, no headings, no code.")


def prefill_digest(a, eng, cfg, say) -> bool:
    """SPD-317 (the prefill design's S0.2): Hy4's record of the prompt pass. For each prompt (--digest-text: its first
    --digest-rows token ids raw, in the engine's chunks; --digest-chat: a chat request, the pack's template, thinking
    off) and each arm of --digest-arms: the sha256 of the last row's logits, of every head input of the last chunk,
    and of the state after it (lat_all, ik_all, win_all, hist, gk_all up to the length). Every arm of a prompt must give
    the same digests; --digest-out writes them, --digest-against compares them with a file recorded before."""
    import hashlib
    from tools.hy4_prefill_profile import set_arm
    prompts = []
    if a.digest_text:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(a.pack, "tokenizer.json"))
        ids = tok.encode(open(a.digest_text).read(), add_special_tokens=False).ids[:a.digest_rows]
        if len(ids) < a.digest_rows:
            raise SystemExit(f"--digest-text holds {len(ids)} tokens, --digest-rows asks for {a.digest_rows}")
        prompts.append(("text", ids))
    if a.digest_chat:
        ids, _ = chat_ids(a.pack, open(a.digest_chat).read(), ESSAY_ASK)
        prompts.append(("chat", ids))
    if not prompts:
        raise SystemExit("prefill-digest: --digest-text or --digest-chat")
    rr, out, ok = cfg.index_kpool, {}, True
    for name, ids in prompts:
        per = {}
        for arm in a.digest_arms.split(","):
            set_arm(eng, arm)
            st = eng.new_state(len(ids) + 64)
            eng.prefill(torch.tensor(ids), st, head=None)
            sync(a.device)
            n = st.length
            d = {"logits_last": _sha(eng.head(eng.prompt_hidden[-1:])), "head_inputs_last_chunk": _sha(eng.prompt_hidden),
                 "lat_all": _sha(st.lat_all[:, :n]), "ik_all": _sha(st.ik_all[:, :n]), "win_all": _sha(st.win_all),
                 "hist": _sha(st.hist), "gk_all": _sha(st.gk_all[:, :n // rr])}
            per[arm] = d
            say("prefill_digest", prompt=name, arm=arm, rows=n, last_chunk_rows=int(eng.prompt_hidden.shape[0]),
                ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(), **d)
            del st
        arms = list(per)
        same = all(per[x] == per[arms[0]] for x in arms)
        ok = ok and same
        say("prefill_digest_arms", prompt=name, arms=arms, equal=same,
            differing={x: sorted(k for k in per[x] if per[x][k] != per[arms[0]][k]) for x in arms[1:]})
        out[name] = per[arms[0]]
    if a.digest_out:
        with open(a.digest_out, "w") as f:
            json.dump({"pack": a.pack, "digests": out}, f, indent=1, sort_keys=True)
    if a.digest_against:
        want = json.load(open(a.digest_against))["digests"]
        for name, d in out.items():
            w = want.get(name)
            bad = None if w is None else sorted(k for k in set(w) | set(d) if w.get(k) != d.get(k))
            ok = ok and w is not None and not bad
            say("prefill_digest_against", prompt=name, recorded=w is not None, equal=w is not None and not bad, differing=bad)
    return ok


def moe_inputs(a, eng) -> list:
    """The routed experts' real inputs of every MoE layer and chunk of --digest-text's first --digest-rows token ids, as
    the prompt pass hands them to `_experts_chunk` on the row kernels' grouped form (gate 0's baseline):
    [(chunk, layer, z, idx, wts)] on the device."""
    from tokenizers import Tokenizer
    from tools.hy4_prefill_profile import set_arm
    if not a.digest_text:
        raise SystemExit("experts-equal / experts-crossover: --digest-text (the prompt whose routing is used)")
    tok = Tokenizer.from_file(os.path.join(a.pack, "tokenizer.json"))
    ids = tok.encode(open(a.digest_text).read(), add_special_tokens=False).ids[:a.digest_rows]
    got, real, chunk = [], eng._experts_chunk, [0]

    def spy(z, idx, wts, li):
        if li == min(eng.banks):
            chunk[0] += 1
        got.append((chunk[0], li, z.clone(), idx.clone(), wts.clone()))
        return real(z, idx, wts, li)
    set_arm(eng, "grouped")
    eng._experts_chunk = spy
    try:
        eng.prefill(torch.tensor(ids), eng.new_state(len(ids) + 64), head=None)
    finally:
        del eng._experts_chunk
    sync(a.device)
    return got


def _expert_paths(eng, z, idx, wts, li, items=(32, 64)) -> dict:
    """One layer's routed experts on the three paths, (pair outputs fp32 before the combine, combined rows) each: the
    pair kernels and the grouped row kernels in EXPERT_SUB launches (the prompt pass's loop), the shared prefill kernel
    with the clamp, one call, at items of 32 and 64."""
    from engine.models.hy4_flash import engine as E_
    from tools import fn_prefill_kernels as PK
    B, out = E_.EXPERT_SUB, {}
    for name, gr in (("pair", False), ("grouped", True)):
        yd = torch.cat([eng.expert_pairs(z[i:i + B], idx[i:i + B], li, grouped=gr) for i in range(0, z.shape[0], B)])
        out[name] = (yd, torch.cat([eng.combine(yd[i * idx.shape[1]:(i + B) * idx.shape[1]], idx[i:i + B], wts[i:i + B])
                                    for i in range(0, z.shape[0], B)]))
    for it in items:
        yd = PK.moe_grouped(z.contiguous(), idx.to(torch.int32).contiguous(), eng.banks[li], rows=it, clamp=eng.cfg.swiglu_limit)
        out[f"group{it}"] = (yd, eng.combine(yd, idx, wts))
    return out


def experts_equal(a, eng, cfg, say) -> bool:
    """SPD-321 (the prefill design's D3): every MoE layer's routed experts on the three paths, on real inputs and
    routing of every chunk of --digest-text, cut to 1, 17, 255, 257, 4,095 and 4,096 rows: torch.equal on the pair
    outputs before the combine and on the combined rows, against the pair kernels. The first unequal layer and row are
    named."""
    caps = moe_inputs(a, eng)
    k = cfg.num_experts_per_tok
    layers, bad = 0, []
    for c, li, z, idx, wts in caps:
        ok_layer = True
        for n in (1, 17, 255, 257, 4095, 4096):
            if n > z.shape[0]:
                continue
            paths = _expert_paths(eng, z[:n], idx[:n], wts[:n], li)
            base_yd, base_y = paths["pair"]
            for name, (yd, y) in paths.items():
                if name == "pair":
                    continue
                eq_yd, eq_y = torch.equal(yd, base_yd), torch.equal(y, base_y)
                if not (eq_yd and eq_y):
                    ok_layer = False
                    rows_yd = (yd != base_yd).any(dim=-1).nonzero()
                    rows_y = (y != base_y).any(dim=-1).nonzero()
                    bad.append({"chunk": c, "layer": li, "rows": n, "path": name, "pairs_equal": eq_yd, "combined_equal": eq_y,
                                "first_pair": int(rows_yd[0]) if rows_yd.numel() else None,
                                "first_row": int(rows_y[0]) if rows_y.numel() else (int(rows_yd[0]) // k if rows_yd.numel() else None),
                                "max_abs": float((y.float() - base_y.float()).abs().max())})
            del paths
        layers += 1
        say("experts_equal_layer", chunk=c, layer=li, rows=int(z.shape[0]), equal=ok_layer)
    say("experts_equal", layer_chunks=layers, unequal=len(bad), first=bad[0] if bad else None, all_equal=not bad,
        paths=["pair", "grouped", "group32", "group64"], row_counts=[1, 17, 255, 257, 4095, 4096])
    return not bad


def experts_crossover(a, eng, cfg, say) -> None:
    """SPD-321 (the prefill design's D6): ms of the three paths at 16 to 1,024 rows of the last chunk's real inputs,
    summed over every MoE layer (median of 5 after 2 warm-ups, a device sync around each), and EXPERTS_MIN: the
    smallest count from which the shared prefill kernel beats the faster row-kernel form at every larger count
    (tools/prefill_crossover.py's rule)."""
    from tools.prefill_crossover import crossover
    caps = moe_inputs(a, eng)
    last = max(c for c, *_ in caps)
    caps = [x for x in caps if x[0] == last]
    from engine.models.hy4_flash import engine as E_
    from tools import fn_prefill_kernels as PK
    B, table = E_.EXPERT_SUB, {}

    def timed(fn) -> float:
        ts = []
        for r in range(7):
            sync(a.device)
            t0 = time.perf_counter()
            fn()
            sync(a.device)
            if r >= 2:
                ts.append(time.perf_counter() - t0)
        return 1e3 * sorted(ts)[len(ts) // 2]
    for n in (16, 32, 64, 128, 256, 512, 1024):
        ms = {"pair": 0.0, "grouped": 0.0, "group": 0.0}
        for _, li, z, idx, wts in caps:
            zn, xn, wn = z[:n], idx[:n], wts[:n]
            for name, gr in (("pair", False), ("grouped", True)):
                ms[name] += timed(lambda: torch.cat([eng.combine(eng.expert_pairs(zn[i:i + B], xn[i:i + B], li, grouped=gr), xn[i:i + B], wn[i:i + B])
                                                     for i in range(0, n, B)]))
            ms["group"] += timed(lambda: eng.combine(PK.moe_grouped(zn.contiguous(), xn.to(torch.int32).contiguous(), eng.banks[li],
                                                                    rows=E_.GROUP_ROWS, clamp=cfg.swiglu_limit), xn, wn))
        table[n] = {k: round(v, 2) for k, v in ms.items()}
        say("experts_crossover_rows", rows=n, layers=len(caps), **table[n])
    best = crossover({n: {"v2": min(t["pair"], t["grouped"]), "unpack": t["group"]} for n, t in table.items()})
    say("experts_crossover", experts_min=best, table=table, rule="the smallest count from which group beats min(pair, grouped) at every larger one")


def table_rows(a, eng, cfg, say) -> bool:
    """SPD-319 (the prefill design's S2 device check): the last chunk's table ids of --digest-text's first --digest-rows
    token ids, each table on a cold cache: the sliced one-submit read against the 16-thread read it replaces (equal packed
    rows, equal decoded rows), and the prompt pass's reader job (`_engram_read`) against `engram_rows`, with its rate
    (blocks read over the job's seconds; the line is 200k reads/s)."""
    import numpy as np
    from tokenizers import Tokenizer
    from engine.models.flashnext.ple import DirectRows
    from engine.models.hy4_flash import engine as E_
    from engine.models.hy4_flash.table import decode_rows
    from tools.hy4_prefill_profile import reset_tables
    if not a.digest_text:
        raise SystemExit("table-rows: --digest-text")
    tok = Tokenizer.from_file(os.path.join(a.pack, "tokenizer.json"))
    ids = tok.encode(open(a.digest_text).read(), add_special_tokens=False).ids[:a.digest_rows]
    C, n = E_.PREFILL_CHUNK, cfg.engram_max_ngram_size
    last = (len(ids) - 1) // C * C
    full = torch.tensor([cfg.engram_pad_id] * max(n - 1 - last, 0) + ids[max(last - (n - 1), 0):], dtype=torch.int64)
    k, T = n - 1, len(ids) - last
    grams = torch.stack([full[k - j: k - j + T] for j in range(n)], dim=-1)
    ok = True
    for li in sorted(eng.tables):
        t = eng.tables[li]
        if t.direct is None:
            say("table_rows", layer=li, skipped="not a cold table")
            continue
        lid = eng.hashes[li].ids(grams)
        uniq = np.unique(lid.reshape(-1).numpy().astype(np.int64))
        reset_tables(eng)
        t0 = time.perf_counter()
        old = DirectRows._read(t.direct, uniq)
        t_old = time.perf_counter() - t0
        t0 = time.perf_counter()
        new = t.direct._read(uniq)
        t_new = time.perf_counter() - t0
        u = torch.from_numpy(uniq)
        dec = lambda p: decode_rows(torch.from_numpy(p), t.head_of(u), t.bias, t.K, t.codebook("cpu"), t.dim)  # noqa: E731
        eq_packed, eq_decoded = bool(np.array_equal(old, new)), bool(torch.equal(dec(old), dec(new)))
        reset_tables(eng)
        b0 = t.direct.stats["blocks_read"]
        t0 = time.perf_counter()
        job = eng._engram_read(li, lid)
        t_job = time.perf_counter() - t0
        reads = t.direct.stats["blocks_read"] - b0
        reset_tables(eng)
        eq_job = bool(torch.equal(job.reshape(*lid.shape, -1), eng.engram_rows(li, lid).cpu()))
        ok = ok and eq_packed and eq_decoded and eq_job
        say("table_rows", layer=li, ids=int(lid.numel()), unique=int(uniq.size), packed_equal=eq_packed, decoded_equal=eq_decoded,
            job_equal=eq_job, threads_reads_s=round(uniq.size / t_old), sliced_reads_s=round(uniq.size / t_new),
            job_s=round(t_job, 3), job_blocks_read=int(reads), job_reads_s=round(reads / t_job) if t_job else None)
    return ok


#: SPD-323: `mla_rows` against its torch reference: max |diff| / max |ref| (the kernel's weights are rounded to bf16
#: before the mix, the reference's are not: within two bf16 steps). This builder's bound, not the design's: the design's
#: lines for the flip are its quality ones (chat2048 KL, `quality_next_token_rule`)
MLA_REL = 1e-2


def v2_micro(a, eng, cfg, say) -> bool:
    """SPD-323 (the prefill design's F1 micro, question 3): `mla_rows` on one window and one sparse layer's real inputs
    of the last 4,096-row chunk of --digest-text at each context of --micro-ctx: ms (median of 7 after 2), the
    reference on 64 of its rows within MLA_REL, and a row alone equal to the same row in the chunk. Pass 80 ms a
    sparse layer and 25 ms a window layer; kill above 150 and 50 (the fallback is then `attend_r` at M rows)."""
    from tokenizers import Tokenizer
    from engine.models.hy4_flash import kernels as K_
    from engine.models.hy4_flash.config import SPARSE
    from tools.hy4_prefill_profile import set_arm
    if not a.digest_text:
        raise SystemExit("v2-micro: --digest-text")
    tok = Tokenizer.from_file(os.path.join(a.pack, "tokenizer.json"))
    text = tok.encode(open(a.digest_text).read(), add_special_tokens=False).ids
    win = next(li for li in range(20, cfg.num_hidden_layers) if cfg.layer_types[li] != SPARSE)
    spa = next(li for li in range(20, cfg.num_hidden_layers) if cfg.layer_types[li] == SPARSE)
    ok = True
    for ctx in (int(c) for c in a.micro_ctx.split(",")):
        if len(text) < ctx:
            raise SystemExit(f"--digest-text holds {len(text)} tokens, --micro-ctx asks for {ctx}")
        got, real, hcs = {}, eng._mla, {}
        real_hc = getattr(eng, "_hc_v2", None)

        def spy(qa, qr, lat, ctl, sel, li):
            if li in (win, spa):
                got[li] = (qa.clone(), qr.clone(), lat.clone(), ctl.clone(), sel.clone())
            return real(qa, qr, lat, ctl, sel, li)

        def spy_hc(h, pend, pre, w):
            if pre == f"layers.{win}.hc_mlp_layer.hc_pre.":
                hcs["x"] = (h.clone(), (pend[0].clone(), pend[1].clone()), pre, w)
            return real_hc(h, pend, pre, w)
        set_arm(eng, "v2")
        eng._mla = spy
        if real_hc is not None:
            eng._hc_v2 = spy_hc
        try:
            eng.prefill(torch.tensor(text[:ctx]), eng.new_state(ctx + 64), head=None)
        finally:
            del eng._mla
            if real_hc is not None:
                del eng._hc_v2
        if "x" in hcs:
            # F2 (question 5): one hyper-connection of 4,096 rows on the row kernels, whole and in sub-batches of 512
            h, pend, pre, w = hcs["x"]
            c = cfg
            args = (*eng._hcw(pre), w, c.rms_norm_eps, c.hc_eps, c.hc_magnitude)
            ts = []
            for r in range(9):
                sync(a.device)
                t0 = time.perf_counter()
                whole = K_.hc(h, pend, *args)
                sync(a.device)
                if r >= 2:
                    ts.append(1e3 * (time.perf_counter() - t0))
            parts = [K_.hc(h[i:i + 512].contiguous(), (pend[0][i:i + 512].contiguous(), pend[1][i:i + 512].contiguous()), *args)
                     for i in range(0, h.shape[0], 512)]
            same = all(torch.equal(torch.cat([p_[k] for p_ in parts]), whole[k]) for k in range(3))
            ms_hc = sorted(ts)[len(ts) // 2]
            ok = ok and same
            say("v2_micro_hc", context=ctx, layer=win, rows=int(h.shape[0]), ms=round(ms_hc, 2), pass_ms=6.0, passed=ms_hc <= 6.0,
                sub_batches_of_512_equal=same)
        for li in (win, spa):
            qa, qr, lat, ctl, sel = got[li]
            sink, scale = eng.sinks[li].reshape(-1), cfg.attn(li).qk ** -0.5
            run = lambda: K_.mla_rows(qa, qr, lat, ctl, sel, sink, scale)  # noqa: E731
            ts = []
            for r in range(9):
                sync(a.device)
                t0 = time.perf_counter()
                out = run()
                sync(a.device)
                if r >= 2:
                    ts.append(1e3 * (time.perf_counter() - t0))
            ms = sorted(ts)[len(ts) // 2]
            rows = torch.linspace(0, qa.shape[0] - 1, 64).long().tolist()
            ref = K_.mla_rows_reference(qa[rows], qr[rows], lat, ctl[rows].contiguous(), sel[rows], sink, scale)
            rel = float((out[rows].float() - ref.float()).abs().max() / ref.float().abs().max())
            alone = all(torch.equal(K_.mla_rows(qa[j:j + 1], qr[j:j + 1], lat, ctl[j:j + 1], sel[j:j + 1], sink, scale)[0], out[j])
                        for j in (0, 17, qa.shape[0] // 2, qa.shape[0] - 1))
            sparse = cfg.layer_types[li] == SPARSE
            line, kill = (80.0, 150.0) if sparse else (25.0, 50.0)
            ok = ok and rel < MLA_REL and alone and ms <= kill
            say("v2_micro_mla", context=ctx, layer=li, layer_kind="sparse" if sparse else "window", rows=int(qa.shape[0]),
                keys_mean=round(float(ctl[:, 1].float().mean()), 1), ms=round(ms, 2), pass_ms=line, kill_ms=kill,
                passed=ms <= line, killed=ms > kill, rel_vs_reference=rel, within=rel < MLA_REL, row_alone_equal=alone,
                launch={"HB": K_.MLA_HB, "BN": K_.MLA_BN, "warps": K_.MLA_WARPS, "stages": K_.MLA_STAGES})
    return ok


def v2_cuts(a, eng, cfg, say) -> bool:
    """SPD-323, SPD-325 (F1 to F3): a row's bits on the prompt pass on fixed-order kernels depend on its chunk only
    through the trunk's projections. With those pinned to a row-invariant form (the row kernels, 32 rows at a time;
    a plain weight on `linr`), the first 3,000 token ids of --digest-text as one chunk, as two chunks cut at 1,000, in
    chunks of 257, and as 2,999 rows then one: every head input, the last row's logits and the state equal. (Without
    the pin the trunk takes the library GEMM above 128 rows, whose bits follow the row count: the design's F4.)"""
    from tokenizers import Tokenizer
    from tools.hy4_prefill_profile import set_arm
    if not a.digest_text:
        raise SystemExit("v2-cuts: --digest-text")
    tok = Tokenizer.from_file(os.path.join(a.pack, "tokenizer.json"))
    ids = tok.encode(open(a.digest_text).read(), add_special_tokens=False).ids[:3000]
    rr = cfg.index_kpool

    def pinned(x, w):
        x2 = x.reshape(-1, x.shape[-1])
        if hasattr(w, "bf16_cached") and x2.dtype == torch.bfloat16:
            y = torch.cat([eng.lin1(x2[i:i + 32], w) for i in range(0, x2.shape[0], 32)])
        else:
            y = eng.linr(x2, w if torch.is_tensor(w) else w.bf16_cached()).to(x.dtype)
        return y.view(*x.shape[:-1], -1)
    set_arm(eng, "v2")
    eng.lin = pinned
    runs = {}
    try:
        for name, cuts in (("one", [3000]), ("cut_1000", [1000, 2000]), ("chunks_257", [257] * 11 + [173]), ("tail_1", [2999, 1])):
            st, hid, i = eng.new_state(3064), [], 0
            for n in cuts:
                hid.append(eng.forward_chunk(torch.tensor(ids[i:i + n]), st).clone())
                i += n
            sync(a.device)
            L = st.length
            runs[name] = {"head_inputs": _sha(torch.cat(hid)), "logits_last": _sha(eng.head(hid[-1][-1:])),
                          "lat_all": _sha(st.lat_all[:, :L]), "ik_all": _sha(st.ik_all[:, :L]), "win_all": _sha(st.win_all),
                          "hist": _sha(st.hist), "gk_all": _sha(st.gk_all[:, :L // rr])}
    finally:
        del eng.lin
    base = runs["one"]
    diff = {k: sorted(x for x in v if v[x] != base[x]) for k, v in runs.items() if k != "one"}
    ok = not any(diff.values())
    say("v2_cuts", rows=len(ids), equal=ok, differing=diff, trunk="row kernels, 32 rows a call (pinned)")
    return ok


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
        if getattr(eng, "vpk", None):
            # one step again with the value half from its BF16 copy: what reading it packed saves
            eng.value_packed, g3 = False, {}
            eng.rows([int(ids[0])], [-1], st, graphs=g3)
            best = 1e9
            for _ in range(5):
                sync(a.device)
                t0 = time.perf_counter()
                for _ in range(8):
                    next(iter(g3.values())).replay()
                sync(a.device)
                best = min(best, (time.perf_counter() - t0) / 8)
            out["replay_ms_1_value_bf16"] = round(1e3 * best, 2)
            out["value_packed_layers"] = len(eng.vpk)
            eng.value_packed = True
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
    ap.add_argument("--prices-at", default="", help="context lengths (rows) to price a verify block at, e.g. 1500,3600,15000,60000")
    ap.add_argument("--prices-rows", type=int, default=8)
    ap.add_argument("--prices-out", default="", help="the price file to write")
    ap.add_argument("--prices-note", default="", help="the file's `measured` line")
    ap.add_argument("--prices-spec", type=float, default=0.0, help="a drafted round's ms a token, as the file's `spec` prior (0: none)")
    ap.add_argument("--prices-draft", type=float, default=11.0, help="the draft's ms, as the file's `draft` prior (with --prices-spec)")
    ap.add_argument("--dfly-cut", default="", help="with --prose --dfly: drafted runs that verify the chain only while its path probability stays at each of these values")
    ap.add_argument("--dfly-dump", default="", help="with --prose --dfly: a few rounds' draft inputs and outputs into this folder")
    ap.add_argument("--dfly-dump-at", default="0,40,80,120,160,200,240,280")
    ap.add_argument("--dfly-accept", default="", help="rows.pt=refdir=R=EVERY: the first drafted token against the pack's and the reference's next token, teacher-forced")
    ap.add_argument("--quality", default="", help="name=rows.pt=refdir=N[,...]: the engine against stored reference logits")
    ap.add_argument("--quality-no-steps", action="store_true", help="score the prompt pass only (the step rows' figures still print)")
    ap.add_argument("--quality-steps", default="", help="R:L: the decode path on the first set's first R rows; torch sums and forced routing on L positions")
    ap.add_argument("--dfly-src", help="the source checkpoint's directory: the prose run again with its unquantized draft layers")
    ap.add_argument("--no-splits", action="store_true", help="the device's rule for every dense split (not the family's table)")
    ap.add_argument("--grouped", action="store_true", help="the prompt's experts on the grouped kernels")
    ap.add_argument("--prose", help="a text file: a chat request to answer it in prose, decode rates from token 50 on")
    ap.add_argument("--answer", type=int, default=600)
    ap.add_argument("--profile", action="store_true", help="one decode step and one drafted round taken apart by part")
    ap.add_argument("--only", help="comma list of checks to run after the load (self,experts,tolerance,decode,dfly,caches,prose,profile,fit,"
                                   "prefill-digest,experts-equal,experts-crossover,table-rows,v2-micro,v2-cuts)")
    ap.add_argument("--prefill-v2", action="store_true", help="the prompt pass on fixed-order kernels (HY4_PREFILL_V2=1)")
    ap.add_argument("--micro-ctx", default="8192,32768", help="v2-micro: the contexts (rows of --digest-text) to time at")
    ap.add_argument("--digest-text", help="prefill-digest: a text file, its first --digest-rows token ids raw")
    ap.add_argument("--digest-rows", type=int, default=8192)
    ap.add_argument("--digest-chat", help="prefill-digest: a text file, as a chat request (the essay ask, thinking off)")
    ap.add_argument("--digest-arms", default="grouped", help="prefill-digest: comma list of the paths (pair, grouped, group, v2)")
    ap.add_argument("--prefill-experts", default="group", choices=("group", "rows"),
                    help="the prompt's experts on the shared prefill kernel (group) or the row kernels (rows; --grouped picks their form)")
    ap.add_argument("--digest-out", help="prefill-digest: write the digests to this JSON file")
    ap.add_argument("--digest-against", help="prefill-digest: compare with a JSON file --digest-out wrote")
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
    eng.prefill_experts = a.prefill_experts
    eng.prefill_v2 = a.prefill_v2
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
        n = min(len(ids), 200)
        rw = ref.forward(torch.tensor(ids[:n]), Hy4State())
        st = eng.new_state(n + 8)
        cut = n - 64
        gs = {} if str(a.device).startswith("cuda") else None
        got = torch.cat((eng.prefill(torch.tensor(ids[:cut]), st, head="all"), torch.cat([eng.step(t, st, graphs=gs).clone() for t in ids[cut:n]])))
        d = (got - rw).abs()
        kl = lambda x, y: float((torch.softmax(x, -1) * (torch.log_softmax(x, -1) - torch.log_softmax(y, -1))).sum(-1).mean())
        say("ref", rows=n, steps=n - cut, max_abs=round(d.max().item(), 4), mean_abs=round(d.mean().item(), 5), logit_scale=round(rw.abs().mean().item(), 3),
            mean_abs_prompt=round(d[:cut].mean().item(), 5), mean_abs_steps=round(d[cut:].mean().item(), 5),
            kl_prompt=round(kl(rw[:cut], got[:cut]), 5), kl_steps=round(kl(rw[cut:], got[cut:]), 5),
            agree_prompt=round((got[:cut].argmax(-1) == rw[:cut].argmax(-1)).float().mean().item(), 4),
            agree_steps=round((got[cut:].argmax(-1) == rw[cut:].argmax(-1)).float().mean().item(), 4),
            fused=bool(getattr(eng, "fused", False)), banks=len(getattr(eng, "banks", {})))

    if only is not None and "prefill-digest" in only:
        res["ok"] = prefill_digest(a, eng, cfg, say) and res["ok"]
    if banks and only is not None and "experts-equal" in only:
        res["ok"] = experts_equal(a, eng, cfg, say) and res["ok"]
    if banks and only is not None and "experts-crossover" in only:
        experts_crossover(a, eng, cfg, say)
    if only is not None and "table-rows" in only:
        res["ok"] = table_rows(a, eng, cfg, say) and res["ok"]
    if banks and only is not None and "v2-micro" in only:
        res["ok"] = v2_micro(a, eng, cfg, say) and res["ok"]
    if only is not None and "v2-cuts" in only:
        res["ok"] = v2_cuts(a, eng, cfg, say) and res["ok"]
    if a.quality and want("quality"):
        quality(a, eng, cfg, say)
    if a.prices_at and want("prices"):
        prices(a, eng, cfg, ids, say)
    if a.dfly_accept and want("accept"):
        dfly_accept(a, eng, cfg, say)
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
