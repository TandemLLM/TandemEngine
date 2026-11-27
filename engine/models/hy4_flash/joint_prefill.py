"""SPD-347: the prompts of a prefill burst in one pass, each prompt's rows bit for bit its own prefill's
(`Hy4FlashEngine.prefill` on the fixed-order prompt pass, `prefill_v2`, one chunk).

Prompts sent at once used to prefill one after another, each reading every layer's routed experts for its own rows. Here
the prompts go through the layers together: everything of a layer up to its MLP runs a prompt at a time in the prompt's
own order and state (`_v2_attn`: the Engram rows, the hyper-connections, the attention on the prompt's own latents and
read lists), and then each MoE layer's routed experts run once over every prompt's rows (`fn_prefill_kernels.moe_grouped`:
the pairs grouped by expert, a pair's output its own row's whatever group or item it lands in; the chunk's own call is
the same kernel, and below `experts_min` rows the row kernels give the same bits a pair, which is why `experts_min` is
not 0 only at the prefill kernel's warp split). The router, the down and up projections, the combine and the shared
expert stay a prompt at a time (`lin_prompt` gives a row its one-row bits, `_rows_mm` and the combine compute a row
alone, so this is order, not arithmetic). A layer's EXL3 slabs are decoded once for all of its prompts
(tools/exl3_linear's slab cache, cleared after the layer: the same tensors, the same products). The Engram rows of every
prompt go to the reader thread layer by layer, so the first layer's rows of the last prompt do not wait behind the
first prompt's last layer.

`ok` says which prompt chunks can join (the fixed-order pass on the device's kernels, one chunk, not a decode step);
`prefill_joint` cuts the jobs into runs of at most RUN_ROWS rows and runs each run of two or more jointly, a lone job
through `fn.prefill` itself.
"""

from __future__ import annotations

import torch

#: rows a joint run takes at most (two full chunks: two 3.5k prompts or four 2k ones in one run)
RUN_ROWS = 8192


def ok(fn, T: int, start: int, st) -> bool:
    """Whether a prompt chunk of T rows at `start` on state `st` can join: its own forward would be one chunk of the
    fixed-order prompt pass on the device's kernels (not the decode step's one token)."""
    from engine.models.hy4_flash import engine as E
    T, start = int(T), int(start)
    return bool(fn.prefill_v2 and fn.fused and 1 <= T <= E.PREFILL_CHUNK and not (T == 1 and start > 0)
                and start + T <= st.cap)


def runs(lens: list, cap: int) -> list:
    """The jobs' indices in order, cut into runs whose rows add up to at most `cap` (a job alone may fill one)."""
    out, cur, n = [], [], 0
    for i, T in enumerate(lens):
        if cur and n + int(T) > cap:
            out.append(cur)
            cur, n = [], 0
        cur.append(i)
        n += int(T)
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def prefill_joint(fn, jobs: list) -> list:
    """jobs [(tokens [T], state)], each `ok`, on distinct states -> [(logits fp32 [1, vocab], head inputs [T, H])] in
    order, every state left as `fn.prefill(tokens, state)` leaves it."""
    assert len({id(st) for _, st in jobs}) == len(jobs), "a state twice in one joint prefill"
    out = [None] * len(jobs)
    for run in runs([int(t.numel()) for t, _ in jobs], RUN_ROWS):
        if len(run) == 1:
            t, st = jobs[run[0]]
            lg = fn.prefill(t.view(-1), st)
            out[run[0]] = (lg, st.prompt_hidden)
            continue
        for i, r in zip(run, _joint(fn, [jobs[i] for i in run])):
            out[i] = r
    return out


def _joint(fn, jobs: list) -> list:
    """One run of jobs through every layer (see the module docstring)."""
    from tools import exl3_linear as X
    cfg = fn.cfg
    toks = [t.reshape(-1).to(torch.int64).cpu() for t, _ in jobs]
    begun = [fn._chunk_begin(t, st, read=False) for t, (_, st) in zip(toks, jobs)]
    # every prompt's Engram rows to the reader layer by layer (the reader takes its jobs in order)
    reads = [{} for _ in jobs]
    if fn.hashes:
        if fn._reader is None:
            from concurrent.futures import ThreadPoolExecutor
            fn._reader = ThreadPoolExecutor(1, thread_name_prefix="hy4-table-rows")
        for li in fn._elayers:
            for k, b in enumerate(begun):
                ids = fn.hashes[li].ids(b[5])
                reads[k][li] = (tuple(ids.shape), fn._reader.submit(fn._engram_read, li, ids))
    cs = [fn._v2_begin(t, st, reads=r) for t, (_, st), r in zip(toks, jobs, reads)]
    X.slab_cache_on()
    try:
        for li in range(cfg.num_hidden_layers):
            xs = []
            for c in cs:
                xs.append(fn._v2_attn(c, li))
            if cfg.is_dense_mlp(li) or li not in fn.banks or not fn._experts_joinable([x for x, _ in xs]):
                for c, (x, post) in zip(cs, xs):
                    c["pend"] = (fn._mlp_chunk_v2(x, li), post)
            else:
                ys = fn._moe_joint([x for x, _ in xs], li)
                for c, (x, post), y in zip(cs, xs, ys):
                    c["pend"] = (y, post)
            X.slab_cache_clear()
            if li + 1 < cfg.num_hidden_layers:
                fn._yield_layer()
    finally:
        X.slab_cache_off()
    res = []
    for c in cs:
        hidden = fn._v2_end(c)
        res.append((fn.head(hidden[-1:]), hidden))
    return res
