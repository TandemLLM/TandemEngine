"""The Hy4-flash engine on a tiny random model, on the CPU: the row path and the prefill against the
reference forward, a verify block's rows against the decode step's (bit for bit), trees, commit of a moved
path, rollback, and the rings, groups and read lists past their first wrap. The tiny byte gate (SPD-317) holds the
bytes of a fixed script against a host's recorded digests (tests/byte_pins.py)."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.models.hy4_flash import engine as E  # noqa: E402
from engine.models.hy4_flash.ref import Hy4FlashRef, Hy4State  # noqa: E402
from tests.hy4_tiny import random_weights, tiny_config  # noqa: E402

BOUND = 2e-5      # fp32: the engine's absorbed keys and fp32 sums against the reference's order
g0 = torch.Generator().manual_seed(11)
PROMPT = torch.cat([torch.tensor([5, 9, 3, 61, 22, 2, 3, 40, 41, 12, 63, 7]), torch.randint(4, 60, (33,), generator=g0)])
_T = {}


def tiny(dtype=torch.float32):
    if dtype not in _T:
        cfg = tiny_config()
        W, tables = random_weights(cfg)
        _T[dtype] = (cfg, Hy4FlashRef(cfg, W, tables, dtype=dtype), E.Hy4FlashEngine(cfg, W, tables, dtype=dtype, max_len=128))
    return _T[dtype]


class _small_chunks:
    """Prompt chunks of 7 rows, attention 5 at a time, read lists 3 at a time: every boundary inside 45 rows."""

    def __enter__(self):
        self.old = (E.PREFILL_CHUNK, E.SUB, E.SUB_LIST)
        E.PREFILL_CHUNK, E.SUB, E.SUB_LIST = 7, 5, 3

    def __exit__(self, *exc):
        E.PREFILL_CHUNK, E.SUB, E.SUB_LIST = self.old


def test_prefill_matches_the_reference():
    cfg, ref, eng = tiny()
    want = ref.forward(PROMPT, Hy4State())
    got = eng.prefill(PROMPT, eng.new_state(), head="all")
    err = (got - want).abs().max().item()
    assert err < BOUND, err
    with _small_chunks():
        got2 = eng.prefill(PROMPT, eng.new_state(), head="all")
    err2 = (got2 - want).abs().max().item()
    assert err2 < BOUND, err2
    return f"45 rows (window 4, lists past row 8): one chunk {err:.1e}, chunks of 7 {err2:.1e}"


def test_decode_rows_match_the_reference():
    cfg, ref, eng = tiny()
    want = ref.forward(PROMPT, Hy4State())
    st = eng.new_state()
    got = torch.cat([eng.step(int(t), st) for t in PROMPT])
    err = (got - want).abs().max().item()
    assert st.length == PROMPT.shape[0] and err < BOUND, err
    # a prompt, then steps: the state the prefill leaves is the one the steps continue
    st = eng.new_state()
    with _small_chunks():
        eng.prefill(PROMPT[:19], st, head=None)
    got = torch.cat([eng.step(int(t), st) for t in PROMPT[19:]])
    err2 = (got - want[19:]).abs().max().item()
    assert err2 < BOUND, err2
    return f"45 decode steps {err:.1e}; 19 prompt rows then 26 steps {err2:.1e}"


def _states_equal(a, b) -> bool:
    n = a.length
    return (n == b.length and all(torch.equal(a.lat[k][:n], b.lat[k][:n]) and torch.equal(a.ik[k][:n], b.ik[k][:n]) for k in a.lat)
            and all(torch.equal(a.win[k], b.win[k]) for k in a.win) and torch.equal(a.hist, b.hist))


def test_verify_rows_are_the_decode_steps():
    for dtype in (torch.float32, torch.bfloat16):
        cfg, ref, eng = tiny(dtype)
        res = eng.self_check(PROMPT[:eng.check_rows()].tolist(), steps=24, widths=(2, 3, 4, 8, 24), log=lambda s: None)
        assert res["ok"], res
        # the replay reached the selected groups, wrapped rings, and groups closed inside a block
        cov = res["covers"]
        assert cov["selection"] and cov["rings_wrapped"] and cov["groups_closed_in_blocks"] > 20, cov
        short = eng.self_check(PROMPT[:6].tolist(), steps=4, widths=(2,), log=lambda s: None)
        assert short["ok"] and not short["covers"]["selection"], short
    return f"{res['rows']} rows (chains of 2, 3, 4, 8 and 24, trees of 5) equal the decode steps bit for bit, fp32 and bf16"


def test_commit_a_moved_path_and_a_rollback():
    cfg, ref, eng = tiny(torch.bfloat16)
    a, b = eng.new_state(), eng.new_state()
    eng.prefill(PROMPT[:10], a, head=None)
    eng.prefill(PROMPT[:10], b, head=None)
    t = PROMPT[10:].tolist()
    want = [eng.step(x, a) for x in t[:6]]
    # a tree whose kept path is rows 0, 3, 4 (a rejected branch 1, 2 sits before it), then only two rows of a chain
    got, _ = eng.rows([t[0], 50, 51, t[1], t[2], 52], [-1, 0, 1, 0, 3, 3], b)
    assert all(torch.equal(got[r], want[i][0]) for i, r in enumerate((0, 3, 4)))
    eng.commit(b, [0, 3, 4])
    got, _ = eng.rows([t[3], t[4], 9, 9], [-1, 0, 1, 2], b)
    eng.commit(b, [0, 1])                               # the rollback: two rows kept of four
    assert torch.equal(got[0], want[3][0]) and torch.equal(got[1], want[4][0])
    assert b.length == a.length - 1
    lg = eng.step(t[5], b)
    assert torch.equal(lg, want[5]) and _states_equal(a, b)
    try:
        eng.rows([1, 2], [-1, 0], b)
        eng.commit(b, [1])
        raise SystemExit("a path that skips the anchor was taken")
    except ValueError:
        pass
    return "a tree's moved path and a partly kept chain leave the state of the plain decode, bit for bit"


def test_long_context_wraps_rings_and_selects_groups():
    """120 rows: the 4-row rings wrap 30 times, 60 groups close, every sparse row past row 8 selects."""
    cfg, ref, eng = tiny()
    g = torch.Generator().manual_seed(2)
    toks = torch.randint(4, 60, (120,), generator=g)
    want = ref.forward(toks, Hy4State())
    st = eng.new_state()
    with _small_chunks():
        got = eng.prefill(toks[:90], st, head="all")
    err = (got - want[:90]).abs().max().item()
    assert err < BOUND, err
    blocks, i = [], 90
    while i < 120:
        blk = toks[i:i + 5].tolist()
        lg, _ = eng.rows(blk, list(range(-1, len(blk) - 1)), st)
        eng.commit(st, list(range(len(blk))))
        blocks.append(lg)
        i += 5
    err2 = (torch.cat(blocks) - want[90:]).abs().max().item()
    assert err2 < BOUND and st.length == 120, err2
    return f"90 prompt rows {err:.1e}, then 6 blocks of 5 {err2:.1e}"


def test_a_block_writes_nothing_until_commit():
    cfg, ref, eng = tiny(torch.bfloat16)
    st, twin = eng.new_state(), eng.new_state()
    eng.prefill(PROMPT[:21], st, head=None)
    eng.prefill(PROMPT[:21], twin, head=None)
    eng.rows(PROMPT[21:29].tolist(), list(range(-1, 7)), st)
    assert _states_equal(st, twin)
    return "state unchanged by a block of 8"


def test_the_row_product_reference():
    """`mm`'s contract on its plain form: out[n] = a[n] @ b[n mod NB], any strides, and an item's bits the same in
    any batch (the kernel itself runs on a device: tools/hy4_check.py compares it there)."""
    from engine.models.hy4_flash import kernels as K
    g = torch.Generator().manual_seed(3)
    a, b = torch.randn(12, 5, 70, generator=g), torch.randn(4, 70, 9, generator=g)
    out = K.mm_reference(a, b)
    want = torch.stack([a[n] @ b[n % 4] for n in range(12)])
    assert out.dtype == torch.float32 and torch.equal(out, want)
    assert torch.equal(K.mm_reference(a[4:6], b), out[4:6]) and torch.equal(K.mm_reference(a[:1], b[:1]), out[:1])
    bt = torch.randn(1, 9, 70, generator=g)
    assert torch.equal(K.mm_reference(a, bt.transpose(1, 2)), torch.stack([a[n] @ bt[0].T for n in range(12)]))
    x = torch.randn(3, 70, generator=g).to(torch.bfloat16)
    cfg, ref, eng = tiny(torch.bfloat16)
    assert torch.equal(eng.sumsq(x), torch.stack([(x[n].float() * x[n].float()).sum() for n in range(3)]).view(3, 1)) or \
        (eng.sumsq(x) - x.float().square().sum(-1, keepdim=True)).abs().max() < 1e-3
    assert K.mm_reference(a[:0], b).shape == (0, 5, 9)
    return "items of any batch equal their one-item call; shared, cycled and strided right operands"


def test_bf16_engram_rows_follow_the_reference_norm():
    # the reference's Engram norms are nn.RMSNorm without an eps, which adds the fp32 epsilon on BF16 too; the
    # BF16 epsilon (2^-7) would halve rows of this size (mean square 0.0025)
    cfg, ref, eng = tiny(torch.bfloat16)
    li = sorted(eng.hashes)[0]
    g = torch.Generator().manual_seed(5)
    M = 3
    h = torch.randn(M, cfg.hc_mult, cfg.hidden_size, generator=g).to(torch.bfloat16)
    rows = (torch.randn(M, cfg.engram_heads, cfg.engram_head_dim, generator=g) * 0.05).to(torch.bfloat16)
    live = torch.ones(M, dtype=torch.bool)
    want = ref.engram_apply(h, rows, li, live).float()
    got = eng.engram_r(h, rows, li, live).float()
    rel = ((got - want).norm() / want.norm()).item()
    assert rel < 0.03, rel
    # and a decode step after a prompt stays near the prompt pass's row (BF16 rounding only)
    st = eng.new_state()
    eng.prefill(PROMPT[:30], st, head=None)
    step = eng.step(int(PROMPT[30]), st).float()
    full = eng.prefill(PROMPT[:31], eng.new_state(), head="last").float()
    d = ((step - full).abs().mean() / full.abs().mean()).item()
    assert d < 0.05, d
    return f"rows of rms 0.05: the row form against the reference's {rel:.4f}; a step against the prompt pass's row {d:.4f} of the mean logit size"


def byte_gate_run(eng=None) -> dict:
    """SPD-317 (S0.5): one fixed script on the tiny model in BF16, every logit and every stored tensor as a digest: a
    120-row prompt in chunks of 16, six decode steps, a tree block with a wrong branch, its commit and one more step."""
    from byte_pins import digest
    if eng is None:
        cfg = tiny_config()
        W, tables = random_weights(cfg)
        eng = E.Hy4FlashEngine(cfg, W, tables, dtype=torch.bfloat16, max_len=256)
    g = torch.Generator().manual_seed(17)
    toks = torch.randint(4, 60, (140,), generator=g)
    st = eng.new_state()
    out = {}
    old = E.PREFILL_CHUNK
    E.PREFILL_CHUNK = 16
    try:
        out["prefill"] = eng.prefill(toks[:120], st, head="all")
    finally:
        E.PREFILL_CHUNK = old
    out["prompt_hidden"] = eng.prompt_hidden
    for i in range(6):
        out[f"step{i}"] = eng.step(int(toks[120 + i]), st)
    t = toks[126:130].tolist()
    lg, hid = eng.rows([t[0], (t[1] + 1) % 64, t[2], t[1], t[2]], [-1, 0, 1, 0, 3], st)
    out["tree"], out["tree_hidden"] = lg, hid
    eng.commit(st, [0, 3, 4])
    out["after_commit"] = eng.step(int(toks[130]), st)
    n = st.length
    out.update(lat=st.lat_all[:, :n], ik=st.ik_all[:, :n], win=st.win_all, hist=st.hist, gk=st.gk_all[:, :n // eng.cfg.index_kpool])
    return {k: digest(v) for k, v in out.items()}


def test_the_tiny_byte_gate():
    """The prompt pass, decode, a tree and a commit keep their bytes across a change that must not move them (the
    form of tests/test_refactor_gate.py; recorded on the tree before the prefill steps, tests/byte_pins.py)."""
    from byte_pins import check
    got = byte_gate_run()
    assert got == byte_gate_run(), "two runs in one process differ: the gate cannot work"
    return check("hy4_byte_gate.json", "tiny_bf16", got)


def _pair_walk(eng, z, idx, li, lim):
    """Every (row, slot) pair through its expert, one row at a time, the SwiGLU clamps where `lim` is set: [M k, latent]
    fp32, the contract of the shared prefill kernel and of the row kernels (on the CPU, the tiny model's weights)."""
    import torch.nn.functional as F
    p = f"layers.{li}.mlp.experts."
    G, U, D = eng.w(p + "gate_proj"), eng.w(p + "up_proj"), eng.w(p + "down_proj")
    out = []
    for r in range(z.shape[0]):
        zr = z[r:r + 1].clone()
        for e in idx[r].tolist():
            g, u = F.linear(zr, G.weight(e, z.dtype)), F.linear(zr, U.weight(e, z.dtype))
            if lim is not None:
                g, u = g.clamp(max=lim), u.clamp(min=-lim, max=lim)
            out.append(F.linear(F.silu(g) * u, D.weight(e, z.dtype)).float())
    return torch.cat(out)


class _Kernels:
    """The shared prefill kernel (tools/fn_prefill_kernels.moe_grouped) and the engine's row kernels (`expert_pairs`)
    replaced by the pair walk, every call recorded."""

    def __init__(self, eng):
        self.eng, self.grouped, self.rows = eng, [], []

    def __enter__(self):
        from tools import fn_prefill_kernels as PK
        self.PK, self.real = PK, PK.moe_grouped

        def moe_grouped(x, ids, bank, rows=64, clamp=None):
            self.grouped.append({"rows": x.shape[0], "dtype": ids.dtype, "items": rows, "clamp": clamp, "layer": bank[1]})
            return _pair_walk(self.eng, x, ids, bank[1], clamp)

        def expert_pairs(z, idx, li, grouped=False):
            self.rows.append({"rows": z.shape[0], "layer": li, "grouped": grouped})
            return _pair_walk(self.eng, z, idx, li, self.eng.cfg.swiglu_limit)
        PK.moe_grouped = moe_grouped
        self.eng.expert_pairs = expert_pairs
        return self

    def __exit__(self, *exc):
        self.PK.moe_grouped = self.real
        del self.eng.expert_pairs


def _banked(warps=None, experts_min=None):
    """The tiny model (fp32) with a kernel bank on every MoE layer (a stand-in the patched kernels read), built with the
    device's MoE warps and EXPERTS_MIN as given."""
    old = (E.moe_warps, E.EXPERTS_MIN)
    if warps is not None:
        E.moe_warps = lambda device: warps
    if experts_min is not None:
        E.EXPERTS_MIN = experts_min
    try:
        cfg = tiny_config()
        W, tables = random_weights(cfg)
        eng = E.Hy4FlashEngine(cfg, W, tables, dtype=torch.float32, max_len=512)
    finally:
        E.moe_warps, E.EXPERTS_MIN = old
    eng.banks = {li: ("bank", li) for li in range(cfg.num_hidden_layers) if not cfg.is_dense_mlp(li)}
    return eng


def test_a_chunks_experts_are_one_grouped_call():
    """SPD-321: on a kernel bank a 300-row chunk's routed experts are one call of the shared prefill kernel a MoE layer,
    with all its rows, int32 ids, items of GROUP_ROWS and the model's SwiGLU limit as the clamp; the row kernels' loop
    (HY4_PREFILL_EXPERTS=rows) gives the same logits."""
    eng = _banked()
    toks = torch.randint(4, 60, (300,), generator=torch.Generator().manual_seed(23))
    with _Kernels(eng) as k:
        eng.prefill_experts = "group"
        a = eng.prefill(toks, eng.new_state(), head="all")
        calls, k.grouped = k.grouped, []
        assert not k.rows, k.rows[:2]
        assert sorted(c["layer"] for c in calls) == sorted(eng.banks), calls
        assert all(c["rows"] == 300 and c["dtype"] == torch.int32 and c["items"] == E.GROUP_ROWS
                   and c["clamp"] == eng.cfg.swiglu_limit for c in calls), calls
        eng.prefill_experts = "rows"
        b = eng.prefill(toks, eng.new_state(), head="all")
        assert not k.grouped and [r["rows"] for r in k.rows] == [256, 44] * len(eng.banks), k.rows
    assert torch.equal(a, b), (a - b).abs().max()
    return f"{len(calls)} MoE layers: one call each of 300 rows, clamp {eng.cfg.swiglu_limit}; the row loop's logits equal"


def test_below_the_crossover_the_row_kernels_run():
    """SPD-321: at the prefill kernel's warp split a chunk below EXPERTS_MIN rows takes the row kernels, a chunk at or
    above it the prefill kernel."""
    eng = _banked(warps=E.GROUP_WARPS, experts_min=64)
    assert eng.experts_min == 64
    toks = torch.randint(4, 60, (300,), generator=torch.Generator().manual_seed(29))
    with _Kernels(eng) as k:
        eng.prefill(toks[:40], eng.new_state(), head=None)
        assert not k.grouped and [r["rows"] for r in k.rows] == [40] * len(eng.banks), (k.grouped[:1], k.rows)
        k.rows = []
        eng.prefill(toks, eng.new_state(), head=None)
        assert not k.rows and [c["rows"] for c in k.grouped] == [300] * len(eng.banks), (k.rows[:1], k.grouped)
    return "40 rows on the row kernels, 300 on the prefill kernel (EXPERTS_MIN 64)"


def test_another_warp_count_takes_one_kernel_at_every_size():
    """SPD-321: where the row kernels split K over another warp count (their bits differ from the prefill kernel's),
    every row count takes the prefill kernel, whatever EXPERTS_MIN says; so does a device without the row kernels."""
    out = []
    toks = torch.randint(4, 60, (40,), generator=torch.Generator().manual_seed(31))
    # 8 warps; and the CPU's GPU adapter, which has no MoE launch shape at all
    for label, eng in (("8 warps", _banked(warps=8, experts_min=64)), ("no launch shape", _banked(experts_min=64))):
        assert eng.experts_min == 0, (label, eng.experts_min)
        with _Kernels(eng) as k:
            eng.prefill(toks, eng.new_state(), head=None)
            assert not k.rows and [c["rows"] for c in k.grouped] == [40] * len(eng.banks), (label, k.rows[:1])
        out.append(f"{label}: 40 rows on the prefill kernel")
    return "; ".join(out)


class _CountingTable:
    """A table whose reads are logged (layer, ids) before they run."""

    def __init__(self, t, li, log):
        self.t, self.li, self.log = t, li, log

    def gather(self, ids, dtype=torch.bfloat16, **kw):
        self.log.append(("read", self.li, int(ids.numel())))
        return self.t.gather(ids, dtype=dtype, **kw)

    def __getattr__(self, k):
        return getattr(self.t, k)


class _NowReader:
    """An executor that runs a job when it is handed over (the order of the log is the order of the hand-overs)."""

    def submit(self, fn, *args):
        from concurrent.futures import Future
        f = Future()
        f.set_result(fn(*args))
        return f


def test_a_chunk_asks_for_its_table_rows_before_the_layers():
    """SPD-319: with counting tables every Engram table's ids are handed to the reader before layer 0 runs, and no
    layer starts a read; the chunk's logits are the ones it gives without the counting."""
    cfg, ref, eng = tiny(torch.bfloat16)
    want = eng.prefill(PROMPT, eng.new_state(), head="all")
    log, real, reader = [], dict(eng.tables), getattr(eng, "_reader", None)
    eng.tables = {li: _CountingTable(t, li, log) for li, t in real.items()}
    eng._reader = _NowReader()
    eng.attention_chunk = lambda x, li, st, slots: (log.append(("layer", li)), E.Hy4FlashEngine.attention_chunk(eng, x, li, st, slots))[1]
    try:
        got = eng.prefill(PROMPT, eng.new_state(), head="all")
    finally:
        eng.tables, eng._reader = real, reader
        del eng.attention_chunk
    first_layer = log.index(("layer", 0))
    reads = [i for i, e in enumerate(log) if e[0] == "read"]
    assert sorted({log[i][1] for i in reads}) == sorted(real), log
    assert reads and max(reads) < first_layer, log[:12]
    assert torch.equal(got, want)
    return f"{len(reads)} reads of {len(real)} tables, all before layer 0; the logits unchanged"


def test_a_big_miss_list_goes_in_slices():
    """SPD-319: a cold reader with a kernel queue given 1,000 missed ids submits 4 slices of at most 256 and returns the
    rows the parent's read returns."""
    import tempfile
    import numpy as np
    from engine.models.flashnext.ple import DirectRows
    from engine.models.hy4_flash.table import BatchRows
    rows, row_bytes = 5000, 124
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rows.bin")
        with open(path, "wb") as f:
            f.write(np.random.default_rng(3).integers(0, 256, rows * row_bytes + 4096, dtype=np.uint8).tobytes())
        r = BatchRows(path, 64, row_bytes, rows, 1 << 20)
        own = r._ctx
        try:
            sent = []
            r._ctx = object()                                     # a kernel queue, as on Linux
            r._submit = lambda ids: (sent.append(int(ids.size)), ids)[1]
            r._collect = lambda job: DirectRows._read(r, job)
            ids = np.random.default_rng(4).choice(rows, 1000, replace=False).astype(np.int64)
            got = r._read(ids)
            want = DirectRows._read(r, ids)
        finally:
            r._ctx = own
            r.close()
    assert sent == [256, 256, 256, 232], sent
    assert np.array_equal(got, want)
    return "1,000 ids in submits of 256, 256, 256, 232: the parent's rows"


def _grams(eng, st, toks: torch.Tensor) -> torch.Tensor:
    """A chunk's n-grams after the state's last tokens (forward_chunk's own)."""
    n, hist = eng.cfg.engram_max_ngram_size, st.hist.cpu()
    full, k = torch.cat((hist, toks)), hist.shape[0]
    return torch.stack([full[k - j: k - j + toks.shape[0]] for j in range(n)], dim=-1)


def test_two_states_interleaved_keep_their_own_rows():
    """SPD-319: two chunks' table jobs handed over together and taken in the other order each give the rows a lone
    read gives; and two states' chunks prefilled in turn equal each one's lone prefill."""
    cfg, ref, eng = tiny(torch.bfloat16)
    g = torch.Generator().manual_seed(37)
    ta, tb = torch.randint(4, 60, (30,), generator=g), torch.randint(4, 60, (30,), generator=g)
    sa, sb = eng.new_state(), eng.new_state()
    ga, gb = _grams(eng, sa, ta), _grams(eng, sb, tb)
    ja, jb = eng._engram_reads(ga), eng._engram_reads(gb)
    for li in reversed(eng._elayers):
        assert torch.equal(eng._engram_take(li, jb[li]), eng.engram_rows(li, eng.hashes[li].ids(gb))), li
        assert torch.equal(eng._engram_take(li, ja[li]), eng.engram_rows(li, eng.hashes[li].ids(ga))), li
    lone = []
    for t in (ta, tb):
        st = eng.new_state()
        with _small_chunks():
            lone.append(eng.prefill(t, st, head="all"))
    got = [[], []]
    with _small_chunks():
        for i in range(0, 30, E.PREFILL_CHUNK):
            for k, (t, st) in enumerate(((ta, sa), (tb, sb))):
                got[k].append(eng.head(eng.forward_chunk(t[i:i + E.PREFILL_CHUNK], st)))
    assert torch.equal(torch.cat(got[0]), lone[0]) and torch.equal(torch.cat(got[1]), lone[1])
    return f"{len(eng._elayers)} tables' jobs crossed; two states' chunks of {E.PREFILL_CHUNK} in turn equal their lone prefills"


def test_a_sparse_row_reads_by_its_own_position():
    """SPD-323 (F1): on the prompt pass on fixed-order kernels, a chunk that crosses the row from which sparse layers
    read lists (index_topk) gives every row the read list the decode step takes for that row, in every sparse layer:
    the rows before the line every row up to theirs, the rows past it their groups' rows."""
    cfg, ref, eng = tiny()
    toks = PROMPT[:35]
    lists, real = {}, eng._sparse_reads

    def spy(x, qres, li, st, slots, cos, sin):
        ctl, sel = real(x, qres, li, st, slots, cos, sin)
        for t, p in enumerate(slots.tolist()):
            base, n, lst = ctl[t].tolist()
            lists[(li, p)] = sel[t, :n].tolist() if lst else list(range(base, base + n))
        return ctl, sel
    eng.prefill_v2, eng._sparse_reads = True, spy
    try:
        st = eng.new_state()
        eng.prefill(toks[:5], st, head=None)
        eng.prefill(toks[5:], st, head=None)          # rows 5 to 34: across row index_topk
    finally:
        eng.prefill_v2 = False
        del eng._sparse_reads
    want, cur, real_sel = {}, {}, eng._select

    def spy_sel(li, st, iq, hw, cos, sin, c):
        out = real_sel(li, st, iq, hw, cos, sin, c)
        cur[li] = out[0].tolist()
        return out
    eng._select = spy_sel
    try:
        st = eng.new_state()
        for p, t in enumerate(toks.tolist()):
            cur.clear()
            eng.step(int(t), st)
            for li in cfg.sparse_layers:
                want[(li, p)] = [q for q in cur[li] if q <= p] if li in cur else list(range(p + 1))
    finally:
        del eng._select
    rows = sorted(k for k in lists if k[1] >= 5)
    bad = [k for k in rows if lists[k] != want[k]]
    assert not bad, [(k, lists[k], want[k]) for k in bad[:3]]
    listed = sum(1 for k in rows if k[1] >= cfg.index_topk)
    assert listed and any(k[1] < cfg.index_topk for k in rows), rows
    return f"{len(rows)} (layer, row) read lists of a chunk across row {cfg.index_topk} equal the decode step's ({listed} lists)"


class _v2:
    """The prompt pass on fixed-order kernels (HY4_PREFILL_V2) for a block of the test."""

    def __init__(self, eng):
        self.eng = eng

    def __enter__(self):
        self.eng.prefill_v2 = True

    def __exit__(self, *exc):
        self.eng.prefill_v2 = False


def test_v2_matches_the_reference():
    """SPD-325 (F2): the prompt pass on fixed-order kernels against the reference forward, in one chunk and in chunks
    of 7 (attention 5 rows at a time, read lists 3), within the fp32 bound of the pass before."""
    cfg, ref, eng = tiny()
    want = ref.forward(PROMPT, Hy4State())
    with _v2(eng):
        got = eng.prefill(PROMPT, eng.new_state(), head="all")
        with _small_chunks():
            got2 = eng.prefill(PROMPT, eng.new_state(), head="all")
    err, err2 = (got - want).abs().max().item(), (got2 - want).abs().max().item()
    assert err < BOUND and err2 < BOUND, (err, err2)
    # and the state it leaves is the one decode steps continue
    st = eng.new_state()
    with _v2(eng), _small_chunks():
        eng.prefill(PROMPT[:19], st, head=None)
    steps = torch.cat([eng.step(int(t), st) for t in PROMPT[19:]])
    err3 = (steps - want[19:]).abs().max().item()
    assert err3 < BOUND, err3
    return f"45 rows: one chunk {err:.1e}, chunks of 7 {err2:.1e}; 19 prompt rows then 26 steps {err3:.1e}"


def test_every_prompt_rows_head_input_is_kept():
    """SPD-325 (F2): `prompt_hidden` holds the head input of every row of the last chunk, one row a token (the shared
    drafter's sync reads them all), and its last row gives the prompt's last logits; on both passes."""
    cfg, ref, eng = tiny()
    out = []
    for v2 in (False, True):
        eng.prefill_v2 = v2
        try:
            lg = eng.prefill(PROMPT, eng.new_state(), head="last")
            assert eng.prompt_hidden.shape == (PROMPT.shape[0], cfg.hidden_size), eng.prompt_hidden.shape
            assert torch.equal(eng.head(eng.prompt_hidden[-1:]), lg)
            with _small_chunks():
                eng.prefill(PROMPT, eng.new_state(), head=None)
            assert eng.prompt_hidden.shape == (PROMPT.shape[0] % 7 or 7, cfg.hidden_size), eng.prompt_hidden.shape
        finally:
            eng.prefill_v2 = False
        out.append(f"{'v2' if v2 else 'before'}: {PROMPT.shape[0]} rows, then the last chunk's {PROMPT.shape[0] % 7 or 7}")
    return "; ".join(out)


def test_the_block_select_gives_the_stable_sorts_groups():
    """SPD-323 (F3): at the packs' shapes (groups of 4, 512 read) the shared block select with a row's own group at +inf
    and the groups past it at -inf (`pick_positions`, its torch form on the CPU, `qsa_select_torch`) reads the 511
    best closed groups by the stable sort (`topk_stable`, ties to the lower group), ascending, then the row's own:
    the decode step's list, ties included."""
    from engine.models.flashnext.ref import topk_stable
    g = torch.Generator().manual_seed(41)
    n = 0
    for G in (600, 2000):
        logit = torch.randint(0, 40, (6, G + 1), generator=g).float() / 7          # many ties
        qg = torch.tensor([G, G - 1, G - 5, 511, 512, (G + 512) // 2])
        got = E.pick_positions(logit, qg, 512, 4)
        for r in range(6):
            q = int(qg[r])
            top = topk_stable(logit[r, :q], 511)[1].sort().values
            want = (torch.cat((top, torch.tensor([q]))).unsqueeze(-1) * 4 + torch.arange(4)).flatten()
            assert torch.equal(got[r].long(), want), (G, r)
            n += 1
    return f"{n} rows over 600 and 2,000 groups with tied scores: the stable sort's 511 groups and the own group"


def test_capacity_is_checked():
    cfg, ref, eng = tiny()
    st = eng.new_state(12)
    eng.prefill(PROMPT[:11], st, head=None)
    eng.step(5, st)
    for call in (lambda: eng.step(5, st), lambda: eng.prefill(PROMPT[:2], st)):
        try:
            call()
            raise SystemExit("a row past the capacity was taken")
        except ValueError:
            pass
    return "a full state refuses a row"


if __name__ == "__main__":
    fails = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            print(f"  {name:<52} ok   {fn() or ''}")
        except AssertionError as e:
            fails += 1
            print(f"  {name:<52} FAIL {e}")
    print(f"{len(tests) - fails} passed" + (f", {fails} FAILED" if fails else ""))
    sys.exit(1 if fails else 0)
