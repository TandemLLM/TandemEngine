"""LFM2 MoE on Apple silicon: the MLX board (engine/models/lfm2_moe/mlx_board.py) against the torch reference.

The tiny random model of tests/test_lfm2_moe_ref.py runs on both. Held: prefill and decode logits equal
the reference's in fp32 within BOUND; a verify tree's rows equal the sequential decode of each row's
path; a commit of a path that leaves the chain continues as the plain decode; bf16 greedy decode picks
the reference's tokens. Skipped where MLX is not installed (any machine but a Mac).
"""

from __future__ import annotations

import functools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import mlx.core as mx
except ImportError:  # not a Mac
    mx = None

from tests.test_lfm2_moe_ref import PROMPT, tiny  # noqa: E402

# fp32 against fp32 torch on the CPU: summation order over 6 layers
BOUND = 1e-4
#: the block sizes a row kernel is held at, bit for bit against its rows alone: every one from the decode
#: step to 32 rows (docs/kernels.md, "Verify twins"), and one past it
ROWS = tuple(range(1, 34))
#: 33 rows of 5 distinct inputs: rows that share an input pick the same experts
SHARED = [0] * 10 + [1, 1, 2, 2, 3, 4] + [0, 1, 2, 3, 4, 4, 3, 2, 1, 0, 0, 2, 4, 1, 3, 0, 4]


def _skip(why: str):
    try:
        import pytest
    except ImportError:                                     # the box's suite: script mode, no pytest
        from scriptmode import Skip
        raise Skip(why) from None
    pytest.skip(why)


def on(device: str):
    """A test on MLX's `device`, skipped where MLX is not installed, its random draws seeded. The algebra
    tests run on "cpu": on the GPU an fp32 matmul of 2 rows or more is computed at about TF32 precision
    (measured on an M5 Max, MLX 0.32.4: 9e-3 off float64 on a 32-wide product whose one-row form is 1e-6
    off), which a 1e-4 bound cannot hold; the "gpu" tests are the bit-for-bit ones."""
    def deco(fn):
        @functools.wraps(fn)
        def test():
            if mx is None:
                _skip("MLX is not installed (Apple silicon only)")
            old = mx.default_device()
            mx.set_default_device(getattr(mx, device))
            mx.random.seed(0)
            try:
                return fn()
            finally:
                mx.set_default_device(old)
        return test
    return deco


def board(dtype=None):
    from engine.models.lfm2_moe.mlx_board import Lfm2MoeMLX
    cfg, W, ref = tiny()
    return cfg, ref, Lfm2MoeMLX.from_torch(cfg, W, dtype or mx.float32)


def np_(a) -> np.ndarray:
    return np.array(a.astype(mx.float32))


@on("cpu")
def test_conv_taps_follow_the_path():
    from engine.models.lfm2_moe.mlx_board import conv_taps
    from engine.models.lfm2_moe.ref import tree_paths
    # S = 2: state rows 0, 1; block rows at 2.. ; tree 0 -> {1 -> 2, 3}
    t = conv_taps(tree_paths([-1, 0, 1, 0]), 2)
    assert t.tolist() == [[0, 1, 2], [1, 2, 3], [2, 3, 4], [1, 2, 5]], t.tolist()


@on("cpu")
def test_prefill_and_decode_match_reference():
    cfg, ref, b = board()
    st, sm = ref.new_state(64), b.new_state(64)
    want, got = ref.prefill(PROMPT, st), np_(b.prefill(PROMPT.numpy(), sm))
    worst = np.abs(got - want.numpy()).max()
    assert worst < BOUND, worst
    for tok in (11, 50, 3):
        d = np.abs(np_(b.decode(tok, sm)) - ref.decode(tok, st).numpy()).max()
        assert d < BOUND, d
        worst = max(worst, d)
    assert sm.length == st.length


@on("cpu")
def test_chunked_prefill_equals_one_block():
    cfg, ref, b = board()
    s1, s2 = b.new_state(64), b.new_state(64)
    a = np_(b.prefill(PROMPT.numpy(), s1))
    c = np_(b.prefill(PROMPT.numpy(), s2, chunk=5))
    assert np.abs(a - c).max() < BOUND


@on("cpu")
def test_verify_tree_equals_sequential_decode():
    import copy
    cfg, ref, b = board()
    st = b.new_state(64)
    b.prefill(PROMPT.numpy(), st)
    tokens = [7, 19, 23, 44, 31, 60]
    parents = [-1, 0, 1, 1, 0, 4]
    snap = copy.copy(st)
    snap.conv, snap.k, snap.v = dict(st.conv), dict(st.k), dict(st.v)
    lg = np_(b.rows(tokens, parents, st))
    assert st.length == snap.length
    paths = {0: [0], 1: [0, 1], 2: [0, 1, 2], 3: [0, 1, 3], 4: [0, 4], 5: [0, 4, 5]}
    for row, path in paths.items():
        s2 = b.new_state(64)
        b.prefill(PROMPT.numpy(), s2)
        for r in path:
            want = np_(b.decode(tokens[r], s2))
        d = np.abs(lg[row] - want[0]).max()
        assert d < BOUND, (row, d)


@on("cpu")
def test_commit_of_a_path_continues_exactly():
    cfg, ref, b = board()
    st, alone = b.new_state(64), b.new_state(64)
    b.prefill(PROMPT.numpy(), st)
    b.prefill(PROMPT.numpy(), alone)
    b.rows([7, 19, 23, 44, 31], [-1, 0, 1, 1, 0], st)
    b.commit(st, [0, 1, 3])
    for t in (7, 19, 44):
        b.decode(t, alone)
    assert st.length == alone.length
    d = np.abs(np_(b.decode(5, st)) - np_(b.decode(5, alone))).max()
    assert d < BOUND, d


@on("cpu")
def test_commit_refuses_a_non_path():
    cfg, ref, b = board()
    st = b.new_state(64)
    b.prefill(PROMPT.numpy(), st)
    b.rows([7, 19, 23], [-1, 0, 0], st)
    try:
        b.commit(st, [0, 1, 2])
    except ValueError as e:
        assert "not a root path" in str(e), e
    else:
        raise AssertionError("a commit of rows that are no root path went through")


@on("gpu")
def test_bf16_greedy_follows_reference():
    """bf16 on the Mac against the fp32 reference, both fed the reference's greedy tokens for 16 steps:
    the same argmax wherever the reference's top two are more than TIE apart (bf16 moves this model's
    logits, up to 17, by up to about 0.8; a free-running comparison forks at the first near tie)."""
    TIE = 1.0
    cfg, ref, b = board(mx.bfloat16)
    st, sm = ref.new_state(64), b.new_state(64)
    r, m = ref.prefill(PROMPT, st)[-1].numpy(), np_(b.prefill(PROMPT.numpy(), sm))[-1]
    for i in range(16):
        top2 = np.sort(r)[-2:]
        assert r.argmax() == m.argmax() or top2[1] - top2[0] < TIE, (i, top2)
        t = int(r.argmax())
        r, m = ref.decode(t, st)[0].numpy(), np_(b.decode(t, sm))[0]


@on("cpu")
def test_generate_equals_decode_loop():
    """The pipelined greedy loop writes what one `decode` at a time writes."""
    cfg, ref, b = board()
    s1, s2 = b.new_state(64), b.new_state(64)
    t0 = int(np_(b.prefill(PROMPT.numpy(), s1))[-1].argmax())
    b.prefill(PROMPT.numpy(), s2)
    want, t = [], t0
    for _ in range(10):
        t = int(np_(b.decode(t, s1))[0].argmax())
        want.append(t)
    assert list(b.generate(t0, s2, 10)) == want
    assert s2.length == s1.length + 1          # generate queued one step past the last token read


@on("gpu")
def test_gpu_verify_tree_rows_are_the_decode_rows_bit_for_bit():
    """On the GPU in bf16 each row of a draft tree is the sequential decode of its path, bit for bit (the
    row-invariant kernels, mlx_kernels.py): greedy speculation writes the plain greedy text."""
    cfg, ref, b = board(mx.bfloat16)
    tokens, parents = [7, 19, 23, 44, 31, 60], [-1, 0, 1, 1, 0, 4]
    st = b.new_state(64)
    b.prefill(PROMPT.numpy(), st)
    lg = np_(b.rows(tokens, parents, st))
    paths = {0: [0], 1: [0, 1], 2: [0, 1, 2], 3: [0, 1, 3], 4: [0, 4], 5: [0, 4, 5]}
    for row, path in paths.items():
        s2 = b.new_state(64)
        b.prefill(PROMPT.numpy(), s2)
        for r in path:
            want = np_(b.decode(tokens[r], s2))
        assert np.array_equal(lg[row], want[0]), (row, np.abs(lg[row] - want[0]).max())


@on("gpu")
def test_gpu_commit_then_decode_is_bit_exact():
    """A tree's path kept, then decoding on: the plain decode's logits bit for bit (conv state and KV rows
    moved by the commit are the decode's own)."""
    cfg, ref, b = board(mx.bfloat16)
    st, alone = b.new_state(64), b.new_state(64)
    b.prefill(PROMPT.numpy(), st)
    b.prefill(PROMPT.numpy(), alone)
    b.rows([7, 19, 23, 44, 31], [-1, 0, 1, 1, 0], st)
    b.commit(st, [0, 1, 3])
    for t in (7, 19, 44):
        b.decode(t, alone)
    for t in (5, 9, 13):
        assert np.array_equal(np_(b.decode(t, st)), np_(b.decode(t, alone))), t


@on("gpu")
def test_gpu_dense_block_rows_are_its_rows_alone_bit_for_bit():
    """Every tabled projection shape (mlx_kernels.MMA): a block's rows on the scalar chain kernel (few rows) and
    on the matrix units (more) are the rows computed one at a time, bit for bit, and close to the fp32 product."""
    from engine.models.lfm2_moe import mlx_kernels as K
    for (N, Kd) in K.MMA:
        if N * Kd > 5e7:
            continue                       # the head: 0.5 GB of weights a case
        w = (mx.random.normal((N, Kd)) * 0.02).astype(mx.bfloat16)
        for M in ROWS:
            x = mx.random.normal((M, Kd)).astype(mx.bfloat16)
            y = K.dense(x, w)
            alone = mx.concatenate([K.dense(x[r:r + 1], w) for r in range(M)])
            assert mx.array_equal(y, alone).item(), (N, Kd, M)
            ref = x.astype(mx.float32) @ w.astype(mx.float32).T
            assert float(mx.abs(y.astype(mx.float32) - ref).max()) < 2e-2 * float(mx.abs(ref).max()), (N, Kd, M)


@on("gpu")
def test_gpu_dense_fused_residual_and_fp32_out_are_bit_exact():
    """Fused residual add and direct fp32 output in rowchain/rowmma are bit-for-bit row-invariant."""
    from engine.models.lfm2_moe import mlx_kernels as K
    w = (mx.random.normal((2048, 2048)) * 0.02).astype(mx.bfloat16)
    for M in ROWS:
        x = mx.random.normal((M, 2048)).astype(mx.bfloat16)
        res = mx.random.normal((M, 2048)).astype(mx.bfloat16)
        y = K.dense(x, w, res=res)
        alone = mx.concatenate([K.dense(x[r:r + 1], w, res=res[r:r + 1]) for r in range(M)])
        assert mx.array_equal(y, alone).item(), ("res", M)
        y32 = K.dense(x, w, out_dtype=mx.float32)
        alone32 = mx.concatenate([K.dense(x[r:r + 1], w, out_dtype=mx.float32) for r in range(M)])
        assert mx.array_equal(y32, alone32).item(), ("fp32", M)


@on("gpu")
def test_gpu_flash_path_attention_is_bit_exact():
    """Split-K FlashAttention-2 path_attention (D=64, GQA=4, bf16) is bit-for-bit identical between
    a 16-node branching tree and single-row sequential decode across BT=32 and BK=256 boundaries."""
    from engine.models.lfm2_moe import mlx_kernels as K
    from engine.models.lfm2_moe.ref import tree_paths
    Hk, Hq, D, gqa = 8, 32, 64, 4
    parents = [-1, 0, 1, 2, 1, 4, 0, 6, 7, 8, 6, 0, 11, 12, 2, 14]
    paths = tree_paths(parents)
    M, A = len(parents), max(len(p) for p in paths)
    anc_np = np.zeros((M, A), dtype=np.int32)
    for i, pa in enumerate(paths):
        anc_np[i, :len(pa)] = pa
    anc = mx.array(anc_np)
    dlen = mx.array([len(p) for p in paths], dtype=mx.int32)
    for L in (0, 30, 254, 513):
        cap = L + M + 8
        kc = mx.random.normal((Hk, cap, D)).astype(mx.bfloat16)
        vc = mx.random.normal((Hk, cap, D)).astype(mx.bfloat16)
        q = mx.random.normal((M, Hq, D)).astype(mx.bfloat16)
        out_tree = K.path_attention(q, kc, vc, anc, dlen, L, gqa)
        for r, pa in enumerate(paths):
            k_seq = mx.concatenate([kc[:, :L], kc[:, L + mx.array(pa, dtype=mx.int32)]], axis=1)
            v_seq = mx.concatenate([vc[:, :L], vc[:, L + mx.array(pa, dtype=mx.int32)]], axis=1)
            out_dec = K.path_attention(q[r:r + 1], k_seq, v_seq, mx.array([[0]], dtype=mx.int32),
                                       mx.array([1], dtype=mx.int32), L + len(pa) - 1, gqa)
            assert mx.array_equal(out_tree[r:r + 1], out_dec).item(), (L, r, pa)


@on("gpu")
def test_gpu_path_attention_rows_are_its_rows_alone_at_every_block_size():
    """A chain block of every size in ROWS: each row's attention is the one-row call at its own position, bit
    for bit, at context lengths on both sides of the 256-token chunk (one chunk, the chunk-then-reduce path)."""
    from engine.models.lfm2_moe import mlx_kernels as K
    Hk, Hq, D, gqa = 8, 32, 64, 4
    one, d1 = mx.array([[0]], dtype=mx.int32), mx.array([1], dtype=mx.int32)
    for L in (0, 250, 600):
        cap = L + max(ROWS) + 8
        kc = mx.random.normal((Hk, cap, D)).astype(mx.bfloat16)
        vc = mx.random.normal((Hk, cap, D)).astype(mx.bfloat16)
        qs = mx.random.normal((max(ROWS), Hq, D)).astype(mx.bfloat16)
        alone = mx.concatenate([K.path_attention(qs[r:r + 1], kc, vc, one, d1, L + r, gqa) for r in range(max(ROWS))])
        for M in ROWS:
            anc = mx.array(np.tril(np.tile(np.arange(M, dtype=np.int32), (M, 1))))
            dlen = mx.arange(1, M + 1, dtype=mx.int32)
            out = K.path_attention(qs[:M], kc, vc, anc, dlen, L, gqa)
            assert mx.array_equal(out, alone[:M]).item(), (L, M)


@on("gpu")
def test_gpu_conv_gate_rows_are_the_decode_steps_at_every_block_size():
    """The short conv of a chain block of every size in ROWS: each row is the decode step's (`next_state`),
    walked one row at a time from the same state, bit for bit."""
    from engine.models.lfm2_moe import mlx_kernels as K
    H, Kt = 256, 3
    cw = (mx.random.normal((H, Kt)) * 0.3).astype(mx.bfloat16)
    state0 = mx.random.normal((Kt - 1, H)).astype(mx.bfloat16)
    bcx = mx.random.normal((max(ROWS), 3 * H)).astype(mx.bfloat16)
    steps, st = [], state0
    for r in range(max(ROWS)):
        cy, st = K.conv_gate(bcx[r:r + 1], st, cw, None, next_state=True)
        mx.eval(cy, st)
        steps.append(cy)
    steps = mx.concatenate(steps)
    for M in ROWS:
        cy, _ = K.conv_gate(bcx[:M], state0, cw)
        assert mx.array_equal(cy, steps[:M]).item(), M


@on("gpu")
def test_gpu_qkv_norm_rope_rows_are_its_rows_alone_at_every_block_size():
    """The fused q/k norm and rotary of a block of every size in ROWS: each row is the one-row call's, bit for
    bit (q, k and v)."""
    from engine.models.lfm2_moe import mlx_kernels as K
    Hq, Hk, D = 32, 8, 64
    n = max(ROWS)
    qkv = mx.random.normal((n, (Hq + 2 * Hk) * D)).astype(mx.bfloat16)
    qn = (mx.random.normal((D,)) * 0.2 + 1.0).astype(mx.bfloat16)
    kn = (mx.random.normal((D,)) * 0.2 + 1.0).astype(mx.bfloat16)
    cos = mx.random.normal((n, 1, D)).astype(mx.bfloat16)
    sin = mx.random.normal((n, 1, D)).astype(mx.bfloat16)
    alone = [K.qkv_norm_rope(qkv[r:r + 1], qn, kn, cos[r:r + 1], sin[r:r + 1], Hq, Hk, D, 1e-5) for r in range(n)]
    q1 = mx.concatenate([a[0] for a in alone])
    k1 = mx.concatenate([a[1] for a in alone], axis=1)
    v1 = mx.concatenate([a[2] for a in alone], axis=1)
    for M in ROWS:
        q, k, v = K.qkv_norm_rope(qkv[:M], qn, kn, cos[:M], sin[:M], Hq, Hk, D, 1e-5)
        assert mx.array_equal(q, q1[:M]).item() and mx.array_equal(k, k1[:, :M]).item() \
            and mx.array_equal(v, v1[:, :M]).item(), M


@on("gpu")
def test_gpu_moe_fast_is_bit_exact():
    """Fused MoE fast path (moegate + moeroute + moemm_r1/moemm_hyb + moewsum) is bit-for-bit row-invariant
    across ns=1, ns=2, and ns>=3..10 (MMA) simultaneously."""
    from engine.models.lfm2_moe import mlx_kernels as K
    H, E, I = 256, 16, 128
    nw = mx.ones((H,), dtype=mx.bfloat16)
    gw = (mx.random.normal((E, H)) * 0.1).astype(mx.bfloat16)
    bias = (mx.random.normal((E,)) * 0.1).astype(mx.float32)
    w13 = (mx.random.normal((E, 2 * I, H)) * 0.05).astype(mx.bfloat16)
    w2 = (mx.random.normal((E, H, I)) * 0.05).astype(mx.bfloat16)
    base = mx.random.normal((5, H)).astype(mx.bfloat16)
    h32 = base[mx.array(SHARED, dtype=mx.int32)]
    for M in ROWS[1:]:
        hm = h32[:M]
        ym = K.moe_fast(hm, nw, gw, bias, w13, w2, 1e-5, 4, 1.0, True)
        alone = mx.concatenate([K.moe_fast(hm[r:r + 1], nw, gw, bias, w13, w2, 1e-5, 4, 1.0, True) for r in range(M)])
        assert mx.array_equal(ym, alone).item(), M


@on("gpu")
def test_gpu_swizzle_roundtrip_and_dense_swiz_bit_exact():
    """512B-coalesced weight swizzle roundtrips exactly, and dense_swiz (plain, FUSE_RMS, ADD_RES)
    is bit-for-bit identical between rowchain_swiz (M<=2) and rowmma_swiz (M>=3)."""
    from engine.models.lfm2_moe import mlx_kernels as K
    for N, Kd in ((64, 128), (128, 256), (384, 128)):
        w = (mx.random.normal((N, Kd)) * 0.02).astype(mx.bfloat16)
        ws = K.swizzle_dense(w)
        assert mx.array_equal(K.unswizzle_dense(ws, N, Kd), w).item(), (N, Kd)
        nw = (mx.random.normal((Kd,)) * 0.2 + 1.0).astype(mx.bfloat16)
        for M in ROWS:
            x = mx.random.normal((M, Kd)).astype(mx.bfloat16)
            res = mx.random.normal((M, N)).astype(mx.bfloat16)
            # Plain
            y = K.dense_swiz(x, ws, N, Kd)
            alone = mx.concatenate([K.dense_swiz(x[r:r + 1], ws, N, Kd) for r in range(M)])
            assert mx.array_equal(y, alone).item(), ("plain", N, Kd, M)
            ref = x.astype(mx.float32) @ w.astype(mx.float32).T
            assert float(mx.abs(y.astype(mx.float32) - ref).max()) < 2e-2 * float(mx.abs(ref).max())
            # Fused RMSNorm
            yr = K.dense_swiz(x, ws, N, Kd, nw=nw, eps=1e-5)
            aloner = mx.concatenate([K.dense_swiz(x[r:r + 1], ws, N, Kd, nw=nw, eps=1e-5) for r in range(M)])
            assert mx.array_equal(yr, aloner).item(), ("rms", N, Kd, M)
            # Fused residual add
            ya = K.dense_swiz(x, ws, N, Kd, res=res)
            alonea = mx.concatenate([K.dense_swiz(x[r:r + 1], ws, N, Kd, res=res[r:r + 1]) for r in range(M)])
            assert mx.array_equal(ya, alonea).item(), ("add", N, Kd, M)


@on("gpu")
def test_gpu_moe_fast_swiz_is_bit_exact():
    """512B-coalesced swizzled MoE pipeline (moegate_rms + moeroute + moemm_r1_swiz/moemm_w2add_swiz vs
    moemm_hyb_swiz + moewsum) roundtrips weights and is bit-for-bit row-invariant at every block size in ROWS."""
    from engine.models.lfm2_moe import mlx_kernels as K
    H, E, I = 256, 16, 128
    nw = (mx.random.normal((H,)) * 0.2 + 1.0).astype(mx.bfloat16)
    gw = (mx.random.normal((E, H)) * 0.1).astype(mx.bfloat16)
    bias = (mx.random.normal((E,)) * 0.1).astype(mx.float32)
    w13 = (mx.random.normal((E, 2 * I, H)) * 0.05).astype(mx.bfloat16)
    w2 = (mx.random.normal((E, H, I)) * 0.05).astype(mx.bfloat16)
    w13s = K.swizzle_moe_w13(w13)
    w2s = K.swizzle_moe_w2(w2)
    assert mx.array_equal(K.unswizzle_moe_w13(w13s, E, I, H), w13).item()
    assert mx.array_equal(K.unswizzle_moe_w2(w2s, E, H, I), w2).item()
    base = mx.random.normal((5, H)).astype(mx.bfloat16)
    h32 = base[mx.array(SHARED, dtype=mx.int32)]
    for M in ROWS[1:]:
        hm = h32[:M]
        ym = K.moe_fast_swiz(hm, nw, gw, bias, w13s, w2s, I, 1e-5, 4, 1.0, True)
        alone = mx.concatenate([K.moe_fast_swiz(hm[r:r + 1], nw, gw, bias, w13s, w2s, I, 1e-5, 4, 1.0, True)
                                for r in range(M)])
        assert mx.array_equal(ym, alone).item(), M
        ref = K.moe_fast(hm, nw, gw, bias, w13, w2, 1e-5, 4, 1.0, True)
        assert float(mx.abs(ym.astype(mx.float32) - ref.astype(mx.float32)).max()) < 0.05 * float(mx.abs(ref.astype(mx.float32)).max())


def _swiz_board(**over):
    """A tiny random BF16 board whose shapes take the swizzled row kernels (the served layout). `over`
    changes the config: head_dim=64 with 8 query and 2 key heads is the served attention shape (GQA 4)."""
    from engine.models.lfm2_moe.config import config_from_dict
    from engine.models.lfm2_moe.mlx_board import Lfm2MoeMLX
    from tests.test_lfm2_moe_ref import TINY, _random_weights
    raw = dict(TINY, vocab_size=64, hidden_size=128, intermediate_size=128, moe_intermediate_size=64,
               num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
               layer_types=["conv", "full_attention", "conv", "full_attention"],
               num_dense_layers=1, num_experts=8, num_experts_per_tok=4)
    raw.update(over)
    cfg = config_from_dict(raw)
    b = Lfm2MoeMLX.from_torch(cfg, _random_weights(cfg, seed=7), mx.bfloat16)
    assert b._swiz is True
    return b


def _served_board():
    return _swiz_board(hidden_size=256, num_attention_heads=8, num_key_value_heads=2, head_dim=64)


@on("gpu")
def test_gpu_swizzled_board_verify_tree_and_batched_prefill():
    """End to end on the swizzled board: a verify tree's rows are the sequential decode's, bit for bit; a
    prompt past ROWS_MAX takes the batched path and gives finite logits of the right shape."""
    b = _swiz_board()
    # 1. Verify tree matches sequential decode bit for bit
    prompt = np.array([1, 5, 9, 13, 17], dtype=np.int32)
    tokens, parents = [7, 19, 23, 44, 31, 60], [-1, 0, 1, 1, 0, 4]
    st = b.new_state(128)
    b.prefill(prompt, st)
    lg = np_(b.rows(tokens, parents, st))
    paths = {0: [0], 1: [0, 1], 2: [0, 1, 2], 3: [0, 1, 3], 4: [0, 4], 5: [0, 4, 5]}
    for row, path in paths.items():
        s2 = b.new_state(128)
        b.prefill(prompt, s2)
        for r in path:
            want = np_(b.decode(tokens[r], s2))
        assert np.array_equal(lg[row], want[0]), (row, np.abs(lg[row] - want[0]).max())
    # 2. a prompt of 70 rows (past ROWS_MAX): the batched kernels
    long_prompt = np.arange(70, dtype=np.int32) % 64
    sb = b.new_state(128)
    lg_batched = np_(b.prefill(long_prompt, sb))
    assert lg_batched.shape == (70, 64) and np.isfinite(lg_batched).all()


@on("gpu")
def test_gpu_self_check_holds_bits_on_chains_a_tree_and_a_long_prompt():
    """The load-time gate (mlx_serving.self_check): `ok` is `bitwise`, over the greedy chain, the chains of
    CHECK_CHAINS rows and the tree after a prompt longer than one attention chunk."""
    from engine.models.lfm2_moe import mlx_serving as S
    res = S.self_check(_swiz_board(), [int(t) for t in np.arange(20) % 64], steps=24, log=None)
    assert res["ok"] and res["bitwise"] and res["picks"] and res["max_diff"] == 0.0, res
    assert res["rows"] == 24 + sum(S.CHECK_CHAINS) + len(S.CHECK_TREE[0]) and res["first_diff"] is None, res
    assert {3, 4, 32} <= set(S.CHECK_CHAINS), "the 4-row attention and router variants, and the cut's widest block"


@on("gpu")
def test_gpu_self_check_refuses_one_unit_in_one_row():
    """The gate itself: a verify row that differs from the decode step's in one logit's last place, with the
    same greedy pick, fails the check and names the row."""
    from engine.models.lfm2_moe import mlx_serving as S
    b = _swiz_board()
    real, calls = b.rows, []

    def rows(tokens, parents, st, last_only=False):
        out = np.array(real(tokens, parents, st, last_only=last_only))
        calls.append(len(tokens))
        if len(calls) == 2 and not last_only:                       # call 1 is the prompt's prefill
            j = int(np.argmin(out[3]))                               # not the pick
            out[3, j] = np.nextafter(out[3, j], np.float32(np.inf))
        return mx.array(out)

    b.rows = rows
    res = S.self_check(b, [int(t) for t in np.arange(20) % 64], steps=24, log=None)
    assert res["ok"] is False and res["bitwise"] is False and res["first_diff"] == 3, res
    assert res["max_diff"] > 0.0 and res["picks"] is True, res


@on("gpu")
def test_gpu_served_attention_shape_rows_are_the_decode_steps():
    """The served attention shape (head_dim 64, GQA 4: the compiled flash path, `_compiled_attn`) through
    the whole board, after a prompt past the 256-token chunk: chains up to ROWS_MAX rows and a tree are the
    sequential decode's rows, bit for bit."""
    from engine.models.lfm2_moe import mlx_kernels as K
    b = _served_board()
    prompt = [int(t) for t in (np.arange(300) * 7) % 64]
    st = b.new_state(300 + K.ROWS_MAX + 8)
    lg = b.prefill(prompt, st, last_only=True)
    L, conv = st.length, dict(st.conv)
    toks, want = [int(mx.argmax(lg[-1]).item())], []
    for _ in range(K.ROWS_MAX):
        want.append(np.array(b.decode(toks[-1], st)[0]))
        toks.append(int(want[-1].argmax()))
    for M in (1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, K.ROWS_MAX):
        st.length, st.conv = L, dict(conv)
        got = np.array(b.rows(toks[:M], list(range(-1, M - 1)), st))
        for r in range(M):
            assert got[r].tobytes() == want[r].tobytes(), (M, r)
    assert b._attn_cc, "the compiled flash attention never ran: the board took the fallback kernel"
    tokens, parents = [7, 19, 23, 44, 31, 60], [-1, 0, 1, 1, 0, 4]
    st.length, st.conv = L, dict(conv)
    got = np.array(b.rows(tokens, parents, st))
    for row, path in {0: [0], 1: [0, 1], 2: [0, 1, 2], 3: [0, 1, 3], 4: [0, 4], 5: [0, 4, 5]}.items():
        st.length, st.conv = L, dict(conv)
        for r in path:
            out = np.array(b.decode(tokens[r], st)[0])
        assert got[row].tobytes() == out.tobytes(), row


@on("gpu")
def test_gpu_dense_swiglu_and_embed_rows_are_their_rows_alone_at_every_block_size():
    """The dense MLP's fused gate and up projection and the embedding lookup on the swizzled layout: a block
    of every size in ROWS is its rows one at a time, bit for bit."""
    from engine.models.lfm2_moe import mlx_kernels as K
    H, I, n = 128, 192, max(ROWS)
    w13 = K.swizzle_moe_w13((mx.random.normal((1, 2 * I, H)) * 0.05).astype(mx.bfloat16))
    x = mx.random.normal((5, H)).astype(mx.bfloat16)[mx.array(SHARED, dtype=mx.int32)]
    alone = mx.concatenate([K.dense_swiglu_swiz(x[r:r + 1], w13, I, H) for r in range(n)])
    emb = (mx.random.normal((64, H))).astype(mx.bfloat16)
    ws = K.swizzle_dense(emb)
    toks = mx.array([(7 * i) % 64 for i in range(n)], dtype=mx.int32)
    for M in ROWS:
        assert mx.array_equal(K.dense_swiglu_swiz(x[:M], w13, I, H), alone[:M]).item(), ("swiglu", M)
        assert mx.array_equal(K.embed_swiz(toks[:M], ws, H), emb[toks[:M]]).item(), ("embed", M)


@on("gpu")
def test_gpu_serving_comes_back_from_an_in_place_ask():
    """The server's in-place ask (server/app.py `_ask_in_place`): settle, forward a hidden question,
    `resume_at` the kept length. The next step is the one a serving that was never asked makes, bit for
    bit: the conv states come back, not only the length. A length nothing kept is refused."""
    from engine.models.lfm2_moe.mlx_serving import Lfm2MoeMLXServing
    prompt = torch.tensor([int(t) for t in np.arange(20) % 64])
    asked, plain = Lfm2MoeMLXServing(_swiz_board(), 128), Lfm2MoeMLXServing(_swiz_board(), 128)
    for srv in (asked, plain):
        srv.forward(prompt, start=0)
    asked.settle()
    n = asked.kv.length
    asked.forward(torch.tensor([5, 9, 11, 3]), start=n)
    asked.resume_at(n)
    assert asked.kv.length == n
    a, b = asked.forward(torch.tensor([7]), start=n), plain.forward(torch.tensor([7]), start=n)
    assert torch.equal(a, b), float((a - b).abs().max())
    for bad in (lambda: asked.resume_at(n - 3), lambda: setattr(asked.kv, "length", n - 3)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("a length nothing kept went through")
    block = torch.arange(65) % 64
    try:
        asked.forward_tree(block, list(range(-1, 64)), start=asked.kv.length)
    except ValueError as e:
        assert "exceeds the row kernels" in str(e), e
    else:
        raise AssertionError("a verify block past ROWS_MAX went through")


def test_the_lfm2_family_picks_its_serving_by_the_adapters_backend():
    """engine/models/lfm2_moe/dispatch.py: the MLX serving where the device's adapter answers "mlx", the
    CUDA serving for "torch"; neither module is imported for the other's device."""
    import types
    import tools.gpu as g
    from engine.models.lfm2_moe import dispatch
    names = ("engine.models.lfm2_moe.mlx_serving", "engine.models.lfm2_moe.serving")
    kept, real, pkg = {n: sys.modules.get(n) for n in names}, g.profile, sys.modules["engine.models.lfm2_moe"]
    attrs = {n: getattr(pkg, n.rsplit(".", 1)[1], None) for n in names}
    try:
        for n in names:
            short = n.rsplit(".", 1)[1]
            m = types.ModuleType(n)
            m.serve = lambda a, ctx, short=short: {"by": short}
            sys.modules[n] = m
            setattr(pkg, short, m)
        for backend, want in (("mlx", "mlx_serving"), ("torch", "serving")):
            g.profile = lambda device=None, backend=backend: types.SimpleNamespace(backend=lambda: backend)
            assert dispatch.serve(None, None) == {"by": want}, backend
    finally:
        g.profile = real
        for n in names:
            short = n.rsplit(".", 1)[1]
            if kept[n] is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = kept[n]
            if attrs[n] is None:
                if hasattr(pkg, short):
                    delattr(pkg, short)
            else:
                setattr(pkg, short, attrs[n])


if __name__ == "__main__":
    from scriptmode import run
    run(globals())
