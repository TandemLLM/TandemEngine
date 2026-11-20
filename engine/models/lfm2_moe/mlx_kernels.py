"""Row-invariant Metal kernels for LFM2 MoE on Apple silicon (the MLX board's block path, mlx_board.py).

MLX's own matmul and attention pick their kernel and tiling by the number of rows, so a draft block's row
and the decode step's row of the same token round differently and a greedy speculative run drifts from the
plain greedy text at near ties (measured: prose, code and chat parted 8 to 56 % into a 512-token answer).
These kernels compute every row from its own inputs in an order fixed by the shapes of the weights and the
row's own keys, never by the row count, so a verified row is the decode step's row bit for bit:

  dense(x, w, res, out_dtype) y = x @ w.T (+ res): scalar chain kernel (few rows) or 8x8 bfloat matrix units
  dense_swiz(...)             512B-coalesced ulong2 swizzled dense projection (`rowchain_swiz` / `rowmma_swiz`)
                              with optional fused 32-lane simd_sum RMSNorm and fused residual add
  moe_fast_swiz(...)          512B-coalesced swizzled MoE pipeline: fused RMSNorm+gate (`moegate`), single-TG
                              router+LPT grouping (`moeroute`), dedicated M=1 w13+SwiGLU+RMSNorm (`moemm_r1_swiz`)
                              and fused w2+wsum+residual (`moemm_w2add_swiz`), or M>1 hybrid (`moemm_hyb_swiz`)
  moe_fast(...)               unswizzled MoE fast path (for tests/unswizzled checkpoints)
  conv_gate(bcx, state, cw, taps)
                              fused double-gated short convolution without mx.concatenate
  qkv_norm_rope(...)          fused per-head Q/K RMSNorm + RoPE + K/V transpose
  path_attention(q, kc, vc, anc, dlen, L, gqa)
                              register-resident FlashAttention-2 Split-K on 8x8 bfloat matrix units (D=64,
                              GQA=4, bf16), sharing K/V across GQA heads and up to 16 rows; `pathattn` fallback
  rowmm(x, w, prow, pexp)     general row-invariant expert/dense projection fallback

Sources are in csrc/mlx_kernels/ (engine/native_build.py's layout). The CUDA board's counterpart is its row
kernels (ENG-287); like them these serve up to `ROWS_MAX` rows, and a longer prompt chunk takes MLX's own
batched kernels (the next token agrees, the bits need not).
"""

from __future__ import annotations


import mlx.core as mx

from engine import native_build as nb

#: rows a block may have on these kernels (the verify width and more; `rowmm` takes pairs up to MAXP)
ROWS_MAX = 64
#: output columns a simdgroup computes, rows a pass reads its weights for, pairs a call may route
TN, R, MAXP = 4, 8, 512
#: rows a dense projection's pass takes (one expert: a block of up to 16 rows reads each weight once)
R_DENSE = 8
#: threads of the fallback attention threadgroup (keys are dealt by logical index modulo NT)
NT = 128
#: logical key chunk size for Split-K FlashAttention path_attention (fixed constant for row invariance)
ATTN_BK = 256

_K: dict = {}


#: every kernel source, by file name: read when its kernel is first built. Literal names, so that
#: tests/test_native_sources.py holds this table to the folder's contents.
_SRC = {
    "convgate": lambda: nb.source(__file__, "convgate.metal"),
    "dense_prefill_swiz": lambda: nb.source(__file__, "dense_prefill_swiz.metal"),
    "embed_swiz": lambda: nb.source(__file__, "embed_swiz.metal"),
    "moegate": lambda: nb.source(__file__, "moegate.metal"),
    "moemm_hyb": lambda: nb.source(__file__, "moemm_hyb.metal"),
    "moemm_hyb_swiz": lambda: nb.source(__file__, "moemm_hyb_swiz.metal"),
    "moemm_prefill_swiz": lambda: nb.source(__file__, "moemm_prefill_swiz.metal"),
    "moemm_r1": lambda: nb.source(__file__, "moemm_r1.metal"),
    "moemm_r1_swiz": lambda: nb.source(__file__, "moemm_r1_swiz.metal"),
    "moemm_w2add_swiz": lambda: nb.source(__file__, "moemm_w2add_swiz.metal"),
    "moeprefill_grp": lambda: nb.source(__file__, "moeprefill_grp.metal"),
    "moeroute": lambda: nb.source(__file__, "moeroute.metal"),
    "moewsum": lambda: nb.source(__file__, "moewsum.metal"),
    "pathattn": lambda: nb.source(__file__, "pathattn.metal"),
    "pathattn_flash": lambda: nb.source(__file__, "pathattn_flash.metal"),
    "pathattn_flash_m2": lambda: nb.source(__file__, "pathattn_flash_m2.metal"),
    "pathattn_reduce": lambda: nb.source(__file__, "pathattn_reduce.metal"),
    "qkv_rope": lambda: nb.source(__file__, "qkv_rope.metal"),
    "rowchain": lambda: nb.source(__file__, "rowchain.metal"),
    "rowchain_swiz": lambda: nb.source(__file__, "rowchain_swiz.metal"),
    "rowmm": lambda: nb.source(__file__, "rowmm.metal"),
    "rowmma": lambda: nb.source(__file__, "rowmma.metal"),
    "rowmma_swiz": lambda: nb.source(__file__, "rowmma_swiz.metal"),
}


def _kernel(name: str, inputs: list, outputs: list, file: str | None = None, header: str = ""):
    k = _K.get(name)
    if k is None:
        src = _SRC[file or name]()
        k = _K[name] = mx.fast.metal_kernel(name=f"lfm2_{name}", input_names=inputs, output_names=outputs,
                                            source=src, header=header)
    return k


# --- 512B-coalesced weight swizzle helpers (BK=16, KS=4, TN=2 -> N_TILE=16, zero extra bytes) ---
def swizzle_dense(w: mx.array) -> mx.array:
    """Repack [N, K] (N % 16 == 0, K % 64 == 0) into 512B-coalesced tile layout [N//16, K//64, 2, 2, 8, 4, 8]."""
    N, Kd = w.shape
    return w.reshape(N // 16, 8, 2, Kd // 64, 4, 2, 8).transpose(0, 3, 5, 2, 1, 4, 6).reshape(-1)


def unswizzle_dense(ws: mx.array, N: int, Kd: int) -> mx.array:
    """Inverse of `swizzle_dense`: [N, K] back (the tests hold the round trip)."""
    return ws.reshape(N // 16, Kd // 64, 2, 2, 8, 4, 8).transpose(0, 4, 3, 1, 5, 2, 6).reshape(N, Kd)


def swizzle_moe_w13(w13: mx.array) -> mx.array:
    """Repack [E, 2*I, K] (I % 16 == 0, K % 64 == 0) with w1 & w3 interleaved inside each 4 KB tile."""
    E, N2, Kd = w13.shape
    I = N2 // 2
    return w13.reshape(E, 2, I // 16, 8, 2, Kd // 64, 4, 2, 8).transpose(0, 2, 5, 1, 7, 4, 3, 6, 8).reshape(-1)


def unswizzle_moe_w13(ws: mx.array, E: int, I: int, Kd: int) -> mx.array:
    """Inverse of `swizzle_moe_w13`: [E, 2*I, K] back (the tests hold the round trip)."""
    return ws.reshape(E, I // 16, Kd // 64, 2, 2, 2, 8, 4, 8).transpose(0, 3, 1, 6, 5, 2, 7, 4, 8).reshape(E, 2 * I, Kd)


def swizzle_moe_w2(w2: mx.array) -> mx.array:
    """Repack [E, H, I] (H % 16 == 0, I % 64 == 0) into 512B-coalesced tile layout."""
    E, H, I = w2.shape
    return w2.reshape(E, H // 16, 8, 2, I // 64, 4, 2, 8).transpose(0, 1, 4, 6, 3, 2, 5, 7).reshape(-1)


def unswizzle_moe_w2(ws: mx.array, E: int, H: int, I: int) -> mx.array:
    """Inverse of `swizzle_moe_w2`: [E, H, I] back (the tests hold the round trip)."""
    return ws.reshape(E, H // 16, I // 64, 2, 2, 8, 4, 8).transpose(0, 1, 5, 4, 2, 6, 3, 7).reshape(E, H, I)


def embed_swiz(tokens: mx.array, w_swiz: mx.array, Kd: int) -> mx.array:
    """Gather rows [M, K] from a 512B-coalesced swizzled [V, K] embedding table."""
    M = tokens.shape[0]
    k8 = Kd // 8
    return _kernel("embed_swiz", ["tokens", "w_swiz"], ["out"])(
        inputs=[tokens, w_swiz], template=[("K", Kd)],
        grid=(k8, M, 1), threadgroup=(min(256, k8), 1, 1),
        output_shapes=[(M, Kd)], output_dtypes=[w_swiz.dtype])[0]


def ksplit(N: int, K: int) -> int:
    """How many ways a threadgroup splits K (1, 2 or 4): by the weight's shape alone, so a row's summation
    order never follows the row count. Narrow outputs split more, to fill the GPU."""
    ks = 4 if N <= 512 else 1 if N <= 2048 else 2
    while ks > 1 and -(-K // 256) < ks:
        ks //= 2
    return ks


def experts_used(pexp: mx.array, E: int) -> mx.array:
    """The experts the pairs use, first, -1 past them ([min(E, P)] int32, on the GPU: no host sync)."""
    used = mx.zeros((E,), dtype=mx.int32).at[pexp].add(1) > 0
    order = mx.argsort(mx.where(used, 0, 1))[:min(E, pexp.shape[0])].astype(mx.int32)
    return mx.where(used[order], order, -1)


def rowmm(x: mx.array, w: mx.array, prow: mx.array, pexp: mx.array, rows: int | None = None,
          elist: mx.array | None = None) -> mx.array:
    """[P, N] in x's dtype: pair p is row prow[p] of x [rows, K] times expert pexp[p] of w [E, N, K]."""
    E, N, K = w.shape
    P = prow.shape[0]
    if P > (MAXP if E > 1 else ROWS_MAX):
        # the kernel's pair list is MAXP long for experts and ROWS_MAX long for one weight (its template)
        raise ValueError(f"rowmm: {P} pairs exceed {MAXP if E > 1 else ROWS_MAX}")
    if elist is None:
        elist = _index(1)[2] if E == 1 else experts_used(pexp, E)
    ks = ksplit(N, K)
    tiles = -(-N // ((4 // ks) * TN))
    want = P if rows is None else rows
    rmax = (R_DENSE if ks <= 2 else 8) if E == 1 else R
    Rn = next(r for r in (1, 2, 4, 8, 16, 32) if r >= min(want, rmax))
    return _kernel("rowmm", ["x", "w", "prow", "pexp", "elist"], ["y"])(
        inputs=[x, w, prow, pexp, elist],
        template=[("T", x.dtype), ("OT", x.dtype), ("K", K), ("N", N), ("TN", TN), ("R", Rn), ("MAXP", MAXP if E > 1 else ROWS_MAX), ("KS", ks)],
        grid=(tiles * 128, elist.shape[0], 1), threadgroup=(128, 1, 1),
        output_shapes=[(P, N)], output_dtypes=[x.dtype])[0]


_IDX: dict = {}


def _index(M: int) -> tuple:
    """(rows 0..M-1, zeros [M], the one-expert list [0]) as int32 arrays, made once a row count."""
    got = _IDX.get(M)
    if got is None:
        got = _IDX[M] = (mx.arange(M, dtype=mx.int32), mx.zeros((M,), dtype=mx.int32), mx.zeros((1,), dtype=mx.int32))
        mx.eval(*got)
    return got


def _scalar_f32(val: float) -> mx.array:
    key = ("f32", float(val))
    got = _IDX.get(key)
    if got is None:
        got = _IDX[key] = mx.array([float(val)], dtype=mx.float32)
        mx.eval(got)
    return got


def pair_rows(M: int, k: int) -> mx.array:
    """[M k] int32: the row of each (row, slot) pair, made once a shape."""
    key = ("pairs", M, k)
    got = _IDX.get(key)
    if got is None:
        got = _IDX[key] = mx.repeat(mx.arange(M, dtype=mx.int32), k)
        mx.eval(got)
    return got


#: (FN, BK, KS, TN, TG) of an unswizzled dense projection by its shape (N, K).
MMA = {
    (6144, 2048): (2, 16, 4, 4, 128),      # conv in_proj
    (2048, 2048): (2, 16, 4, 4, 128),      # conv and attention out_proj
    (3072, 2048): (2, 16, 4, 4, 128),      # q, k, v
    (23552, 2048): (2, 32, 2, 4, 64),      # dense MLP w1 and w3
    (2048, 11776): (2, 16, 4, 4, 128),     # dense MLP w2
    (128000, 2048): (2, 32, 2, 4, 64),     # head
}
#: rows a block may have on the scalar kernel; more take the matrix units
CHAIN_ROWS = 2


def mma(x: mx.array, w: mx.array, FN: int, BK: int, KS: int, res: mx.array | None = None,
        out_dtype=None) -> mx.array:
    """x [M, K] @ w [N, K].T (+ res) on 8x8 bfloat simdgroup matrices."""
    M, Kd = x.shape
    N = w.shape[0]
    FM = 1 if M <= 8 else 2
    od = out_dtype or x.dtype
    if res is not None:
        k = _kernel("rowmma_add", ["x", "w", "res"], ["y"], file="rowmma", header="#define ADD_RES 1\n")
        inps = [x, w, res]
    else:
        k = _kernel("rowmma", ["x", "w"], ["y"], file="rowmma", header="#define ADD_RES 0\n")
        inps = [x, w]
    return k(
        inputs=inps,
        template=[("T", x.dtype), ("OT", od), ("K", Kd), ("N", N), ("FM", FM), ("FN", FN), ("BK", BK), ("KS", KS)],
        grid=(-(-N // ((4 // KS) * 8 * FN)) * 128, -(-M // (FM * 8)), 1), threadgroup=(128, 1, 1),
        output_shapes=[(M, N)], output_dtypes=[od])[0]


def chain(x: mx.array, w: mx.array, BK: int, KS: int, TN: int, TG: int, res: mx.array | None = None,
          out_dtype=None) -> mx.array:
    """x [M, K] @ w [N, K].T (+ res) on the scalar kernel: `mma`'s rows for the same (BK, KS), bit for bit."""
    M, Kd = x.shape
    N = w.shape[0]
    R = next(r for r in (1, 2, 4, 8) if r >= min(M, CHAIN_ROWS))
    od = out_dtype or x.dtype
    if res is not None:
        k = _kernel("rowchain_add", ["x", "w", "res"], ["y"], file="rowchain", header="#define ADD_RES 1\n")
        inps = [x, w, res]
    else:
        k = _kernel("rowchain", ["x", "w"], ["y"], file="rowchain", header="#define ADD_RES 0\n")
        inps = [x, w]
    return k(
        inputs=inps,
        template=[("T", x.dtype), ("OT", od), ("K", Kd), ("N", N), ("TN", TN), ("R", R), ("KS", KS), ("BK", BK), ("TG", TG)],
        grid=(-(-N // ((TG // KS) * TN)) * TG, -(-M // R), 1), threadgroup=(TG, 1, 1),
        output_shapes=[(M, N)], output_dtypes=[od])[0]


def dense(x: mx.array, w: mx.array, res: mx.array | None = None, out_dtype=None) -> mx.array:
    """x [M, K] @ w [N, K].T (+ res), row-invariant."""
    cfg = MMA.get((w.shape[0], w.shape[1]))
    if cfg is not None and w.shape[1] % (cfg[1] * cfg[2]) == 0 and cfg[1] % 8 == 0 and cfg[4] % cfg[2] == 0:
        FN, BK, KS, TN, TG = cfg
        return (chain(x, w, BK, KS, TN, TG, res=res, out_dtype=out_dtype) if x.shape[0] <= CHAIN_ROWS
                else mma(x, w, FN, BK, KS, res=res, out_dtype=out_dtype))
    rows, zero, one = _index(x.shape[0])
    out = rowmm(x, w[None], rows, zero, elist=one)
    if res is not None:
        out = res + out
    if out_dtype is not None and out.dtype != out_dtype:
        out = out.astype(out_dtype)
    return out


def dense_swiz(x: mx.array, w_swiz: mx.array, N: int, Kd: int, nw: mx.array | None = None,
               eps: float = 1e-5, res: mx.array | None = None, out_dtype=None) -> mx.array:
    """512B-coalesced swizzled row-invariant dense projection (`rowchain_swiz` for M<=2, `rowmma_swiz` for M>=3),
    with optional fused 32-lane simd_sum RMSNorm (`nw`) and optional fused residual add (`res`)."""
    M = x.shape[0]
    od = out_dtype or x.dtype
    has_rms = nw is not None
    has_res = res is not None
    tag = f"{'_rms' if has_rms else ''}{'_add' if has_res else ''}"
    hdr = f"#define FUSE_RMS {int(has_rms)}\n#define ADD_RES {int(has_res)}\n"
    in_names = ["x", "w_swiz"] + (["nw", "eps"] if has_rms else []) + (["res"] if has_res else [])
    inps = [x, w_swiz] + ([nw, _scalar_f32(eps)] if has_rms else []) + ([res] if has_res else [])
    if M <= CHAIN_ROWS:
        R_n = 1 if M == 1 else 2
        TG = 128
        ntg = -(-N // ((TG // 32) * 16))
        k = _kernel(f"rowchain_swiz{tag}", in_names, ["y"], file="rowchain_swiz", header=hdr)
        return k(inputs=inps,
                 template=[("T", x.dtype), ("OT", od), ("K", Kd), ("N", N), ("R", R_n), ("TG", TG)],
                 grid=(ntg * TG, -(-M // R_n), 1), threadgroup=(TG, 1, 1),
                 output_shapes=[(M, N)], output_dtypes=[od])[0]
    FM = 1 if M <= 8 else 2
    k = _kernel(f"rowmma_swiz{tag}", in_names, ["y"], file="rowmma_swiz", header=hdr)
    return k(inputs=inps,
             template=[("T", x.dtype), ("OT", od), ("K", Kd), ("N", N), ("FM", FM)],
             grid=((N // 16) * 128, -(-M // (FM * 8)), 1), threadgroup=(128, 1, 1),
             output_shapes=[(M, N)], output_dtypes=[od])[0]


_NAX_HDR = (
    "#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400\n"
    "#include <metal_tensor>\n"
    "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
    "#endif\n"
)


def dense_prefill_swiz(x: mx.array, w_swiz: mx.array, N: int, Kd: int, out_dtype=None) -> mx.array:
    """High-throughput batched prefill GEMM directly on 512B-coalesced w_swiz (M5 NAX / Metal 4 MPP, zero unswizzle!)."""
    M = x.shape[0]
    od = out_dtype or x.dtype
    return _kernel("dense_prefill_swiz", ["x", "w_swiz"], ["y"], header=_NAX_HDR)(
        inputs=[x, w_swiz],
        template=[("T", x.dtype), ("OT", od), ("K", Kd), ("N", N)],
        grid=(-(-M // 64) * 128, -(-N // 64), 1), threadgroup=(128, 1, 1),
        output_shapes=[(M, N)], output_dtypes=[od])[0]


def moe_prefill_swiz(h: mx.array, ids: mx.array, wts: mx.array, w13_swiz: mx.array, w2_swiz: mx.array,
                     E: int, I: int, topk: int) -> mx.array:
    """High-throughput batched prefill MoE directly on 512B-coalesced w13_swiz & w2_swiz (M5 NAX, zero unswizzle!)."""
    M, H = h.shape
    P = M * topk
    pexp = ids.reshape(-1)
    bm = 32
    maxt = P // bm + E
    tile_e, tile_off, tile_nr, sel = _kernel("moeprefill_grp", ["pexp"], ["tile_e", "tile_off", "tile_nr", "sel"])(
        inputs=[pexp], template=[("E", E), ("BM", bm), ("MAXT", maxt)],
        grid=(256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(maxt,), (maxt,), (maxt,), (P,)], output_dtypes=[mx.int32, mx.int32, mx.int32, mx.int32])
    bn13, bn2 = 64, 64
    act = _kernel("moemm_prefill_swiz", ["x", "w_swiz", "tile_e", "tile_off", "tile_nr", "sel"], ["y"], header=_NAX_HDR)(
        inputs=[h, w13_swiz, tile_e, tile_off, tile_nr, sel],
        template=[("T", h.dtype), ("K", H), ("NOUT", I), ("BM", bm), ("BN", bn13), ("TOPK", topk), ("SWIGLU", 1), ("PROW_IDENTITY", 0)],
        grid=(maxt * 128, -(-I // bn13), 1), threadgroup=(128, 1, 1),
        output_shapes=[(P, I)], output_dtypes=[h.dtype])[0]
    y = _kernel("moemm_prefill_swiz", ["x", "w_swiz", "tile_e", "tile_off", "tile_nr", "sel"], ["y"], header=_NAX_HDR)(
        inputs=[act, w2_swiz, tile_e, tile_off, tile_nr, sel],
        template=[("T", h.dtype), ("K", I), ("NOUT", H), ("BM", bm), ("BN", bn2), ("TOPK", topk), ("SWIGLU", 0), ("PROW_IDENTITY", 1)],
        grid=(maxt * 128, -(-H // bn2), 1), threadgroup=(128, 1, 1),
        output_shapes=[(P, H)], output_dtypes=[h.dtype])[0].reshape(M, topk, H)
    wt = wts.astype(y.dtype)
    out = y[:, 0] * wt[:, 0:1]
    for j in range(1, topk):
        out = out + y[:, j] * wt[:, j:j + 1]
    return out.astype(h.dtype)


def dense_swiglu_swiz(x: mx.array, w13_swiz: mx.array, I: int, Kd: int) -> mx.array:
    """512B-coalesced swizzled fused SwiGLU dense MLP projection (w13_swiz = swizzle_moe_w13(w13[None])).
    Reuses moemm_r1_swiz (M==1) and moemm_hyb_swiz (M>1) for bit-exact row invariance and 560+ GB/s BW."""
    M = x.shape[0]
    if M == 1:
        tg13 = 64
        _, _, one = _index(1)
        return _kernel("moemm_r1_swiz_pexp", ["x", "w_swiz", "pexp"], ["y"],
                       file="moemm_r1_swiz", header="#define FUSE_ROUTE 0\n")(
            inputs=[x, w13_swiz, one],
            template=[("T", x.dtype), ("K", Kd), ("NOUT", I), ("TG", tg13)],
            grid=(-(-I // ((tg13 // 32) * 16)) * tg13, 1, 1), threadgroup=(tg13, 1, 1),
            output_shapes=[(1, I)], output_dtypes=[x.dtype])[0]
    rows, _, one = _index(M)
    eoff = one
    ecnt_key = ("ecnt", M)
    ecnt = _IDX.get(ecnt_key)
    if ecnt is None:
        ecnt = _IDX[ecnt_key] = mx.array([M], dtype=mx.int32)
        mx.eval(ecnt)
    fm_max = 1 if M <= 8 else 2
    return _kernel("moemm_hyb_swiz", ["x", "w_swiz", "elist", "eoff", "ecnt", "sel"], ["y"])(
        inputs=[x, w13_swiz, one, eoff, ecnt, rows],
        template=[("T", x.dtype), ("K", Kd), ("NOUT", I), ("FM_MAX", fm_max), ("TOPK", 1), ("SWIGLU", 1), ("PROW_IDENTITY", 1)],
        grid=((I // 16) * 128, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(M, I)], output_dtypes=[x.dtype])[0]


def moe_fast_swiz(h: mx.array, nw: mx.array, gw: mx.array, bias: mx.array, w13_swiz: mx.array, w2_swiz: mx.array,
                  I: int, eps: float, topk: int, scale: float, use_bias: bool) -> mx.array:
    """512B-coalesced swizzled MoE block (ONLY 3 launches per layer at M=1!):
      M == 1: moegate_rms_out (gate+RMSNorm->n) -> moemm_r1_swiz_rt (route+w13+SwiGLU) -> moemm_w2add_swiz_rt (route+w2+wsum+res)
      M >  1: moegate_rms_out (gate+RMSNorm->n) -> moeroute_grp -> moemm_hyb_swiz (w13+SwiGLU) -> moemm_hyb_swiz (w2) -> moewsum"""
    M, H = h.shape
    E = gw.shape[0]
    P = M * topk
    emax = min(E, P)
    eps_arr = _scalar_f32(eps)
    sc_arr = _scalar_f32(scale)
    Rg = min(8, next(r for r in (1, 2, 4, 8) if r >= min(M, 8)))
    logits, n = _kernel("moegate_rms_out", ["x", "w", "nw", "eps"], ["out", "n_out"],
                        file="moegate", header="#define FUSE_RMS 1\n#define WRITE_NORM 1\n")(
        inputs=[h, gw, nw, eps_arr], template=[("T", h.dtype), ("K", H), ("E", E), ("R", Rg)],
        grid=(E * 64, 1, 1), threadgroup=(64, 1, 1),
        output_shapes=[(M, E), (M, H)], output_dtypes=[h.dtype, h.dtype])
    if M == 1:
        # In-register exact top-TOPK + 512B-coalesced w13+SwiGLU (zero rms_norm, zero moeroute launch!)
        tg13 = 32
        act = _kernel("moemm_r1_swiz_rt", ["x", "w_swiz", "logits", "bias"], ["y"],
                      file="moemm_r1_swiz", header="#define FUSE_ROUTE 1\n")(
            inputs=[n, w13_swiz, logits, bias],
            template=[("T", h.dtype), ("K", H), ("NOUT", I), ("TG", tg13), ("E", E), ("TOPK", topk), ("USE_BIAS", int(use_bias))],
            grid=(-(-I // ((tg13 // 32) * 16)) * tg13, topk, 1), threadgroup=(tg13, 1, 1),
            output_shapes=[(P, I)], output_dtypes=[h.dtype])[0]
        # In-register exact top-TOPK + 4-load 512B-coalesced w2 + ascending wsum + residual add (zero moewsum launch!)
        tg2 = 32
        return _kernel("moemm_w2add_swiz_rt", ["x", "w_swiz", "logits", "bias", "scale", "h"], ["out"],
                       file="moemm_w2add_swiz", header="#define FUSE_ROUTE 1\n")(
            inputs=[act, w2_swiz, logits, bias, sc_arr, h],
            template=[("T", h.dtype), ("K", I), ("NOUT", H), ("TG", tg2), ("E", E), ("TOPK", topk), ("USE_BIAS", int(use_bias))],
            grid=(-(-H // ((tg2 // 32) * 16)) * tg2, 1, 1), threadgroup=(tg2, 1, 1),
            output_shapes=[(M, H)], output_dtypes=[h.dtype])[0]
    tg_route = 64 if max(M, E) <= 64 else 128
    pexp, wts, elist, eoff, ecnt, sel = _kernel(
        "moeroute_grp", ["logits", "bias", "scale"], ["pexp", "wts", "elist", "eoff", "ecnt", "sel"],
        file="moeroute", header="#define GROUP 1\n")(
        inputs=[logits, bias, sc_arr],
        template=[("T", h.dtype), ("E", E), ("TOPK", topk), ("MAXP", max(P, 64)), ("EMAX", emax), ("USE_BIAS", int(use_bias))],
        grid=(tg_route, 1, 1), threadgroup=(tg_route, 1, 1),
        output_shapes=[(P,), (M, topk), (emax,), (emax,), (emax,), (P,)],
        output_dtypes=[mx.int32, h.dtype, mx.int32, mx.int32, mx.int32, mx.int32])
    fm_max = 1 if M <= 8 else 2
    act = _kernel("moemm_hyb_swiz", ["x", "w_swiz", "elist", "eoff", "ecnt", "sel"], ["y"])(
        inputs=[n, w13_swiz, elist, eoff, ecnt, sel],
        template=[("T", h.dtype), ("K", H), ("NOUT", I), ("FM_MAX", fm_max), ("TOPK", topk), ("SWIGLU", 1), ("PROW_IDENTITY", 0)],
        grid=((I // 16) * 128, emax, 1), threadgroup=(128, 1, 1),
        output_shapes=[(P, I)], output_dtypes=[h.dtype])[0]
    y = _kernel("moemm_hyb_swiz", ["x", "w_swiz", "elist", "eoff", "ecnt", "sel"], ["y"])(
        inputs=[act, w2_swiz, elist, eoff, ecnt, sel],
        template=[("T", h.dtype), ("K", I), ("NOUT", H), ("FM_MAX", fm_max), ("TOPK", topk), ("SWIGLU", 0), ("PROW_IDENTITY", 1)],
        grid=((H // 16) * 128, emax, 1), threadgroup=(128, 1, 1),
        output_shapes=[(P, H)], output_dtypes=[h.dtype])[0]
    return _kernel("moewsum", ["h", "y", "wts"], ["out"])(
        inputs=[h, y, wts], template=[("T", h.dtype), ("N", H), ("TOPK", topk)],
        grid=(H // 4, M, 1), threadgroup=(min(256, H // 4), 1, 1),
        output_shapes=[(M, H)], output_dtypes=[h.dtype])[0]


def moe_fast(h: mx.array, nw: mx.array, gw: mx.array, bias: mx.array, w13: mx.array, w2: mx.array,
             eps: float, topk: int, scale: float, use_bias: bool) -> mx.array:
    """Unswizzled fused row-invariant MoE block (bfloat16, K % 64 == 0, E <= 128)."""
    M, H = h.shape
    E = gw.shape[0]
    I = w2.shape[2]
    P = M * topk
    emax = min(E, P)
    n = mx.fast.rms_norm(h, nw, eps)
    Rg = min(8, next(r for r in (1, 2, 4, 8) if r >= min(M, 8)))
    logits = _kernel("moegate", ["x", "w"], ["out"], file="moegate", header="#define FUSE_RMS 0\n#define WRITE_NORM 0\n")(
        inputs=[n, gw], template=[("T", n.dtype), ("K", H), ("E", E), ("R", Rg)],
        grid=(E * 64, 1, 1), threadgroup=(64, 1, 1),
        output_shapes=[(M, E)], output_dtypes=[n.dtype])[0]
    tg_route = 64 if max(M, E) <= 64 else 128
    BK, KS = 16, 4
    if M == 1:
        pexp, wts = _kernel("moeroute_m1", ["logits", "bias", "scale"], ["pexp", "wts"],
                            file="moeroute", header="#define GROUP 0\n")(
            inputs=[logits, bias, _scalar_f32(scale)],
            template=[("T", n.dtype), ("E", E), ("TOPK", topk), ("USE_BIAS", int(use_bias))],
            grid=(tg_route, 1, 1), threadgroup=(tg_route, 1, 1),
            output_shapes=[(P,), (M, topk)],
            output_dtypes=[mx.int32, n.dtype])
        act = _kernel("moemm_r1", ["x", "w", "pexp"], ["y"])(
            inputs=[n, w13, pexp],
            template=[("T", n.dtype), ("K", H), ("NOUT", I), ("BK", BK), ("KS", KS), ("TN", 1), ("TG", 128), ("SWIGLU", 1), ("PROW_IDENTITY", 0)],
            grid=(-(-I // 32) * 128, topk, 1), threadgroup=(128, 1, 1),
            output_shapes=[(P, I)], output_dtypes=[n.dtype])[0]
        y = _kernel("moemm_r1", ["x", "w", "pexp"], ["y"])(
            inputs=[act, w2, pexp],
            template=[("T", n.dtype), ("K", I), ("NOUT", H), ("BK", BK), ("KS", KS), ("TN", 4), ("TG", 64), ("SWIGLU", 0), ("PROW_IDENTITY", 1)],
            grid=(-(-H // 64) * 64, topk, 1), threadgroup=(64, 1, 1),
            output_shapes=[(P, H)], output_dtypes=[n.dtype])[0]
    else:
        pexp, wts, elist, eoff, ecnt, sel = _kernel(
            "moeroute_grp", ["logits", "bias", "scale"], ["pexp", "wts", "elist", "eoff", "ecnt", "sel"],
            file="moeroute", header="#define GROUP 1\n")(
            inputs=[logits, bias, _scalar_f32(scale)],
            template=[("T", n.dtype), ("E", E), ("TOPK", topk), ("MAXP", max(P, 64)), ("EMAX", emax), ("USE_BIAS", int(use_bias))],
            grid=(tg_route, 1, 1), threadgroup=(tg_route, 1, 1),
            output_shapes=[(P,), (M, topk), (emax,), (emax,), (emax,), (P,)],
            output_dtypes=[mx.int32, n.dtype, mx.int32, mx.int32, mx.int32, mx.int32])
        fn, fm_max = 2, (1 if M <= 8 else 2)
        ntile = (4 // KS) * fn * 8
        act = _kernel("moemm_hyb", ["x", "w", "elist", "eoff", "ecnt", "sel"], ["y"])(
            inputs=[n, w13, elist, eoff, ecnt, sel],
            template=[("T", n.dtype), ("K", H), ("NOUT", I), ("BK", BK), ("KS", KS), ("FN", fn), ("FM_MAX", fm_max), ("TOPK", topk), ("SWIGLU", 1), ("PROW_IDENTITY", 0)],
            grid=(-(-I // ntile) * 128, emax, 1), threadgroup=(128, 1, 1),
            output_shapes=[(P, I)], output_dtypes=[n.dtype])[0]
        y = _kernel("moemm_hyb", ["x", "w", "elist", "eoff", "ecnt", "sel"], ["y"])(
            inputs=[act, w2, elist, eoff, ecnt, sel],
            template=[("T", n.dtype), ("K", I), ("NOUT", H), ("BK", BK), ("KS", KS), ("FN", fn), ("FM_MAX", fm_max), ("TOPK", topk), ("SWIGLU", 0), ("PROW_IDENTITY", 1)],
            grid=(-(-H // ntile) * 128, emax, 1), threadgroup=(128, 1, 1),
            output_shapes=[(P, H)], output_dtypes=[n.dtype])[0]
    return _kernel("moewsum", ["h", "y", "wts"], ["out"])(
        inputs=[h, y, wts], template=[("T", n.dtype), ("N", H), ("TOPK", topk)],
        grid=(H // 4, M, 1), threadgroup=(min(256, H // 4), 1, 1),
        output_shapes=[(M, H)], output_dtypes=[n.dtype])[0]


def conv_gate(bcx: mx.array, state: mx.array, cw: mx.array, taps: mx.array | None = None,
              next_state: bool = False) -> tuple[mx.array, mx.array]:
    """Fused double-gated short convolution: returns (C * y, B * x) or (C * y, next_state) in ascending tap order."""
    M = bcx.shape[0]
    H, Kt = cw.shape
    tg = min(256, H)
    if next_state:
        k = _kernel("convgate_step", ["bcx", "state", "cw"], ["cy", "next_state"],
                    file="convgate", header="#define HAS_TAPS 0\n#define NEXT_STATE 1\n")
        return k(inputs=[bcx, state, cw], template=[("T", bcx.dtype), ("H", H), ("KT", Kt)],
                 grid=(H, M, 1), threadgroup=(tg, 1, 1),
                 output_shapes=[(M, H), (Kt - 1, H)], output_dtypes=[bcx.dtype, bcx.dtype])
    if taps is not None:
        k = _kernel("convgate_taps", ["bcx", "state", "cw", "taps"], ["cy", "bx"],
                    file="convgate", header="#define HAS_TAPS 1\n#define NEXT_STATE 0\n")
        inps = [bcx, state, cw, taps]
    else:
        k = _kernel("convgate_chain", ["bcx", "state", "cw"], ["cy", "bx"],
                    file="convgate", header="#define HAS_TAPS 0\n#define NEXT_STATE 0\n")
        inps = [bcx, state, cw]
    cy, bx = k(inputs=inps, template=[("T", bcx.dtype), ("H", H), ("KT", Kt)],
               grid=(H, M, 1), threadgroup=(tg, 1, 1),
               output_shapes=[(M, H), (M, H)], output_dtypes=[bcx.dtype, bcx.dtype])
    return cy, bx


def qkv_norm_rope(qkv: mx.array, qn: mx.array, kn: mx.array, cos: mx.array, sin: mx.array,
                  Hq: int, Hk: int, D: int, eps: float) -> tuple[mx.array, mx.array, mx.array]:
    """Fused per-head Q/K RMSNorm + RoPE + K/V transpose: returns (q [M, Hq, D], k [Hk, M, D], v [Hk, M, D])."""
    M = qkv.shape[0]
    q, k, v = _kernel("qkv_rope", ["qkv", "qn", "kn", "cos", "sin", "eps"], ["q_out", "k_out", "v_out"])(
        inputs=[qkv, qn, kn, cos, sin, _scalar_f32(eps)],
        template=[("T", qkv.dtype), ("Hq", Hq), ("Hk", Hk), ("D", D)],
        grid=((Hq + 2 * Hk) * 32, M, 1), threadgroup=(32, 1, 1),
        output_shapes=[(M, Hq, D), (Hk, M, D), (Hk, M, D)],
        output_dtypes=[qkv.dtype, qkv.dtype, qkv.dtype])
    return q, k, v


def path_attention(q: mx.array, kc: mx.array, vc: mx.array, anc: mx.array, dlen: mx.array, L: int,
                   gqa: int, lb: mx.array | None = None, nchunks: int | None = None) -> mx.array:
    """q [M, Hq, D] (rotated), the caches [Hk, cap, D] holding the block's rows at L.., anc [M, A] int32 each
    row's path (block rows, root first, padded), dlen [M] its length: out [M, Hq, D]."""
    M, Hq, D = q.shape
    if lb is None:
        lb = mx.array([L], dtype=mx.int32)
    if D == 64 and gqa == 4 and q.dtype == mx.bfloat16:
        Hk = Hq // gqa
        if nchunks is None:
            nchunks = (L + anc.shape[1] + ATTN_BK - 1) // ATTN_BK
        if M <= 2:
            if nchunks == 1:
                return _kernel("pathattn_flash_m2_single", ["q", "kc", "vc", "anc", "dl", "lb"], ["o"],
                               file="pathattn_flash_m2", header="#define SINGLE_CHUNK 1\n")(
                    inputs=[q, kc, vc, anc, dlen, lb],
                    template=[("T", q.dtype), ("D", D), ("GQA", gqa), ("BK", ATTN_BK), ("BM", 2)],
                    grid=(Hk * 128, 1, 1), threadgroup=(128, 1, 1),
                    output_shapes=[(M, Hq, D)], output_dtypes=[q.dtype])[0]
            pm, pl, pa = _kernel("pathattn_flash_m2_chunk", ["q", "kc", "vc", "anc", "dl", "lb"], ["pm", "pl", "pa"],
                                 file="pathattn_flash_m2", header="#define SINGLE_CHUNK 0\n")(
                inputs=[q, kc, vc, anc, dlen, lb],
                template=[("T", q.dtype), ("D", D), ("GQA", gqa), ("BK", ATTN_BK), ("BM", 2)],
                grid=(Hk * 128, nchunks, 1), threadgroup=(128, 1, 1),
                output_shapes=[(M, Hq, nchunks), (M, Hq, nchunks), (M, Hq, nchunks, D)],
                output_dtypes=[mx.float32, mx.float32, mx.float32])
            return _kernel("pathattn_reduce", ["pm", "pl", "pa", "dl", "lb"], ["o"])(
                inputs=[pm, pl, pa, dlen, lb],
                template=[("T", q.dtype), ("D", D), ("BK", ATTN_BK)],
                grid=(Hq * 32, M, 1), threadgroup=(32, 1, 1),
                output_shapes=[(M, Hq, D)], output_dtypes=[q.dtype])[0]
        if M <= 4:
            bm, nsg = 4, 2
        elif M <= 8:
            bm, nsg = 8, 4
        else:
            bm, nsg = 16, 8
        if nchunks == 1:
            return _kernel("pathattn_flash_single", ["q", "kc", "vc", "anc", "dl", "lb"], ["o"],
                           file="pathattn_flash", header="#define SINGLE_CHUNK 1\n")(
                inputs=[q, kc, vc, anc, dlen, lb],
                template=[("T", q.dtype), ("D", D), ("GQA", gqa), ("BK", ATTN_BK), ("BM", bm), ("NSG", nsg)],
                grid=(Hk * nsg * 32, 1, -(-M // bm)), threadgroup=(nsg * 32, 1, 1),
                output_shapes=[(M, Hq, D)], output_dtypes=[q.dtype])[0]
        pm, pl, pa = _kernel("pathattn_flash_chunk", ["q", "kc", "vc", "anc", "dl", "lb"], ["pm", "pl", "pa"],
                             file="pathattn_flash", header="#define SINGLE_CHUNK 0\n")(
            inputs=[q, kc, vc, anc, dlen, lb],
            template=[("T", q.dtype), ("D", D), ("GQA", gqa), ("BK", ATTN_BK), ("BM", bm), ("NSG", nsg)],
            grid=(Hk * nsg * 32, nchunks, -(-M // bm)), threadgroup=(nsg * 32, 1, 1),
            output_shapes=[(M, Hq, nchunks), (M, Hq, nchunks), (M, Hq, nchunks, D)],
            output_dtypes=[mx.float32, mx.float32, mx.float32])
        return _kernel("pathattn_reduce", ["pm", "pl", "pa", "dl", "lb"], ["o"])(
            inputs=[pm, pl, pa, dlen, lb],
            template=[("T", q.dtype), ("D", D), ("BK", ATTN_BK)],
            grid=(Hq * 32, M, 1), threadgroup=(32, 1, 1),
            output_shapes=[(M, Hq, D)], output_dtypes=[q.dtype])[0]
    return _kernel("pathattn", ["q", "kc", "vc", "anc", "dl", "lb"], ["o"])(
        inputs=[q, kc, vc, anc, dlen, lb],
        template=[("T", q.dtype), ("D", D), ("GQA", gqa), ("NT", NT)],
        grid=(Hq * NT, M, 1), threadgroup=(NT, 1, 1),
        output_shapes=[(M, Hq, D)], output_dtypes=[q.dtype])[0]
