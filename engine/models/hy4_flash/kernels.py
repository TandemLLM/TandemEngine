"""Hy4-flash's row kernel: one batched product whose every element is computed alone.

A block's rows must have the decode step's bits whatever the block: a row computed alone and the same row
among 31 others. Library products and reductions do not promise that (on the GB10 a row of `F.linear`, of a
batched `bmm` or of a mean over 3,072 values differs from its one-row call), so every sum the row pass takes
goes through `mm`:

    mm(a [B, I, K], b [NB, K, J]) -> fp32 [B, I, J]        out[n] = a[n] @ b[n mod NB]

One program a batch item and output tile. A tile's elements are each the sum over K in blocks of BK, the
blocks in order, so an element's bits depend on K and the tile sizes and never on B, on the other items or on
what the masked lanes hold. The row pass puts a block's rows (or its rows times heads) in the batch axis; the
rest of a row is element-wise operations, gathers and a stable sort, which are exact.

Both operands bf16: the products run on the tensor cores and are exact (two 8-bit mantissas), summed in fp32.
Otherwise both are widened to fp32 first. `b` may be any strided view (a transpose costs nothing), shared by
every item (NB 1), one a head (the absorbed key and value halves) or one an item (a row's own read list).

`mm_reference` is the same contract in plain torch, one item a call: the CPU tests' path and the debugging
knob's (`HY4_ROW_TORCH`). Its sums are torch's, so its bits are not the kernel's.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:          # noqa: BLE001  (a Mac, a CPU venv: the reference only)
    HAVE_TRITON = False
try:
    import triton.language.extra.libdevice as _LD
except Exception:          # noqa: BLE001
    _LD = None

#: SPD-345: a hyper-connection in one launch (`_hc_fused`) from this many rows, else the three launches (the same
#: bits either way: the fused program runs their statements in their shapes). Below it the three launches keep more of
#: the device busy (the mixes of a row are NMIX programs there, one here)
HC_FUSED_MIN = 16

#: the K block (fixed: an element's summation order)
BK = 64

if HAVE_TRITON:

    @triton.jit
    def _mm(A, B, C, I, J, K, NB, sa_b, sa_i, sa_k, sb_b, sb_k, sb_j, sc_b, sc_i, sc_j,
            BI: tl.constexpr, BJ: tl.constexpr, BK_: tl.constexpr, BF16: tl.constexpr):
        pb = tl.program_id(0).to(tl.int64)
        oi = tl.program_id(1) * BI + tl.arange(0, BI)
        oj = tl.program_id(2) * BJ + tl.arange(0, BJ)
        a_ptr = A + pb * sa_b
        b_ptr = B + (pb % NB) * sb_b
        acc = tl.zeros([BI, BJ], dtype=tl.float32)
        for k0 in range(0, K, BK_):
            ok = k0 + tl.arange(0, BK_)
            a = tl.load(a_ptr + oi[:, None].to(tl.int64) * sa_i + ok[None, :].to(tl.int64) * sa_k,
                        mask=(oi[:, None] < I) & (ok[None, :] < K), other=0.0)
            b = tl.load(b_ptr + ok[:, None].to(tl.int64) * sb_k + oj[None, :].to(tl.int64) * sb_j,
                        mask=(ok[:, None] < K) & (oj[None, :] < J), other=0.0)
            if BF16:
                acc += tl.dot(a, b)
            else:
                acc += tl.dot(a.to(tl.float32), b.to(tl.float32))
        tl.store(C + pb * sc_b + oi[:, None].to(tl.int64) * sc_i + oj[None, :].to(tl.int64) * sc_j, acc,
                 mask=(oi[:, None] < I) & (oj[None, :] < J))


def mm(a: torch.Tensor, b: torch.Tensor, bj: int | None = None) -> torch.Tensor:
    """a [B, I, K] @ b [NB, K, J] (NB divides B) -> fp32 [B, I, J], every element its own sum (above).
    `bj`: the output tile's width (default by J); two calls whose elements must agree take the same one."""
    Bn, I, K = a.shape
    NB, Kb, J = b.shape
    if Kb != K or Bn % NB:
        raise ValueError(f"mm: {tuple(a.shape)} @ {tuple(b.shape)}")
    out = torch.empty(Bn, I, J, dtype=torch.float32, device=a.device)
    if Bn == 0 or I == 0 or J == 0:
        return out
    BJ = bj or (16 if J <= 16 else 32 if J <= 32 else 64)
    both16 = a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    _mm[(Bn, triton.cdiv(I, 16), triton.cdiv(J, BJ))](
        a, b, out, I, J, K, NB, a.stride(0), a.stride(1), a.stride(2), b.stride(0), b.stride(1), b.stride(2),
        out.stride(0), out.stride(1), out.stride(2), BI=16, BJ=BJ, BK_=BK, BF16=both16, num_warps=4)
    return out


def mm_reference(a: torch.Tensor, b: torch.Tensor, bj: int | None = None) -> torch.Tensor:
    """`mm` in plain torch, one item a call (so an item's bits do not depend on the batch either)."""
    NB = b.shape[0]
    if a.shape[0] == 0:
        return torch.empty(0, a.shape[1], b.shape[2], dtype=torch.float32, device=a.device)
    return torch.stack([a[n].float() @ b[n % NB].float() for n in range(a.shape[0])])


# ------------------------------------------------------------------------------------ the joint pass's lane reads
# SPD-349: a joint verify pass holds the rows of several sequences; each row reads its own sequence's committed rows.
# The sequences' buffers are separate tensors (paged or not, each reserved as one address range), so a row's lane names
# a base address in a table (int64 device pointers, one a lane) and the kernels read there: the pass never depends on
# which tensors the lanes hold, and a graph of it replays for any lanes. Both kernels compute what their one-lane forms
# do (a copy; `_mm`'s statements in its shapes), so a row's bits are its lone pass's.

if HAVE_TRITON:

    @triton.jit
    def _gather_lanes(OUT, IDX, FLAG, LANE, APTR, a_off, a_stride, BSRC, b_stride, n, C,
                      BN: tl.constexpr, CB: tl.constexpr):
        """out[r, j, :C] = FLAG[r, j] ? BSRC[IDX[r, j]] : lane(r)'s A[a_off + IDX[r, j]] (rows of C bf16)."""
        r = tl.program_id(0).to(tl.int64)
        j = tl.program_id(1) * BN + tl.arange(0, BN)
        jm = j < n
        c = tl.arange(0, CB)
        cm = c < C
        idx = tl.load(IDX + r * n + j, mask=jm, other=0).to(tl.int64)
        fl = tl.load(FLAG + r * n + j, mask=jm, other=0) != 0
        lane = tl.load(LANE + r)
        abase = tl.load(APTR + lane).to(tl.pointer_type(tl.bfloat16))
        m2 = jm[:, None] & cm[None, :]
        va = tl.load(abase + a_off + idx[:, None] * a_stride + c[None, :], mask=m2 & ~fl[:, None], other=0.0)
        vb = tl.load(BSRC + idx[:, None] * b_stride + c[None, :], mask=m2 & fl[:, None], other=0.0)
        tl.store(OUT + (r * n + j[:, None]) * C + c[None, :], tl.where(fl[:, None], vb, va), mask=m2)

    @triton.jit
    def _mm_lanes(A, BPTR, LANE, C, I, J, K, b_off, sa_b, sa_i, sa_k, sb_k, sb_j, sc_b, sc_i, sc_j,
                  BI: tl.constexpr, BJ: tl.constexpr, BK_: tl.constexpr):
        """`_mm` (bf16 both, NB 1) with item n's b at its lane's base address + b_off: the same statements."""
        pb = tl.program_id(0).to(tl.int64)
        oi = tl.program_id(1) * BI + tl.arange(0, BI)
        oj = tl.program_id(2) * BJ + tl.arange(0, BJ)
        a_ptr = A + pb * sa_b
        b_ptr = tl.load(BPTR + tl.load(LANE + pb)).to(tl.pointer_type(tl.bfloat16)) + b_off
        acc = tl.zeros([BI, BJ], dtype=tl.float32)
        for k0 in range(0, K, BK_):
            ok = k0 + tl.arange(0, BK_)
            a = tl.load(a_ptr + oi[:, None].to(tl.int64) * sa_i + ok[None, :].to(tl.int64) * sa_k,
                        mask=(oi[:, None] < I) & (ok[None, :] < K), other=0.0)
            b = tl.load(b_ptr + ok[:, None].to(tl.int64) * sb_k + oj[None, :].to(tl.int64) * sb_j,
                        mask=(ok[:, None] < K) & (oj[None, :] < J), other=0.0)
            acc += tl.dot(a, b)
        tl.store(C + pb * sc_b + oi[:, None].to(tl.int64) * sc_i + oj[None, :].to(tl.int64) * sc_j, acc,
                 mask=(oi[:, None] < I) & (oj[None, :] < J))


def gather_lanes(idx: torch.Tensor, flag: torch.Tensor, lane: torch.Tensor, aptr: torch.Tensor, a_off: int, a_stride: int,
                 bsrc: torch.Tensor, C: int) -> torch.Tensor:
    """idx int64 [R, n] (contiguous), flag bool [R, n], lane int64 [R], aptr int64 [lanes] (bf16 bases) -> bf16 [R, n, C]:
    a row's entries from its lane's buffer at a_off + idx * a_stride (flag 0) or from bsrc's row idx (flag 1)."""
    R, n = idx.shape
    out = torch.empty(R, n, C, dtype=torch.bfloat16, device=idx.device)
    if R and n:
        assert bsrc.stride(-1) == 1 and idx.is_contiguous() and flag.is_contiguous()
        BN = 8 if C > 256 else 32
        _gather_lanes[(R, triton.cdiv(n, BN))](out, idx, flag, lane, aptr, a_off, a_stride, bsrc, bsrc.stride(0), n, C,
                                               BN=BN, CB=_pow2(C), num_warps=4)
    return out


def mm_lanes(a: torch.Tensor, bptr: torch.Tensor, lane: torch.Tensor, b_off: int, K: int, J: int, sb_k: int, sb_j: int,
             bj: int | None = None) -> torch.Tensor:
    """`mm(a, b)` where item n's b [K, J] (bf16, strides sb_k, sb_j) is at its lane's base + b_off -> fp32 [B, I, J], the
    bits of `mm(a[n:n+1], b_lane)` for every n (a bf16)."""
    Bn, I, Ka = a.shape
    assert Ka == K and a.dtype == torch.bfloat16
    out = torch.empty(Bn, I, J, dtype=torch.float32, device=a.device)
    if Bn == 0 or I == 0 or J == 0:
        return out
    BJ = bj or (16 if J <= 16 else 32 if J <= 32 else 64)
    _mm_lanes[(Bn, triton.cdiv(I, 16), triton.cdiv(J, BJ))](
        a, bptr, lane, out, I, J, K, b_off, a.stride(0), a.stride(1), a.stride(2), sb_k, sb_j,
        out.stride(0), out.stride(1), out.stride(2), BI=16, BJ=BJ, BK_=BK, num_warps=4)
    return out


# ------------------------------------------------------------------------------------ fused row kernels
# One launch for a chain of a row's operations. A program owns one row (or one output element of a row); a sum
# runs over the program's whole block of threads in the order the block's shape gives (`tl.sum`), so it is parallel
# and its bits depend on the shape alone, never on how many rows the launch holds. bf16 activations only (the
# served dtype); the fp32 tests and the debugging knob take `mm`.

def _pow2(n: int) -> int:
    p = 16
    while p < n:
        p *= 2
    return p


if HAVE_TRITON:

    @triton.jit
    def _norm(X, W, OUT, K, s_x, s_o, eps, KB: tl.constexpr, HASW: tl.constexpr):
        m = tl.program_id(0).to(tl.int64)
        k = tl.arange(0, KB)
        msk = k < K
        x = tl.load(X + m * s_x + k, mask=msk, other=0.0).to(tl.float32)
        n = (x * (1.0 / tl.sqrt(tl.sum(x * x, axis=0) / K + eps))).to(tl.bfloat16)
        if HASW:
            n = (tl.load(W + k, mask=msk, other=0.0).to(tl.float32) * n.to(tl.float32)).to(tl.bfloat16)
        tl.store(OUT + m * s_o + k, n, mask=msk)

    @triton.jit
    def _gemv(X, W, OUT, K, N, s_x, s_w, KB: tl.constexpr, SIG: tl.constexpr, BF: tl.constexpr):
        m = tl.program_id(0).to(tl.int64)
        j = tl.program_id(1).to(tl.int64)
        acc = tl.zeros([1], dtype=tl.float32)
        for k0 in range(0, K, KB):
            k = k0 + tl.arange(0, KB)
            msk = k < K
            x = tl.load(X + m * s_x + k, mask=msk, other=0.0).to(tl.float32)
            w = tl.load(W + j * s_w + k, mask=msk, other=0.0).to(tl.float32)
            acc += tl.sum(x * w, axis=0)
        if SIG:
            acc = tl.sigmoid(acc)
        if BF:
            tl.store(OUT + m * N + j + tl.arange(0, 1), acc.to(tl.bfloat16))
        else:
            tl.store(OUT + m * N + j + tl.arange(0, 1), acc)

    @triton.jit
    def _router_topk(SC, BIAS, IDS, WTS, E, scale, EB: tl.constexpr, KK: tl.constexpr, NORM: tl.constexpr):
        m = tl.program_id(0).to(tl.int64)
        e = tl.arange(0, EB)
        msk = e < E
        s = tl.load(SC + m * E + e, mask=msk, other=0.0)
        c = tl.where(msk, s + tl.load(BIAS + e, mask=msk, other=0.0), float("-inf"))
        tot = tl.zeros([1], dtype=tl.float32)
        for i in tl.static_range(KK):
            j = tl.argmax(c, axis=0)
            w = tl.sum(tl.where(e == j, s, 0.0), axis=0)
            tl.store(IDS + m * KK + i + tl.arange(0, 1), j.to(tl.int32) + tl.zeros([1], dtype=tl.int32))
            tl.store(WTS + m * KK + i + tl.arange(0, 1), w + tl.zeros([1], dtype=tl.float32))
            tot += w
            c = tl.where(e == j, float("-inf"), c)
        kk = tl.arange(0, KK)
        wv = tl.load(WTS + m * KK + kk)
        if NORM:
            wv = wv / (tot + 1e-20)
        tl.store(WTS + m * KK + kk, wv * scale)

    @triton.jit
    def _softmax_sink(S_, VALID, SINK, P, n, heads, NB: tl.constexpr):
        m = tl.program_id(0).to(tl.int64)
        h = tl.program_id(1).to(tl.int64)
        j = tl.arange(0, NB)
        msk = j < n
        s = tl.load(S_ + (m * heads + h) * n + j, mask=msk, other=0.0)
        v = tl.load(VALID + m * n + j, mask=msk, other=0)
        s = tl.where(msk & (v != 0), s, float("-inf"))
        sink = tl.load(SINK + h + tl.arange(0, 1))
        mx = tl.maximum(tl.max(s, axis=0), sink)
        e = tl.exp(s - mx)
        z = tl.sum(e, axis=0) + tl.exp(sink - mx)
        tl.store(P + (m * heads + h) * n + j, (e / z).to(tl.bfloat16), mask=msk)

    @triton.jit
    def _rot(X, COS, SIN, s_m, s_g, D0, P_, PB: tl.constexpr):
        m = tl.program_id(0).to(tl.int64)
        g = tl.program_id(1).to(tl.int64)
        i = tl.arange(0, PB)
        msk = i < P_
        base = X + m * s_m + g * s_g + D0
        a = tl.load(base + 2 * i, mask=msk, other=0.0).to(tl.float32)
        b = tl.load(base + 2 * i + 1, mask=msk, other=0.0).to(tl.float32)
        c = tl.load(COS + m * P_ + i, mask=msk, other=0.0).to(tl.float32)
        s = tl.load(SIN + m * P_ + i, mask=msk, other=0.0).to(tl.float32)
        tl.store(base + 2 * i, (a * c - b * s).to(tl.bfloat16), mask=msk)
        tl.store(base + 2 * i + 1, (b * c + a * s).to(tl.bfloat16), mask=msk)

    @triton.jit
    def _inject(Hh, Y, POST, OUT, S: tl.constexpr, HD, KB: tl.constexpr):
        m = tl.program_id(0).to(tl.int64)
        s = tl.program_id(1).to(tl.int64)
        k = tl.arange(0, KB)
        msk = k < HD
        h = tl.load(Hh + (m * S + s) * HD + k, mask=msk, other=0.0).to(tl.float32)
        y = tl.load(Y + m * HD + k, mask=msk, other=0.0).to(tl.float32)
        p = tl.load(POST + m * S + s + tl.arange(0, 1))
        tl.store(OUT + (m * S + s) * HD + k, (p * y + h).to(tl.bfloat16), mask=msk)

    @triton.jit
    def _hc_sq(Hh, Y, POST, HOUT, MIX, HD, NMIX, S: tl.constexpr, NB: tl.constexpr, INJECT: tl.constexpr):
        """The sum of squares of a row's streams (after the inject of the layer before, which it also stores)."""
        m = tl.program_id(0).to(tl.int64)
        o = tl.arange(0, NB)
        msk = o < S * HD
        f = tl.load(Hh + m * S * HD + o, mask=msk, other=0.0)
        if INJECT:
            y = tl.load(Y + m * HD + o % HD, mask=msk, other=0.0).to(tl.float32)
            p = tl.load(POST + m * S + o // HD, mask=msk, other=0.0)
            f = (p * y + f.to(tl.float32)).to(tl.bfloat16)
            tl.store(HOUT + m * S * HD + o, f, mask=msk)
        ff = f.to(tl.float32)
        tl.store(MIX + m * (NMIX + 1) + NMIX + tl.arange(0, 1), tl.sum(ff * ff, axis=0) + tl.zeros([1], dtype=tl.float32))

    @triton.jit
    def _hc_mix(Hh, Y, POST, FN, MIX, HD, NMIX, S: tl.constexpr, NB: tl.constexpr, INJECT: tl.constexpr):
        """One gate's mix of a row: the flattened streams times one row of the gate matrix."""
        m = tl.program_id(0).to(tl.int64)
        j = tl.program_id(1).to(tl.int64)
        o = tl.arange(0, NB)
        msk = o < S * HD
        f = tl.load(Hh + m * S * HD + o, mask=msk, other=0.0)
        if INJECT:
            y = tl.load(Y + m * HD + o % HD, mask=msk, other=0.0).to(tl.float32)
            p = tl.load(POST + m * S + o // HD, mask=msk, other=0.0)
            f = (p * y + f.to(tl.float32)).to(tl.bfloat16)
        w = tl.load(FN + j * S * HD + o, mask=msk, other=0.0)
        tl.store(MIX + m * (NMIX + 1) + j + tl.arange(0, 1), tl.sum(f.to(tl.float32) * w, axis=0) + tl.zeros([1], dtype=tl.float32))

    @triton.jit
    def _hc_apply(Hh, MIX, BASE, SCALE, W, XN, POSTO, HD, NMIX, hc_eps, mag, eps, S: tl.constexpr, HB: tl.constexpr, HEAD: tl.constexpr):
        """The gates from the mixes, the streams' weighted sum, its norm with the weight; the post gates out."""
        m = tl.program_id(0).to(tl.int64)
        js = tl.arange(0, S)
        ms = tl.sum(tl.load(MIX + m * (NMIX + 1) + NMIX + tl.arange(0, 1)), axis=0) / (S * HD)
        r = 1.0 / tl.sqrt(ms + eps)
        s0 = tl.sum(tl.load(SCALE + tl.arange(0, 1)), axis=0)
        pre = tl.sigmoid(tl.load(MIX + m * (NMIX + 1) + js) * r * s0 + tl.load(BASE + js)) + hc_eps
        if not HEAD:
            s1 = tl.sum(tl.load(SCALE + 1 + tl.arange(0, 1)), axis=0)
            post = mag * tl.sigmoid(tl.load(MIX + m * (NMIX + 1) + S + js) * r * s1 + tl.load(BASE + S + js)) + hc_eps
            tl.store(POSTO + m * S + js, post)
        k = tl.arange(0, HB)
        km = k < HD
        h = tl.load(Hh + m * S * HD + js[:, None] * HD + k[None, :], mask=km[None, :], other=0.0).to(tl.float32)
        xf = tl.sum(pre[:, None] * h, axis=0).to(tl.bfloat16).to(tl.float32)
        n = (xf * (1.0 / tl.sqrt(tl.sum(xf * xf, axis=0) / HD + eps))).to(tl.bfloat16).to(tl.float32)
        w = tl.load(W + k, mask=km, other=0.0).to(tl.float32)
        tl.store(XN + m * HD + k, (w * n).to(tl.bfloat16), mask=km)

    @triton.jit
    def _hc_fused(Hh, Y, POST, HOUT, FN, MIX, BASE, SCALE, W, XN, POSTO, HD, NMIX, hc_eps, mag, eps,
                  S: tl.constexpr, NB: tl.constexpr, HB: tl.constexpr, INJECT: tl.constexpr, HEAD: tl.constexpr):
        """SPD-345: `_hc_sq`, every `_hc_mix` and `_hc_apply` of a row in one program, statement for statement: the
        same loads, the same block shapes and warps, so every `tl.sum` is the three launches' tree and the bits are
        theirs. The mixes and the injected streams go through global memory between the phases, as before, behind a
        barrier of the program's threads (a CTA's writes are seen by its own threads after `bar.sync`)."""
        m = tl.program_id(0).to(tl.int64)
        o = tl.arange(0, NB)
        msk = o < S * HD
        f = tl.load(Hh + m * S * HD + o, mask=msk, other=0.0)
        if INJECT:
            y = tl.load(Y + m * HD + o % HD, mask=msk, other=0.0).to(tl.float32)
            p = tl.load(POST + m * S + o // HD, mask=msk, other=0.0)
            f = (p * y + f.to(tl.float32)).to(tl.bfloat16)
            tl.store(HOUT + m * S * HD + o, f, mask=msk)
        ff = f.to(tl.float32)
        tl.store(MIX + m * (NMIX + 1) + NMIX + tl.arange(0, 1), tl.sum(ff * ff, axis=0) + tl.zeros([1], dtype=tl.float32))
        for j in range(NMIX):
            w = tl.load(FN + j * S * HD + o, mask=msk, other=0.0)
            tl.store(MIX + m * (NMIX + 1) + j + tl.arange(0, 1), tl.sum(ff * w, axis=0) + tl.zeros([1], dtype=tl.float32))
        tl.debug_barrier()
        js = tl.arange(0, S)
        ms = tl.sum(tl.load(MIX + m * (NMIX + 1) + NMIX + tl.arange(0, 1)), axis=0) / (S * HD)
        r = 1.0 / tl.sqrt(ms + eps)
        s0 = tl.sum(tl.load(SCALE + tl.arange(0, 1)), axis=0)
        pre = tl.sigmoid(tl.load(MIX + m * (NMIX + 1) + js) * r * s0 + tl.load(BASE + js)) + hc_eps
        if not HEAD:
            s1 = tl.sum(tl.load(SCALE + 1 + tl.arange(0, 1)), axis=0)
            post = mag * tl.sigmoid(tl.load(MIX + m * (NMIX + 1) + S + js) * r * s1 + tl.load(BASE + S + js)) + hc_eps
            tl.store(POSTO + m * S + js, post)
        k = tl.arange(0, HB)
        km = k < HD
        h = tl.load(HOUT + m * S * HD + js[:, None] * HD + k[None, :], mask=km[None, :], other=0.0).to(tl.float32)
        xf = tl.sum(pre[:, None] * h, axis=0).to(tl.bfloat16).to(tl.float32)
        n = (xf * (1.0 / tl.sqrt(tl.sum(xf * xf, axis=0) / HD + eps))).to(tl.bfloat16).to(tl.float32)
        w = tl.load(W + k, mask=km, other=0.0).to(tl.float32)
        tl.store(XN + m * HD + k, (w * n).to(tl.bfloat16), mask=km)

    @triton.jit
    def _mul_sigmoid(X, G, OUT, N, BLK: tl.constexpr):
        """out = x * sigmoid(g), bf16, torch's two operations' bits: the sigmoid 1 / (1 + exp(-g)) in fp32 with the
        full-precision exp and an IEEE division (torch's CUDA sigmoid), rounded to bf16, the product in fp32 rounded."""
        i = tl.program_id(0).to(tl.int64) * BLK + tl.arange(0, BLK)
        msk = i < N
        g = tl.load(G + i, mask=msk, other=0.0).to(tl.float32)
        s = tl.math.div_rn(tl.full([BLK], 1.0, tl.float32), 1.0 + _LD.exp(-g)).to(tl.bfloat16)
        x = tl.load(X + i, mask=msk, other=0.0).to(tl.float32)
        tl.store(OUT + i, (x * s.to(tl.float32)).to(tl.bfloat16), mask=msk)


if HAVE_TRITON:

    @triton.jit
    def _wsum_relu(SC, HW, OUT, G, NH: tl.constexpr, GB: tl.constexpr):
        """out[m, g] = sum over heads of hw[m, h] * relu(sc[m, h, g]): the indexer's score of a group."""
        m = tl.program_id(0).to(tl.int64)
        g = tl.program_id(1) * GB + tl.arange(0, GB)
        hs = tl.arange(0, NH)
        msk = g < G
        sc = tl.load(SC + (m * NH + hs[:, None]) * G + g[None, :], mask=msk[None, :], other=0.0)
        hw = tl.load(HW + m * NH + hs).to(tl.float32)
        tl.store(OUT + m * G + g, tl.sum(hw[:, None] * tl.maximum(sc, 0.0), axis=0), mask=msk)


def wsum_relu(sc: torch.Tensor, hw: torch.Tensor) -> torch.Tensor:
    """sc fp32 [B, heads, G], hw [B, heads] -> fp32 [B, G] (heads a power of two)."""
    B, NH, G = sc.shape
    out = torch.empty(B, G, dtype=torch.float32, device=sc.device)
    _wsum_relu[(B, triton.cdiv(G, 256))](sc, hw, out, G, NH=NH, GB=256, num_warps=4)
    return out


def norm(x: torch.Tensor, w: torch.Tensor | None, eps: float, out: torch.Tensor | None = None) -> torch.Tensor:
    """The RMS norm of rows x [B, K] bf16 with its weight (bf16 [K]) in one launch; `out` may be x itself."""
    B, K = x.shape
    if out is None:
        out = torch.empty(B, K, dtype=torch.bfloat16, device=x.device)
    _norm[(B,)](x, x if w is None else w, out, K, x.stride(0), out.stride(0), eps, KB=_pow2(K), HASW=w is not None, num_warps=8)
    return out


def gemv(x: torch.Tensor, w: torch.Tensor, sigmoid: bool = False, bf16: bool = False) -> torch.Tensor:
    """x [B, K] bf16 @ w [N, K]^T (as stored: bf16, fp16 or fp32) -> [B, N] fp32 (or bf16): one program an element."""
    B, K = x.shape
    N = w.shape[0]
    out = torch.empty(B, N, dtype=torch.bfloat16 if bf16 else torch.float32, device=x.device)
    _gemv[(B, N)](x, w, out, K, N, x.stride(0), w.stride(0), KB=min(_pow2(K), 4096), SIG=sigmoid, BF=bf16, num_warps=8)
    return out


def router(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, k: int, scale: float, renorm: bool) -> tuple:
    """Rows x [B, H] -> their k routed experts (best first, ties to the lower id) int32 [B, k] and weights fp32."""
    B, E = x.shape[0], w.shape[0]
    sc = gemv(x, w, sigmoid=True)
    ids = torch.empty(B, k, dtype=torch.int32, device=x.device)
    wts = torch.empty(B, k, dtype=torch.float32, device=x.device)
    _router_topk[(B,)](sc, bias, ids, wts, E, scale, EB=_pow2(E), KK=k, NORM=renorm, num_warps=4)
    return ids, wts


def softmax_sink(s: torch.Tensor, valid: torch.Tensor, sinks: torch.Tensor) -> torch.Tensor:
    """s fp32 [B, heads, n] with valid [B, n] and one sink logit a head -> the keys' weights bf16 [B, heads, n]."""
    B, heads, n = s.shape
    p = torch.empty(B, heads, n, dtype=torch.bfloat16, device=s.device)
    _softmax_sink[(B, heads)](s, valid, sinks, p, n, heads, NB=_pow2(n), num_warps=4)
    return p


def rot(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, d0: int) -> None:
    """The rotary on x [B, G, D] bf16 in place, on the channels from d0 (pairs; cos, sin bf16 [B, pairs])."""
    B, G, D = x.shape
    P_ = cos.shape[1]
    _rot[(B, G)](x, cos, sin, x.stride(0), x.stride(1), d0, P_, PB=_pow2(P_), num_warps=1)


def rot_exact(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, d0: int) -> None:
    """SPD-347: `rot` with every product rounded before the sum (no fused multiply-add): the bits of ref.py's
    `rotate` (x * cos + rotate_half(x) * sin in fp32 as three torch operations, back to bf16), in place. The prompt
    pass's rotary, which `rotate` defines; the row pass keeps `rot`."""
    B, G, D = x.shape
    P_ = cos.shape[1]
    assert cos.is_contiguous() and sin.is_contiguous() and x.stride(2) == 1
    _rot[(B, G)](x, cos, sin, x.stride(0), x.stride(1), d0, P_, PB=_pow2(P_), num_warps=1, enable_fp_fusion=False)


if HAVE_TRITON:

    @triton.jit
    def _engram_decode(PK, HEADS, BIAS, CB, OUT, W, DIM, K: tl.constexpr, DB: tl.constexpr):
        """One packed Engram row -> its DIM values bf16, table.py `decode_rows` statement for statement: the 16 state
        bits of position i from stream bits ((i - m // K) mod DIM) K + m % K, the mul1 codebook value (fp16) times the
        row's fp16 scale plus its head's bias, the product rounded before the sum (launched without fused
        multiply-add), then bf16."""
        n = tl.program_id(0).to(tl.int64)
        i = tl.arange(0, DB)
        msk = i < DIM
        base = PK + n * W
        scale = tl.load(base + tl.arange(0, 1)).to(tl.float16, bitcast=True).to(tl.float32)
        state = tl.zeros([DB], dtype=tl.int32)
        for m in tl.static_range(16):
            s = ((i - m // K + DIM) % DIM) * K + m % K
            wd = tl.load(base + 1 + s // 16, mask=msk, other=0).to(tl.int32) & 0xFFFF
            state = state | (((wd >> (s % 16)) & 1) << m)
        cb = tl.load(CB + state, mask=msk, other=0.0).to(tl.float32)
        hd = tl.load(HEADS + n).to(tl.int64)
        b = tl.load(BIAS + hd * DIM + i, mask=msk, other=0.0).to(tl.float32)
        v = cb * tl.sum(scale, axis=0) + b
        tl.store(OUT + n * DIM + i, v.to(tl.bfloat16), mask=msk)


def engram_decode(packed: torch.Tensor, heads: torch.Tensor, bias: torch.Tensor, K: int, codebook: torch.Tensor,
                  dim: int) -> torch.Tensor:
    """packed int16 [N, 1 + dim K / 16] on the device, heads int [N], bias fp16 [n_heads, dim], codebook fp16 [65536]
    -> bf16 [N, dim]: `table.decode_rows(...).to(torch.bfloat16)`'s bits in one launch."""
    N, W = packed.shape
    out = torch.empty(N, dim, dtype=torch.bfloat16, device=packed.device)
    if N:
        _engram_decode[(N,)](packed, heads, bias, codebook, out, W, dim, K=K, DB=_pow2(dim), num_warps=1,
                             enable_fp_fusion=False)
    return out


def inject(h: torch.Tensor, y: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
    B, S, H = h.shape
    out = torch.empty_like(h)
    _inject[(B, S)](h, y, post, out, HD=H, S=S, KB=_pow2(H), num_warps=4)
    return out


def hc(h: torch.Tensor, pend: tuple | None, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, w: torch.Tensor,
       eps: float, hc_eps: float, mag: float, head: bool = False) -> tuple:
    """A hyper-connection in three launches. h [B, S, H] bf16 (contiguous); `pend` = (y [B, H] bf16, post [B, S] fp32):
    the block output of the layer before, injected first. -> the normed, weighted stream mix [B, H], the post gates
    [B, S] (None for the head form), the streams after the inject."""
    B, S, H = h.shape
    NMIX = fn.shape[0]
    mix = torch.empty(B, NMIX + 1, dtype=torch.float32, device=h.device)
    y, post = pend if pend is not None else (h, mix)
    hout = torch.empty_like(h) if pend is not None else h
    NB = _pow2(S * H)
    xn = torch.empty(B, H, dtype=torch.bfloat16, device=h.device)
    posto = None if head else torch.empty(B, S, dtype=torch.float32, device=h.device)
    if HC_FUSED_MIN and B >= HC_FUSED_MIN:
        _hc_fused[(B,)](h, y, post, hout, fn, mix, base, scale, w, xn, mix if head else posto, H, NMIX, hc_eps, mag, eps,
                        S=S, NB=NB, HB=_pow2(H), INJECT=pend is not None, HEAD=head, num_warps=8)
        return xn, posto, hout
    _hc_sq[(B,)](h, y, post, hout, mix, H, NMIX, S=S, NB=NB, INJECT=pend is not None, num_warps=8)
    _hc_mix[(B, NMIX)](h, y, post, fn, mix, H, NMIX, S=S, NB=NB, INJECT=pend is not None, num_warps=8)
    _hc_apply[(B,)](hout, mix, base, scale, w, xn, mix if head else posto, H, NMIX, hc_eps, mag, eps, S=S, HB=_pow2(H), HEAD=head, num_warps=8)
    return xn, posto, hout


def mul_sigmoid(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """x * torch.sigmoid(g) for bf16 x, g of one shape (contiguous) in one launch, the two torch operations' bits."""
    x, g = x.contiguous(), g.contiguous()
    out = torch.empty_like(x)
    N = x.numel()
    _mul_sigmoid[(triton.cdiv(N, 1024),)](x, g, out, N, BLK=1024, num_warps=4)
    return out


def mul_sigmoid_ok() -> bool:
    return HAVE_TRITON and _LD is not None and hasattr(tl.math, "div_rn")


# ------------------------------------------------------------------------------------ the prompt pass's attention
# SPD-323 (the prefill design's F1): Hy4's latent attention over a prompt chunk in one launch, both kinds. A program owns
# one row and a tile of MLA_HB heads. A row's visible keys are its virtual columns 0..n-1: position base + c (a dense or
# window row) or SEL[row, c] (a row that reads a list), read in place from one latent buffer [rows, R + RP] by position,
# MLA_BN keys a step in the row's own order. Scores (qa . k[:R] + qr . k[R:]) * scale on the tensor cores in fp32; a
# running softmax over the steps (fn_seq_kernels `_attn_rows_split`'s); the weights rounded to bf16 times k[:R], summed
# in fp32; at the end the sink joins the denominator. A row's sums depend on its own keys in its own order, never on
# the other rows of the launch. The launch shape is part of the bits (a sum's order follows its block's shape): the
# constants below are frozen with tests/gpu/test_hy4_mla_rows_gpu.py's digests.

#: heads a program takes (the query tile of every dot)
MLA_HB = 16
#: keys a step
MLA_BN = 32
#: warps and pipeline stages of the launch
MLA_WARPS = 8
MLA_STAGES = 2

if HAVE_TRITON:

    @triton.jit
    def _mla_rows(QA, QR, LAT, CTL, SEL, SINK, OUT, heads, scale, s_qa_t, s_qa_h, s_qr_t, s_qr_h, s_lat, s_sel,
                  s_o_t, s_o_h, R: tl.constexpr, RP: tl.constexpr, HB: tl.constexpr, BN: tl.constexpr):
        t = tl.program_id(0).to(tl.int64)
        hs = tl.program_id(1) * HB + tl.arange(0, HB)
        hm = hs < heads
        base = tl.load(CTL + t * 3)
        n = tl.load(CTL + t * 3 + 1)
        lst = tl.load(CTL + t * 3 + 2)
        r = tl.arange(0, R)
        rp = tl.arange(0, RP)
        qa = tl.load(QA + t * s_qa_t + hs[:, None].to(tl.int64) * s_qa_h + r[None, :], mask=hm[:, None], other=0.0)
        qr = tl.load(QR + t * s_qr_t + hs[:, None].to(tl.int64) * s_qr_h + rp[None, :], mask=hm[:, None], other=0.0)
        m_i = tl.full([HB], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([HB], dtype=tl.float32)
        acc = tl.zeros([HB, R], dtype=tl.float32)
        for c0 in range(0, n, BN):
            cols = c0 + tl.arange(0, BN)
            cm = cols < n
            sel = tl.load(SEL + t * s_sel + cols, mask=cm & (lst != 0), other=0)
            pos = tl.where(lst != 0, sel, base + cols).to(tl.int64)
            k = tl.load(LAT + pos[:, None] * s_lat + r[None, :], mask=cm[:, None], other=0.0)
            kr = tl.load(LAT + pos[:, None] * s_lat + R + rp[None, :], mask=cm[:, None], other=0.0)
            s = tl.dot(qa, tl.trans(k)) + tl.dot(qr, tl.trans(kr))
            s = tl.where(cm[None, :], s * scale, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            p = tl.exp(s - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), k)
            m_i = m_new
        # the sink as one more logit: everything rescaled to the larger of the keys' maximum and the sink, so no term
        # overflows (the same quotient as mix / (l + exp(sink - m)))
        sink = tl.load(SINK + hs, mask=hm, other=0.0)
        mx = tl.maximum(m_i, sink)
        a = tl.exp(m_i - mx)
        o = acc * (a / (l_i * a + tl.exp(sink - mx)))[:, None]
        tl.store(OUT + t * s_o_t + hs[:, None].to(tl.int64) * s_o_h + r[None, :], o.to(tl.bfloat16), mask=hm[:, None])


def mla_rows(qa: torch.Tensor, qr: torch.Tensor, lat: torch.Tensor, ctl: torch.Tensor, sel: torch.Tensor,
             sink: torch.Tensor, scale: float) -> torch.Tensor:
    """Every row of a prompt chunk over its own keys. qa [T, heads, R] bf16 (the absorbed key half of the query), qr
    [T, heads, RP] bf16 (its rotated rope half), lat [N, R + RP] bf16 (the latents by position, any row stride),
    ctl int32 [T, 3] (base, n, reads a list), sel int32 [T, *] (a list row's positions, ascending), sink fp32 [heads]
    -> the weighted latents [T, heads, R] bf16, the sink's weight dropped."""
    T, H, R = qa.shape
    RP = qr.shape[2]
    assert lat.shape[1] == R + RP and lat.stride(1) == 1 and qa.stride(2) == 1 and qr.stride(2) == 1
    assert ctl.dtype == torch.int32 and ctl.is_contiguous() and sel.dtype == torch.int32 and sel.stride(1) == 1
    out = torch.empty(T, H, R, dtype=torch.bfloat16, device=qa.device)
    if T == 0:
        return out
    _mla_rows[(T, triton.cdiv(H, MLA_HB))](
        qa, qr, lat, ctl, sel, sink.contiguous(), out, H, float(scale), qa.stride(0), qa.stride(1), qr.stride(0), qr.stride(1),
        lat.stride(0), sel.stride(0), out.stride(0), out.stride(1), R=R, RP=RP, HB=MLA_HB, BN=MLA_BN,
        num_warps=MLA_WARPS, num_stages=MLA_STAGES)
    return out


def mla_rows_reference(qa: torch.Tensor, qr: torch.Tensor, lat: torch.Tensor, ctl: torch.Tensor, sel: torch.Tensor,
                       sink: torch.Tensor, scale: float) -> torch.Tensor:
    """`mla_rows` in plain torch, one row a call, fp32 (the CPU's path and the kernel's reference): the softmax with
    the sink as one more logit, exact, its weights unrounded -> qa's dtype."""
    T, H, R = qa.shape
    out = torch.empty(T, H, R, dtype=qa.dtype, device=qa.device)
    sk = sink.reshape(-1, 1).float()
    for t in range(T):
        base, n, lst = (int(v) for v in ctl[t].tolist())
        pos = sel[t, :n].long() if lst else torch.arange(base, base + n, device=lat.device)
        k = lat[pos].float()
        s = (qa[t].float() @ k[:, :R].T + qr[t].float() @ k[:, R:].T) * scale
        z = torch.cat((s, sk), dim=1)
        e = torch.exp(z - z.max(dim=1, keepdim=True).values)
        out[t] = ((e[:, :-1] / e.sum(dim=1, keepdim=True)) @ k[:, :R]).to(qa.dtype)
    return out
