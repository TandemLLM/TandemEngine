"""The prompt order of an EXL3 projection: one summation order for every row count of a prompt pass (R83).

`tools/exl3_linear.py` runs a projection on three kernel classes picked by the call's rows, and each sums K in its
own order: the row kernel up to 32 rows, the skinny GEMM `exl3_gemm` at `pick`'s split up to EXL3_CHUNK_UNTIL rows,
then a decoded fp16 slab on the library GEMM. A prompt row's bits then depend on the size of the chunk it is read in,
so a cache resume off the prefill grid is not the cold read.

Here a prompt pass's products sum in one order, given as arguments and never read from a table: S 1, eight K slices
of KT / 8 tiles each, each slice a chain of k16 products from zero in K order, the slices added as one chain from the
first one's value. Two kernels compute it, and both read the input rotated by `exl3_linear.rotate_in`:

    1 .. CROSS rows   the skinny GEMM with the order passed in, `exl3_matmul(x, w, S=1, wk=8)` (its 32-row chunks
                      in one launch from 33 rows on, as today; past EXL3_CHUNK_UNTIL rows in calls of that many)
    more rows         the slab decoded as `exl3_linear` decodes it, then `_slab_rot_s1` at L = KT / 8: the same
                      chain, with the output rotation in its epilogue (`tools/exl3_slab_gemm.py`, the pair the 27B's
                      `skinny_matmul` proves on its shapes)

so a row's bits do not depend on the call: `linear(x, w)[m] == linear(x[m:m+1], w)` at every row count. That is R for
the linear op; a family whose decode step reads the same projections on another kernel (Hy4's `dense`) keeps that
kernel for its row pass. `_slab_rot_s1` and its Walsh-Hadamard are copied here, not imported, so an edit of the 27B's
module does not move a family that adopts this one (the GPU guard holds the copy to the original's bits).

A half-integer rate (1.5, 2.5, 3.5 bits; Hy4 3.56's indexer query projections) cannot run on the skinny GEMM: it
takes the slab at every row count, which is the same order.

At bind (`bind`, at a shape's first call past ROWS rows or a half-integer shape's first call, or by the family at
load) each shape is checked once: 33 rows, CROSS rows and CROSS + 1 rows against the same rows in one-row calls.
Unequal raises `RowOrderError`, and `FAILED` holds the shape for the health report; so does a rate the kernels do not
decode: there is no silent fallback.

The order is a value (`Order`): its key is the chain of slices, equal for two launches exactly when they add the same
partials in the same order, whatever kernel runs them. The CPU test reads the order of each launch the dispatch makes
(tests/exl3_launches.py) and holds every row count to one.

Off GB10's inline extensions (the CPU, another arch) `linear` is `exl3_linear.exl3_matmul`: that path is unchanged.
A family's fingerprint covers this module once it imports it (the manifest's import walk); `tools/exl3_slab_gemm.py`
is not imported, so an edit of it moves no family that uses this one.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from tools import exl3_linear as X  # noqa: E402
from tools import gpu as _gpu  # noqa: E402

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                            # pragma: no cover
    HAVE_TRITON = False

#: rows a row kernel call takes
ROWS = 32
#: the prompt order: S ranks, WK K slices a rank (the skinny GEMM's binding takes wk 4 or 8)
S_PROMPT, WK_PROMPT = 1, 8
#: rows above which a call takes the slab (for speed only: both kernels give the order's bits): the rowinv design's
#: Gate 0 c, on Hy4's trunk, the first measured row count at which the slab beat the skinny GEMM at S1 w8 (512 rows:
#: the skinny GEMM; 1,024: the slab). Above EXL3_CHUNK_UNTIL the skinny GEMM runs in calls of that many rows
CROSS = 1024


class RowOrderError(RuntimeError):
    """A bound shape's prompt kernels do not give one order's bits."""


# ------------------------------------------------------------------------------------------ the order
@dataclass(frozen=True)
class Order:
    """The order one output element of a weight's product sums K in: `key`, a tuple of groups, each a tuple of
    slices (k0, k1) in 16-row K tiles; a slice is a chain of k16 products from zero in K order, a group its slices
    added in order from the first one's value, the product its groups added in order. Canonical: an empty slice or
    group is dropped (it adds +0, which no partial of a chain from +0 changes), and a group of one slice after the
    first is merged into the first while the first is all there is before it ((a + b) + c is the chain a, b, c).
    `form` names the launch it was read from and is not part of equality. A library product has no order of ours:
    its key names the call's shape, so it equals no order but a library product of the same shape."""

    key: tuple
    form: str = field(default="", compare=False)

    @staticmethod
    def chain(groups, form: str = "") -> "Order":
        gs = [tuple((int(a), int(b)) for a, b in g if int(b) > int(a)) for g in groups]
        gs = [g for g in gs if g]
        while len(gs) > 1 and len(gs[1]) == 1:
            gs = [gs[0] + gs[1]] + gs[2:]
        return Order(tuple(gs), form)

    @staticmethod
    def dense(KT: int, S: int, W: int) -> "Order":
        """`dense` / `dense_rows` at split (S, W): kernels.cu `dense_body` and `RowGemm`'s integer arithmetic, the
        S W partials added as one chain, rank-major, in warp order."""
        assert S >= 1 and KT % (8 * S) == 0 and W in (4, 8), (KT, S, W)
        per_r, sl = KT // S, []
        for r in range(S):
            ka, kb = r * per_r, r * per_r + per_r
            per = (kb - ka + W - 1) // W
            for w in range(W):
                k0 = min(kb, ka + w * per)
                sl.append((k0, min(kb, k0 + per)))
        return Order.chain([sl], f"dense S{S} w{W}")

    @staticmethod
    def skinny(KT: int, S: int, WK: int) -> "Order":
        """The skinny GEMM `exl3_gemm` at (S, wk): exl3.cu's ranks of ceil(KT / S) tiles, each over its warps in
        slices of ceil, the warps of a rank added in order, then the ranks in rank order."""
        per_r, groups = (KT + S - 1) // S, []
        for r in range(S):
            r0 = min(KT, r * per_r)
            r1 = min(KT, r0 + per_r)
            per_w = (r1 - r0 + WK - 1) // WK
            g = []
            for w in range(WK):
                k0 = min(r1, r0 + w * per_w)
                g.append((k0, min(r1, k0 + per_w)))
            groups.append(g)
        return Order.chain(groups, f"skinny S{S} wk{WK}")

    @staticmethod
    def s1(KT: int, L: int) -> "Order":
        """`_slab_rot_s1`'s: chains of L tiles over [0, KT) (the last one shorter), added as one chain."""
        return Order.chain([[(k, min(KT, k + L)) for k in range(0, KT, L)]], f"s1 L{L}")

    @staticmethod
    def sliced(bounds, S: int, WK: int) -> "Order":
        """`exl3_slab_gemm._slab_rot_sliced`'s: bounds (k0, k1) in tiles, S WK of them (a tensor, a flat list or
        pairs), WK a rank, the ranks added in order."""
        v = bounds.reshape(-1).tolist() if torch.is_tensor(bounds) else [int(t) for ab in bounds for t in
                                                                           (ab if isinstance(ab, (tuple, list)) else (ab,))]
        assert len(v) == 2 * S * WK, (len(v), S, WK)
        b = [(int(v[2 * i]), int(v[2 * i + 1])) for i in range(S * WK)]
        return Order.chain([b[r * WK:(r + 1) * WK] for r in range(S)], f"sliced S{S} wk{WK}")

    @staticmethod
    def library(M: int, K: int, N: int) -> "Order":
        return Order((("library", int(M), int(K), int(N)),), f"library M{M}")

    @property
    def ours(self) -> bool:
        return not (self.key and self.key[0] and self.key[0][0] == "library")

    def s1_launch(self) -> tuple | None:
        """(KT, L, Q) of the `_slab_rot_s1` launch that sums in this order (one chain of L-tile slices from 0, the
        last shorter; Q tiles a step, dividing every slice), or None where that kernel cannot."""
        if not self.ours or len(self.key) != 1:
            return None
        sl = self.key[0]
        L, KT = sl[0][1] - sl[0][0], sl[-1][1]
        if sl[0][0] != 0 or not all(a == i * L and b == min(KT, a + L) for i, (a, b) in enumerate(sl)):
            return None
        last = sl[-1][1] - sl[-1][0]
        return KT, L, next(q for q in (4, 2, 1) if L % q == 0 and last % q == 0)

    def __str__(self) -> str:
        return self.form or repr(self.key)


def prompt_order(w) -> Order:
    """The prompt order of `w`: the skinny GEMM's at S_PROMPT, WK_PROMPT (whatever `pick` binds for its decode)."""
    return Order.skinny(w.K // 16, S_PROMPT, WK_PROMPT)


# ------------------------------------------------------------------------------------------ s1, copied
RSQRT128 = tl.constexpr(0.08838834764831845) if HAVE_TRITON else 0.08838834764831845

if HAVE_TRITON:
    # tools/exl3_slab_gemm.py's `_wht_stage`, `_wht128_rows` and `_slab_rot_s1`, statement for statement

    @triton.jit
    def _wht_stage(x, R: tl.constexpr, H: tl.constexpr):
        """One butterfly stage over [R * 2 * H] values viewed as (R, 2, H): low + high, low - high."""
        x = tl.reshape(x, (R, 2, H))
        x = tl.permute(x, (0, 2, 1))
        a, b = tl.split(x)
        x = tl.join(a + b, a - b)
        return tl.permute(x, (0, 2, 1))

    @triton.jit
    def _wht128_rows(x, BM: tl.constexpr, BN: tl.constexpr):
        """The unnormalised natural-order Walsh-Hadamard over each 128-column block of x [BM, BN]:
        stage s pairs column i with i ^ 2^s, low + high and low - high, bit 0 first (`wht128`)."""
        R: tl.constexpr = BM * (BN // 128)
        x = tl.reshape(x, (R * 64, 2))
        a, b = tl.split(x)
        x = tl.join(a + b, a - b)
        x = _wht_stage(x, R * 32, 2)
        x = _wht_stage(x, R * 16, 4)
        x = _wht_stage(x, R * 8, 8)
        x = _wht_stage(x, R * 4, 16)
        x = _wht_stage(x, R * 2, 32)
        x = _wht_stage(x, R, 64)
        return tl.reshape(x, (BM, BN))

    @triton.jit
    def _slab_rot_s1(A, B, SV, C, M, N, KT, L, s_am, s_bk, s_cm,
                     BM: tl.constexpr, BN: tl.constexpr, Q: tl.constexpr, GROUP: tl.constexpr,
                     OUT_F32: tl.constexpr):
        """One pass over K, the chain of a slice of L K tiles (the last one shorter) added to the running sum at
        its end and begun again from zero; then the output rotation and svh."""
        pid = tl.program_id(0)
        npm = tl.cdiv(M, BM)
        npn = tl.cdiv(N, BN)
        gid = pid // (GROUP * npn)
        fm = gid * GROUP
        gs = min(npm - fm, GROUP)
        pm = fm + (pid % gs)
        pn = (pid % (GROUP * npn)) // gs
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, 16 * Q)
        mm = rm[:, None] < M
        a = A + rm[:, None] * s_am + rk[None, :]
        b = B + rk[:, None] * s_bk + rn[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        part = tl.zeros((BM, BN), dtype=tl.float32)
        for kt in range(0, KT, Q):
            x = tl.load(a, mask=mm, other=0.0)
            y = tl.load(b)
            acc = tl.dot(x, y, acc)
            a += 16 * Q
            b += 16 * Q * s_bk
            if ((kt + Q) % L == 0) | (kt + Q == KT):
                part += acc
                acc = tl.zeros((BM, BN), dtype=tl.float32)
        v = _wht128_rows(part, BM, BN)
        sv = tl.load(SV + rn).to(tl.float32)
        o = v * RSQRT128 * sv[None, :]
        if OUT_F32:
            tl.store(C + rm[:, None] * s_cm + rn[None, :], o, mask=mm)
        else:
            tl.store(C + rm[:, None] * s_cm + rn[None, :], o.to(tl.bfloat16), mask=mm)


def _kernel():
    """The copied `_slab_rot_s1`, looked up at the call (the CPU test fakes it)."""
    return _slab_rot_s1


def tile(K: int) -> tuple:
    """(BM, BN, warps, stages): `exl3_slab_gemm.sliced_tile`'s (the tiling never moves a bit: each element's chain
    is its own); 128 x 128 on 8 warps, a fourth stage for a long K."""
    return (128, 128, 8, 4 if K > 8192 else 3)


def product(xh: torch.Tensor, wi: torch.Tensor, svh: torch.Tensor, out: torch.Tensor, KT: int, L: int, Q: int,
            tiling: tuple | None = None) -> torch.Tensor:
    """out[M, n] = rot_out(xh[M, K] @ wi[K, n]) times svh[n], K summed in chains of L tiles from 0 (Q a step);
    xh, wi fp16 with unit column stride, out bf16 or fp32 (any row stride, unit column stride), n % 128 == 0."""
    M, K = xh.shape
    N = wi.shape[1]
    assert KT * 16 == K and L % Q == 0 and (KT % L) % Q == 0, (KT, K, L, Q)
    assert N % 128 == 0 and xh.stride(1) == 1 and wi.stride(1) == 1 and out.stride(1) == 1
    bm, bn, nw, st = tiling or tile(K)
    grid = (((M + bm - 1) // bm) * (N // bn),)
    _kernel()[grid](xh, wi, svh, out, M, N, KT, L, xh.stride(0), wi.stride(0), out.stride(0),
                    BM=bm, BN=bn, Q=Q, GROUP=8, OUT_F32=out.dtype == torch.float32, num_warps=nw, num_stages=st)
    return out


# ------------------------------------------------------------------------------------------ the two kernels
def _out(M: int, w, dev, out=None, dtype=None) -> torch.Tensor:
    if out is not None:
        return out
    return torch.empty(M, w.N, dtype=dtype or (torch.float32 if w.out_fp32 else torch.bfloat16), device=dev)


def skinny(x: torch.Tensor, w, out: torch.Tensor | None = None) -> torch.Tensor:
    """The prompt order on the skinny GEMM, at any row count (in calls of at most EXL3_CHUNK_UNTIL rows)."""
    M = x.shape[0]
    step = X.CHUNK_UNTIL
    if M <= step:
        return X.exl3_matmul(x, w, S=S_PROMPT, wk=WK_PROMPT, out=out)
    out = _out(M, w, x.device, out)
    for r0 in range(0, M, step):
        X.exl3_matmul(x[r0:r0 + step], w, S=S_PROMPT, wk=WK_PROMPT, out=out[r0:r0 + step])
    return out


def slab(x: torch.Tensor | None, w, *, xh: torch.Tensor | None = None, out: torch.Tensor | None = None,
         dtype=None, tiling: tuple | None = None) -> torch.Tensor:
    """The prompt order on the slab, at any row count: the input rotated as `exl3_linear` rotates it (unless `xh`),
    each slab decoded as it decodes it (its slab cache when on), the copied `_slab_rot_s1` at L = KT / WK_PROMPT."""
    if xh is None:
        if x.stride(1) != 1 or x.stride(0) % 4 or x.data_ptr() % 8:
            x = x.contiguous()
        xh = X.rotate_in(x, w)
    M = xh.shape[1]
    out = _out(M, w, xh.device, out, dtype)
    KT, L, Q = prompt_order(w).s1_launch()
    segs = X._slabs(w) or [(g, n0, n1, None) for g, n0, n1 in X.segments(w, X.SLAB_N)]
    for g, n0, n1, wi in segs:
        product(xh[g], wi if wi is not None else w.inner(n0, n1), w.svh[n0:n1], out[:, n0:n1], KT, L, Q, tiling)
    return out


# ------------------------------------------------------------------------------------------ bind and dispatch
#: (tile_n, K, kb, out_fp32, device) -> the check's record, for every bound shape
_BOUND: dict = {}
#: the shapes whose check failed (the health report reads it)
FAILED: dict = {}


def _on_kernels(x: torch.Tensor) -> bool:
    return x.is_cuda and _gpu.profile(x.device).inline("exl3_linear")


def _bkey(w, dev) -> tuple:
    return (w.tile_n, w.K, w.kb, bool(w.out_fp32), str(dev))


def half(w) -> bool:
    """A half-integer rate (1.5, 2.5, 3.5 bits): the skinny GEMM cannot decode it, so its prompt order runs on the
    slab at every row count (the same order: one chain of eight slices of KT / 8 tiles)."""
    return w.kb >= 10


def _known(w) -> None:
    """A projection the prompt order is defined for: an EXL3 rate the kernels decode, K in eight whole slices.
    Anything else is not one order's bits here: `RowOrderError`, never a quiet path."""
    if w.kb not in X.KS + X.HALF_KS:
        raise RowOrderError(f"{w.N}x{w.K}: {w.kb} bits is no EXL3 rate the kernels decode")
    if w.K % (16 * WK_PROMPT):
        raise RowOrderError(f"{w.N}x{w.K}: K is not {WK_PROMPT} whole slices of 16-row tiles")


def _ref(x: torch.Tensor, w) -> torch.Tensor:
    """Each row its one-row call's bits: the skinny GEMM at S1 w8 in 32-row calls (its rows are their one-row calls'
    up to 32 rows), or for a half-integer rate the slab's s1 in 32-row calls (each element its own chain)."""
    if half(w):
        return torch.cat([slab(x[i:i + ROWS], w) for i in range(0, x.shape[0], ROWS)])
    return torch.cat([X.exl3_matmul(x[i:i + ROWS], w, S=S_PROMPT, wk=WK_PROMPT) for i in range(0, x.shape[0], ROWS)])


def check(w, seed: int = 0) -> dict:
    """The bind-time check of one shape, against the same rows in one-row calls (`_ref`, and one row alone): 33 rows
    and CROSS rows on the skinny GEMM (above EXL3_CHUNK_UNTIL in calls of that many), and CROSS + 1 rows on the slab;
    a half-integer rate the slab at 33 and CROSS + 1 rows."""
    dev = w.t.device
    n = max(ROWS + 1, CROSS + 1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(n, w.K, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    ref = _ref(x, w)
    one = slab(x[n - 1:], w) if half(w) else X.exl3_matmul(x[n - 1:], w, S=S_PROMPT, wk=WK_PROMPT)
    short = slab if half(w) else skinny
    out = {"short_equal": bool(torch.equal(short(x[:ROWS + 1], w), ref[:ROWS + 1])),
           "slab_equal": bool(torch.equal(slab(x, w), ref)), "one_row_equal": bool(torch.equal(one, ref[-1:])),
           "rows": [ROWS + 1, n], "order": str(prompt_order(w)), "half_integer": half(w)}
    if not half(w):
        out["cross_equal"] = bool(torch.equal(skinny(x[:CROSS], w), ref[:CROSS]))
    return out


def bind(w) -> dict:
    """Check the shape once a device (and compile its slab product); `RowOrderError` when it is unequal."""
    key = _bkey(w, w.t.device)
    got = _BOUND.get(key)
    if got is None:
        _known(w)
        got = check(w)
        if not all(v for k, v in got.items() if k.endswith("_equal")):
            FAILED[key] = got
            raise RowOrderError(f"{w.tile_n}x{w.K} at {X.rate(w.kb):g} bits: the prompt order's kernels are not one "
                                f"order's bits ({got}); the prompt pass cannot run on it")
        _BOUND[key] = got
    return got


def linear(x: torch.Tensor, w, out: torch.Tensor | None = None) -> torch.Tensor:
    """x [M, K] bf16 @ EXL3Block^T -> [M, N] (bf16; fp32 for an `out_fp32` block) in the prompt order: every row
    the bits of its own one-row call, at every M. Up to CROSS rows the skinny GEMM, more the slab; a half-integer
    rate the slab at every row count."""
    if not _on_kernels(x):
        return X.exl3_matmul(x, w, out=out)
    _known(w)
    M = x.shape[0]
    if M > ROWS or half(w):
        bind(w)
    if M <= CROSS and not half(w):
        return skinny(x, w, out=out)
    return slab(x, w, out=out)
