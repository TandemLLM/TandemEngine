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
