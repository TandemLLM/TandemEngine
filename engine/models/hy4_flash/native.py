"""Hy4-flash's own CUDA kernel: the attention's value half read packed.

`kv_b_proj` is one EXL3 projection [heads (nope + v), R]; the row path needs, for every head, the head's
weighted latent sum [R] through the head's v value rows. `vdense` reads those rows packed (K bits a weight
instead of a BF16 copy), each head with its own input. The kernel is built on the shared EXL3 row product of
tools/fn_kernels (its source is compiled in as a library; nothing of it is changed).
"""

from __future__ import annotations

import torch

from engine import native_build as nb
from engine.settings import SETTINGS as _S

_CUDA = nb.source(__file__, "vdense.cu")
_CPP = nb.source(__file__, "vdense.cpp")
_MOD = None


def available(device) -> bool:
    """Whether this device builds the inline kernels (the row path keeps the BF16 value half where not)."""
    from tools import gpu as _gpu
    try:
        return torch.device(device).type == "cuda" and _gpu.profile(device).inline("fn_kernels")
    except Exception:  # noqa: BLE001 - no profile for this device: the plain path
        return False


def _module():
    global _MOD
    if _MOD is None:
        from tools import fn_kernels as FK
        from tools import gpu as _gpu
        src = "#define EXL3_ST 3\n" + FK._HELPERS + FK._CUDA + _CUDA
        cpp = FK._CPP + _CPP
        _MOD = nb.load(f"hy4_native_{nb.tag(cpp, src)}", cpp_sources=[cpp], cuda_sources=[src],
                       functions=["hy4_vdense", "fn_set_pdl"],
                       extra_cuda_cflags=["-O3", "-lineinfo", *_gpu.arch_flags("fn_kernels")], verbose=False)
        _MOD.fn_set_pdl(int(_S.get("FN_PDL") == "1" and _gpu.profile().pdl()))
    return _MOD


def fits(w, heads: int, per: int, off: int, V: int) -> bool:
    """Whether the packed projection `w` can serve heads of `per` columns with V value columns from `off`."""
    return (hasattr(w, "t") and w.N == heads * per and w.K % 128 == 0 and w.N % 128 == 0 and per % 4 == 0
            and off % 4 == 0 and V % 4 == 0 and w.t.shape[2] % 16 == 0 and w.suh.numel() == w.K)


def vdense(x: torch.Tensor, w, heads: int, per: int, off: int, V: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """x [M heads, R] bf16 (row m's head h at m heads + h) -> [M heads, V] bf16: each head's input through the
    head's V columns of the packed projection `w` (an EXL3 block). One program a (row, head, 128-column block)."""
    if out is None:
        out = torch.empty(x.shape[0], V, dtype=torch.bfloat16, device=x.device)
    _module().hy4_vdense(x, w.t, w.suh.reshape(-1), w.svh, out, heads, per, off)
    return out
