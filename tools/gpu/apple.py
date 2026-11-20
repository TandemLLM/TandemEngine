"""Apple silicon (M-series, Metal through MLX): unified memory, no CUDA, no Triton.

The Mac runs the models that have an MLX board (engine/models/lfm2_moe/mlx_board.py) and nothing that
needs an inline CUDA extension or a Triton kernel: every `INLINE` kind raises `NoInlineBuild`, as on the
base. Claimed by name ("Apple M5 Max", from `chip_name`), never by a CUDA capability, so it is picked only
where torch has no CUDA device. Per-device files are keyed by the chip's name ("apple-m5-max"): a price
table measured on one chip is not another's.

Measured on an M5 Max (MLX 0.32.4): an fp32 matmul of 2 rows or more runs at about TF32 precision on the
GPU (one row is exact), so MLX's own matmul rounds a verify row and a decode row of the same token
differently. The board therefore runs its own Metal kernels, whose rows do not depend on the row count
(engine/models/lfm2_moe/mlx_kernels.py); the tests hold a block's rows to its rows alone, bit for bit
(tests/test_lfm2_moe_mlx.py).
"""

from __future__ import annotations

import functools
import platform
import subprocess

from tools.gpu.base import GpuProfile


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


@functools.lru_cache(maxsize=None)
def chip_name() -> str | None:
    """"Apple M5 Max" on Apple silicon, None elsewhere (sysctl, no torch or MLX import)."""
    if not is_apple_silicon():
        return None
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                              timeout=5).stdout.strip() or "Apple silicon"
    except (OSError, subprocess.SubprocessError):
        return "Apple silicon"


class AppleSilicon(GpuProfile):
    id = "apple"
    about = "Apple silicon (Metal via MLX): unified memory, MLX boards only, no inline CUDA or Triton"
    names = ("APPLE",)
    unified_memory = True

    def label(self) -> str:
        return self.name or "Apple silicon"

    def backend(self) -> str:
        return "mlx"

    def bf16(self) -> bool:
        return True
