"""The lfm2_moe family's server builder: the MLX serving where the device's adapter says the models run on
MLX (tools/gpu `backend()`: Apple silicon, mlx_serving.py), the CUDA one elsewhere (serving.py). The CUDA
serving is not imported for a Mac's build beyond its module (mlx_serving.py reads the family's conductor fit
from it, which loads engine.py and kernels.py; they import without Triton and build nothing until used), and
MLX is never imported off Apple silicon."""

from __future__ import annotations


def serve(a, ctx) -> dict:
    from tools import gpu as _gpu
    if _gpu.profile().backend() == "mlx":
        from engine.models.lfm2_moe import mlx_serving
        return mlx_serving.serve(a, ctx)
    from engine.models.lfm2_moe import serving
    return serving.serve(a, ctx)
