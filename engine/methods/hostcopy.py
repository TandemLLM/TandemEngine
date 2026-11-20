"""`h2d` for the server's generation loop on a machine without Triton (Apple silicon,
engine/models/lfm2_moe/mlx_serving.py): engine/model.py owns `h2d` and `HOST_ASYNC` and cannot be imported
there. Off CUDA the copy is the plain one engine/model.py's `h2d` makes; on CUDA it is engine/model.py's."""

from __future__ import annotations

import torch


def h2d(values, dtype: torch.dtype, device) -> torch.Tensor:
    """A Python list as a device tensor (engine/model.py `h2d`; its pinned path is CUDA's alone)."""
    if torch.device(device).type != "cuda":
        return torch.tensor(values, dtype=dtype).to(device)
    from engine.model import h2d as on_cuda
    return on_cuda(values, dtype, device)
