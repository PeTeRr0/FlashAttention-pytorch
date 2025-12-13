from __future__ import annotations

import torch
from torch import Tensor


def resolve_softmax_scale(d_k: int, softmax_scale: float | None) -> float:
    """Return provided scale or the default 1/sqrt(d_k)."""
    return softmax_scale if softmax_scale is not None else d_k ** -0.5


def maybe_float(x: Tensor) -> Tensor:
    """Promote to float32 for numerically stable accumulations."""
    if not x.is_floating_point():
        raise TypeError("Attention inputs must be floating point tensors.")
    return x.float()


def apply_dropout(probs: Tensor, p: float, training: bool, generator=None) -> Tensor:
    if p == 0.0 or not training:
        return probs
    keep = torch.rand_like(probs, dtype=torch.float, generator=generator) >= p
    return probs * keep / (1.0 - p)
