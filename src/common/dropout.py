from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def _validate_p(p: float) -> None:
    """Validate dropout probability."""
    if p < 0 or p >= 1:
        raise ValueError(f"dropout probability has to be in [0, 1), got {p}")


def apply_dropout(
    x: Tensor,
    p: float,
    *,
    training: bool = True,
    inplace: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Apply inverted dropout and return both output and mask.

    Returning the mask is handy for reproducibility/debugging and mirrors what
    fused CUDA kernels typically expose.
    """
    _validate_p(p)
    if not training or p == 0:
        return x, None

    mask = torch.rand_like(x, dtype=torch.float, generator=generator) >= p
    scale = 1.0 / (1.0 - p)

    if inplace:
        x.mul_(mask)
        x.mul_(scale)
        return x, mask

    out = (x * mask) * scale
    return out, mask


def dropout_add(
    x: Tensor,
    residual: Tensor,
    p: float,
    *,
    training: bool = True,
    inplace: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Dropout followed by residual addition, optionally in-place on the input."""
    dropped, mask = apply_dropout(
        x,
        p,
        training=training,
        inplace=inplace,
        generator=generator,
    )

    if inplace:
        dropped.add_(residual)
        return dropped, mask

    return dropped + residual, mask
