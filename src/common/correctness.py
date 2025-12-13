from __future__ import annotations

import torch
from torch import Tensor


def validate_inputs(q: Tensor, k: Tensor, v: Tensor) -> None:
    """Ensure tensors share the expected shape semantics."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must have shape (batch, heads, length, dim)")
    if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:
        raise ValueError("batch and head dimensions must match for q, k, v")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v sequence lengths must match")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, v must be on the same device")
