from __future__ import annotations

import torch
from torch import Tensor


def make_causal_mask(q_len: int, k_len: int, device: torch.device) -> Tensor:
    """Return a lower-triangular causal mask of shape (q_len, k_len)."""
    q_idx = torch.arange(q_len, device=device).unsqueeze(1)
    k_idx = torch.arange(k_len, device=device).unsqueeze(0)
    return k_idx <= q_idx


def apply_masks(
    scores: Tensor,
    *,
    causal: bool,
    attn_mask: Tensor | None,
    q_start: int,
    k_start: int,
) -> Tensor:
    """Apply causal and user masks to a score block."""
    q_len = scores.shape[-2]
    k_len = scores.shape[-1]

    if causal:
        q_idx = torch.arange(q_start, q_start + q_len, device=scores.device).view(q_len, 1)
        k_idx = torch.arange(k_start, k_start + k_len, device=scores.device).view(1, k_len)
        causal_block = k_idx <= q_idx
        scores = scores.masked_fill(~causal_block, float("-inf"))

    if attn_mask is not None:
        if attn_mask.dim() == 2:
            mask_block = attn_mask[q_start : q_start + q_len, k_start : k_start + k_len].unsqueeze(0).unsqueeze(0)
        elif attn_mask.dim() == 3:
            mask_block = attn_mask[:, q_start : q_start + q_len, k_start : k_start + k_len].unsqueeze(1)
        else:
            raise ValueError("attn_mask must have shape (q, k) or (b, q, k)")
        scores = scores.masked_fill(~mask_block, float("-inf"))

    return scores
