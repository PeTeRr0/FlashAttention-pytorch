from typing import Optional

import torch
import torch.nn.functional as F

from common.correctness import validate_inputs
from common.mask import apply_masks
from common.utils import resolve_softmax_scale


def flashattention3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
    softmax_scale: Optional[float] = None,
    softcap: Optional[float] = None,
    block_size: int = 128,
) -> torch.Tensor:
    """FlashAttention-3 forward pass with the double-buffered streaming softmax."""
    validate_inputs(q, k, v)
    b, h, q_len, d_k = q.shape
    _, _, k_len, _ = k.shape

    softmax_scale = resolve_softmax_scale(d_k, softmax_scale)

    stats_dtype = torch.float32
    out = torch.zeros((b, h, q_len, v.size(-1)), device=q.device, dtype=stats_dtype)
    m_i = torch.full((b, h, q_len, 1), float("-inf"), device=q.device, dtype=stats_dtype)
    l_i = torch.zeros((b, h, q_len, 1), device=q.device, dtype=stats_dtype)

    next_k = k[:, :, 0 : min(block_size, k_len), :]
    next_v = v[:, :, 0 : min(block_size, k_len), :]

    for start in range(0, k_len, block_size):
        end = min(start + block_size, k_len)
        k_block = next_k
        v_block = next_v

        if end < k_len:
            next_end = min(end + block_size, k_len)
            next_k = k[:, :, end:next_end, :]
            next_v = v[:, :, end:next_end, :]

        scores = torch.matmul(q, k_block.transpose(-2, -1)) * softmax_scale
        scores = scores.to(stats_dtype)

        if softcap is not None:
            scores = torch.clamp(scores, min=-softcap, max=softcap)

        scores = apply_masks(
            scores,
            causal=causal,
            attn_mask=attn_mask,
            q_start=0,
            k_start=start,
        )

        block_max = scores.amax(dim=-1, keepdim=True)
        m_new = torch.maximum(m_i, block_max)

        exp_m = torch.exp(m_i - m_new)
        exp_scores = torch.exp(scores - m_new)
        l_new = exp_m * l_i + exp_scores.sum(dim=-1, keepdim=True)

        p = exp_scores / l_new
        p = F.dropout(p, p=dropout_p, training=training)

        carried = (exp_m * l_i / l_new) * out
        out = carried + torch.matmul(p, v_block.to(stats_dtype))

        m_i = m_new
        l_i = l_new

    return out.to(v.dtype)
