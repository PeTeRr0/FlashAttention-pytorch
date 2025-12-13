import torch

def causal_block_skip(row_start, br, col_start):
    return col_start >= row_start + br

def apply_causal_mask(scores, row_start, col_start):
    br = scores.shape[0]
    bc = scores.shape[1]
    r = torch.arange(row_start, row_start + br, device=scores.device)[:, None]
    c = torch.arange(col_start, col_start + bc, device=scores.device)[None, :]
    mask = c > r
    return scores.masked_fill(mask, float("-inf"))

# aliases that mirror helpers used inside the faX torch paths
_causal_block_skip = causal_block_skip
_apply_causal_mask = apply_causal_mask
