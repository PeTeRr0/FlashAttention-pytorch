from __future__ import annotations

import torch
from common.utils import merge_bh


def make_qkv(batch, heads, seqlen, head_dim, device, dtype, merge_heads=False):
    shape = (batch, heads, seqlen, head_dim)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)
    if merge_heads:
        q, _ = merge_bh(q)
        k, _ = merge_bh(k)
        v, _ = merge_bh(v)
    return q, k, v


def flatten_output(x):
    if x.dim() == 4:
        return x.reshape(-1, x.shape[-2], x.shape[-1])
    return x


def flatten_lse(x):
    if x.dim() == 3:
        return x.reshape(-1, x.shape[-1])
    return x


def dtype_tolerances(dtype):
    if dtype == torch.float16:
        return {"rtol": 5e-2, "atol": 5e-2}
    if dtype == torch.bfloat16:
        return {"rtol": 5e-2, "atol": 5e-2}
    return {"rtol": 1e-4, "atol": 1e-4}
