import torch
import triton
from .kernels_fwd import fa3_fwd_kernel

def _merge_bh(x):
    if x.dim() == 3:
        return x
    b, h, n, d = x.shape
    return x.reshape(b * h, n, d)

def fa3_triton(q, k, v, causal, softmax_scale, spec):
    qb = _merge_bh(q).contiguous()
    kb = _merge_bh(k).contiguous()
    vb = _merge_bh(v).contiguous()

    bh, n, d = qb.shape
    o = torch.empty_like(qb)
    lse = torch.empty((bh, n), device=q.device, dtype=torch.float32)

    grid = (bh, triton.cdiv(n, spec.br))
    fa3_fwd_kernel[grid](
        qb, kb, vb, o, lse,
        qb.stride(1), qb.stride(2),
        kb.stride(1), kb.stride(2),
        vb.stride(1), vb.stride(2),
        o.stride(1), o.stride(2),
        lse.stride(1),
        n_ctx=n, softmax_scale=softmax_scale,
        d_head=d, BR=spec.br, BC=spec.bc,
        CAUSAL=causal, num_warps=spec.num_warps
    )
    return o, lse
