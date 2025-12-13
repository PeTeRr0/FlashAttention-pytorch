import torch
from .spec import pick_fa2_spec
from .torch.impl import fa2_torch
from .triton.impl import fa2_triton
from .cuda.impl import fa2_cuda

def fa2_attention(q, k, v, causal=False, softmax_scale=None, backend="auto"):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    
    head_dim = q.shape[-1]
    spec = pick_fa2_spec(head_dim)

    if backend == "auto":
        if q.is_cuda:
            try:
                return fa2_cuda(q, k, v, causal, softmax_scale, spec)
            except Exception:
                return fa2_triton(q, k, v, causal, softmax_scale, spec)
        return fa2_torch(q, k, v, causal, softmax_scale, spec)

    if backend == "cuda":
        return fa2_cuda(q, k, v, causal, softmax_scale, spec)
    if backend == "triton":
        return fa2_triton(q, k, v, causal, softmax_scale, spec)
    if backend == "torch":
        return fa2_torch(q, k, v, causal, softmax_scale, spec)

    raise ValueError(backend)