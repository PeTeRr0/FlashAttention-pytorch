import torch
from .spec import pick_fa3_spec
from .torch.impl import fa3_torch
from .triton.impl import fa3_triton
from .cuda.impl import fa3_cuda

def fa3_attention(q, k, v, causal=False, softmax_scale=None, backend="auto", fp8=False):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    head_dim = q.shape[-1]
    spec = pick_fa3_spec(head_dim)

    if backend == "auto":
        if q.is_cuda:
            try:
                return fa3_cuda(q, k, v, causal, softmax_scale, spec, fp8)
            except Exception:
                return fa3_triton(q, k, v, causal, softmax_scale, spec)
        return fa3_torch(q, k, v, causal, softmax_scale, spec, fp8)

    if backend == "cuda":
        return fa3_cuda(q, k, v, causal, softmax_scale, spec, fp8)
    if backend == "triton":
        return fa3_triton(q, k, v, causal, softmax_scale, spec)
    if backend == "torch":
        return fa3_torch(q, k, v, causal, softmax_scale, spec, fp8)

    raise ValueError(backend)
