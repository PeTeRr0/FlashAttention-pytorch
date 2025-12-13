import torch
from .spec import pick_fa1_spec
from .torch.impl import fa1_torch
from .triton.impl import fa1_triton
from .cuda.impl import fa1_cuda

def fa1_attention(q, k, v, causal=False, softmax_scale=None, backend="auto"):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    head_dim = q.shape[-1]
    spec = pick_fa1_spec(head_dim)

    if backend == "auto":
        if q.is_cuda:
            try:
                return fa1_cuda(q, k, v, causal, softmax_scale, spec)
            except Exception:
                return fa1_triton(q, k, v, causal, softmax_scale, spec)
        return fa1_torch(q, k, v, causal, softmax_scale, spec)

    if backend == "cuda":
        return fa1_cuda(q, k, v, causal, softmax_scale, spec)
    if backend == "triton":
        return fa1_triton(q, k, v, causal, softmax_scale, spec)
    if backend == "torch":
        return fa1_torch(q, k, v, causal, softmax_scale, spec)

    raise ValueError(backend)
