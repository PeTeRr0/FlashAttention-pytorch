import torch
from importlib import import_module

_ext = None

def _load_ext():
    global _ext
    if _ext is not None:
        return _ext
    for name in ("flashattention_lab_cuda", "flashattention_lab._C"):
        try:
            _ext = import_module(name)
            return _ext
        except Exception:
            pass
    raise ImportError("CUDA extension module not found")

def _merge_bh(x):
    if x.dim() == 3:
        return x, None
    b, h, n, d = x.shape
    return x.reshape(b * h, n, d), (b, h)

def _split_bh(x, bh_shape):
    if bh_shape is None:
        return x
    b, h = bh_shape
    bh, n, d = x.shape
    return x.reshape(b, h, n, d)

def _split_bh_lse(lse, bh_shape):
    if bh_shape is None:
        return lse
    b, h = bh_shape
    bh, n = lse.shape
    return lse.reshape(b, h, n)

class _FA3CudaFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, softmax_scale, br, bc, stages, fp8):
        ext = _load_ext()

        if not q.is_cuda or not k.is_cuda or not v.is_cuda:
            raise RuntimeError("Inputs must be CUDA tensors")

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        o, lse = ext.fa3_forward(
            q, k, v,
            bool(causal), float(softmax_scale),
            int(br), int(bc), int(stages),
            bool(fp8)
        )

        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = bool(causal)
        ctx.softmax_scale = float(softmax_scale)
        ctx.br = int(br)
        ctx.bc = int(bc)
        ctx.stages = int(stages)
        ctx.fp8 = bool(fp8)

        return o, lse

    @staticmethod
    def backward(ctx, do, dlse):
        ext = _load_ext()

        q, k, v, o, lse = ctx.saved_tensors

        do = do.contiguous()

        dq, dk, dv = ext.fa3_backward(
            q, k, v, o, do, lse,
            bool(ctx.causal), float(ctx.softmax_scale),
            int(ctx.br), int(ctx.bc), int(ctx.stages),
            bool(ctx.fp8)
        )

        return dq, dk, dv, None, None, None, None, None, None

def fa3_cuda(q, k, v, causal, softmax_scale, spec, fp8):
    qb, bh_shape = _merge_bh(q)
    kb, _ = _merge_bh(k)
    vb, _ = _merge_bh(v)

    o, lse = _FA3CudaFn.apply(
        qb, kb, vb,
        causal, softmax_scale,
        spec.br, spec.bc, spec.stages,
        fp8
    )

    o = _split_bh(o, bh_shape)
    lse = _split_bh_lse(lse, bh_shape)

    return o, lse
