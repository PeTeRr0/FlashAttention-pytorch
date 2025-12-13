import torch
import triton
from .kernels_fwd import fa1_fwd_kernel
from .kernels_bwd import fa1_bwd_d_kernel, fa1_bwd_dk_dv_kernel

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

class _FA1TritonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, softmax_scale, br, bc, num_warps):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        bh, n, d = q.shape
        o = torch.empty_like(q)
        lse = torch.empty((bh, n), device=q.device, dtype=torch.float32)

        grid = (bh, triton.cdiv(n, br))
        fa1_fwd_kernel[grid](
            q, k, v, o, lse,
            q.stride(1), q.stride(2),
            k.stride(1), k.stride(2),
            v.stride(1), v.stride(2),
            o.stride(1), o.stride(2),
            lse.stride(1),
            n, softmax_scale,
            d_head=d, BR=br, BC=bc, CAUSAL=causal,
            num_warps=num_warps
        )

        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = bool(causal)
        ctx.softmax_scale = float(softmax_scale)
        ctx.br = int(br)
        ctx.bc = int(bc)
        ctx.num_warps = int(num_warps)

        return o, lse

    @staticmethod
    def backward(ctx, do, dlse):
        q, k, v, o, lse = ctx.saved_tensors
        do = do.contiguous()

        bh, n, d = q.shape

        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k, dtype=torch.float32)
        dv = torch.empty_like(v, dtype=torch.float32)
        dvec = torch.empty((bh, n), device=q.device, dtype=torch.float32)

        grid_d = (bh, triton.cdiv(n, ctx.br))
        fa1_bwd_d_kernel[grid_d](
            o, do, dvec,
            o.stride(1), o.stride(2),
            do.stride(1), do.stride(2),
            dvec.stride(1),
            n,
            d_head=d, BR=ctx.br,
            num_warps=ctx.num_warps
        )

        grid_bwd = (bh, triton.cdiv(n, ctx.bc))
        fa1_bwd_dk_dv_kernel[grid_bwd](
            q, k, v, o,
            do, lse, dvec, dq, dk, dv,
            q.stride(1), q.stride(2),
            k.stride(1), k.stride(2),
            v.stride(1), v.stride(2),
            o.stride(1), o.stride(2),
            do.stride(1), do.stride(2),
            lse.stride(1), dvec.stride(1),
            dq.stride(1), dq.stride(2),
            dk.stride(1), dk.stride(2),
            dv.stride(1), dv.stride(2),
            n, ctx.softmax_scale,
            d_head=d, BR=ctx.br, BC=ctx.bc, CAUSAL=ctx.causal,
            num_warps=ctx.num_warps
        )

        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None, None

def fa1_triton(q, k, v, causal, softmax_scale, spec):
    qb, bh_shape = _merge_bh(q)
    kb, _ = _merge_bh(k)
    vb, _ = _merge_bh(v)

    o, lse = _FA1TritonFn.apply(qb, kb, vb, causal, softmax_scale, spec.br, spec.bc, spec.num_warps)

    o = _split_bh(o, bh_shape)
    lse = _split_bh_lse(lse, bh_shape)

    return o, lse
