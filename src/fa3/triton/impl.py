import torch

from .kernels_bwd import flash_bwd
from .kernels_fwd import flash_fwd


class FlashAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, dropout_p, softmax_scale, softcap):
        output, aux = flash_fwd(
            q, k, v,
            causal=causal,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            softcap=softcap,
        )
        ctx.save_for_backward(q, k, v)
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.softcap = softcap
        ctx.aux = aux
        return output

    @staticmethod
    def backward(ctx, doutput):
        q, k, v = ctx.saved_tensors
        dq, dk, dv = flash_bwd(
            q, k, v, doutput,
            causal=ctx.causal,
            dropout_p=ctx.dropout_p,
            softmax_scale=ctx.softmax_scale,
            softcap=ctx.softcap,
        )
        return dq, dk, dv, None, None, None, None


def flashattention3(q, k, v, causal=False, attn_mask=None, dropout_p=0.0, training=False, softmax_scale=None, softcap=0.0):
    if attn_mask is not None:
        raise NotImplementedError("attn_mask is not supported in FlashAttention-3 Triton kernel.")
    if not training and dropout_p > 0.0:
        dropout_p = 0.0
    output = FlashAttentionFn.apply(q, k, v, causal, dropout_p, softmax_scale, softcap)
    return output