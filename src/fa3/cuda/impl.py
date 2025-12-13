import torch
from flashattention_lab import _cuda_ext


class _FlashAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, dropout_p, softmax_scale, softcap):
        output, aux = _cuda_ext.fa3_fwd(q, k, v, causal, dropout_p, softmax_scale, softcap)
        ctx.save_for_backward(q, k, v, aux)
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.softcap = softcap
        return output

    @staticmethod
    def backward(ctx, doutput):
        q, k, v, aux = ctx.saved_tensors
        dq, dk, dv = _cuda_ext.fa3_bwd(q, k, v, doutput, aux, ctx.causal, ctx.dropout_p, ctx.softmax_scale, ctx.softcap)
        return dq, dk, dv, None, None, None, None


def flashattention3(q, k, v, causal=False, attn_mask=None, dropout_p=0.0, training=False, softmax_scale=None, softcap=0.0):
    if attn_mask is not None:
        raise NotImplementedError("attn_mask is not supported in FlashAttention-3 CUDA path.")
    if not training and dropout_p > 0.0:
        dropout_p = 0.0
    output = _FlashAttnFn.apply(q, k, v, causal, dropout_p, softmax_scale, softcap)
    return output