import torch
from .mask import apply_causal_mask
from .utils import merge_bh, split_bh, split_bh_lse

def reference_attention(q, k, v, causal=False, softmax_scale=None):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    qb, bh_shape = merge_bh(q)
    kb, _ = merge_bh(k)
    vb, _ = merge_bh(v)

    scores = torch.matmul(qb.to(torch.float32), kb.transpose(-2, -1).to(torch.float32)) * softmax_scale

    if causal:
        scores = apply_causal_mask(scores, 0, 0)

    p = torch.softmax(scores, dim=-1)
    o = torch.matmul(p, vb.to(torch.float32)).to(q.dtype)
    lse = torch.logsumexp(scores, dim=-1)

    o = split_bh(o, bh_shape)
    lse = split_bh_lse(lse, bh_shape)
    return o, lse

def reference_backward(q, k, v, do, causal=False, softmax_scale=None):
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    o_ref, lse_ref = reference_attention(q_ref, k_ref, v_ref, causal, softmax_scale)
    o_ref.backward(do)

    return q_ref.grad, k_ref.grad, v_ref.grad, o_ref.detach(), lse_ref.detach()

def assert_allclose(actual, expected, rtol=1e-3, atol=1e-3, msg=None):
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=msg)
