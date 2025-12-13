import pytest

torch = pytest.importorskip("torch")

from common.correctness import assert_allclose, reference_attention, reference_backward
from fa1.cuda.impl import fa1_cuda
from fa1.spec import pick_fa1_spec
from fa1.torch.impl import fa1_backward_torch, fa1_forward_torch, fa1_torch
from tests.utils import dtype_tolerances, flatten_lse, flatten_output, make_qkv


@pytest.mark.parametrize("shape", [(1, 2, 16, 32), (2, 1, 33, 64)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("merge_heads", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fa1_torch_forward_matches_reference(shape, causal, merge_heads, dtype, device):
    torch.manual_seed(0)
    batch, heads, seqlen, head_dim = shape
    q, k, v = make_qkv(batch, heads, seqlen, head_dim, device=device, dtype=dtype, merge_heads=merge_heads)
    softmax_scale = head_dim ** -0.5

    spec = pick_fa1_spec(head_dim)
    o, lse = fa1_torch(q, k, v, causal, softmax_scale, spec)
    o_ref, lse_ref = reference_attention(q, k, v, causal=causal, softmax_scale=softmax_scale)

    o = flatten_output(o)
    lse = flatten_lse(lse)
    o_ref = flatten_output(o_ref)
    lse_ref = flatten_lse(lse_ref)

    tol = dtype_tolerances(dtype)
    assert_allclose(o, o_ref, **tol)
    assert_allclose(lse, lse_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("causal", [False, True])
def test_fa1_torch_backward_matches_reference(causal, device):
    torch.manual_seed(1)
    dtype = torch.float32
    batch, heads, seqlen, head_dim = 1, 2, 12, 32
    q, k, v = make_qkv(batch, heads, seqlen, head_dim, device=device, dtype=dtype, merge_heads=True)
    softmax_scale = head_dim ** -0.5
    spec = pick_fa1_spec(head_dim)

    o, lse = fa1_forward_torch(q, k, v, causal, softmax_scale, spec.br, spec.bc)
    do = torch.randn_like(o)

    dq, dk, dv = fa1_backward_torch(q, k, v, o, do, lse, causal, softmax_scale, spec.br, spec.bc)
    dq_ref, dk_ref, dv_ref, _, _ = reference_backward(q, k, v, do, causal, softmax_scale)

    assert_allclose(dq, dq_ref, rtol=1e-3, atol=1e-3)
    assert_allclose(dk, dk_ref, rtol=1e-3, atol=1e-3)
    assert_allclose(dv, dv_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("causal", [False, True])
def test_fa1_triton_forward_matches_reference(causal, triton_available):
    if not triton_available:
        pytest.skip("Triton backend is unavailable on this machine")

    torch.manual_seed(2)
    dtype = torch.float16
    device = "cuda"
    batch, heads, seqlen, head_dim = 1, 2, 64, 64
    q, k, v = make_qkv(batch, heads, seqlen, head_dim, device=device, dtype=dtype, merge_heads=True)
    softmax_scale = head_dim ** -0.5

    from fa1.triton.impl import fa1_triton

    spec = pick_fa1_spec(head_dim)
    o, lse = fa1_triton(q, k, v, causal, softmax_scale, spec)
    o_ref, lse_ref = reference_attention(q, k, v, causal=causal, softmax_scale=softmax_scale)

    o = flatten_output(o)
    lse = flatten_lse(lse)
    o_ref = flatten_output(o_ref)
    lse_ref = flatten_lse(lse_ref)

    tol = dtype_tolerances(dtype)
    assert_allclose(o, o_ref, **tol)
    assert_allclose(lse, lse_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("causal", [False, True])
def test_fa1_cuda_backward_matches_reference(causal, cuda_extension_available):
    if not cuda_extension_available:
        pytest.skip("CUDA extension for FlashAttention-1 not built or no GPU available")

    torch.manual_seed(3)
    dtype = torch.float16
    device = "cuda"
    batch, heads, seqlen, head_dim = 1, 2, 24, 64
    q, k, v = make_qkv(batch, heads, seqlen, head_dim, device=device, dtype=dtype, merge_heads=True)
    q = q.requires_grad_(True)
    k = k.requires_grad_(True)
    v = v.requires_grad_(True)

    softmax_scale = head_dim ** -0.5
    spec = pick_fa1_spec(head_dim)
    o, _ = fa1_cuda(q, k, v, causal, softmax_scale, spec)
    do = torch.randn_like(o)

    loss = (o * do).sum()
    loss.backward()

    dq_ref, dk_ref, dv_ref, _, _ = reference_backward(q.detach(), k.detach(), v.detach(), do.detach(), causal, softmax_scale)
    tol = dtype_tolerances(dtype)
    assert_allclose(q.grad, dq_ref, **tol)
    assert_allclose(k.grad, dk_ref, **tol)
    assert_allclose(v.grad, dv_ref, **tol)


def test_fa1_backend_consistency(triton_available, cuda_extension_available):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    backends = ["torch"]
    if triton_available:
        backends.append("triton")
    if cuda_extension_available:
        backends.append("cuda")

    if len(backends) < 2:
        pytest.skip("Only one FlashAttention-1 backend available on this machine")

    torch.manual_seed(4)
    batch, heads, seqlen, head_dim = 1, 2, 24, 32
    q, k, v = make_qkv(batch, heads, seqlen, head_dim, device=device, dtype=dtype, merge_heads=False)
    softmax_scale = head_dim ** -0.5

    outputs = {}
    for backend in backends:
        spec = pick_fa1_spec(head_dim)
        if backend == "torch":
            o, lse = fa1_torch(q, k, v, True, softmax_scale, spec)
        elif backend == "triton":
            from fa1.triton.impl import fa1_triton

            o, lse = fa1_triton(q, k, v, True, softmax_scale, spec)
        else:
            o, lse = fa1_cuda(q, k, v, True, softmax_scale, spec)
        outputs[backend] = (flatten_output(o), flatten_lse(lse))

    ref_backend = backends[0]
    ref_o, ref_lse = outputs[ref_backend]
    tol = dtype_tolerances(dtype)
    for backend, (o, lse) in outputs.items():
        if backend == ref_backend:
            continue
        assert_allclose(o, ref_o, **tol)
        assert_allclose(lse, ref_lse, rtol=1e-3, atol=1e-3)
