import pytest

torch = pytest.importorskip("torch")

from common.correctness import assert_allclose, reference_attention, reference_backward
from fa3.cuda.impl import fa3_cuda
from fa3.spec import pick_fa3_spec
from fa3.torch.impl import fa3_torch
from tests.utils import dtype_tolerances, flatten_lse, flatten_output, make_qkv


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fa3_torch_forward_matches_reference(causal, fp8, dtype, device):
    torch.manual_seed(20)
    batch, heads, seqlen, head_dim = 1, 2, 24, 32
    q, k, v = make_qkv(batch, heads, seqlen, head_dim, device=device, dtype=dtype, merge_heads=True)
    softmax_scale = head_dim ** -0.5

    spec = pick_fa3_spec(head_dim)
    o, lse = fa3_torch(q, k, v, causal, softmax_scale, spec, fp8)
    o_ref, lse_ref = reference_attention(q, k, v, causal=causal, softmax_scale=softmax_scale)

    o = flatten_output(o)
    lse = flatten_lse(lse)
    o_ref = flatten_output(o_ref)
    lse_ref = flatten_lse(lse_ref)

    tol = dtype_tolerances(dtype)
    if fp8:
        tol = {"rtol": 1e-1, "atol": 1e-1}
    assert_allclose(o, o_ref, **tol)
    assert_allclose(lse, lse_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("causal", [False, True])
def test_fa3_triton_forward_matches_reference(causal, triton_available):
    if not triton_available:
        pytest.skip("Triton backend is unavailable on this machine")

    torch.manual_seed(21)
    dtype = torch.float16
    device = "cuda"
    batch, heads, seqlen, head_dim = 1, 2, 64, 64
    q, k, v = make_qkv(batch, heads, seqlen, head_dim, device=device, dtype=dtype, merge_heads=True)
    softmax_scale = head_dim ** -0.5

    from fa3.triton.impl import fa3_triton

    spec = pick_fa3_spec(head_dim)
    o, lse = fa3_triton(q, k, v, causal, softmax_scale, spec)
    o_ref, lse_ref = reference_attention(q, k, v, causal=causal, softmax_scale=softmax_scale)

    o = flatten_output(o)
    lse = flatten_lse(lse)
    o_ref = flatten_output(o_ref)
    lse_ref = flatten_lse(lse_ref)

    tol = dtype_tolerances(dtype)
    assert_allclose(o, o_ref, **tol)
    assert_allclose(lse, lse_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("fp8", [False, True])
def test_fa3_cuda_backward_matches_reference(causal, fp8, cuda_extension_available):
    if not cuda_extension_available:
        pytest.skip("CUDA extension for FlashAttention-3 not built or no GPU available")

    torch.manual_seed(22)
    dtype = torch.float16
    device = "cuda"
    batch, heads, seqlen, head_dim = 1, 2, 32, 32
    q, k, v = make_qkv(batch, heads, seqlen, head_dim, device=device, dtype=dtype, merge_heads=True)
    q = q.requires_grad_(True)
    k = k.requires_grad_(True)
    v = v.requires_grad_(True)

    softmax_scale = head_dim ** -0.5
    spec = pick_fa3_spec(head_dim)
    o, _ = fa3_cuda(q, k, v, causal, softmax_scale, spec, fp8)
    do = torch.randn_like(o)

    loss = (o * do).sum()
    loss.backward()

    dq_ref, dk_ref, dv_ref, _, _ = reference_backward(q.detach(), k.detach(), v.detach(), do.detach(), causal, softmax_scale)
    tol = {"rtol": 1e-1, "atol": 1e-1} if fp8 else dtype_tolerances(dtype)
    assert_allclose(q.grad, dq_ref, **tol)
    assert_allclose(k.grad, dk_ref, **tol)
    assert_allclose(v.grad, dv_ref, **tol)


def test_fa3_backend_consistency(triton_available, cuda_extension_available):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    backends = ["torch"]
    if triton_available:
        backends.append("triton")
    if cuda_extension_available:
        backends.append("cuda")

    if len(backends) < 2:
        pytest.skip("Only one FlashAttention-3 backend available on this machine")

    torch.manual_seed(23)
    batch, heads, seqlen, head_dim = 1, 2, 20, 32
    q, k, v = make_qkv(batch, heads, seqlen, head_dim, device=device, dtype=dtype, merge_heads=False)
    softmax_scale = head_dim ** -0.5

    outputs = {}
    for backend in backends:
        spec = pick_fa3_spec(head_dim)
        if backend == "torch":
            o, lse = fa3_torch(q, k, v, True, softmax_scale, spec, False)
        elif backend == "triton":
            from fa3.triton.impl import fa3_triton

            o, lse = fa3_triton(q, k, v, True, softmax_scale, spec)
        else:
            o, lse = fa3_cuda(q, k, v, True, softmax_scale, spec, False)
        outputs[backend] = (flatten_output(o), flatten_lse(lse))

    ref_backend = backends[0]
    ref_o, ref_lse = outputs[ref_backend]
    tol = dtype_tolerances(dtype)
    for backend, (o, lse) in outputs.items():
        if backend == ref_backend:
            continue
        assert_allclose(o, ref_o, **tol)
        assert_allclose(lse, ref_lse, rtol=1e-3, atol=1e-3)
