import torch

def merge_bh(x):
    if x.dim() == 3:
        return x, None
    b, h, n, d = x.shape
    return x.reshape(b * h, n, d), (b, h)

def split_bh(x, bh_shape):
    if bh_shape is None:
        return x
    b, h = bh_shape
    bh, n, d = x.shape
    return x.reshape(b, h, n, d)

def split_bh_lse(lse, bh_shape):
    if bh_shape is None:
        return lse
    b, h = bh_shape
    bh, n = lse.shape
    return lse.reshape(b, h, n)

def block_absmax_scale(x, block, dim=-1, eps=1e-6):
    n = x.shape[-2]
    nb = (n + block - 1) // block
    scales = torch.empty((x.shape[0], nb), device=x.device, dtype=torch.float32)
    for i in range(nb):
        s = i * block
        e = min(s + block, n)
        blk = x[:, s:e, :]
        m = torch.amax(torch.abs(blk).to(torch.float32), dim=(1, 2))
        scales[:, i] = m.clamp_min(eps)
    return scales

def block_quant_dequant(x, scales, block):
    xq = torch.empty_like(x, dtype=torch.float16)
    n = x.shape[-2]
    nb = scales.shape[1]
    for i in range(nb):
        s = i * block
        e = min(s + block, n)
        sc = scales[:, i].to(torch.float16)[:, None, None]
        y = x[:, s:e, :].to(torch.float16) / sc
        y = torch.clamp(y, -1.0, 1.0)
        xq[:, s:e, :] = y * sc
    return xq

def hadamard_inplace(x):
    b, n, d = x.shape
    h = 1
    y = x
    while h < d:
        a = y[..., 0::2*h]
        c = y[..., h::2*h]
        y[..., 0::2*h] = a + c
        y[..., h::2*h] = a - c
        h *= 2
    return y

def incoherent_process(q, k, seed=0):
    b, n, d = q.shape
    if (d & (d - 1)) != 0:
        return q, k
    g = torch.Generator(device=q.device)
    g.manual_seed(int(seed))
    s = torch.randint(0, 2, (d,), device=q.device, generator=g, dtype=torch.int32)
    s = (s * 2 - 1).to(q.dtype)
    q2 = q * s[None, None, :]
    k2 = k * s[None, None, :]
    q2 = hadamard_inplace(q2.to(torch.float32)).to(q.dtype)
    k2 = hadamard_inplace(k2.to(torch.float32)).to(k.dtype)
    q2 = q2 / (d ** 0.5)
    k2 = k2 / (d ** 0.5)
    return q2, k2

# aliases that mirror helpers used inside the faX torch paths
_merge_bh = merge_bh
_split_bh = split_bh
_split_bh_lse = split_bh_lse
