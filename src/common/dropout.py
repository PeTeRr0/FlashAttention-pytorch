import torch

def dropout_mask(shape, p, device=None, dtype=torch.bool, generator=None):
    if p == 0.0:
        return torch.ones(shape, device=device, dtype=dtype)
    mask = torch.rand(shape, device=device, generator=generator) > p
    return mask.to(dtype)

def apply_dropout(x, p, training=True, generator=None):
    if (not training) or p == 0.0:
        return x, None
    mask = torch.rand_like(x, dtype=torch.float32, generator=generator) > p
    scale = 1.0 / (1.0 - p)
    out = x * mask.to(x.dtype) * scale
    return out, mask
