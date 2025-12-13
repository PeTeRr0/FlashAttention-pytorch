from dataclasses import dataclass

@dataclass(frozen=True)
class FA1Spec:
    br: int
    bc: int
    num_warps: int

def pick_fa1_spec(head_dim: int) -> FA1Spec:
    if head_dim <= 64:
        return FA1Spec(br=128, bc=128, num_warps=8)
    return FA1Spec(br=64, bc=128, num_warps=8)
