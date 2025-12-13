from dataclasses import dataclass

@dataclass(frozen=True)
class FA2Spec:
    br: int
    bc: int
    num_warps: int

def pick_fa2_spec(head_dim: int) -> FA2Spec:
    if head_dim <= 64:
        return FA2Spec(br=128, bc=128, num_warps=8)
    return FA2Spec(br=64, bc=128, num_warps=8)