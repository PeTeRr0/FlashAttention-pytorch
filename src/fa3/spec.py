from dataclasses import dataclass

@dataclass(frozen=True)
class FA3Spec:
    br: int
    bc: int
    num_warps: int
    stages: int

def pick_fa3_spec(head_dim: int) -> FA3Spec:
    if head_dim <= 64:
        return FA3Spec(br=128, bc=128, num_warps=8, stages=2)
    return FA3Spec(br=64, bc=128, num_warps=8, stages=2)
