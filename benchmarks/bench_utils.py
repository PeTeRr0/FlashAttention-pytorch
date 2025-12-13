from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import torch
except Exception as exc:  # pragma: no cover - torch is required at runtime
    raise RuntimeError("PyTorch must be installed to run benchmarks") from exc


def has_cuda() -> bool:
    return torch.cuda.is_available()


def has_triton() -> bool:
    if not has_cuda():
        return False
    try:
        import triton  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def has_fa1_cuda() -> bool:
    if not has_cuda():
        return False
    try:
        from fa1.cuda.impl import _load_ext

        _load_ext()
    except Exception:
        return False
    return True


def has_fa2_cuda() -> bool:
    if not has_cuda():
        return False
    try:
        from fa2.cuda.impl import _load_ext

        _load_ext()
    except Exception:
        return False
    return True


def has_fa3_cuda() -> bool:
    if not has_cuda():
        return False
    try:
        from fa3.cuda.impl import _load_ext

        _load_ext()
    except Exception:
        return False
    return True


def make_qkv(
    batch: int,
    heads: int,
    seqlen: int,
    head_dim: int,
    device: str,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (batch, heads, seqlen, head_dim)
    g = torch.Generator(device=device)
    g.manual_seed(0)
    q = torch.randn(shape, device=device, dtype=dtype, generator=g)
    k = torch.randn(shape, device=device, dtype=dtype, generator=g)
    v = torch.randn(shape, device=device, dtype=dtype, generator=g)
    return q, k, v


def benchmark_fn(
    fn: Callable[[], torch.Tensor],
    device: str,
    warmup: int,
    iters: int,
) -> Tuple[float, float, Optional[float]]:
    """Benchmark callable returning a tensor. Returns (mean_ms, std_ms, peak_mem_mb)."""
    if device == "cuda":
        torch.cuda.synchronize()
    with torch.inference_mode():
        for _ in range(warmup):
            out = fn()
            if torch.is_tensor(out):
                _ = out.sum().item()
            elif isinstance(out, (list, tuple)) and torch.is_tensor(out[0]):
                _ = out[0].sum().item()

        if device == "cuda":
            torch.cuda.synchronize()

        times: List[float] = []
        peak_mem_mb: Optional[float] = None
        for _ in range(iters):
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                start_mem = torch.cuda.memory_allocated()
            start = time.perf_counter()
            out = fn()
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            if torch.is_tensor(out):
                _ = out.sum().item()
            elif isinstance(out, (list, tuple)) and torch.is_tensor(out[0]):
                _ = out[0].sum().item()
            times.append((end - start) * 1e3)
            if device == "cuda":
                peak = torch.cuda.max_memory_allocated()
                used = peak - start_mem
                mem_mb = used / (1024 * 1024)
                peak_mem_mb = mem_mb if peak_mem_mb is None else max(peak_mem_mb, mem_mb)

    mean_ms = statistics.mean(times)
    std_ms = statistics.pstdev(times) if len(times) > 1 else 0.0
    return mean_ms, std_ms, peak_mem_mb


def maybe_cast_dtype(dtype: str) -> Optional[torch.dtype]:
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return mapping.get(dtype.lower(), None)


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(row: Sequence[str]) -> str:
        return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in col_widths)
    parts = [fmt_row(headers), sep]
    parts.extend(fmt_row(r) for r in rows)
    return "\n".join(parts)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda" if has_cuda() else "cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seqlen", type=int, nargs="+", default=[128, 512, 1024], help="Sequence lengths")
    parser.add_argument("--head-dim", type=int, nargs="+", default=[64], help="Head dimensions")
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 2], help="Batch sizes")
    parser.add_argument("--num-heads", type=int, nargs="+", default=[4], help="Number of heads")
    parser.add_argument("--causal", action="store_true", help="Run only causal mode (defaults to both)")
    parser.add_argument("--non-causal-only", action="store_true", help="Run only non-causal mode")
    parser.add_argument("--dtypes", type=str, nargs="+", default=["fp16", "bf16"], help="Dtypes: fp16 bf16 fp32")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations")


def iter_causal_flags(args: argparse.Namespace) -> Iterable[bool]:
    if args.causal:
        return [True]
    if args.non_causal_only:
        return [False]
    return [False, True]


def format_result_row(entry: Dict[str, Optional[str]]) -> List[str]:
    return [
        entry.get("backend", ""),
        entry.get("dtype", ""),
        entry.get("shape", ""),
        entry.get("causal", ""),
        entry.get("mean", ""),
        entry.get("std", ""),
        entry.get("mem", ""),
        entry.get("status", ""),
    ]
