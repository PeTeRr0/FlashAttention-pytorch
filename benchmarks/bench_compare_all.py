import argparse
from typing import Dict, List

from bench_utils import (
    add_common_args,
    benchmark_fn,
    has_fa1_cuda,
    has_fa2_cuda,
    has_fa3_cuda,
    has_triton,
    iter_causal_flags,
    make_qkv,
    maybe_cast_dtype,
)

import torch

from fa1.op import fa1_attention
from fa2.op import fa2_attention
from fa3.op import fa3_attention


def available_backends(algo: str, device: str) -> List[str]:
    backends = ["torch"]
    if device != "cuda":
        return backends
    if has_triton():
        backends.append("triton")
    if algo == "fa1" and has_fa1_cuda():
        backends.append("cuda")
    if algo == "fa2" and has_fa2_cuda():
        backends.append("cuda")
    if algo == "fa3" and has_fa3_cuda():
        backends.append("cuda")
    return backends


def main():
    parser = argparse.ArgumentParser(description="Benchmark FA1/FA2/FA3 across backends")
    add_common_args(parser)
    parser.add_argument("--fp8", action="store_true", help="Include fp8 path for FA3 (if supported)")
    args = parser.parse_args()

    device = args.device
    algos = {
        "fa1": fa1_attention,
        "fa2": fa2_attention,
        "fa3": fa3_attention,
    }

    rows = []
    causal_flags = iter_causal_flags(args)
    for seqlen in args.seqlen:
        for head_dim in args.head_dim:
            softmax_scale = head_dim ** -0.5
            for batch in args.batch_size:
                for heads in args.num_heads:
                    for causal in causal_flags:
                        for dtype_str in args.dtypes:
                            dtype = maybe_cast_dtype(dtype_str)
                            if dtype is None:
                                continue
                            try:
                                q, k, v = make_qkv(batch, heads, seqlen, head_dim, device, dtype)
                            except Exception as exc:
                                for algo in algos:
                                    rows.append(
                                        {
                                            "algo": algo,
                                            "backend": "-",
                                            "dtype": dtype_str,
                                            "fp8": "-",
                                            "shape": f"B{batch} H{heads} N{seqlen} D{head_dim}",
                                            "causal": str(causal),
                                            "mean": "-",
                                            "std": "-",
                                            "mem": "-",
                                            "status": f"skip (dtype/device unsupported: {exc})",
                                        }
                                    )
                                continue

                            for algo, attn_fn in algos.items():
                                for backend in available_backends(algo, device):
                                    fp8_flags = [True, False] if (algo == "fa3" and args.fp8) else [False]
                                    for fp8 in fp8_flags:
                                        def _call():
                                            if algo == "fa3":
                                                o, _ = attn_fn(
                                                    q,
                                                    k,
                                                    v,
                                                    causal=causal,
                                                    softmax_scale=softmax_scale,
                                                    backend=backend,
                                                    fp8=fp8,
                                                )
                                            else:
                                                o, _ = attn_fn(
                                                    q,
                                                    k,
                                                    v,
                                                    causal=causal,
                                                    softmax_scale=softmax_scale,
                                                    backend=backend,
                                                )
                                            return o

                                        try:
                                            mean_ms, std_ms, peak_mem = benchmark_fn(
                                                _call, device, args.warmup, args.iters
                                            )
                                            rows.append(
                                                {
                                                    "algo": algo,
                                                    "backend": backend,
                                                    "dtype": dtype_str,
                                                    "fp8": str(fp8) if algo == "fa3" else "-",
                                                    "shape": f"B{batch} H{heads} N{seqlen} D{head_dim}",
                                                    "causal": str(causal),
                                                    "mean": f"{mean_ms:.2f} ms",
                                                    "std": f"{std_ms:.2f} ms",
                                                    "mem": f"{peak_mem:.1f} MB" if peak_mem is not None else "-",
                                                    "status": "ok",
                                                }
                                            )
                                        except Exception as exc:
                                            rows.append(
                                                {
                                                    "algo": algo,
                                                    "backend": backend,
                                                    "dtype": dtype_str,
                                                    "fp8": str(fp8) if algo == "fa3" else "-",
                                                    "shape": f"B{batch} H{heads} N{seqlen} D{head_dim}",
                                                    "causal": str(causal),
                                                    "mean": "-",
                                                    "std": "-",
                                                    "mem": "-",
                                                    "status": f"skip ({exc})",
                                                }
                                            )

    headers = ["algo", "backend", "dtype", "fp8", "shape", "causal", "mean", "std", "peak_mem", "status"]
    table_rows = []
    col_widths = [len(h) for h in headers]
    for r in rows:
        row = [
            r.get("algo", ""),
            r.get("backend", ""),
            r.get("dtype", ""),
            r.get("fp8", ""),
            r.get("shape", ""),
            r.get("causal", ""),
            r.get("mean", ""),
            r.get("std", ""),
            r.get("mem", ""),
            r.get("status", ""),
        ]
        table_rows.append(row)
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt(row: List[str]) -> str:
        return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in col_widths)
    print(fmt(headers))
    print(sep)
    for row in table_rows:
        print(fmt(row))


if __name__ == "__main__":
    main()
