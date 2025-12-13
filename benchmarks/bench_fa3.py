import argparse
from typing import List

from bench_utils import (
    add_common_args,
    benchmark_fn,
    has_fa3_cuda,
    has_triton,
    iter_causal_flags,
    make_qkv,
    maybe_cast_dtype,
)

import torch

from fa3.op import fa3_attention


def available_backends(device: str) -> List[str]:
    backends = ["torch"]
    if device == "cuda":
        if has_triton():
            backends.append("triton")
        if has_fa3_cuda():
            backends.append("cuda")
    return backends


def main():
    parser = argparse.ArgumentParser(description="Benchmark FlashAttention-3 backends")
    add_common_args(parser)
    parser.add_argument("--fp8", action="store_true", help="Include fp8 quantized path for FA3")
    args = parser.parse_args()

    device = args.device
    backends = available_backends(device)
    if not backends:
        raise SystemExit("No available backends for FlashAttention-3 on this machine")

    rows = []
    causal_flags = iter_causal_flags(args)
    fp8_flags = [False, True] if args.fp8 else [False]
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
                            for fp8 in fp8_flags:
                                try:
                                    q, k, v = make_qkv(batch, heads, seqlen, head_dim, device, dtype)
                                except Exception as exc:
                                    rows.append(
                                        {
                                            "backend": "-",
                                            "dtype": dtype_str,
                                            "fp8": str(fp8),
                                            "shape": f"B{batch} H{heads} N{seqlen} D{head_dim}",
                                            "causal": str(causal),
                                            "mean": "-",
                                            "std": "-",
                                            "mem": "-",
                                            "status": f"skip (dtype/device unsupported: {exc})",
                                        }
                                    )
                                    continue

                                for backend in backends:
                                    def _call():
                                        o, _ = fa3_attention(
                                            q, k, v, causal=causal, softmax_scale=softmax_scale, backend=backend, fp8=fp8
                                        )
                                        return o

                                    try:
                                        mean_ms, std_ms, peak_mem = benchmark_fn(
                                            _call, device, args.warmup, args.iters
                                        )
                                        rows.append(
                                            {
                                                "backend": backend,
                                                "dtype": dtype_str,
                                                "fp8": str(fp8),
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
                                                "backend": backend,
                                                "dtype": dtype_str,
                                                "fp8": str(fp8),
                                                "shape": f"B{batch} H{heads} N{seqlen} D{head_dim}",
                                                "causal": str(causal),
                                                "mean": "-",
                                                "std": "-",
                                                "mem": "-",
                                                "status": f"skip ({exc})",
                                            }
                                        )

    headers = ["backend", "dtype", "fp8", "shape", "causal", "mean", "std", "peak_mem", "status"]
    col_widths = [len(h) for h in headers]
    table_rows = []
    for r in rows:
        row = [
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

    def fmt(row):
        return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in col_widths)
    print(fmt(headers))
    print(sep)
    for row in table_rows:
        print(fmt(row))


if __name__ == "__main__":
    main()
