import argparse
from typing import List

from bench_utils import (
    add_common_args,
    benchmark_fn,
    format_result_row,
    format_table,
    has_fa2_cuda,
    has_triton,
    iter_causal_flags,
    make_qkv,
    maybe_cast_dtype,
)

import torch

from fa2.op import fa2_attention


def available_backends(device: str) -> List[str]:
    backends = ["torch"]
    if device == "cuda":
        if has_triton():
            backends.append("triton")
        if has_fa2_cuda():
            backends.append("cuda")
    return backends


def main():
    parser = argparse.ArgumentParser(description="Benchmark FlashAttention-2 backends")
    add_common_args(parser)
    args = parser.parse_args()

    device = args.device
    backends = available_backends(device)
    if not backends:
        raise SystemExit("No available backends for FlashAttention-2 on this machine")

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
                                rows.append(
                                    {
                                        "backend": "-",
                                        "dtype": dtype_str,
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
                                    o, _ = fa2_attention(
                                        q, k, v, causal=causal, softmax_scale=softmax_scale, backend=backend
                                    )
                                    return o

                                try:
                                    mean_ms, std_ms, peak_mem = benchmark_fn(_call, device, args.warmup, args.iters)
                                    rows.append(
                                        {
                                            "backend": backend,
                                            "dtype": dtype_str,
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
                                            "shape": f"B{batch} H{heads} N{seqlen} D{head_dim}",
                                            "causal": str(causal),
                                            "mean": "-",
                                            "std": "-",
                                            "mem": "-",
                                            "status": f"skip ({exc})",
                                        }
                                    )

    headers = ["backend", "dtype", "shape", "causal", "mean", "std", "peak_mem", "status"]
    table = format_table(headers, [format_result_row(r) for r in rows])
    print(table)


if __name__ == "__main__":
    main()
