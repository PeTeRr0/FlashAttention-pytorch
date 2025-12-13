import argparse
from typing import List

import torch

from bench_utils import (
    BenchmarkRecord,
    add_common_args,
    benchmark_fn,
    compute_tflops,
    has_fa2_cuda,
    has_triton,
    is_oom_error,
    iter_causal_flags,
    make_qkv,
    maybe_cast_dtype,
    write_results,
)
from plotting import default_forward_fig_path, default_table_path, plot_forward_figure, render_ablation_table

from fa2.op import fa2_attention


def available_backends(device: str) -> List[str]:
    backends = ["torch"]
    if device == "cuda":
        if has_triton():
            backends.append("triton")
        if has_fa2_cuda():
            backends.append("cuda")
    return backends


def method_name(backend: str) -> str:
    if backend == "torch":
        return "Standard attention"
    if backend == "triton":
        return "FlashAttention-2 Triton"
    if backend == "cuda":
        return "FlashAttention-2"
    return f"fa2-{backend}"


def main():
    parser = argparse.ArgumentParser(description="Benchmark FlashAttention-2 backends")
    add_common_args(parser)
    parser.add_argument("--tag", type=str, default="fa2", help="Base name for result files")
    parser.add_argument("--config-label", type=str, default=None, help="Optional config name for tables")
    parser.add_argument("--plot-dtype", type=str, default=None, help="dtype to use when plotting (default: best)")
    parser.add_argument("--no-plot", action="store_true", help="Skip figure/table generation")
    args = parser.parse_args()

    device = args.device
    backends = available_backends(device)
    if not backends:
        raise SystemExit("No available backends for FlashAttention-2 on this machine")

    records: List[BenchmarkRecord] = []
    causal_flags = list(iter_causal_flags(args))
    figure_seqlens = [512, 1024, 2048, 4096, 8192, 16384]
    active_seqlens = [s for s in figure_seqlens if s in args.seqlen] or args.seqlen

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
                                status = "oom" if is_oom_error(exc) else "error"
                                for backend in backends:
                                    records.append(
                                        BenchmarkRecord(
                                            method=method_name(backend),
                                            algo="fa2",
                                            backend=backend,
                                            direction="forward",
                                            dtype=dtype_str,
                                            causal=causal,
                                            seqlen=seqlen,
                                            head_dim=head_dim,
                                            batch_size=batch,
                                            num_heads=heads,
                                            mean_ms=None,
                                            std_ms=None,
                                            tflops=None,
                                            peak_mem_mb=None,
                                            status=status,
                                            fp8=None,
                                            config=args.config_label,
                                            error=str(exc),
                                        )
                                    )
                                if device == "cuda":
                                    torch.cuda.empty_cache()
                                continue

                            for backend in backends:
                                def _call():
                                    out, _ = fa2_attention(
                                        q, k, v, causal=causal, softmax_scale=softmax_scale, backend=backend
                                    )
                                    return out

                                try:
                                    mean_ms, std_ms, peak_mem = benchmark_fn(_call, device, args.warmup, args.iters)
                                    tflops = compute_tflops(batch, heads, seqlen, head_dim, mean_ms, "forward")
                                    records.append(
                                        BenchmarkRecord(
                                            method=method_name(backend),
                                            algo="fa2",
                                            backend=backend,
                                            direction="forward",
                                            dtype=dtype_str,
                                            causal=causal,
                                            seqlen=seqlen,
                                            head_dim=head_dim,
                                            batch_size=batch,
                                            num_heads=heads,
                                            mean_ms=mean_ms,
                                            std_ms=std_ms,
                                            tflops=tflops,
                                            peak_mem_mb=peak_mem,
                                            status="ok",
                                            fp8=None,
                                            config=args.config_label,
                                            error=None,
                                        )
                                    )
                                except Exception as exc:
                                    status = "oom" if is_oom_error(exc) else "error"
                                    if device == "cuda" and status == "oom":
                                        torch.cuda.empty_cache()
                                    records.append(
                                        BenchmarkRecord(
                                            method=method_name(backend),
                                            algo="fa2",
                                            backend=backend,
                                            direction="forward",
                                            dtype=dtype_str,
                                            causal=causal,
                                            seqlen=seqlen,
                                            head_dim=head_dim,
                                            batch_size=batch,
                                            num_heads=heads,
                                            mean_ms=None,
                                            std_ms=None,
                                            tflops=None,
                                            peak_mem_mb=None,
                                            status=status,
                                            fp8=None,
                                            config=args.config_label,
                                            error=str(exc),
                                        )
                                    )

    headers = [
        "method",
        "backend",
        "dtype",
        "shape",
        "causal",
        "mean_ms",
        "std_ms",
        "tflops",
        "peak_mem_mb",
        "status",
    ]
    col_widths = [len(h) for h in headers]
    table_rows = []
    for rec in records:
        row = [
            rec.method,
            rec.backend,
            rec.dtype,
            f"B{rec.batch_size} H{rec.num_heads} N{rec.seqlen} D{rec.head_dim}",
            "causal" if rec.causal else "non-causal",
            f"{rec.mean_ms:.2f}" if rec.mean_ms is not None else "-",
            f"{rec.std_ms:.2f}" if rec.std_ms is not None else "-",
            f"{rec.tflops:.2f}" if rec.tflops is not None else "-",
            f"{rec.peak_mem_mb:.1f}" if rec.peak_mem_mb is not None else "-",
            rec.status if not rec.error else f"{rec.status} ({rec.error})",
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

    paths = write_results(args.tag, records)
    print(f"\nSaved structured results to {paths['json']} and {paths['csv']}")

    if not args.no_plot:
        plot_forward_figure(
            records,
            seqlens=active_seqlens,
            dtype=args.plot_dtype,
            config=args.config_label,
            output_path=default_forward_fig_path(f"{args.tag}_forward"),
        )
        render_ablation_table(
            records,
            output_path=default_table_path(args.tag),
            markdown_path=default_table_path(args.tag).with_suffix(".md"),
            latex_path=default_table_path(args.tag).with_suffix(".tex"),
            config=args.config_label,
        )


if __name__ == "__main__":
    main()
