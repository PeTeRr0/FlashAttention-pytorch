import argparse
from typing import Dict, List

import torch

from bench_utils import (
    BenchmarkRecord,
    add_common_args,
    benchmark_fn,
    compute_tflops,
    has_fa1_cuda,
    has_fa2_cuda,
    has_fa3_cuda,
    has_triton,
    is_oom_error,
    iter_causal_flags,
    make_qkv,
    maybe_cast_dtype,
    write_results,
)
from plotting import (
    default_forward_fig_path,
    default_mixed_fig_path,
    default_table_path,
    plot_forward_figure,
    plot_mixed_figure,
    render_ablation_table,
)

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


def display_method(algo: str, backend: str) -> str:
    if backend == "torch":
        return "Standard attention"
    if algo == "fa1":
        return "FlashAttention-1" if backend == "cuda" else "FlashAttention-1 Triton"
    if algo == "fa2":
        return "FlashAttention-2" if backend == "cuda" else "FlashAttention-2 Triton"
    if algo == "fa3":
        if backend == "cuda":
            return "FlashAttention-3"
        if backend == "triton":
            return "FlashAttention-3 Triton"
    return f"{algo}-{backend}"


def method_label(algo: str, backend: str, fp8: bool = False) -> str:
    base = display_method(algo, backend)
    return f"{base} FP8" if fp8 else base


def main():
    parser = argparse.ArgumentParser(description="Benchmark FA1/FA2/FA3 across backends (forward/backward)")
    add_common_args(parser)
    parser.add_argument("--fp8", action="store_true", help="Include fp8 path for FA3 (if supported)")
    parser.add_argument(
        "--directions",
        nargs="+",
        default=["forward"],
        choices=["forward", "backward"],
        help="Benchmark forward, backward, or both",
    )
    parser.add_argument("--tag", type=str, default="compare_all", help="Base name for result files")
    parser.add_argument("--config-label", type=str, default=None, help="Optional config name for tables")
    parser.add_argument("--plot-dtype", type=str, default=None, help="dtype to use when plotting (default: best)")
    parser.add_argument("--no-plot", action="store_true", help="Skip figure/table generation")
    parser.add_argument(
        "--caption",
        type=str,
        default="Figure 6: Attention backward speed (FP16/BF16) on H100 GPU.",
        help="Caption text for the mixed panel figure",
    )
    args = parser.parse_args()

    device = args.device
    algos = {
        "fa1": fa1_attention,
        "fa2": fa2_attention,
        "fa3": fa3_attention,
    }

    records: List[BenchmarkRecord] = []
    causal_flags = list(iter_causal_flags(args))
    figure_seqlens = [512, 1024, 2048, 4096, 8192, 16384]
    active_seqlens = [s for s in figure_seqlens if s in args.seqlen] or args.seqlen

    for direction in args.directions:
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
                                    q_base, k_base, v_base = make_qkv(batch, heads, seqlen, head_dim, device, dtype)
                                except Exception as exc:
                                    status = "oom" if is_oom_error(exc) else "error"
                                    for algo in algos:
                                        for backend in available_backends(algo, device):
                                            fp8_flags = [True, False] if (algo == "fa3" and args.fp8) else [False]
                                            for fp8 in fp8_flags:
                                                records.append(
                                                    BenchmarkRecord(
                                                        method=method_label(algo, backend, fp8=fp8),
                                                        algo=algo,
                                                        backend=backend,
                                                        direction=direction,
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
                                                        fp8=fp8 if algo == "fa3" else None,
                                                        config=args.config_label,
                                                        error=str(exc),
                                                    )
                                                )
                                    if device == "cuda":
                                        torch.cuda.empty_cache()
                                    continue

                                for algo, attn_fn in algos.items():
                                    for backend in available_backends(algo, device):
                                        fp8_flags = [True, False] if (algo == "fa3" and args.fp8) else [False]
                                        for fp8 in fp8_flags:
                                            def _call():
                                                q = q_base
                                                k = k_base
                                                v = v_base
                                                if direction == "backward":
                                                    q = q_base.clone().detach().requires_grad_(True)
                                                    k = k_base.clone().detach().requires_grad_(True)
                                                    v = v_base.clone().detach().requires_grad_(True)
                                                if algo == "fa3":
                                                    out, _ = attn_fn(
                                                        q,
                                                        k,
                                                        v,
                                                        causal=causal,
                                                        softmax_scale=softmax_scale,
                                                        backend=backend,
                                                        fp8=fp8,
                                                    )
                                                else:
                                                    out, _ = attn_fn(
                                                        q,
                                                        k,
                                                        v,
                                                        causal=causal,
                                                        softmax_scale=softmax_scale,
                                                        backend=backend,
                                                    )
                                                if direction == "backward":
                                                    loss = out.sum()
                                                    loss.backward()
                                                    return q.grad
                                                return out

                                            _call._requires_grad = direction == "backward"  # type: ignore[attr-defined]

                                            try:
                                                mean_ms, std_ms, peak_mem = benchmark_fn(
                                                    _call, device, args.warmup, args.iters
                                                )
                                                tflops = compute_tflops(batch, heads, seqlen, head_dim, mean_ms, direction)
                                                records.append(
                                                    BenchmarkRecord(
                                                        method=method_label(algo, backend, fp8=fp8),
                                                        algo=algo,
                                                        backend=backend,
                                                        direction=direction,
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
                                                        fp8=fp8 if algo == "fa3" else None,
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
                                                        method=method_label(algo, backend, fp8=fp8),
                                                        algo=algo,
                                                        backend=backend,
                                                        direction=direction,
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
                                                        fp8=fp8 if algo == "fa3" else None,
                                                        config=args.config_label,
                                                        error=str(exc),
                                                    )
                                                )

    headers = [
        "method",
        "backend",
        "direction",
        "dtype",
        "shape",
        "causal",
        "mean_ms",
        "std_ms",
        "tflops",
        "peak_mem_mb",
        "status",
    ]
    table_rows = []
    col_widths = [len(h) for h in headers]
    for rec in records:
        row = [
            rec.method,
            rec.backend,
            rec.direction,
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
        plot_dtype = args.plot_dtype
        plot_forward_figure(
            records,
            seqlens=active_seqlens,
            dtype=plot_dtype,
            config=args.config_label,
            output_path=default_forward_fig_path(),
        )
        if "backward" in args.directions:
            plot_mixed_figure(
                records,
                seqlens=active_seqlens,
                dtype=plot_dtype,
                config=args.config_label,
                output_path=default_mixed_fig_path(),
                caption=args.caption,
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
