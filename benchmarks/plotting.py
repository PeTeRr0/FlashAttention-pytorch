from __future__ import annotations

import itertools
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bench_utils import BenchmarkRecord, FIGURES_DIR, TABLES_DIR


Palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _method_colors(methods: Sequence[str]) -> Dict[str, str]:
    return {m: Palette[i % len(Palette)] for i, m in enumerate(methods)}


def _select_records(
    records: Sequence[BenchmarkRecord],
    *,
    direction: str,
    head_dim: int,
    causal: bool,
    dtype: Optional[str],
    seqlens: Sequence[int],
    config: Optional[str] = None,
) -> Dict[Tuple[str, int], BenchmarkRecord]:
    by_key: Dict[Tuple[str, int], BenchmarkRecord] = {}
    for rec in records:
        if rec.direction != direction:
            continue
        if rec.head_dim != head_dim or rec.causal != causal:
            continue
        if dtype and rec.dtype != dtype:
            continue
        if config and rec.config != config:
            continue
        if rec.seqlen not in seqlens:
            continue
        key = (rec.method, rec.seqlen)
        existing = by_key.get(key)
        # If multiple entries exist for the same key, keep the faster one.
        if existing is None or (
            rec.tflops is not None
            and (existing.tflops is None or rec.tflops > existing.tflops)
        ):
            by_key[key] = rec
    return by_key


def _panel_values(
    records: Sequence[BenchmarkRecord],
    *,
    direction: str,
    head_dim: int,
    causal: bool,
    seqlens: Sequence[int],
    dtype: Optional[str],
    config: Optional[str],
) -> Tuple[List[str], Dict[str, List[Optional[float]]], Dict[str, List[str]]]:
    lookup = _select_records(
        records,
        direction=direction,
        head_dim=head_dim,
        causal=causal,
        dtype=dtype,
        seqlens=seqlens,
        config=config,
    )

    methods = sorted({rec.method for rec in lookup.values()})
    x_labels = [str(s) if s < 1000 else f"{int(s/1000)}k" for s in seqlens]
    values: Dict[str, List[Optional[float]]] = {m: [] for m in methods}
    statuses: Dict[str, List[str]] = {m: [] for m in methods}

    for method in methods:
        for seqlen in seqlens:
            rec = lookup.get((method, seqlen))
            if rec is None:
                values[method].append(None)
                statuses[method].append("missing")
                continue
            if rec.status.lower() == "oom":
                values[method].append(None)
                statuses[method].append("oom")
            elif rec.status.lower() != "ok":
                values[method].append(None)
                statuses[method].append(rec.status)
            else:
                values[method].append(rec.tflops)
                statuses[method].append("ok")

    return x_labels, values, statuses


def _bar_panel(
    ax,
    x_labels: List[str],
    values: Mapping[str, List[Optional[float]]],
    statuses: Mapping[str, List[str]],
    title: str,
    show_legend: bool,
) -> List[Tuple[str, Any]]:
    methods = list(values.keys())
    colors = _method_colors(methods)
    base_x = list(range(len(x_labels)))
    width = 0.8 / max(len(methods), 1)
    handles: List[Tuple[str, any]] = []
    max_height = 0.0

    for idx, method in enumerate(methods):
        offsets = [x - 0.4 + width / 2 + idx * width for x in base_x]
        bar_vals = values[method]
        bars = []
        for pos, val in zip(offsets, bar_vals):
            if val is not None:
                bar = ax.bar(pos, val, width=width, color=colors[method], label=method if show_legend else None)
                bars.append(bar[0])
                max_height = max(max_height, val)
                ax.text(pos, val + 0.05, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
            else:
                bars.append(None)
        for pos, status in zip(offsets, statuses[method]):
            if status == "oom":
                ax.text(pos, 0.05, "OOM", ha="center", va="bottom", fontsize=8, color="#444444")
        if show_legend:
            handles.append((method, bars[0] if bars and bars[0] is not None else None))

    ax.set_xticks(base_x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Speed (TFLOPs/s)")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    if max_height > 0:
        ax.set_ylim(0, max_height * 1.25)
    return handles


def plot_forward_figure(
    records: Sequence[BenchmarkRecord],
    *,
    seqlens: Sequence[int],
    dtype: Optional[str],
    config: Optional[str],
    output_path: Path,
) -> None:
    panels = [
        {"head_dim": 64, "causal": False},
        {"head_dim": 64, "causal": True},
        {"head_dim": 128, "causal": False},
        {"head_dim": 128, "causal": True},
        {"head_dim": 256, "causal": False},
        {"head_dim": 256, "causal": True},
    ]
    panel_captions = [
        "(a) Forward, without causal mask, head dim 64",
        "(b) Forward, with causal mask, head dim 64",
        "(c) Forward, without causal mask, head dim 128",
        "(d) Forward, with causal mask, head dim 128",
        "(e) Forward, without causal mask, head dim 256",
        "(f) Forward, with causal mask, head dim 256",
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharey=False)
    legend_handles: Dict[str, any] = {}

    for idx, (ax, panel) in enumerate(zip(itertools.chain.from_iterable(axes), panels)):
        x_labels, values, statuses = _panel_values(
            records,
            direction="forward",
            head_dim=panel["head_dim"],
            causal=panel["causal"],
            seqlens=seqlens,
            dtype=dtype,
            config=config,
        )
        handles = _bar_panel(
            ax,
            x_labels,
            values,
            statuses,
            title=f"Attention forward speed, head dim {panel['head_dim']} (H100 80GB SXM5)",
            show_legend=not legend_handles,
        )
        ax.set_xlabel(panel_captions[idx], fontsize=9)
        ax.xaxis.set_label_coords(0.5, -0.2)
        for name, handle in handles:
            if handle is not None and name not in legend_handles:
                legend_handles[name] = handle

    fig.tight_layout(rect=[0, 0.08, 1, 0.98])
    if legend_handles:
        fig.legend(legend_handles.values(), legend_handles.keys(), loc="upper center", ncol=len(legend_handles))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_mixed_figure(
    records: Sequence[BenchmarkRecord],
    *,
    seqlens: Sequence[int],
    dtype: Optional[str],
    config: Optional[str],
    output_path: Path,
    caption: str,
) -> None:
    panels = [
        {
            "direction": "backward",
            "head_dim": 64,
            "causal": False,
            "title": "Attention backward speed, head dim 64 (H100 80GB SXM5)",
            "caption": "(a) Backward, without causal mask, head dim 64",
        },
        {
            "direction": "backward",
            "head_dim": 128,
            "causal": False,
            "title": "Attention backward speed, head dim 128 (H100 80GB SXM5)",
            "caption": "(b) Backward, without causal mask, head dim 128",
        },
        {
            "direction": "forward",
            "head_dim": 256,
            "causal": False,
            "title": "Attention forward speed, head dim 256 (H100 80GB SXM5)",
            "caption": "(a) Forward, without causal mask, head dim 256",
        },
        {
            "direction": "forward",
            "head_dim": 256,
            "causal": True,
            "title": "Attention forward speed, head dim 256 (H100 80GB SXM5)",
            "caption": "(b) Forward, with causal mask, head dim 256",
        },
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=False)
    legend_handles: Dict[str, any] = {}

    for ax, panel in zip(itertools.chain.from_iterable(axes), panels):
        x_labels, values, statuses = _panel_values(
            records,
            direction=panel["direction"],
            head_dim=panel["head_dim"],
            causal=panel["causal"],
            seqlens=seqlens,
            dtype=dtype,
            config=config,
        )
        handles = _bar_panel(
            ax,
            x_labels,
            values,
            statuses,
            title=panel["title"],
            show_legend=not legend_handles,
        )
        ax.set_xlabel(panel["caption"], fontsize=9)
        ax.xaxis.set_label_coords(0.5, -0.18)
        for name, handle in handles:
            if handle is not None and name not in legend_handles:
                legend_handles[name] = handle

    fig.tight_layout(rect=[0, 0.1, 1, 0.92])
    if legend_handles:
        fig.legend(legend_handles.values(), legend_handles.keys(), loc="upper center", ncol=len(legend_handles))
    fig.text(0.5, 0.52, caption, ha="center", fontsize=11)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def render_ablation_table(
    records: Sequence[BenchmarkRecord],
    *,
    output_path: Path,
    markdown_path: Optional[Path] = None,
    latex_path: Optional[Path] = None,
    config: Optional[str] = None,
) -> None:
    filtered: List[BenchmarkRecord] = []
    for rec in records:
        if rec.status != "ok":
            continue
        if config and rec.config != config:
            continue
        filtered.append(rec)

    targets: List[Tuple[str, Any]] = [
        ("FLASHATTENTION-3", lambda r: r.algo == "fa3"),
        ("No GEMM-Softmax Pipelining, Warp-Specialization", lambda r: r.algo == "fa2"),
        ("GEMM-Softmax Pipelining, No Warp-Specialization", lambda r: r.algo == "fa1"),
    ]

    def summarize(label: str, predicate) -> Tuple[str, Optional[float], Optional[float]]:
        matches = [r for r in filtered if (r.config == label or predicate(r))]
        latencies = [r.mean_ms for r in matches if r.mean_ms is not None]
        speeds = [r.tflops for r in matches if r.tflops is not None]
        if not latencies or not speeds:
            return label, None, None
        return label, statistics.mean(latencies), statistics.mean(speeds)

    rows: List[Tuple[str, Optional[float], Optional[float]]] = [summarize(label, pred) for label, pred in targets]

    fig, ax = plt.subplots(figsize=(6, 0.6 * (len(rows) + 2)))
    ax.axis("off")
    cell_text = [
        [
            name,
            f"{latency:.3f} ms" if latency is not None else "-",
            f"{tflops:.3f}" if tflops is not None else "-",
        ]
        for name, latency, tflops in rows
    ]
    table = ax.table(
        cellText=cell_text,
        colLabels=["Configuration", "Time", "TFLOPs/s"],
        cellLoc="center",
        colColours=["#f0f0f0", "#f0f0f0", "#f0f0f0"],
        loc="center",
    )
    ax.set_title("Table 2: Pipelining ablation measurements", pad=12, fontsize=12, weight="bold")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(weight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if markdown_path:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        with markdown_path.open("w", encoding="utf-8") as f:
            f.write("| Configuration | Time (ms) | TFLOPs/s |\n")
            f.write("|---|---|---|\n")
            for name, latency, tflops in rows:
                latency_str = f"{latency:.3f}" if latency is not None else "-"
                tflops_str = f"{tflops:.3f}" if tflops is not None else "-"
                f.write(f"| {name} | {latency_str} | {tflops_str} |\n")

    if latex_path:
        latex_path.parent.mkdir(parents=True, exist_ok=True)
        with latex_path.open("w", encoding="utf-8") as f:
            f.write("\\begin{tabular}{lcc}\n")
            f.write("\\textbf{Configuration} & \\textbf{Time (ms)} & \\textbf{TFLOPs/s}\\\\\\hline\n")
            for name, latency, tflops in rows:
                latency_str = f"{latency:.3f}" if latency is not None else "-"
                tflops_str = f"{tflops:.3f}" if tflops is not None else "-"
                f.write(f"{name} & {latency_str} & {tflops_str}\\\\\n")
            f.write("\\end{tabular}\n")


def default_forward_fig_path(name: str = "figure_a_forward") -> Path:
    return FIGURES_DIR / f"{name}.png"


def default_mixed_fig_path(name: str = "figure_b_mixed") -> Path:
    return FIGURES_DIR / f"{name}.png"


def default_table_path(name: str = "ablation_table") -> Path:
    return TABLES_DIR / f"{name}.png"
