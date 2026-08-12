"""Create portfolio figures from a completed training run.

Example:
    python -m src.report --metrics artifacts/model_metrics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot model-comparison results.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()

    metrics = pd.read_csv(args.metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    long_metrics = metrics.melt(
        id_vars="model", value_vars=["roc_auc", "precision", "recall", "f1"],
        var_name="metric", value_name="score",
    )

    # SVG avoids relying on a system-specific Matplotlib binary and renders in
    # GitHub's file viewer and modern browsers. Each model is one group of bars.
    metrics_to_plot = ["roc_auc", "precision", "recall", "f1"]
    colors = ["#2563eb", "#0f766e", "#d97706", "#7c3aed"]
    width, height = 1100, 620
    left, top, bottom, chart_height = 85, 85, 120, 385
    group_width = 220
    bar_width = 30
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:26px;font-weight:700}.label{font-size:14px}.small{font-size:12px;fill:#4b5563}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text class="title" x="85" y="45">Credit-default model comparison</text>',
    ]
    for tick in range(6):
        value = tick / 5
        y = top + chart_height * (1 - value)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="1040" y2="{y:.1f}" stroke="#e5e7eb"/>')
        svg.append(f'<text class="small" x="45" y="{y + 4:.1f}">{value:.1f}</text>')

    for index, row in metrics.iterrows():
        x_start = left + index * group_width + 25
        for metric_index, metric in enumerate(metrics_to_plot):
            score = float(row[metric])
            bar_height = score * chart_height
            x = x_start + metric_index * (bar_width + 8)
            y = top + chart_height - bar_height
            svg.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" rx="3" fill="{colors[metric_index]}"/>')
        center = x_start + 2 * (bar_width + 8) - 8
        svg.append(f'<text class="label" text-anchor="middle" x="{center}" y="{top + chart_height + 28}">{row["model"]}</text>')

    for index, metric in enumerate(metrics_to_plot):
        x = 610 + index * 115
        svg.append(f'<rect x="{x}" y="{height - 65}" width="14" height="14" rx="2" fill="{colors[index]}"/>')
        svg.append(f'<text class="small" x="{x + 20}" y="{height - 53}">{metric}</text>')
    svg.append('</svg>')
    output = args.output_dir / "model_comparison.svg"
    output.write_text("\n".join(svg), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
