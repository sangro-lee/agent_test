#!/usr/bin/env python
"""
Plot mean_actual vs guidance weight (w) per model from screening_comparison.csv.

Usage:
  python plot/plot_screening_w.py --csv plot/screening_comparison.csv
  python plot/plot_screening_w.py --csv plot/screening_comparison.csv --out outputs/w_sweep.png
  python plot/plot_screening_w.py --csv plot/screening_comparison.csv --metric mean_actual hit_rate
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_w(run_tag: str) -> float | None:
    m = re.search(r"cfg_w([\d.]+)", run_tag)
    return float(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",  type=str, required=True)
    parser.add_argument("--out",  type=str, default=None)
    parser.add_argument("--metric", nargs="+",
                        default=["mean_actual", "hit_rate", "n"],
                        help="Columns to plot (default: mean_actual hit_rate n)")
    parser.add_argument("--save_csv", type=str, default=None,
                        help="Path to save results as CSV (default: same dir as --out, w_sweep.csv)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df["w"] = df["run_tag"].apply(parse_w)
    df = df.dropna(subset=["w"])

    metrics = args.metric
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5), squeeze=False)

    for ax, metric in zip(axes[0], metrics):
        for exp, grp in df.groupby("exp"):
            grp = grp.sort_values("w")
            ax.plot(grp["w"], grp[metric], marker="o", label=exp)
            for _, row in grp.iterrows():
                ax.annotate(f"w={row['w']:.1f}",
                            (row["w"], row[metric]),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=6, alpha=0.7)

        ax.set_xlabel("Guidance weight (w)")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs w")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    csv_path = Path(args.csv)
    out_path = Path(args.out) if args.out else csv_path.parent / "w_sweep.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_screening_w] Saved: {out_path}")

    # Save CSV
    save_cols = ["exp", "w"] + [m for m in metrics if m in df.columns]
    result_df = df[save_cols].sort_values(["exp", "w"]).reset_index(drop=True)
    csv_out = Path(args.save_csv) if args.save_csv else out_path.with_suffix(".csv")
    result_df.to_csv(csv_out, index=False)
    print(f"[plot_screening_w] CSV: {csv_out}")


if __name__ == "__main__":
    main()
