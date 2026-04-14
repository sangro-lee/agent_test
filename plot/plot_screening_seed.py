#!/usr/bin/env python
"""
Plot mean ± std across seeds per base model vs guidance weight (w).

Seed suffix rules:
  - No suffix or _random → seed 42
  - _<number>            → that seed
  e.g. rgcn_mlp_z4, rgcn_mlp_z4_random → base=rgcn_mlp_z4 (seed 42)
       rgcn_mlp_z4_43                   → base=rgcn_mlp_z4 (seed 43)

Usage:
  # Combined plot (mean±std, all methods in one figure)
  python plot/plot_screening_seed.py --csv plot/screening_comparison_cosine.csv

  # Per-method PNGs (individual seed lines + mean, one PNG per base model)
  python plot/plot_screening_seed.py --csv plot/screening_comparison_cosine.csv --per_method

  # Both at once
  python plot/plot_screening_seed.py --csv plot/screening_comparison_cosine.csv --per_method \\
      --metric mean_actual hit_rate
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_w(run_tag: str) -> float | None:
    m = re.search(r"cfg_w([\d.]+)", run_tag)
    return float(m.group(1)) if m else None


def parse_base_exp(exp: str) -> str:
    """Strip trailing seed suffix: _<number> or _random → base model name."""
    return re.sub(r"_(\d+|random)$", "", exp)


def parse_seed(exp: str) -> str:
    """Return seed label for legend: number suffix or '42' if absent/random."""
    m = re.search(r"_(\d+|random)$", exp)
    if m is None:
        return "42"
    val = m.group(1)
    return "42" if val == "random" else val


def _plot_combined(df: pd.DataFrame, metrics: list[str],
                   out_path: Path, base_exps: list[str]) -> None:
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(base_exps)))

    for ax, metric in zip(axes[0], metrics):
        if metric not in df.columns:
            ax.set_title(f"{metric} (not found)")
            continue

        for base, color in zip(base_exps, colors):
            grp = df[df["base_exp"] == base]
            agg = (grp.groupby("w")[metric]
                      .agg(["mean", "std"])
                      .reset_index()
                      .sort_values("w"))
            ws   = agg["w"].values
            mean = agg["mean"].values
            std  = agg["std"].fillna(0).values

            ax.plot(ws, mean, marker="o", label=base, color=color)
            ax.fill_between(ws, mean - std, mean + std, alpha=0.15, color=color)

        ax.set_xlabel("Guidance weight (w)")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs w (mean±std over seeds)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_screening_seed] Saved: {out_path}")


def _plot_per_method(df: pd.DataFrame, metrics: list[str],
                     out_dir: Path, base_exps: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_metrics = len(metrics)

    for base in base_exps:
        grp = df[df["base_exp"] == base]
        seeds = sorted(grp["exp"].unique(), key=parse_seed)
        seed_colors = plt.cm.tab10(np.linspace(0, 0.9, len(seeds)))

        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5), squeeze=False)
        fig.suptitle(base, fontsize=11)

        for ax, metric in zip(axes[0], metrics):
            if metric not in grp.columns:
                ax.set_title(f"{metric} (not found)")
                continue

            # Individual seed lines
            for exp, color in zip(seeds, seed_colors):
                s = grp[grp["exp"] == exp].sort_values("w")
                ax.plot(s["w"], s[metric], marker=".", linestyle="--",
                        color=color, alpha=0.6, linewidth=1,
                        label=f"seed {parse_seed(exp)}")

            # Bold mean line
            agg = (grp.groupby("w")[metric]
                      .agg(["mean", "std"])
                      .reset_index()
                      .sort_values("w"))
            ws   = agg["w"].values
            mean = agg["mean"].values
            std  = agg["std"].fillna(0).values
            ax.plot(ws, mean, marker="o", color="black",
                    linewidth=2, label="mean", zorder=5)
            ax.fill_between(ws, mean - std, mean + std,
                            alpha=0.12, color="black")

            ax.set_xlabel("Guidance weight (w)")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} vs w")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        out_path = out_dir / f"seed_{base}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot_screening_seed] Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        type=str, required=True)
    parser.add_argument("--out",        type=str, default=None,
                        help="Output PNG for combined plot (default: <csv_dir>/seed_sweep.png)")
    parser.add_argument("--metric",     nargs="+",
                        default=["mean_actual", "hit_rate", "n"],
                        help="Columns to plot (default: mean_actual hit_rate n)")
    parser.add_argument("--per_method", action="store_true",
                        help="Also generate one PNG per base model showing individual seeds")
    parser.add_argument("--save_csv",   type=str, default=None,
                        help="Path to save aggregated CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df["w"]        = df["run_tag"].apply(parse_w)
    df["base_exp"] = df["exp"].apply(parse_base_exp)
    df = df.dropna(subset=["w"])

    metrics   = [m for m in args.metric if m in df.columns]
    base_exps = sorted(df["base_exp"].unique())
    csv_path  = Path(args.csv)
    out_path  = Path(args.out) if args.out else csv_path.parent / "seed_sweep.png"

    # Combined mean±std plot
    _plot_combined(df, metrics, out_path, base_exps)

    # Per-method plots
    if args.per_method:
        _plot_per_method(df, metrics, out_path.parent, base_exps)

    # Save aggregated CSV
    rows = []
    for base in base_exps:
        grp = df[df["base_exp"] == base]
        agg = grp.groupby("w")[metrics].agg(["mean", "std"]).reset_index()
        agg.columns = ["w"] + [f"{m}_{s}" for m in metrics for s in ("mean", "std")]
        agg.insert(0, "base_exp", base)
        rows.append(agg)

    result_df = pd.concat(rows, ignore_index=True).sort_values(["base_exp", "w"])
    csv_out = Path(args.save_csv) if args.save_csv else out_path.with_suffix(".csv")
    result_df.to_csv(csv_out, index=False)
    print(f"[plot_screening_seed] CSV: {csv_out}")


if __name__ == "__main__":
    main()
