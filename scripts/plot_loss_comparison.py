#!/usr/bin/env python
"""Compare best train_loss and val_loss across diffusion experiments.

Finds all loss_history.csv under outputs/runs/*/diffusion/*/loss_history.csv,
extracts best train_loss and best val_loss per experiment, and saves a grouped
bar chart to outputs/loss_comparison.png.

Usage:
  python scripts/plot_loss_comparison.py
  python scripts/plot_loss_comparison.py --runs_dir outputs/runs --out outputs/loss_comparison.png
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")


def collect_results(runs_dir: str) -> pd.DataFrame:
    rows = []
    for exp in sorted(os.listdir(runs_dir)):
        diff_dir = os.path.join(runs_dir, exp, "diffusion")
        if not os.path.isdir(diff_dir):
            continue
        for run in sorted(os.listdir(diff_dir)):
            csv_path = os.path.join(diff_dir, run, "loss_history.csv")
            if not os.path.isfile(csv_path):
                continue
            df = pd.read_csv(csv_path)
            best_train = float(df["train_loss"].min())
            best_val   = float(df["val_loss"].min()) if "val_loss" in df.columns else float("nan")
            rows.append({"exp": exp, "run": run, "best_train": best_train, "best_val": best_val})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default=os.path.join(ROOT, "outputs", "runs"))
    parser.add_argument("--out",      default=os.path.join(ROOT, "outputs", "loss_comparison.png"))
    args = parser.parse_args()

    df = collect_results(args.runs_dir)
    if df.empty:
        print("No loss_history.csv found.")
        sys.exit(1)

    # If multiple runs per exp, keep the best train among them
    df = (
        df.sort_values("best_train")
          .groupby("exp", sort=False)
          .first()
          .reset_index()
          .sort_values("best_train")
    )

    exps       = df["exp"].tolist()
    best_train = df["best_train"].tolist()
    best_val   = df["best_val"].tolist()

    x     = np.arange(len(exps))
    width = 0.35
    has_val = any(not np.isnan(v) for v in best_val)

    fig, ax = plt.subplots(figsize=(max(8, len(exps) * 1.2), 5))

    bars_train = ax.bar(x - (width / 2 if has_val else 0), best_train, width,
                        label="best train loss", color="steelblue")

    if has_val:
        bars_val = ax.bar(x + width / 2, best_val, width,
                          label="best val loss", color="tomato")
        # value labels
        for bar in bars_val:
            if not np.isnan(bar.get_height()):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7)

    for bar in bars_train:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(exps, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Loss")
    ax.set_title("Best Train / Val Loss per Experiment")
    if has_val:
        ax.legend()
    ax.set_ylim(0, max(best_train + [v for v in best_val if not np.isnan(v)]) * 1.15)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")

    # Print table
    print(df[["exp", "run", "best_train", "best_val"]].to_string(index=False))


if __name__ == "__main__":
    main()
