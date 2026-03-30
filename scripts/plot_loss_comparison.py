#!/usr/bin/env python
"""Plot train/val loss and gradient norms from loss_history.csv files.

Finds all loss_history.csv under outputs/runs/*/diffusion/**/loss_history.csv,
and saves a loss_curve.png in the same folder as each CSV.

Usage:
  python scripts/plot_loss_comparison.py
  python scripts/plot_loss_comparison.py --runs_dir outputs/runs
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent.parent


def _title(csv_path: Path) -> str:
    # exp_name / date / run_folder
    parts = csv_path.parts
    diff_idx = next((i for i, p in enumerate(parts) if p == "diffusion"), None)
    if diff_idx is not None:
        exp_name = parts[diff_idx - 1]
        run_name = csv_path.parent.name
        return f"{exp_name}  /  {run_name}"
    return csv_path.parent.name


def plot_one(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    has_val      = "val_loss"            in df.columns and df["val_loss"].notna().any()
    has_vqc_grad = "vqc_grad_norm"       in df.columns and df["vqc_grad_norm"].notna().any()
    has_cls_grad = "classical_grad_norm" in df.columns and df["classical_grad_norm"].notna().any()
    has_grad     = has_vqc_grad or has_cls_grad

    epochs = df["epoch"] if "epoch" in df.columns else df.index
    title  = _title(csv_path)

    # ── Loss curve ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, df["train_loss"], label="train loss", color="steelblue", linewidth=1.4)
    if has_val:
        ax.plot(epochs, df["val_loss"], label="val loss", color="tomato",
                linewidth=1.4, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_out = csv_path.parent / "loss_curve.png"
    fig.savefig(loss_out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {loss_out}")

    # ── Gradient curve ────────────────────────────────────────────────────
    if has_grad:
        fig, ax = plt.subplots(figsize=(9, 4))
        if has_vqc_grad:
            ax.plot(epochs, df["vqc_grad_norm"], label="vqc grad",
                    color="mediumpurple", linewidth=1.4)
        if has_cls_grad:
            ax.plot(epochs, df["classical_grad_norm"], label="classical grad",
                    color="darkorange", linewidth=1.4, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient Norm")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        grad_out = csv_path.parent / "grad_curve.png"
        fig.savefig(grad_out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {grad_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default=str(ROOT / "outputs" / "runs"))
    args = parser.parse_args()

    csvs = sorted(Path(args.runs_dir).glob("*/diffusion/**/loss_history.csv"))
    if not csvs:
        print("No loss_history.csv found.")
        sys.exit(1)

    for csv_path in csvs:
        plot_one(csv_path)


if __name__ == "__main__":
    main()
