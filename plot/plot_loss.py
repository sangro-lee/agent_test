#!/usr/bin/env python
"""
Compare training/validation loss across experiments from CSV files.

Usage:
  # All CSVs in current directory
  python plot/plot_loss.py --csv_dir plot/

  # Specific files
  python plot/plot_loss.py --csv_dir plot/ --files mlp_z4.csv ortho_z4.csv vqc_z4_loss.csv

  # Custom output path
  python plot/plot_loss.py --csv_dir plot/ --out outputs/loss_comparison.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_loss_comparison(
    csv_paths: list[Path],
    out_path: Path,
    smooth: int = 1,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_train, ax_val = axes

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        label = csv_path.stem

        if smooth > 1:
            df["train_loss"] = df["train_loss"].rolling(smooth, min_periods=1).mean()
            df["val_loss"]   = df["val_loss"].rolling(smooth, min_periods=1).mean()

        ax_train.plot(df["epoch"], df["train_loss"], label=label)
        ax_val.plot(df["epoch"], df["val_loss"],     label=label)

    for ax, title in zip(axes, ["Train Loss", "Val Loss"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_loss] Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, required=True,
                        help="Directory containing loss CSV files")
    parser.add_argument("--files", nargs="+", default=None,
                        help="Specific CSV filenames to include (default: all *.csv in csv_dir)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG path (default: csv_dir/loss_comparison.png)")
    parser.add_argument("--smooth", type=int, default=1,
                        help="Rolling average window for smoothing (default: 1 = no smoothing)")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    if args.files:
        csv_paths = [csv_dir / f for f in args.files]
    else:
        csv_paths = sorted(csv_dir.glob("*.csv"))

    csv_paths = [p for p in csv_paths if p.exists()]
    if not csv_paths:
        print(f"[plot_loss] No CSV files found in {csv_dir}")
        return

    print(f"[plot_loss] Plotting {len(csv_paths)} file(s): {[p.name for p in csv_paths]}")

    out_path = Path(args.out) if args.out else csv_dir / "loss_comparison.png"
    plot_loss_comparison(csv_paths, out_path, smooth=args.smooth)


if __name__ == "__main__":
    main()
