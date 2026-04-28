#!/usr/bin/env python
"""
Check VQC angle encoding aliasing in latent space.

AngleEmbedding: RY(z_i) per qubit — period 2π.
If span(z[:, d]) > 2π for any dimension d, different molecules may produce
identical quantum states, making them indistinguishable to the denoiser.

Usage:
  python -m src.utils.check_vqc_aliasing --run_dir outputs/runs/sme_vqc_reupload_z4_random
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TWO_PI = 2 * math.pi


def check_aliasing(
    z: np.ndarray,
    y: np.ndarray,
    pic50_threshold: float = 0.5,
    angle_threshold: float = 0.1,
    out_dir: Path | None = None,
    split: str = "train",
) -> None:
    n_mols, latent_dim = z.shape
    print(f"\n[{split}] {n_mols} molecules, latent_dim={latent_dim}  (2π = {TWO_PI:.4f})")
    print(f"\n{'Dim':>4}  {'min':>8}  {'max':>8}  {'span':>8}  status")
    print("-" * 50)

    aliased_dims = []
    for d in range(latent_dim):
        z_min  = float(z[:, d].min())
        z_max  = float(z[:, d].max())
        span   = z_max - z_min
        status = "⚠ ALIASED (span > 2π)" if span > TWO_PI else "OK"
        print(f"  {d:>2}  {z_min:>8.4f}  {z_max:>8.4f}  {span:>8.4f}  {status}")
        if span > TWO_PI:
            aliased_dims.append(d)

    if not aliased_dims or out_dir is None:
        return

    # For each aliased dim: find molecule pairs that are close in wrapped angle
    # but have different pIC50 — these are the "confused" pairs for the denoiser
    for d in aliased_dims:
        wrapped = ((z[:, d] + math.pi) % TWO_PI) - math.pi  # map to [-π, π]

        angle_diff = np.abs(wrapped[:, None] - wrapped[None, :])
        angle_diff = np.minimum(angle_diff, TWO_PI - angle_diff)  # circular
        pic50_diff = np.abs(y[:, None] - y[None, :])

        mask = (angle_diff < angle_threshold) & (pic50_diff > pic50_threshold)
        np.fill_diagonal(mask, False)
        rows, cols = np.where(mask)
        n_pairs = sum(1 for i, j in zip(rows, cols) if i < j)

        print(f"\n[dim {d}] aliased pairs (|Δangle|<{angle_threshold}, |ΔpIC50|>{pic50_threshold}): {n_pairs}")
        for i, j in zip(rows[:10], cols[:10]):
            if i < j:
                print(f"  mol {i:4d} pIC50={y[i]:.2f} angle={wrapped[i]:.3f} ↔ "
                      f"mol {j:4d} pIC50={y[j]:.2f} angle={wrapped[j]:.3f}  "
                      f"Δangle={angle_diff[i,j]:.4f} ΔpIC50={pic50_diff[i,j]:.2f}")

        # scatter: wrapped angle vs pIC50
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(wrapped, y, c=y, cmap="coolwarm", s=8, alpha=0.6)
        plt.colorbar(sc, ax=ax, label="pIC50")
        ax.set_xlabel(f"z[{d}] wrapped to [-π, π]  (raw span={z[:,d].max()-z[:,d].min():.2f})")
        ax.set_ylabel("pIC50")
        ax.set_title(f"VQC aliasing — {split} set, dim {d}")
        ax.axvline(-math.pi, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline( math.pi, color="gray", linestyle="--", linewidth=0.8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = out_dir / f"aliasing_{split}_dim{d}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir",         type=str, required=True)
    parser.add_argument("--out",             type=str, default=None)
    parser.add_argument("--pic50_threshold", type=float, default=0.5)
    parser.add_argument("--angle_threshold", type=float, default=0.1)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir / "aliasing_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    df        = pd.read_csv(run_dir / "cleaned_dataset.csv")
    label_col = "pIC50"

    for split in ("train", "val", "test"):
        idx_path = run_dir / "splits" / f"{split}_idx.npy"
        z_path   = run_dir / f"latents_{split}.npy"
        if not idx_path.exists() or not z_path.exists():
            continue
        idx = np.load(idx_path)
        z   = np.load(z_path).astype(np.float32)
        y   = df[label_col].values[idx].astype(np.float32)
        check_aliasing(z, y,
                       pic50_threshold=args.pic50_threshold,
                       angle_threshold=args.angle_threshold,
                       out_dir=out_dir,
                       split=split)


if __name__ == "__main__":
    main()
