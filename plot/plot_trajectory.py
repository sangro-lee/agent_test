#!/usr/bin/env python
"""
Plot denoising trajectory in UMAP space.

Loads z_trajectory.npy (shape: n_checkpoints × n_samples × latent_dim),
projects each checkpoint through a UMAP fitted on train latents,
and draws sample paths from noise (t=T) to final (t=0).

Usage:
  python plot/plot_trajectory.py \
      --run_dir outputs/runs/rgcn_mlp_z4 \
      --traj outputs/runs/rgcn_mlp_z4/diffusion/T1000_ep200/cfg_w3.0_ddim/z_trajectory.npy

  # limit to first 30 samples for clarity
  python plot/plot_trajectory.py \
      --run_dir outputs/runs/rgcn_mlp_z4 \
      --traj .../z_trajectory.npy \
      --n_show 30 --out outputs/traj.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.plots import make_umap_reducer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir",  type=str, required=True,
                        help="Run directory (for latents_train.npy)")
    parser.add_argument("--traj",     type=str, required=True,
                        help="Path to z_trajectory.npy")
    parser.add_argument("--n_show",   type=int, default=50,
                        help="Number of sample trajectories to draw (default: 50)")
    parser.add_argument("--n_neighbors", type=int,   default=30)
    parser.add_argument("--min_dist",    type=float, default=0.3)
    parser.add_argument("--out",      type=str, default=None,
                        help="Output PNG path (default: same dir as traj)")
    args = parser.parse_args()

    run_dir  = Path(args.run_dir)
    traj_path = Path(args.traj)

    # ── Load ──────────────────────────────────────────────────────────────
    trajectory = np.load(traj_path).astype(np.float32)
    # shape: (n_checkpoints, n_samples, latent_dim)
    n_checkpoints, n_samples, latent_dim = trajectory.shape
    print(f"[plot_trajectory] trajectory: {trajectory.shape}")

    z_train_path = run_dir / "latents_train.npy"
    if not z_train_path.exists():
        print(f"[plot_trajectory] latents_train.npy not found in {run_dir}")
        sys.exit(1)
    z_train = np.load(z_train_path).astype(np.float32)
    y_train_path = run_dir / "y_train.npy"
    y_train = np.load(y_train_path).astype(np.float32) if y_train_path.exists() else np.zeros(len(z_train))

    # ── Fit UMAP on train ─────────────────────────────────────────────────
    print(f"[plot_trajectory] Fitting UMAP on train (n={len(z_train)})...")
    reducer = make_umap_reducer(z_train, n_neighbors=args.n_neighbors, min_dist=args.min_dist)
    if reducer is None:
        print("[plot_trajectory] umap-learn not installed. Run: pip install umap-learn")
        sys.exit(1)

    emb_train = reducer.transform(z_train)

    # ── Project all trajectory checkpoints ───────────────────────────────
    n_show = min(args.n_show, n_samples)
    traj_sub = trajectory[:, :n_show, :]  # (n_checkpoints, n_show, latent_dim)

    print(f"[plot_trajectory] Projecting {n_checkpoints} checkpoints × {n_show} samples...")
    # Stack all frames for batch transform
    flat = traj_sub.reshape(-1, latent_dim)           # (n_checkpoints*n_show, latent_dim)
    emb_flat = reducer.transform(flat)                # (n_checkpoints*n_show, 2)
    emb_traj = emb_flat.reshape(n_checkpoints, n_show, 2)  # (n_checkpoints, n_show, 2)

    # ── Plot ──────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=(8, 6))

    # Train background
    sc = ax.scatter(emb_train[:, 0], emb_train[:, 1],
                    c=y_train, cmap="Blues", s=5, alpha=0.3,
                    vmin=y_train.min(), vmax=y_train.max(), zorder=1)
    fig.colorbar(sc, ax=ax, location="right", pad=0.01, label="pIC50 train")

    # Draw trajectories: color from blue (noise) → red (final)
    cmap = plt.get_cmap("cool")
    for i in range(n_show):
        pts = emb_traj[:, i, :]          # (n_checkpoints, 2)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)  # (n_checkpoints-1, 2, 2)
        colors = [cmap(k / max(n_checkpoints - 1, 1)) for k in range(n_checkpoints - 1)]
        lc = LineCollection(segs, colors=colors, linewidths=0.7, alpha=0.6, zorder=2)
        ax.add_collection(lc)

    # Mark start (noise) and end (final) points
    ax.scatter(emb_traj[0, :, 0],  emb_traj[0, :, 1],
               s=20, c="cyan",  alpha=0.7, zorder=3, label=f"start (noise, n={n_show})")
    ax.scatter(emb_traj[-1, :, 0], emb_traj[-1, :, 1],
               s=20, c="red",   alpha=0.9, zorder=4, label=f"end (sampled, n={n_show})")

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"Denoising Trajectory — {traj_path.parent.name}\n"
                 f"({n_checkpoints} checkpoints, {n_show} samples shown)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = traj_path.parent / "umap_trajectory.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_trajectory] Saved: {out_path}")


if __name__ == "__main__":
    main()
