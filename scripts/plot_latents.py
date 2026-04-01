#!/usr/bin/env python
"""Plot t-SNE and UMAP from saved latents (no model re-run needed).

Usage:
  python scripts/plot_latents.py --config configs/experiments/rgcn_mlp_z4.yaml
  python scripts/plot_latents.py --run_dir outputs/runs/rgcn_mlp_z4
  python scripts/plot_latents.py --run_dir outputs/runs/rgcn_mlp_z4 --no_tsne
  python scripts/plot_latents.py --run_dir outputs/runs/rgcn_mlp_z4 --no_umap
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.plots import plot_latent_tsne, plot_latent_umap


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config",  type=str, help="Config YAML path (to resolve run_dir)")
    group.add_argument("--run_dir", type=str, help="Run directory containing latents_*.npy")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Splits to plot (default: train val test)")
    parser.add_argument("--no_tsne",        action="store_true")
    parser.add_argument("--no_umap",        action="store_true")
    parser.add_argument("--umap_n_neighbors", type=int, default=15)
    args = parser.parse_args()

    if args.config:
        from src.utils.config import parse_config_args
        from src.utils.io import resolve_run_dir
        cfg, _ = parse_config_args("plot_latents", argv=["--config", args.config])
        run_dir = Path(resolve_run_dir(cfg, create_if_missing=False))
    else:
        run_dir = Path(args.run_dir)

    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    all_latents, all_y = [], []
    for split in args.splits:
        z_path = run_dir / f"latents_{split}.npy"
        y_path = run_dir / f"y_{split}.npy"
        if not z_path.exists():
            print(f"[plot_latents] {z_path.name} not found, skipping {split}.")
            continue
        latents = np.load(z_path).astype(np.float32)
        y = np.load(y_path).astype(np.float32) if y_path.exists() else np.zeros(len(latents))

        print(f"[plot_latents] {split}: {latents.shape}")
        if not args.no_tsne:
            plot_latent_tsne(latents, y, eval_dir / f"tsne_{split}.png", split_name=split)
            print(f"  Saved: tsne_{split}.png")
        if not args.no_umap:
            plot_latent_umap(latents, y, eval_dir / f"umap_{split}.png",
                             split_name=split, n_neighbors=args.umap_n_neighbors)
            print(f"  Saved: umap_{split}.png")

        all_latents.append(latents)
        all_y.append(y)

    if len(all_latents) > 1:
        combined = np.concatenate(all_latents)
        combined_y = np.concatenate(all_y)
        print(f"[plot_latents] combined: {combined.shape}")
        if not args.no_tsne:
            plot_latent_tsne(combined, combined_y, eval_dir / "tsne_all.png", split_name="all splits")
            print("  Saved: tsne_all.png")
        if not args.no_umap:
            plot_latent_umap(combined, combined_y, eval_dir / "umap_all.png",
                             split_name="all splits", n_neighbors=args.umap_n_neighbors)
            print("  Saved: umap_all.png")


if __name__ == "__main__":
    main()
