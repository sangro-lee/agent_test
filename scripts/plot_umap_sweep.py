#!/usr/bin/env python
"""
Run plot_latents.py (UMAP, combined all splits) across multiple (n_neighbors, min_dist).
Output: evaluation/umap_all_n{nn}_d{md}.png

Usage:
  python scripts/plot_umap_sweep.py --run_dir outputs/runs/rgcn_mlp_z4
  python scripts/plot_umap_sweep.py --run_dir outputs/runs/rgcn_mlp_z4 \
      --n_neighbors 10 20 30 50 --min_dist 0.1
"""
from __future__ import annotations

import argparse
import itertools
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run_dir", type=str)
    group.add_argument("--config",  type=str)
    parser.add_argument("--n_neighbors", type=int,   nargs="+", default=[10, 20, 30, 50])
    parser.add_argument("--min_dist",    type=float, nargs="+", default=[0.1],
                        help="Single value applied to all, or one per n_neighbors")
    args = parser.parse_args()

    nn_list = args.n_neighbors
    md_list = args.min_dist

    if args.run_dir:
        run_dir = Path(args.run_dir)
        loc_args = ["--run_dir", args.run_dir]
    else:
        sys.path.insert(0, str(ROOT))
        from src.utils.config import parse_config_args
        from src.utils.io import resolve_run_dir
        cfg, _ = parse_config_args("sweep", argv=["--config", args.config])
        run_dir = Path(resolve_run_dir(cfg, create_if_missing=False))
        loc_args = ["--config", args.config]

    eval_dir = run_dir / "evaluation"

    base_cmd = [sys.executable, str(ROOT / "scripts" / "plot_latents.py"),
                *loc_args, "--no_tsne", "--splits", "train", "test"]

    for nn, md in itertools.product(nn_list, md_list):
        tag = f"n{nn}_d{md}"
        print(f"\n[sweep] n_neighbors={nn}  min_dist={md}")
        subprocess.run(base_cmd + ["--umap_n_neighbors", str(nn),
                                   "--umap_min_dist",    str(md)], check=True)

        src = eval_dir / "umap_all.png"
        dst = eval_dir / f"umap_all_{tag}.png"
        if src.exists():
            shutil.move(str(src), str(dst))
            print(f"  → {dst.name}")


if __name__ == "__main__":
    main()
