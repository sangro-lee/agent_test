#!/usr/bin/env python
"""
Plot UMAP for a single experiment: train/test background + sampled latents overlay.

Usage:
  # Fit UMAP on train latents and plot
  python scripts/evaluate_plot.py --run_dir outputs/runs/rgcn_mlp_z4

  # Load pre-fitted reducer (fixed coordinate frame)
  python scripts/evaluate_plot.py --run_dir outputs/runs/rgcn_vqc_z4 \\
      --load_reducer outputs/umap_reducer.pkl

  # Save fitted reducer for reuse
  python scripts/evaluate_plot.py --run_dir outputs/runs/rgcn_mlp_z4 \\
      --save_reducer outputs/umap_reducer.pkl

  # Specify a z_samples.npy explicitly
  python scripts/evaluate_plot.py --run_dir outputs/runs/rgcn_mlp_z4 \\
      --z_samples outputs/runs/rgcn_mlp_z4/diffusion/T1000_ep200/cfg_w3.0_ddim/z_samples.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.plots import make_umap_reducer


def _find_z_samples(run_dir: Path) -> list[Path]:
    """Auto-detect all z_samples.npy under diffusion/."""
    return sorted(run_dir.glob("diffusion/**/z_samples.npy"))


def plot_umap_sampled(
    run_dir: Path,
    z_samples_path: Path,
    out_path: Path,
    reducer=None,
    n_neighbors: int = 30,
    min_dist: float = 0.3,
) -> object:
    """Plot UMAP with train/test background and sampled latents overlay.

    Returns the fitted reducer (for saving/reuse).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Load background latents ───────────────────────────────────────────
    z_train_path = run_dir / "latents_train.npy"
    z_test_path  = run_dir / "latents_test.npy"
    y_train_path = run_dir / "y_train.npy"

    if not z_train_path.exists():
        raise FileNotFoundError(f"latents_train.npy not found in {run_dir}. Run evaluate.py first.")

    z_train = np.load(z_train_path).astype(np.float32)
    z_test  = np.load(z_test_path).astype(np.float32)  if z_test_path.exists()  else None
    y_train = np.load(y_train_path).astype(np.float32) if y_train_path.exists() else np.zeros(len(z_train))
    y_test_path = run_dir / "y_test.npy"
    y_test  = np.load(y_test_path).astype(np.float32)  if (z_test is not None and y_test_path.exists()) else None

    # ── Load sampled latents ──────────────────────────────────────────────
    z_samples = np.load(z_samples_path).astype(np.float32)
    print(f"[evaluate_plot] z_samples: {z_samples.shape}  from {z_samples_path.parent.name}")

    # ── Load retrieved test SMILES (from top_candidates_test.csv) ─────────
    retrieved_test_idx: list[int] = []
    smiles_test: list[str] = []
    _cand_path = z_samples_path.parent / "top_candidates_test.csv"
    _smiles_test_path = run_dir / "smiles_test.npy"
    if _cand_path.exists() and _smiles_test_path.exists():
        import pandas as pd
        retrieved_smiles = set(pd.read_csv(_cand_path)["smiles"].astype(str).tolist())
        smiles_test = list(np.load(_smiles_test_path, allow_pickle=True).astype(str))
        retrieved_test_idx = [i for i, s in enumerate(smiles_test) if s in retrieved_smiles]
        print(f"[evaluate_plot] retrieved test: {len(retrieved_test_idx)} molecules highlighted")

    # ── Fit or use provided reducer ───────────────────────────────────────
    if reducer is None:
        print(f"[evaluate_plot] Fitting UMAP on train (n={len(z_train)})...")
        reducer = make_umap_reducer(z_train, n_neighbors=n_neighbors, min_dist=min_dist)
        if reducer is None:
            raise ImportError("umap-learn not installed. Run: pip install umap-learn")

    emb_train   = reducer.transform(z_train)
    emb_test    = reducer.transform(z_test)    if z_test    is not None else None
    emb_samples = reducer.transform(z_samples)

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))

    # background: train colored by pIC50 (Blues)
    sc = ax.scatter(emb_train[:, 0], emb_train[:, 1],
                    c=y_train, cmap="Blues", s=6, alpha=0.4,
                    vmin=y_train.min(), vmax=y_train.max(),
                    label=f"train ({len(z_train)})", zorder=1)

    # test colored by pIC50 (Oranges) if available, else fixed orange
    if emb_test is not None:
        if y_test is not None:
            ax.scatter(emb_test[:, 0], emb_test[:, 1],
                       c=y_test, cmap="Oranges", s=10, alpha=0.6,
                       vmin=y_train.min(), vmax=y_train.max(),
                       label=f"test ({len(z_test)})", zorder=2)
        else:
            ax.scatter(emb_test[:, 0], emb_test[:, 1],
                       s=10, alpha=0.5, c="orange", label=f"test ({len(z_test)})", zorder=2)
        # retrieved test molecules (gold stars)
        if retrieved_test_idx:
            emb_retrieved = emb_test[retrieved_test_idx]
            ax.scatter(emb_retrieved[:, 0], emb_retrieved[:, 1],
                       s=80, alpha=1.0, c="gold", marker="*",
                       label=f"retrieved test ({len(retrieved_test_idx)})", zorder=5)

    # sampled latents
    ax.scatter(emb_samples[:, 0], emb_samples[:, 1],
               s=18, alpha=0.8, c="tomato", label=f"sampled ({len(z_samples)})", zorder=3)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("pIC50 (Blues=train / Oranges=test)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"UMAP — {run_dir.name}\n{z_samples_path.parent.name}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate_plot] Saved: {out_path}")

    return reducer


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run_dir", type=str, help="Experiment run directory")
    group.add_argument("--config",  type=str, help="Config YAML path")

    parser.add_argument("--z_samples",    type=str, default=None,
                        help="Path to z_samples.npy. If omitted, auto-detect all under diffusion/")
    parser.add_argument("--load_reducer", type=str, default=None,
                        help="Path to pre-fitted UMAP reducer (.pkl)")
    parser.add_argument("--save_reducer", type=str, default=None,
                        help="Path to save fitted UMAP reducer (.pkl)")
    parser.add_argument("--out",          type=str, default=None,
                        help="Output PNG path. Default: run_dir/evaluation/umap_sampled_<tag>.png")
    parser.add_argument("--n_neighbors",  type=int,   default=30)
    parser.add_argument("--min_dist",     type=float, default=0.3)
    args = parser.parse_args()

    # ── Resolve run_dir ───────────────────────────────────────────────────
    if args.config:
        from src.utils.config import parse_config_args
        from src.utils.io import resolve_run_dir
        cfg, _ = parse_config_args("evaluate_plot", argv=["--config", args.config])
        run_dir = Path(resolve_run_dir(cfg, create_if_missing=False))
    else:
        run_dir = Path(args.run_dir)

    # ── Load reducer if provided ──────────────────────────────────────────
    reducer = None
    if args.load_reducer:
        import joblib
        reducer = joblib.load(args.load_reducer)
        print(f"[evaluate_plot] Loaded reducer: {args.load_reducer}")

    # ── Resolve z_samples paths ───────────────────────────────────────────
    if args.z_samples:
        z_paths = [Path(args.z_samples)]
    else:
        z_paths = _find_z_samples(run_dir)
        if not z_paths:
            print(f"[evaluate_plot] No z_samples.npy found under {run_dir}/diffusion/")
            return
        print(f"[evaluate_plot] Found {len(z_paths)} z_samples file(s).")

    eval_dir = run_dir / "evaluation"

    # ── Plot each z_samples ───────────────────────────────────────────────
    for z_path in z_paths:
        tag = z_path.parent.name  # e.g. cfg_w3.0_ddim
        if args.out:
            out_path = Path(args.out)
        else:
            out_path = eval_dir / f"umap_sampled_{tag}.png"

        fitted = plot_umap_sampled(
            run_dir=run_dir,
            z_samples_path=z_path,
            out_path=out_path,
            reducer=reducer,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
        )
        # Reuse the first fitted reducer for subsequent z_samples
        if reducer is None:
            reducer = fitted

    # ── Save reducer ──────────────────────────────────────────────────────
    if args.save_reducer and reducer is not None:
        import joblib
        Path(args.save_reducer).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(reducer, args.save_reducer)
        print(f"[evaluate_plot] Saved reducer: {args.save_reducer}")


if __name__ == "__main__":
    main()
