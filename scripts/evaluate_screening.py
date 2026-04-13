#!/usr/bin/env python
"""Compare screening results across diffusion experiments.

Finds all top_candidates_test.csv under outputs/runs/*/diffusion/**/
and computes per-experiment metrics (test-set source only).

Metrics:
  mean_actual  — mean actual_pIC50 of retrieved test candidates
  max_actual   — max  actual_pIC50 of retrieved test candidates
  hit_rate     — fraction with actual_pIC50 >= threshold
  mean_pred    — mean pred_pIC50 (model confidence)
  n            — number of candidates retrieved from test set

Usage:
  python scripts/evaluate_screening.py
  python scripts/evaluate_screening.py --threshold 7.0 --runs_dir outputs/runs
  python scripts/evaluate_screening.py --csv top_candidates.csv  # use combined pool
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent


def _exp_name(csv_path: Path) -> str:
    """Extract experiment name from path: runs/<exp>/diffusion/.../top_candidates_test.csv"""
    parts = csv_path.parts
    diff_idx = next((i for i, p in enumerate(parts) if p == "diffusion"), None)
    if diff_idx is not None:
        return parts[diff_idx - 1]
    return csv_path.parent.parent.name


def _run_tag(csv_path: Path) -> str:
    """Return a short tag: <exp>/<w_tag> for display."""
    parts = csv_path.parts
    diff_idx = next((i for i, p in enumerate(parts) if p == "diffusion"), None)
    if diff_idx is not None:
        exp = parts[diff_idx - 1]
        w_tag = csv_path.parent.name   # e.g. cfg_w3.0_ddim
        return f"{exp}\n{w_tag}"
    return csv_path.parent.name


def _run_root(csv_path: Path) -> Path:
    """Return experiment root dir (parent of 'diffusion' folder)."""
    parts = csv_path.parts
    diff_idx = next((i for i, p in enumerate(parts) if p == "diffusion"), None)
    if diff_idx is not None:
        return Path(*parts[:diff_idx])
    return csv_path.parent.parent.parent


def plot_umap(rows: list[dict], out_dir: Path, n_neighbors: int = 15, min_dist: float = 0.1) -> None:
    """Plot UMAP for each experiment group: train+test as background, each run's sampled latents colored."""
    try:
        import umap as umap_lib
    except ImportError:
        print("[UMAP] umap-learn not installed. Run: pip install umap-learn")
        return

    exp_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        exp_groups[row["exp"]].append(row)

    for exp, runs in exp_groups.items():
        root = _run_root(Path(runs[0]["path"]))
        z_train_path = root / "latents_train.npy"
        z_test_path  = root / "latents_test.npy"
        if not z_train_path.exists():
            print(f"[UMAP] latents_train.npy not found for {exp}, skipping.")
            continue

        z_train = np.load(z_train_path).astype(np.float32)
        z_test  = np.load(z_test_path).astype(np.float32) if z_test_path.exists() else None

        run_samples = []
        for row in runs:
            z_path = Path(row["path"]).parent / "z_samples.npy"
            if z_path.exists():
                run_samples.append((row["run_tag"], np.load(z_path).astype(np.float32)))
        if not run_samples:
            print(f"[UMAP] No z_samples.npy found for {exp}, skipping.")
            continue

        print(f"[UMAP] Fitting on {exp} (train={len(z_train)})...")
        reducer = umap_lib.UMAP(n_components=2, random_state=42, init="random",
                                n_neighbors=min(n_neighbors, len(z_train) - 1), min_dist=min_dist)
        reducer.fit(z_train)

        emb_train = reducer.transform(z_train)
        emb_test  = reducer.transform(z_test) if z_test is not None else None

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(emb_train[:, 0], emb_train[:, 1], s=2, c="lightgray", alpha=0.5,
                   label=f"train ({len(z_train)})", zorder=1)
        if emb_test is not None:
            ax.scatter(emb_test[:, 0], emb_test[:, 1], s=4, c="orange", alpha=0.6,
                       label=f"test ({len(z_test)})", zorder=2)

        colors = plt.cm.tab10(np.linspace(0, 0.9, len(run_samples)))
        for (tag, z_s), color in zip(run_samples, colors):
            emb_s = reducer.transform(z_s)
            ax.scatter(emb_s[:, 0], emb_s[:, 1], s=6, color=color, alpha=0.7,
                       label=tag.replace("\n", "/"), zorder=3)

        ax.set_title(f"UMAP — {exp}")
        ax.legend(fontsize=7, markerscale=2)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()

        umap_out = out_dir / f"umap_{exp}.png"
        fig.savefig(umap_out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {umap_out}")


def compute_metrics(df: pd.DataFrame, threshold: float) -> dict:
    df = df.copy()
    # filter rows with valid actual_pIC50
    valid = df[df["actual_pIC50"].notna() & (df["actual_pIC50"] > 0)]
    if valid.empty:
        return {
            "n": 0,
            "mean_actual": float("nan"),
            "max_actual":  float("nan"),
            "hit_rate":    float("nan"),
            "mean_pred":   float(df["pred_pIC50"].mean()) if len(df) else float("nan"),
        }
    if "cosine_sim" in valid.columns:
        mean_sim = float(valid["cosine_sim"].mean())
    elif "euclidean_dist" in valid.columns:
        mean_sim = float(valid["euclidean_dist"].mean())
    else:
        mean_sim = float("nan")
    return {
        "n":           len(valid),
        "mean_actual": float(valid["actual_pIC50"].mean()),
        "max_actual":  float(valid["actual_pIC50"].max()),
        "hit_rate":    float((valid["actual_pIC50"] >= threshold).mean()),
        "mean_pred":   float(valid["pred_pIC50"].mean()),
        "mean_sim":    mean_sim,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir",  default=str(ROOT / "outputs" / "runs"))
    parser.add_argument("--threshold", type=float, default=7.0,
                        help="pIC50 threshold for hit_rate (default: 7.0)")
    parser.add_argument("--retrieval_metric", type=str, default=None,
                        choices=["cosine", "euclidean"],
                        help="Retrieval metric used during sampling; sets default --csv to "
                             "top_candidates_test_{metric}.csv")
    parser.add_argument("--csv",       default=None,
                        help="CSV filename to look for (default: top_candidates_test_{retrieval_metric}.csv "
                             "or top_candidates_test.csv)")
    parser.add_argument("--out",       default=str(ROOT / "outputs" / "screening_comparison.png"))
    parser.add_argument("--umap",      action="store_true",
                        help="Generate UMAP plots per experiment (requires umap-learn)")
    parser.add_argument("--umap_n_neighbors", type=int,   default=15,
                        help="UMAP n_neighbors (default: 15)")
    parser.add_argument("--umap_min_dist",    type=float, default=0.1,
                        help="UMAP min_dist (default: 0.1)")
    args = parser.parse_args()

    if args.csv is None:
        if args.retrieval_metric:
            args.csv = f"top_candidates_test_{args.retrieval_metric}.csv"
        else:
            args.csv = "top_candidates_test.csv"

    pattern = f"*/diffusion/**/{args.csv}"
    csvs = sorted(Path(args.runs_dir).glob(pattern))
    if not csvs:
        print(f"No {args.csv} found under {args.runs_dir}")
        sys.exit(1)

    rows = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        m = compute_metrics(df, args.threshold)
        m["exp"]     = _exp_name(csv_path)
        m["run_tag"] = _run_tag(csv_path)
        m["path"]    = str(csv_path)
        _div_path = csv_path.parent / "diversity.json"
        if _div_path.exists():
            with open(_div_path) as _f:
                _div = json.load(_f)
            m["topk_ratio"] = _div.get("topk_ratio")   # top-k selected latents
            m["div_ratio"]  = _div.get("ratio")         # all samples (reference)
        else:
            m["topk_ratio"] = None
            m["div_ratio"]  = None
        rows.append(m)

    result = pd.DataFrame(rows).sort_values("mean_actual", ascending=False)

    # ── Print table ────────────────────────────────────────────────────────
    print(f"\n{'Experiment/Run':<45} {'n':>4}  {'mean_actual':>11}  {'max_actual':>10}  "
          f"{'hit_rate':>8}  {'mean_pred':>9}  {'mean_sim':>8}  {'topk_div':>8}  {'all_div':>7}")
    print("-" * 126)
    for _, r in result.iterrows():
        tag = r['run_tag'].replace('\n', '/')
        topk_str = f"{r['topk_ratio']:>8.3f}" if pd.notna(r.get('topk_ratio')) else f"{'N/A':>8}"
        div_str  = f"{r['div_ratio']:>7.3f}"  if pd.notna(r.get('div_ratio'))  else f"{'N/A':>7}"
        sim_str  = f"{r['mean_sim']:>8.4f}"   if pd.notna(r.get('mean_sim'))   else f"{'N/A':>8}"
        print(f"{tag:<45} {int(r['n']) if not math.isnan(r['n']) else 0:>4}  "
              f"{r['mean_actual']:>11.4f}  {r['max_actual']:>10.4f}  "
              f"{r['hit_rate']:>8.2%}  {r['mean_pred']:>9.4f}  {sim_str}  {topk_str}  {div_str}")

    # ── Bar chart ──────────────────────────────────────────────────────────
    valid_rows = result[result["n"] > 0].reset_index(drop=True)
    if valid_rows.empty:
        print("No valid rows to plot.")
        return

    labels = valid_rows["run_tag"].tolist()
    x = np.arange(len(labels))
    width = 0.28

    fig, axes = plt.subplots(1, 3, figsize=(max(10, len(labels) * 2.2), 5))

    # mean / max actual pIC50
    ax = axes[0]
    ax.bar(x - width/2, valid_rows["mean_actual"], width, label="mean actual", color="steelblue")
    ax.bar(x + width/2, valid_rows["max_actual"],  width, label="max actual",  color="tomato")
    ax.axhline(args.threshold, color="gray", linestyle="--", linewidth=0.8, label=f"threshold={args.threshold}")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("pIC50"); ax.set_title("Actual pIC50 (test set)"); ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # hit rate
    ax = axes[1]
    ax.bar(x, valid_rows["hit_rate"] * 100, color="mediumseagreen")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(f"Hit Rate (%) @ pIC50≥{args.threshold}")
    ax.set_title("Hit Rate"); ax.grid(True, alpha=0.3)
    for i, v in enumerate(valid_rows["hit_rate"]):
        ax.text(i, v * 100 + 0.5, f"{v:.1%}", ha="center", va="bottom", fontsize=7)

    # pred pIC50
    ax = axes[2]
    ax.bar(x, valid_rows["mean_pred"], color="mediumpurple")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("mean pred_pIC50"); ax.set_title("Model Confidence"); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Screening Evaluation — test set (threshold={args.threshold})", fontsize=10)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {args.out}")

    # Save summary CSV
    csv_out = Path(args.out).with_suffix(".csv")
    result_csv = result.copy()
    result_csv["run_tag"] = result_csv["run_tag"].str.replace('\n', '/', regex=False)
    result_csv.drop(columns=["path"]).to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")

    # ── UMAP ───────────────────────────────────────────────────────────────
    if args.umap:
        plot_umap(rows, Path(args.out).parent, n_neighbors=args.umap_n_neighbors,
                  min_dist=args.umap_min_dist)


if __name__ == "__main__":
    main()
