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
import math
import sys
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
    return {
        "n":           len(valid),
        "mean_actual": float(valid["actual_pIC50"].mean()),
        "max_actual":  float(valid["actual_pIC50"].max()),
        "hit_rate":    float((valid["actual_pIC50"] >= threshold).mean()),
        "mean_pred":   float(valid["pred_pIC50"].mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir",  default=str(ROOT / "outputs" / "runs"))
    parser.add_argument("--threshold", type=float, default=7.0,
                        help="pIC50 threshold for hit_rate (default: 7.0)")
    parser.add_argument("--csv",       default="top_candidates_test.csv",
                        help="CSV filename to look for (default: top_candidates_test.csv)")
    parser.add_argument("--out",       default=str(ROOT / "outputs" / "screening_comparison.png"))
    args = parser.parse_args()

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
        rows.append(m)

    result = pd.DataFrame(rows).sort_values("mean_actual", ascending=False)

    # ── Print table ────────────────────────────────────────────────────────
    print(f"\n{'Experiment':<35} {'n':>4}  {'mean_actual':>11}  {'max_actual':>10}  "
          f"{'hit_rate':>8}  {'mean_pred':>9}")
    print("-" * 85)
    for _, r in result.iterrows():
        print(f"{r['exp']:<35} {int(r['n']) if not math.isnan(r['n']) else 0:>4}  "
              f"{r['mean_actual']:>11.4f}  {r['max_actual']:>10.4f}  "
              f"{r['hit_rate']:>8.2%}  {r['mean_pred']:>9.4f}")

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
    result.drop(columns=["run_tag", "path"]).to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")


if __name__ == "__main__":
    main()
