#!/usr/bin/env python
"""
Encoder top-k baseline: rank test set by pred_pIC50 and compute mean actual pIC50.

Compares encoder direct ranking vs diffusion retrieval as a sanity check.

Usage:
  python scripts/eval_encoder_topk.py --run_dir outputs/runs/sme_random
  python scripts/eval_encoder_topk.py --run_dir outputs/runs/sme_random --top_k 30 50 100
"""
import argparse
from pathlib import Path

import pandas as pd


def eval_topk(df: pd.DataFrame, top_k: int, threshold: float) -> dict:
    top = df.nlargest(top_k, "y_pred")
    return {
        "top_k":        top_k,
        "mean_actual":  float(top["y_true"].mean()),
        "max_actual":   float(top["y_true"].max()),
        "hit_rate":     float((top["y_true"] >= threshold).mean()),
        "mean_pred":    float(top["y_pred"].mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir",   type=str, required=True)
    parser.add_argument("--top_k",     type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--threshold", type=float, default=7.0)
    args = parser.parse_args()

    run_dir  = Path(args.run_dir)
    csv_path = run_dir / "predictions" / "test_preds.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[test set] {len(df)} molecules  threshold={args.threshold}")
    print(f"\n{'top_k':>6}  {'mean_actual':>11}  {'max_actual':>10}  {'hit_rate':>8}  {'mean_pred':>9}")
    print("-" * 55)

    rows = []
    for k in args.top_k:
        k = min(k, len(df))
        m = eval_topk(df, k, args.threshold)
        rows.append(m)
        print(f"{m['top_k']:>6}  {m['mean_actual']:>11.4f}  {m['max_actual']:>10.4f}  "
              f"{m['hit_rate']:>8.2%}  {m['mean_pred']:>9.4f}")

    out_path = run_dir / "predictions" / "encoder_topk.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
