#!/usr/bin/env python
"""
Encoder top-k baseline: rank molecules by pred_pIC50 and compute metrics.

Two input modes:
  1. Test set  : --run_dir (reads predictions/test_preds.csv)
  2. Screening : --screening_preds + --screening_pic50 (npy from encode_screening.py)

Usage:
  # test set
  python scripts/eval_encoder_topk.py --run_dir outputs/runs/sme_random

  # screening library
  python scripts/eval_encoder_topk.py \\
      --screening_preds screening/sub_4/preds_screening.npy \\
      --screening_pic50 screening/sub_4/y_screening.npy \\
      --out             outputs/runs/sme_random/screening/sub_4/encoder_topk.csv

  # both
  python scripts/eval_encoder_topk.py \\
      --run_dir         outputs/runs/sme_random \\
      --screening_preds screening/sub_4/preds_screening.npy \\
      --screening_pic50 screening/sub_4/y_screening.npy
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def eval_topk(y_true: np.ndarray, y_pred: np.ndarray,
              top_k: int, threshold: float) -> dict:
    idx = np.argsort(y_pred)[::-1][:top_k]
    top_true = y_true[idx]
    return {
        "top_k":       top_k,
        "n":           len(idx),
        "mean_actual": float(top_true.mean()),
        "max_actual":  float(top_true.max()),
        "hit_rate":    float((top_true >= threshold).mean()),
        "mean_pred":   float(y_pred[idx].mean()),
    }


def print_and_collect(y_true, y_pred, top_ks, threshold, label):
    print(f"\n[{label}] {len(y_true)} molecules  threshold={threshold}")
    print(f"\n{'top_k':>6}  {'n':>4}  {'mean_actual':>11}  {'max_actual':>10}  "
          f"{'hit_rate':>8}  {'mean_pred':>9}")
    print("-" * 60)
    rows = []
    for k in top_ks:
        m = eval_topk(y_true, y_pred, min(k, len(y_true)), threshold)
        m["source"] = label
        rows.append(m)
        print(f"{m['top_k']:>6}  {m['n']:>4}  {m['mean_actual']:>11.4f}  "
              f"{m['max_actual']:>10.4f}  {m['hit_rate']:>8.2%}  {m['mean_pred']:>9.4f}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir",          type=str, default=None,
                        help="Run directory (reads predictions/test_preds.csv)")
    parser.add_argument("--screening_preds",  type=str, default=None,
                        help="preds_screening.npy from encode_screening.py")
    parser.add_argument("--screening_pic50",  type=str, default=None,
                        help="y_screening.npy from encode_screening.py")
    parser.add_argument("--screening_name",   type=str, default="screening",
                        help="Label for screening source (default: screening)")
    parser.add_argument("--top_k",            type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--threshold",        type=float, default=7.0)
    parser.add_argument("--out",              type=str, default=None,
                        help="Output CSV path (default: auto)")
    args = parser.parse_args()

    if args.run_dir is None and args.screening_preds is None:
        raise ValueError("Provide --run_dir and/or --screening_preds")

    all_rows = []

    # ── Test set ──────────────────────────────────────────────────────────
    if args.run_dir:
        run_dir  = Path(args.run_dir)
        csv_path = run_dir / "predictions" / "test_preds.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Not found: {csv_path}")
        df = pd.read_csv(csv_path)
        rows = print_and_collect(
            df["y_true"].values, df["y_pred"].values,
            args.top_k, args.threshold, "test",
        )
        all_rows.extend(rows)

    # ── Screening ─────────────────────────────────────────────────────────
    if args.screening_preds:
        if not args.screening_pic50:
            raise ValueError("--screening_pic50 required with --screening_preds")
        y_pred = np.load(args.screening_preds).astype(np.float32).ravel()
        y_true = np.load(args.screening_pic50).astype(np.float32).ravel()
        rows = print_and_collect(
            y_true, y_pred,
            args.top_k, args.threshold, args.screening_name,
        )
        all_rows.extend(rows)

    # ── Save ──────────────────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
    elif args.run_dir:
        out_path = Path(args.run_dir) / "predictions" / "encoder_topk.csv"
    else:
        out_path = Path(args.screening_preds).parent / "encoder_topk.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
