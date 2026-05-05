#!/usr/bin/env python
"""
Rank screening library by encoder-predicted pIC50 and save top-k.

Expects outputs from encode_screening.py:
  <screening_dir>/latents_screening.npy
  <screening_dir>/smiles_screening.npy
  <screening_dir>/preds_screening.npy

Usage:
  python scripts/top_k_screening.py \
      --screening_dir outputs/runs/MY_RUN/screening/generated_lib \
      --run_dir       outputs/runs/MY_RUN \
      --top_k         1000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screening_dir", required=True,
                        help="Directory with preds_screening.npy and smiles_screening.npy")
    parser.add_argument("--run_dir", default=None,
                        help="Run dir containing y_scaler.json (for denormalization). "
                             "Omit if normalize_y was not used.")
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--out", default=None,
                        help="Output CSV path (default: <screening_dir>/top_k_candidates.csv)")
    args = parser.parse_args()

    scr_dir = Path(args.screening_dir)
    preds  = np.load(scr_dir / "preds_screening.npy")
    smiles = np.load(scr_dir / "smiles_screening.npy", allow_pickle=True).astype(str)

    # Denormalize if y_scaler.json exists
    y_mean, y_std = 0.0, 1.0
    if args.run_dir:
        scaler_path = Path(args.run_dir) / "y_scaler.json"
        if scaler_path.exists():
            with open(scaler_path) as f:
                sc = json.load(f)
            y_mean = float(sc.get("y_mean", sc.get("mean", 0.0)))
            y_std  = float(sc.get("y_std",  sc.get("std",  1.0)))
            if y_std != 1.0:
                preds = preds * y_std + y_mean
                print(f"[top_k] denormalized with y_mean={y_mean:.4f} y_std={y_std:.4f}")

    order = np.argsort(-preds)[:args.top_k]
    df = pd.DataFrame({
        "smiles":     smiles[order],
        "pred_pIC50": preds[order],
    })

    out = Path(args.out) if args.out else scr_dir / "top_k_candidates.csv"
    df.to_csv(out, index=False)
    print(f"[top_k] {len(df)} candidates saved → {out}")
    print(f"[top_k] pred range: {df['pred_pIC50'].min():.3f} ~ {df['pred_pIC50'].max():.3f}")


if __name__ == "__main__":
    main()
