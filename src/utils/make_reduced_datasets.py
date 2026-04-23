"""
Split a cleaned CSV into stratified reduced subsets for learning curve experiments.

For fraction p, creates n = round(1/p) independent subsets saved as CSV files.
Each subset contains p fraction of the data, stratified by pIC50.

Output structure:
    data/BACE1/reduced/
        frac_0.20/
            subset_0.csv
            subset_1.csv
            ...  (5 total)
        frac_0.10/
            ...  (10 total)

Usage:
    python scripts/make_reduced_datasets.py \\
        --csv data/BACE1/bace1_clean_pic50.csv \\
        --frac 0.2 \\
        [--out_dir data/BACE1/reduced] [--n_bins 4] [--seed 0]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def stratified_sample(
    df: pd.DataFrame,
    pIC50_col: str,
    frac: float,
    n_bins: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bins = pd.qcut(df[pIC50_col], q=n_bins, labels=False, duplicates="drop")
    sampled_idx = []
    for b in np.unique(bins):
        group = df.index[bins == b].tolist()
        k = max(1, round(len(group) * frac))
        chosen = rng.choice(group, size=min(k, len(group)), replace=False)
        sampled_idx.extend(chosen.tolist())
    return df.loc[sorted(sampled_idx)].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input CSV (e.g. data/BACE1/bace1_clean_pic50.csv)")
    parser.add_argument("--frac", type=float, required=True,
                        help="Fraction to keep per subset (e.g. 0.2 → 5 subsets)")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: same folder as input CSV / reduced)")
    parser.add_argument("--n_bins", type=int, default=4,
                        help="pIC50 quantile bins for stratification (default: 4)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed; subset_i uses seed+i (default: 0)")
    parser.add_argument("--pIC50_col", default="pIC50",
                        help="Column name for pIC50 values (default: pIC50)")
    args = parser.parse_args()

    if not (0 < args.frac < 1):
        raise ValueError(f"--frac must be in (0, 1), got {args.frac}")

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent / "reduced"
    frac_dir = out_dir / f"frac_{args.frac:.2f}"
    frac_dir.mkdir(parents=True, exist_ok=True)

    n_subsets = round(1.0 / args.frac)
    print(f"Fraction {args.frac} → {n_subsets} subsets (~{round(len(df) * args.frac)} rows each)")
    print(f"Output: {frac_dir}\n")

    for i in range(n_subsets):
        subset = stratified_sample(df, args.pIC50_col, args.frac, args.n_bins, seed=args.seed + i)
        out_path = frac_dir / f"subset_{i}.csv"
        subset.to_csv(out_path, index=False)
        print(f"  subset_{i}.csv: {len(subset)} rows  (seed={args.seed + i})")

    print(f"\nDone. Point raw_csv in config to e.g. {frac_dir}/subset_0.csv")


if __name__ == "__main__":
    main()
