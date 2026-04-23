"""
Split a cleaned CSV into non-overlapping stratified subsets (k-fold style).

For fraction p, creates k = round(1/p) non-overlapping subsets.
Data is shuffled once, then partitioned bin-by-bin so each subset
gets an equal share of each pIC50 quantile. Subsets together cover
the full dataset with no duplicates.

Output structure:
    data/BACE1/reduced/
        frac_0.20/
            subset_0.csv  (~20% of data, no overlap with others)
            subset_1.csv
            ...  (5 total)

Usage:
    python -m src.utils.make_reduced_datasets \\
        --csv data/BACE1/bace1_clean_pic50.csv \\
        --frac 0.2 \\
        [--out_dir data/BACE1/reduced] [--n_bins 4] [--seed 0]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def stratified_kfold_split(
    df: pd.DataFrame,
    pIC50_col: str,
    k: int,
    n_bins: int,
    seed: int,
) -> list[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    bins = pd.qcut(df[pIC50_col], q=n_bins, labels=False, duplicates="drop")

    fold_indices: list[list] = [[] for _ in range(k)]
    for b in np.unique(bins):
        group = df.index[bins == b].tolist()
        rng.shuffle(group)
        for i, idx in enumerate(group):
            fold_indices[i % k].append(idx)

    return [df.loc[sorted(idxs)].reset_index(drop=True) for idxs in fold_indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True,
                        help="Input CSV (e.g. data/BACE1/bace1_clean_pic50.csv)")
    parser.add_argument("--frac", type=float, required=True,
                        help="Fraction per subset (e.g. 0.2 → 5 non-overlapping subsets)")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: <csv parent>/reduced)")
    parser.add_argument("--n_bins", type=int, default=4,
                        help="pIC50 quantile bins for stratification (default: 4)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Shuffle seed (default: 0)")
    parser.add_argument("--pIC50_col", default="pIC50",
                        help="pIC50 column name (default: pIC50)")
    args = parser.parse_args()

    if not (0 < args.frac < 1):
        raise ValueError(f"--frac must be in (0, 1), got {args.frac}")

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    k = round(1.0 / args.frac)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent / "reduced"
    frac_dir = out_dir / f"frac_{args.frac:.2f}"
    frac_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fraction {args.frac} → {k} non-overlapping subsets (~{len(df)//k} rows each)")
    print(f"Output: {frac_dir}\n")

    subsets = stratified_kfold_split(df, args.pIC50_col, k, args.n_bins, args.seed)
    for i, subset in enumerate(subsets):
        out_path = frac_dir / f"subset_{i}.csv"
        subset.to_csv(out_path, index=False)
        print(f"  subset_{i}.csv: {len(subset)} rows")

    total = sum(len(s) for s in subsets)
    print(f"\nTotal rows across subsets: {total} (original: {len(df)})")
    print(f"Done. Point raw_csv in config to e.g. {frac_dir}/subset_0.csv")


if __name__ == "__main__":
    main()
