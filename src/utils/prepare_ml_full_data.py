import argparse
import os
import pandas as pd
import numpy as np

FILE_PATH = "./화합물정리_0423_malachite_extracted_sm.csv"
BASE_DIR  = "/home/sangro/git/agent_test/data"


def undersample(df, keep_neg_mid, keep_neg, keep_low_pos, seed):
    mask_neg_mid = (df["y"] >= -20) & (df["y"] < -10)
    mask_neg     = (df["y"] >= -10) & (df["y"] <   0)
    mask_low_pos = (df["y"] >=   0) & (df["y"] <  10)
    mask_other   = ~(mask_neg_mid | mask_neg | mask_low_pos)

    rng = np.random.default_rng(seed)

    def sample_group(sub, n):
        if len(sub) <= n:
            return sub
        return sub.iloc[rng.choice(len(sub), size=n, replace=False)]

    return pd.concat([
        sample_group(df[mask_neg_mid], keep_neg_mid),
        sample_group(df[mask_neg],     keep_neg),
        sample_group(df[mask_low_pos], keep_low_pos),
        df[mask_other],
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)


def main(args):
    df = pd.read_csv(FILE_PATH)
    print(f"원본 데이터: {len(df)}개")

    has_smiles = df[df["SMILES"].notna() & (df["SMILES"].astype(str).str.strip() != "nan")].copy()
    has_smiles = has_smiles.dropna(subset=["Malachite green assay_50uM (%)"]).reset_index(drop=True)
    has_smiles = has_smiles.rename(columns={"Malachite green assay_50uM (%)": "y"})
    print(f"유효 데이터: {len(has_smiles)}개")

    before = len(has_smiles)
    has_smiles = undersample(has_smiles, args.keep_neg_mid, args.keep_neg, args.keep_low_pos, args.seed)
    print(f"Undersampling (seed={args.seed}): {before}개 → {len(has_smiles)}개")
    print(f"  -20~-10: 최대 {args.keep_neg_mid}개")
    print(f"   -10~0 : 최대 {args.keep_neg}개")
    print(f"    0~10 : 최대 {args.keep_low_pos}개")

    # 10% 단위 분포
    print("\n구간별 분포:")
    bins10 = list(range(int(has_smiles["y"].min())//10*10 - 10,
                        int(has_smiles["y"].max())//10*10 + 20, 10))
    for i in range(len(bins10) - 1):
        lo, hi = bins10[i], bins10[i+1]
        n = ((has_smiles["y"] >= lo) & (has_smiles["y"] < hi)).sum()
        if n > 0:
            print(f"  {lo:>4}~{hi:<4}: {n:4d}개  {'█' * min(n // 3, 20)}")

    out_path = os.path.join(
        BASE_DIR,
        f"ml_full_under_{args.keep_neg_mid}_{args.keep_neg}_{args.keep_low_pos}_seed{args.seed}.csv"
    )
    has_smiles.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep_neg_mid",  type=int, default=50, help="-20~-10 (default: 50)")
    parser.add_argument("--keep_neg",      type=int, default=50, help="-10~0   (default: 50)")
    parser.add_argument("--keep_low_pos",  type=int, default=50, help="0~10    (default: 50)")
    parser.add_argument("--seed",          type=int, default=42, help="random seed (default: 42)")
    args = parser.parse_args()

    main(args)
