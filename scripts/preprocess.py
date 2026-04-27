#!/usr/bin/env python
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.cleaner import clean_activity_dataframe
from src.data.loader import load_activity_csv
from src.data.splitter import random_split, scaffold_split
from src.utils.config import parse_config_args, save_config
from src.utils.io import create_run_dir, save_numpy


def main():
    cfg, args = parse_config_args(
        "Preprocess molecular activity data",
        extra_arg_fn=lambda p: (
            p.add_argument("--train_csv", default=None),
            p.add_argument("--val_csv",   default=None),
            p.add_argument("--test_csv",  default=None),
        ),
    )
    data_cfg = cfg["data"]
    tr_cfg = cfg["training"]

    if args.train_csv:
        # Pre-split mode: user provides train/val(/test) CSVs directly
        train_df = pd.read_csv(args.train_csv)
        val_df   = pd.read_csv(args.val_csv)
        test_df  = pd.read_csv(args.test_csv) if args.test_csv else pd.DataFrame(columns=train_df.columns)

        cleaned = pd.concat([train_df, val_df, test_df], ignore_index=True)
        n_tr, n_val = len(train_df), len(val_df)
        train_idx = np.arange(0, n_tr)
        val_idx   = np.arange(n_tr, n_tr + n_val)
        test_idx  = np.arange(n_tr + n_val, len(cleaned))
        print(f"[pre-split] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    else:
        raw_df, report = load_activity_csv(
            csv_path=data_cfg["raw_csv"],
            smiles_col=data_cfg["smiles_col"],
            activity_col=data_cfg["activity_col"],
        )
        print("[load]", report)

        cleaned = clean_activity_dataframe(
            raw_df,
            smiles_col=data_cfg["smiles_col"],
            activity_col=data_cfg["activity_col"],
            activity_type=data_cfg.get("activity_type", "pIC50"),
            unit_col=data_cfg.get("unit_col", None),
        )
        print(f"[clean] rows: {len(raw_df)} -> {len(cleaned)}")

        split_type = str(tr_cfg.get("split_type", "scaffold")).lower()
        if split_type == "random":
            train_idx, val_idx, test_idx = random_split(
                cleaned,
                val_frac=float(tr_cfg["val_fraction"]),
                test_frac=float(tr_cfg["test_fraction"]),
                seed=int(tr_cfg["seed"]),
            )
        else:
            train_idx, val_idx, test_idx = scaffold_split(
                cleaned,
                smiles_col=data_cfg["smiles_col"],
                val_frac=float(tr_cfg["val_fraction"]),
                test_frac=float(tr_cfg["test_fraction"]),
                seed=int(tr_cfg["seed"]),
            )
        print(f"[split] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} (type={split_type})")

    run_dir = create_run_dir(cfg)
    (run_dir / "splits").mkdir(parents=True, exist_ok=True)

    cleaned.to_csv(run_dir / "cleaned_dataset.csv", index=False)
    save_numpy(run_dir / "splits" / "train_idx.npy", train_idx)
    save_numpy(run_dir / "splits" / "val_idx.npy", val_idx)
    save_numpy(run_dir / "splits" / "test_idx.npy", test_idx)
    save_config(cfg, str(run_dir / "config_resolved.yaml"))

    print(f"[save] run_dir: {run_dir}")


if __name__ == "__main__":
    main()
