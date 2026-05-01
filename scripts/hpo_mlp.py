#!/usr/bin/env python
"""
Optuna HPO for fingerprint/unimol + FingerprintMLP encoder.

Usage (from existing run_dir):
  python scripts/hpo_mlp.py \
      --config outputs/runs/fp_random/config_resolved.yaml \
      --n_trials 50

Usage (from raw CSV directly — splits generated per trial):
  python scripts/hpo_mlp.py \
      --config configs/experiments/unimol_mlp_z4_random.yaml \
      --csv data/BACE1/bace1_clean_pic50.csv \
      --n_trials 50

  # resume interrupted study
  python scripts/hpo_mlp.py \
      --config outputs/runs/fp_random/config_resolved.yaml \
      --n_trials 50 --study_name fp_random_hpo

Multi-GPU (same study_name, different devices):
  CUDA_VISIBLE_DEVICES=0 python scripts/hpo_mlp.py --config ... --n_trials 25 &
  CUDA_VISIBLE_DEVICES=1 python scripts/hpo_mlp.py --config ... --n_trials 25 &
"""
import argparse
import copy
import shutil
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score

from src.data.splitter import random_split, scaffold_split
from src.features.descriptors import smiles_to_descriptors
from src.features.fingerprints import smiles_to_fp
from src.models.mlp import FingerprintMLP
from src.training.scheduler import get_scheduler
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.io import load_numpy, resolve_run_dir, save_json
from src.utils.seed import set_seed

optuna.logging.set_verbosity(optuna.logging.WARNING)

_SEEDS = [0, 1, 42, 123, 999]
_KEEP  = 50  # fixed undersampling count per range


def _undersample_indices(y_vals: np.ndarray, seed: int, keep: int = _KEEP) -> np.ndarray:
    """Return original indices after undersampling three low-activity ranges."""
    mask_neg_mid = (y_vals >= -20) & (y_vals <  -10)
    mask_neg     = (y_vals >= -10) & (y_vals <    0)
    mask_low_pos = (y_vals >=   0) & (y_vals <   10)
    mask_other   = ~(mask_neg_mid | mask_neg | mask_low_pos)

    rng = np.random.default_rng(seed)

    def _pick(mask):
        idxs = np.where(mask)[0]
        return rng.choice(idxs, size=min(keep, len(idxs)), replace=False)

    selected = np.concatenate([
        _pick(mask_neg_mid), _pick(mask_neg), _pick(mask_low_pos),
        np.where(mask_other)[0],
    ])
    rng.shuffle(selected)
    return selected

# Intermediate layers before the fixed latent_dim=4.
# "" means no intermediate layer → hidden_dims = [4].
_HIDDEN_DIMS_GRID = [
    # 0 intermediate
    "",
    # 1 intermediate
    "512", "256", "128", "64",
    # 2 intermediate
    "512_256", "512_128", "512_64",
    "256_128", "256_64",
    "128_64",
    # 3 intermediate
    "512_256_128", "512_256_64", "512_128_64",
    "256_128_64",
]


def resolve_device(device_cfg: str) -> torch.device:
    d = device_cfg.lower()
    if d == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


def load_features(feat_cfg: dict, smiles_all: list) -> np.ndarray:
    feature_type = str(feat_cfg.get("type", "fingerprint")).lower()
    if feature_type == "fingerprint":
        x = smiles_to_fp(
            smiles_all,
            bits=int(feat_cfg.get("fp_bits", 2048)),
            radius=int(feat_cfg.get("fp_radius", 2)),
            use_chirality=bool(feat_cfg.get("use_chirality", False)),
        )
        if bool(feat_cfg.get("use_descriptors", False)):
            x = np.concatenate([x, smiles_to_descriptors(smiles_all)], axis=1)
        return x
    elif feature_type == "unimol":
        from src.features.unimol import smiles_to_unimol
        print("[hpo_mlp] extracting Uni-Mol embeddings (one-time)...")
        return smiles_to_unimol(smiles_all)
    else:
        raise ValueError(
            f"hpo_mlp.py only supports feature_type 'fingerprint' or 'unimol', got: {feature_type}"
        )


def get_splits(cfg: dict, df: pd.DataFrame, seed: int, csv_mode: bool):
    """Return (train_idx, val_idx) arrays.

    csv_mode=True  → generate fresh splits from df using config fractions + trial seed.
    csv_mode=False → load pre-saved npy splits from run_dir.
    """
    if csv_mode:
        tr_cfg = cfg["training"]
        data_cfg = cfg["data"]
        split_type = str(tr_cfg.get("split_type", "random")).lower()
        val_frac  = float(tr_cfg.get("val_fraction", 0.2))
        test_frac = float(tr_cfg.get("test_fraction", 0.1))
        if split_type == "scaffold":
            train_idx, val_idx, _ = scaffold_split(
                df, smiles_col=data_cfg["smiles_col"],
                val_frac=val_frac, test_frac=test_frac, seed=seed,
            )
        else:
            train_idx, val_idx, _ = random_split(
                df, val_frac=val_frac, test_frac=test_frac, seed=seed,
            )
        return train_idx, val_idx
    else:
        base_run_dir = resolve_run_dir(cfg, create_if_missing=False)
        splits_dir = base_run_dir / "splits"
        return (
            load_numpy(splits_dir / "train_idx.npy"),
            load_numpy(splits_dir / "val_idx.npy"),
        )


def run_trial(
    cfg: dict,
    x_all: np.ndarray,
    y_all: np.ndarray,
    df: pd.DataFrame,
    trial_dir: Path,
    trial_seed: int,
    csv_mode: bool,
) -> float:
    tr_cfg    = cfg["training"]
    model_cfg = cfg["model"]

    set_seed(trial_seed)
    device = resolve_device(str(tr_cfg.get("device", "auto")))

    train_idx, val_idx = get_splits(cfg, df, seed=trial_seed, csv_mode=csv_mode)

    normalize_y = bool(tr_cfg.get("normalize_y", False))
    if normalize_y:
        y_mean = float(y_all[train_idx].mean())
        y_std  = float(y_all[train_idx].std()) or 1.0
        y_norm = (y_all - y_mean) / y_std
    else:
        y_norm = y_all

    batch_size  = int(tr_cfg["batch_size"])
    hidden_dims = list(model_cfg["hidden_dims"])

    def make_loader(indices, shuffle=False):
        x = torch.tensor(x_all[indices], dtype=torch.float32)
        y = torch.tensor(y_norm[indices], dtype=torch.float32)
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader   = make_loader(val_idx,   shuffle=False)

    model = FingerprintMLP(
        input_dim=x_all.shape[1],
        hidden_dims=hidden_dims,
        dropout=float(model_cfg.get("dropout", 0.2)),
        activation=str(model_cfg.get("activation", "relu")),
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=float(tr_cfg["lr"]),
        weight_decay=float(tr_cfg["weight_decay"]),
    )
    scheduler = get_scheduler(optimizer, cfg)

    trial_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.MSELoss(),
        device=device,
        scheduler=scheduler,
        is_graph=False,
        early_stopping_patience=int(tr_cfg.get("early_stopping_patience", 15)),
        checkpoint_every=0,
        run_dir=trial_dir,
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(tr_cfg.get("epochs", 100)),
    )

    from src.utils.io import load_checkpoint
    load_checkpoint(trial_dir / "checkpoints" / "best.pt", model=model, map_location=device)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            pred, _ = model(x_batch)
            preds.append(pred.cpu())
            trues.append(y_batch)
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(trues).numpy()
    return float(r2_score(y_true, y_pred))


def make_objective(
    base_cfg: dict,
    all_x: list,
    all_y: list,
    all_df: list,
    arch_dir: Path,
    hidden_str: str,
    csv_mode: bool,
):
    def objective(trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(base_cfg)
        tr_cfg    = cfg["training"]
        model_cfg = cfg["model"]

        # ── Training hyperparameters ──────────────────────────────────────
        tr_cfg["lr"]           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        tr_cfg["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        tr_cfg["batch_size"]   = trial.suggest_categorical("batch_size", [8, 16, 32, 64])

        # ── Model hyperparameters ─────────────────────────────────────────
        model_cfg["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)

        # hidden_dims: fixed intermediate layers + latent_dim=4
        intermediate = [int(d) for d in hidden_str.split("_")] if hidden_str else []
        model_cfg["hidden_dims"] = intermediate + [4]

        if csv_mode:
            # us_seed: controls which molecules survive undersampling (keep=50 fixed)
            # split_seed: controls train/val split (also used as model init seed)
            us_seed    = trial.suggest_categorical("us_seed",    _SEEDS)
            split_seed = trial.suggest_categorical("split_seed", _SEEDS)
            us_idx     = _undersample_indices(all_y[0], seed=us_seed)
            x_sel = all_x[0][us_idx]
            y_sel = all_y[0][us_idx]
            df_sel = all_df[0].iloc[us_idx].reset_index(drop=True)
            trial_seed = split_seed
        else:
            x_sel, y_sel, df_sel = all_x[0], all_y[0], all_df[0]
            trial_seed = trial.number

        trial_dir = arch_dir / f"trial_{trial.number}"
        val_r2 = run_trial(cfg, x_sel, y_sel, df_sel, trial_dir, trial_seed, csv_mode)
        print(f"    trial {trial.number:3d}  val_r2={val_r2:.4f}  params={trial.params}")
        return val_r2

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True,  type=str)
    parser.add_argument("--csv",        default=None,   type=str,
                        help="Path to raw CSV. If given, splits are generated per trial.")
    parser.add_argument("--label_col",  default=None,   type=str,
                        help="Override label column name (e.g. 'Malachite green assay_50uM (%%)').")
    parser.add_argument("--smiles_col", default=None,   type=str,
                        help="Override SMILES column name.")
    parser.add_argument("--n_trials",   default=50,     type=int)
    parser.add_argument("--study_name", default=None,   type=str)
    parser.add_argument("--timeout",    default=None,   type=int, help="seconds")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    data_cfg = base_cfg["data"]
    feat_cfg = base_cfg["features"]

    # Determine data source
    csv_mode = args.csv is not None
    base_run_dir = resolve_run_dir(base_cfg, create_if_missing=True)
    hpo_root = base_run_dir / "hpo_mlp"

    if csv_mode:
        df = pd.read_csv(args.csv)
    else:
        df = pd.read_csv(base_run_dir / "cleaned_dataset.csv")

    hpo_root.mkdir(parents=True, exist_ok=True)

    label_col  = data_cfg.get("label_col", "pIC50")
    smiles_col = data_cfg["smiles_col"]

    # Drop rows with missing SMILES or label
    before = len(df)
    df = df.dropna(subset=[smiles_col, label_col]).reset_index(drop=True)
    df = df[df[smiles_col].astype(str).str.strip().ne("nan")].reset_index(drop=True)
    if len(df) < before:
        print(f"[hpo_mlp] dropped {before - len(df)} rows with missing SMILES/label")

    smiles_all = df[smiles_col].astype(str).tolist()

    # Extract features once for the full dataset
    x_all = load_features(feat_cfg, smiles_all)
    y_all = df[label_col].astype(float).values
    print(f"[hpo_mlp] features shape: {x_all.shape}  csv_mode={csv_mode}")

    all_x  = [x_all]
    all_y  = [y_all]
    all_df = [df]

    base_study_name = args.study_name or Path(args.config).stem

    print(f"Results → {hpo_root}")
    print(f"Architectures: {len(_HIDDEN_DIMS_GRID)}  ×  n_trials: {args.n_trials}\n")

    all_results = []

    for hidden_str in _HIDDEN_DIMS_GRID:
        arch_label = hidden_str + "_4" if hidden_str else "4"
        arch_dir   = hpo_root / f"arch_{arch_label}"
        arch_dir.mkdir(parents=True, exist_ok=True)

        study_name = f"{base_study_name}_{arch_label}"
        storage    = f"sqlite:///{arch_dir}/study.db"

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=0),
        )

        print(f"[arch {arch_label}]  hidden_dims={([int(d) for d in hidden_str.split('_')] if hidden_str else []) + [4]}")
        study.optimize(
            make_objective(base_cfg, all_x, all_y, all_df, arch_dir, hidden_str, csv_mode),
            n_trials=args.n_trials,
            timeout=args.timeout,
        )

        best = study.best_trial
        best_trial_dir = arch_dir / f"trial_{best.number}"
        print(f"  → best val_r2={best.value:.4f}  params={best.params}\n")
        all_results.append({
            "arch": arch_label, "val_r2": best.value, "params": best.params,
            "trial_dir": str(best_trial_dir),
        })

    all_results.sort(key=lambda r: r["val_r2"], reverse=True)

    print("=== Overall Best ===")
    top = all_results[0]
    print(f"  arch     : {top['arch']}")
    print(f"  val_r2   : {top['val_r2']:.4f}")
    for k, v in top["params"].items():
        print(f"  {k:20s}: {v}")

    # Copy best model checkpoint to hpo_root
    best_ckpt_src = Path(top["trial_dir"]) / "checkpoints" / "best.pt"
    best_ckpt_dst = hpo_root / "best_model.pt"
    if best_ckpt_src.exists():
        shutil.copy2(best_ckpt_src, best_ckpt_dst)
        print(f"Best model → {best_ckpt_dst}")
    else:
        print(f"[warning] best checkpoint not found: {best_ckpt_src}")

    save_json(hpo_root / "best_params.json", all_results[0])
    save_json(hpo_root / "all_results.json", all_results)
    print(f"Saved → {hpo_root / 'best_params.json'}")

    if csv_mode and "us_seed" in top["params"]:
        best_us_seed    = top["params"]["us_seed"]
        best_split_seed = top["params"]["split_seed"]

        us_idx   = _undersample_indices(all_y[0], seed=best_us_seed)
        best_df  = all_df[0].iloc[us_idx].reset_index(drop=True)

        train_idx, val_idx = get_splits(base_cfg, best_df, seed=best_split_seed, csv_mode=True)

        best_df.to_csv(hpo_root / "best_dataset.csv", index=False)
        best_df.iloc[train_idx].reset_index(drop=True).to_csv(hpo_root / "best_train.csv", index=False)
        best_df.iloc[val_idx].reset_index(drop=True).to_csv(hpo_root / "best_val.csv", index=False)
        np.save(hpo_root / "best_train_idx.npy", train_idx)
        np.save(hpo_root / "best_val_idx.npy",   val_idx)

        print(f"Best dataset → {hpo_root / 'best_dataset.csv'}  "
              f"({len(best_df)} total, train={len(train_idx)}, val={len(val_idx)}, "
              f"us_seed={best_us_seed}, split_seed={best_split_seed})")


if __name__ == "__main__":
    main()
