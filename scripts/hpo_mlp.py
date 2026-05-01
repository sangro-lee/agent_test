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
        val_frac  = float(tr_cfg.get("val_fraction", 0.1))
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
    x_all: np.ndarray,
    y_all: np.ndarray,
    df: pd.DataFrame,
    hpo_dir: Path,
    csv_mode: bool,
):
    def objective(trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(base_cfg)
        tr_cfg    = cfg["training"]
        model_cfg = cfg["model"]

        # ── Training hyperparameters ──────────────────────────────────────
        tr_cfg["lr"]           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        tr_cfg["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        tr_cfg["batch_size"]   = trial.suggest_categorical("batch_size", [8,16, 32, 64])

        # ── Model hyperparameters ─────────────────────────────────────────
        model_cfg["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)

        # Intermediate hidden layers (before latent_dim) — strictly decreasing from {512,256,128,64}
        hidden_str = trial.suggest_categorical(
            "hidden_dims",
            [
                # 1 layer
                "512", "256", "128", "64", "4",
                # 2 layers
                "512_256", "512_128", "512_64",
                "256_128", "256_64",
                "128_64", "128_4",
                "64_4",
                # 3 layers
                "512_256_128", "512_256_64", "512_128_64",
                "256_128_64", "256_128_4", "256_64_4",
                "128_64_4",
            ],
        )
        model_cfg["hidden_dims"] = [int(d) for d in hidden_str.split("_")]

        # Seed (controls model init; in csv_mode also controls data split)
        trial_seed = trial.suggest_categorical("seed", _SEEDS)

        trial_dir = hpo_dir / f"trial_{trial.number}"
        val_r2 = run_trial(cfg, x_all, y_all, df, trial_dir, trial_seed, csv_mode)
        print(f"  trial {trial.number:3d}  val_r2={val_r2:.4f}  params={trial.params}")
        return val_r2

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True,  type=str)
    parser.add_argument("--csv",        default=None,   type=str,
                        help="Path to raw CSV. If given, splits are generated per trial.")
    parser.add_argument("--n_trials",   default=50,     type=int)
    parser.add_argument("--study_name", default=None,   type=str)
    parser.add_argument("--timeout",    default=None,   type=int, help="seconds")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    data_cfg = base_cfg["data"]
    feat_cfg = base_cfg["features"]

    # Determine data source
    csv_mode = args.csv is not None
    if csv_mode:
        df = pd.read_csv(args.csv)
        # Save path used for study storage
        hpo_root = Path(args.csv).parent / "hpo_mlp"
    else:
        base_run_dir = resolve_run_dir(base_cfg, create_if_missing=False)
        df = pd.read_csv(base_run_dir / "cleaned_dataset.csv")
        hpo_root = base_run_dir / "hpo_mlp"

    hpo_root.mkdir(parents=True, exist_ok=True)

    smiles_all = df[data_cfg["smiles_col"]].astype(str).tolist()
    label_col  = data_cfg.get("label_col", "pIC50")
    y_all      = df[label_col].astype(float).values

    # Extract features once before all trials
    x_all = load_features(feat_cfg, smiles_all)
    print(f"[hpo_mlp] features shape: {x_all.shape}  csv_mode={csv_mode}")

    study_name = args.study_name or (
        Path(args.config).stem + "_hpo_mlp"
    )
    storage = f"sqlite:///{hpo_root}/study.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=0),
    )

    print(f"Study: {study_name}  (n_trials={args.n_trials})")
    print(f"Results → {hpo_root}\n")

    study.optimize(
        make_objective(base_cfg, x_all, y_all, df, hpo_root, csv_mode),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    print("\n=== Best Trial ===")
    best = study.best_trial
    print(f"  val_r2   : {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k:20s}: {v}")

    save_json(hpo_root / "best_params.json", {
        "val_r2": best.value,
        "params": best.params,
    })
    print(f"\nSaved → {hpo_root / 'best_params.json'}")


if __name__ == "__main__":
    main()
