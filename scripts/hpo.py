#!/usr/bin/env python
"""
Optuna HPO for sme_rgcn encoder.

Usage:
  python scripts/hpo.py \
      --config outputs/runs/sme_random/config_resolved.yaml \
      --n_trials 50

  # resume interrupted study
  python scripts/hpo.py \
      --config outputs/runs/sme_random/config_resolved.yaml \
      --n_trials 50 --study_name sme_random_hpo
"""
import argparse
import copy
from pathlib import Path

import optuna
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam

from src.features.graph import SMEMolDataset, SME_NODE_DIM
from src.models.gnn import SMERGCNModel
from src.training.scheduler import get_scheduler
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.io import load_numpy, resolve_run_dir, save_json
from src.utils.seed import set_seed

try:
    from torch_geometric.loader import DataLoader as PyGDataLoader
except Exception:
    PyGDataLoader = None

optuna.logging.set_verbosity(optuna.logging.WARNING)


def resolve_device(device_cfg: str) -> torch.device:
    d = device_cfg.lower()
    if d == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


def run_trial(cfg: dict, trial_dir: Path, trial_seed: int) -> float:
    data_cfg = cfg["data"]
    feat_cfg  = cfg["features"]
    model_cfg = cfg["model"]
    tr_cfg    = cfg["training"]

    set_seed(trial_seed)
    device = resolve_device(str(tr_cfg.get("device", "auto")))

    base_run_dir = resolve_run_dir(cfg, create_if_missing=False)
    df        = pd.read_csv(base_run_dir / "cleaned_dataset.csv")
    train_idx = load_numpy(base_run_dir / "splits" / "train_idx.npy")
    val_idx   = load_numpy(base_run_dir / "splits" / "val_idx.npy")

    smiles_all = df[data_cfg["smiles_col"]].astype(str).tolist()
    label_col  = data_cfg.get("label_col", "pIC50")
    y_all      = df[label_col].astype(float).values

    normalize_y = bool(tr_cfg.get("normalize_y", False))
    if normalize_y:
        y_mean = float(y_all[train_idx].mean())
        y_std  = float(y_all[train_idx].std()) or 1.0
        y_all_norm = (y_all - y_mean) / y_std
    else:
        y_all_norm = y_all

    smask_type    = str(feat_cfg.get("smask_type", "brics"))
    use_chirality = bool(feat_cfg.get("use_chirality", True))
    batch_size    = int(tr_cfg["batch_size"])
    latent_dim    = int(model_cfg.get("latent_dim", 128))

    def make_loader(indices, shuffle=False):
        smiles = [smiles_all[i] for i in indices]
        y = [float(y_all_norm[i]) for i in indices]
        ds = SMEMolDataset(smiles, y, smask_type=smask_type, use_chirality=use_chirality)
        return PyGDataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader   = make_loader(val_idx,   shuffle=False)

    _ffn_dims_cfg = model_cfg.get("sme_ffn_dims")
    if _ffn_dims_cfg is not None:
        sme_kwargs = {"ffn_dims": [int(d) for d in _ffn_dims_cfg] + [latent_dim]}
    else:
        sme_kwargs = {"ffn_hidden": int(model_cfg["sme_ffn_hidden"])}

    model = SMERGCNModel(
        in_feats=SME_NODE_DIM,
        hidden_feats=list(model_cfg["sme_hidden_feats"]),
        rgcn_dropout=float(model_cfg["sme_rgcn_dropout"]),
        ffn_dropout=float(model_cfg["sme_ffn_dropout"]),
        **sme_kwargs,
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
        is_graph=True,
        early_stopping_patience=int(tr_cfg.get("early_stopping_patience", 15)),
        checkpoint_every=0,   # 중간 체크포인트 저장 안 함
        run_dir=trial_dir,
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(tr_cfg.get("epochs", 100)),
    )
    return float(history.get("best_val_loss", float("inf")))


def make_objective(base_cfg: dict, hpo_dir: Path, base_seed: int):
    def objective(trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(base_cfg)
        tr_cfg    = cfg["training"]
        model_cfg = cfg["model"]

        # ── Training hyperparameters ──────────────────────────────────────
        tr_cfg["lr"]           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        tr_cfg["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        tr_cfg["batch_size"]   = trial.suggest_categorical("batch_size", [32, 64, 128])

        # ── Model hyperparameters ─────────────────────────────────────────
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        model_cfg["sme_rgcn_dropout"] = dropout
        model_cfg["sme_ffn_dropout"]  = dropout

        model_cfg["sme_ffn_hidden"] = trial.suggest_categorical(
            "sme_ffn_hidden", [128, 256, 512]
        )

        # hidden_feats: Optuna는 list를 직접 못 다루므로 문자열로
        hidden_feats_str = trial.suggest_categorical(
            "sme_hidden_feats",
            ["128_1", "128_2", "256_1", "256_2", "256_3"],
        )
        dim, n = hidden_feats_str.split("_")
        model_cfg["sme_hidden_feats"] = [int(dim)] * int(n)

        trial_dir  = hpo_dir / f"trial_{trial.number}"
        trial_seed = base_seed + trial.number

        val_loss = run_trial(cfg, trial_dir, trial_seed)
        print(f"  trial {trial.number:3d}  val_loss={val_loss:.4f}  params={trial.params}")
        return val_loss

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True, type=str)
    parser.add_argument("--n_trials",   default=50,    type=int)
    parser.add_argument("--study_name", default=None,  type=str)
    parser.add_argument("--timeout",    default=None,  type=int, help="seconds")
    parser.add_argument("--seed",       default=42,    type=int)
    args = parser.parse_args()

    base_cfg     = load_config(args.config)
    base_run_dir = resolve_run_dir(base_cfg, create_if_missing=False)
    hpo_dir      = base_run_dir / "hpo"
    hpo_dir.mkdir(parents=True, exist_ok=True)

    study_name = args.study_name or f"{base_run_dir.name}_hpo"
    storage    = f"sqlite:///{hpo_dir}/study.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    print(f"Study: {study_name}  (n_trials={args.n_trials})")
    print(f"Results → {hpo_dir}\n")

    study.optimize(
        make_objective(base_cfg, hpo_dir, args.seed),
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    print("\n=== Best Trial ===")
    best = study.best_trial
    print(f"  val_loss : {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k:20s}: {v}")

    save_json(hpo_dir / "best_params.json", {
        "val_loss": best.value,
        "params":   best.params,
    })
    print(f"\nSaved → {hpo_dir / 'best_params.json'}")


if __name__ == "__main__":
    main()
