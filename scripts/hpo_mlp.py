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

Per-arch outputs (arch_dir/):
  scatter_{r2|loss}_{val|train|test}.png
  dataset_{r2|loss}.csv  /  {train|val|test}_idx_{r2|loss}.npy
  y_scaler_{r2|loss}.json

Global outputs (hpo_root/):
  best_model.pt  best_y_scaler.json  best_params.json  all_results.json
  best_dataset.csv  best_{train|val|test}.csv  best_{train|val|test}_idx.npy
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


def _undersample_indices(
    y_vals: np.ndarray,
    seed: int,
    keep_neg_mid: int = 50,
    keep_neg: int = 50,
    keep_low_pos: int = 50,
) -> np.ndarray:
    """Return original indices after undersampling three low-activity ranges."""
    mask_neg_mid = (y_vals >= -20) & (y_vals <  -10)
    mask_neg     = (y_vals >= -10) & (y_vals <    0)
    mask_low_pos = (y_vals >=   0) & (y_vals <   10)
    mask_other   = ~(mask_neg_mid | mask_neg | mask_low_pos)

    rng = np.random.default_rng(seed)

    def _pick(mask, n):
        idxs = np.where(mask)[0]
        return rng.choice(idxs, size=min(n, len(idxs)), replace=False)

    selected = np.concatenate([
        _pick(mask_neg_mid, keep_neg_mid),
        _pick(mask_neg,     keep_neg),
        _pick(mask_low_pos, keep_low_pos),
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
    """Return (train_idx, val_idx, test_idx) arrays.

    csv_mode=True  → generate fresh splits from df using config fractions + trial seed.
    csv_mode=False → load pre-saved npy splits from run_dir.
    """
    if csv_mode:
        tr_cfg   = cfg["training"]
        data_cfg = cfg["data"]
        split_type = str(tr_cfg.get("split_type", "random")).lower()
        val_frac   = float(tr_cfg.get("val_fraction", 0.2))
        test_frac  = float(tr_cfg.get("test_fraction", 0.1))
        if split_type == "scaffold":
            train_idx, val_idx, test_idx = scaffold_split(
                df, smiles_col=data_cfg["smiles_col"],
                val_frac=val_frac, test_frac=test_frac, seed=seed,
            )
        else:
            train_idx, val_idx, test_idx = random_split(
                df, val_frac=val_frac, test_frac=test_frac, seed=seed,
            )
        return train_idx, val_idx, test_idx
    else:
        base_run_dir = resolve_run_dir(cfg, create_if_missing=False)
        splits_dir   = base_run_dir / "splits"
        test_path    = splits_dir / "test_idx.npy"
        test_idx     = load_numpy(test_path) if test_path.exists() else np.array([], dtype=int)
        return (
            load_numpy(splits_dir / "train_idx.npy"),
            load_numpy(splits_dir / "val_idx.npy"),
            test_idx,
        )


def run_trial(
    cfg: dict,
    x_all: np.ndarray,
    y_all: np.ndarray,
    df: pd.DataFrame,
    trial_dir: Path,
    trial_seed: int,
    csv_mode: bool,
) -> tuple:
    """Train one trial; return (val_r2, val_loss) evaluated on best checkpoint."""
    tr_cfg    = cfg["training"]
    model_cfg = cfg["model"]

    set_seed(trial_seed)
    device = resolve_device(str(tr_cfg.get("device", "auto")))

    train_idx, val_idx, _ = get_splits(cfg, df, seed=trial_seed, csv_mode=csv_mode)

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

    criterion_eval = nn.MSELoss()
    val_loss_total = 0.0
    n_val = 0
    preds, trues = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            pred, _ = model(x_batch)
            val_loss_total += criterion_eval(pred, y_batch.to(device)).item() * len(y_batch)
            n_val += len(y_batch)
            preds.append(pred.cpu())
            trues.append(y_batch)

    val_loss = val_loss_total / max(n_val, 1)
    y_pred   = torch.cat(preds).numpy()
    y_true   = torch.cat(trues).numpy()
    return float(r2_score(y_true, y_pred)), val_loss


def _reconstruct_trial_data(
    trial: optuna.Trial,
    base_cfg: dict,
    all_x: list,
    all_y: list,
    all_df: list,
    csv_mode: bool,
    keep_neg_mid: int,
    keep_neg: int,
    keep_low_pos: int,
):
    """Reconstruct (cfg, x_sel, y_sel, df_sel, trial_seed) from trial params."""
    cfg       = copy.deepcopy(base_cfg)
    tr_cfg    = cfg["training"]
    model_cfg = cfg["model"]

    tr_cfg["lr"]           = trial.params["lr"]
    tr_cfg["weight_decay"] = trial.params["weight_decay"]
    tr_cfg["batch_size"]   = trial.params["batch_size"]
    model_cfg["dropout"]   = trial.params["dropout"]

    if csv_mode:
        us_seed    = trial.params["us_seed"]
        split_seed = trial.params["split_seed"]
        us_idx  = _undersample_indices(all_y[0], seed=us_seed,
                                       keep_neg_mid=keep_neg_mid,
                                       keep_neg=keep_neg,
                                       keep_low_pos=keep_low_pos)
        x_sel   = all_x[0][us_idx]
        y_sel   = all_y[0][us_idx]
        df_sel  = all_df[0].iloc[us_idx].reset_index(drop=True)
        trial_seed = split_seed
    else:
        x_sel, y_sel, df_sel = all_x[0], all_y[0], all_df[0]
        trial_seed = trial.number

    return cfg, x_sel, y_sel, df_sel, trial_seed


def scatter_best_trial(
    trial: optuna.Trial,
    trial_dir: Path,
    base_cfg: dict,
    all_x: list,
    all_y: list,
    all_df: list,
    hidden_str: str,
    arch_label: str,
    csv_mode: bool,
    keep_neg_mid: int,
    keep_neg: int,
    keep_low_pos: int,
    arch_dir: Path,
    tag: str,
):
    """Plot val/train/test scatter; save dataset, split indices, y_scaler per arch.

    tag: "r2" (best by val R²) or "loss" (best by val_loss).
    Outputs written to arch_dir:
      scatter_{tag}_{val|train|test}.png
      dataset_{tag}.csv  /  {train|val|test}_idx_{tag}.npy
      y_scaler_{tag}.json
    """
    from src.evaluation.plots import plot_scatter
    from src.utils.io import load_checkpoint

    cfg, x_sel, y_sel, df_sel, trial_seed = _reconstruct_trial_data(
        trial, base_cfg, all_x, all_y, all_df, csv_mode,
        keep_neg_mid, keep_neg, keep_low_pos,
    )
    tr_cfg    = cfg["training"]
    model_cfg = cfg["model"]
    label_col = cfg["data"].get("label_col", "y")

    intermediate = [int(d) for d in hidden_str.split("_")] if hidden_str else []
    model_cfg["hidden_dims"] = intermediate + [4]

    device = resolve_device(str(tr_cfg.get("device", "auto")))

    normalize_y = bool(tr_cfg.get("normalize_y", False))
    train_idx, val_idx, test_idx = get_splits(cfg, df_sel, seed=trial_seed, csv_mode=csv_mode)

    if normalize_y:
        y_mean = float(y_sel[train_idx].mean())
        y_std  = float(y_sel[train_idx].std()) or 1.0
        y_norm = (y_sel - y_mean) / y_std
    else:
        y_norm, y_mean, y_std = y_sel, 0.0, 1.0

    # ── persist scaler, dataset, indices ──────────────────────────────────
    save_json(arch_dir / f"y_scaler_{tag}.json", {
        "y_mean": y_mean, "y_std": y_std, "normalize_y": normalize_y,
    })
    df_sel.to_csv(arch_dir / f"dataset_{tag}.csv", index=False)
    np.save(arch_dir / f"train_idx_{tag}.npy", train_idx)
    np.save(arch_dir / f"val_idx_{tag}.npy",   val_idx)
    np.save(arch_dir / f"test_idx_{tag}.npy",  test_idx)

    # ── load model ────────────────────────────────────────────────────────
    batch_size = int(tr_cfg["batch_size"])

    def make_loader(idx):
        return DataLoader(
            TensorDataset(torch.tensor(x_sel[idx], dtype=torch.float32),
                          torch.tensor(y_norm[idx], dtype=torch.float32)),
            batch_size=batch_size, shuffle=False,
        )

    val_loader   = make_loader(val_idx)
    train_loader = make_loader(train_idx)
    has_test     = len(test_idx) > 0
    test_loader  = make_loader(test_idx) if has_test else None

    model = FingerprintMLP(
        input_dim=x_sel.shape[1],
        hidden_dims=model_cfg["hidden_dims"],
        dropout=float(model_cfg["dropout"]),
        activation=str(model_cfg.get("activation", "relu")),
    ).to(device)
    load_checkpoint(trial_dir / "checkpoints" / "best.pt", model=model, map_location=device)
    model.eval()

    def _infer(loader):
        preds, trues = [], []
        with torch.no_grad():
            for x_b, y_b in loader:
                pred, _ = model(x_b.to(device))
                preds.append(pred.cpu())
                trues.append(y_b)
        yp = torch.cat(preds).numpy()
        yt = torch.cat(trues).numpy()
        if normalize_y:
            yp = yp * y_std + y_mean
            yt = yt * y_std + y_mean
        return yt, yp

    # ── scatter plots ──────────────────────────────────────────────────────
    y_true_val,   y_pred_val   = _infer(val_loader)
    y_true_train, y_pred_train = _infer(train_loader)

    plot_scatter(y_true_val,   y_pred_val,
                 title=f"Val [{tag}] — arch {arch_label}",
                 save_path=arch_dir / f"scatter_{tag}_val.png",
                 label=label_col)
    plot_scatter(y_true_train, y_pred_train,
                 title=f"Train [{tag}] — arch {arch_label}",
                 save_path=arch_dir / f"scatter_{tag}_train.png",
                 label=label_col)
    if has_test:
        y_true_test, y_pred_test = _infer(test_loader)
        plot_scatter(y_true_test, y_pred_test,
                     title=f"Test [{tag}] — arch {arch_label}",
                     save_path=arch_dir / f"scatter_{tag}_test.png",
                     label=label_col)


def make_objective(
    base_cfg: dict,
    all_x: list,
    all_y: list,
    all_df: list,
    arch_dir: Path,
    hidden_str: str,
    csv_mode: bool,
    keep_neg_mid: int = 50,
    keep_neg: int = 50,
    keep_low_pos: int = 50,
):
    def objective(trial: optuna.Trial) -> float:
        cfg       = copy.deepcopy(base_cfg)
        tr_cfg    = cfg["training"]
        model_cfg = cfg["model"]

        tr_cfg["lr"]           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        tr_cfg["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        tr_cfg["batch_size"]   = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        model_cfg["dropout"]   = trial.suggest_float("dropout", 0.0, 0.5)

        intermediate = [int(d) for d in hidden_str.split("_")] if hidden_str else []
        model_cfg["hidden_dims"] = intermediate + [4]

        if csv_mode:
            us_seed    = trial.suggest_categorical("us_seed",    _SEEDS)
            split_seed = trial.suggest_categorical("split_seed", _SEEDS)
            us_idx  = _undersample_indices(all_y[0], seed=us_seed,
                                           keep_neg_mid=keep_neg_mid,
                                           keep_neg=keep_neg,
                                           keep_low_pos=keep_low_pos)
            x_sel   = all_x[0][us_idx]
            y_sel   = all_y[0][us_idx]
            df_sel  = all_df[0].iloc[us_idx].reset_index(drop=True)
            trial_seed = split_seed
        else:
            x_sel, y_sel, df_sel = all_x[0], all_y[0], all_df[0]
            trial_seed = trial.number

        trial_dir = arch_dir / f"trial_{trial.number}"
        val_r2, val_loss = run_trial(cfg, x_sel, y_sel, df_sel, trial_dir, trial_seed, csv_mode)
        trial.set_user_attr("val_loss", val_loss)
        print(f"    trial {trial.number:3d}  val_r2={val_r2:.4f}  val_loss={val_loss:.4f}  params={trial.params}")
        return val_r2

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True, type=str)
    parser.add_argument("--csv",         default=None,  type=str,
                        help="Path to raw CSV. If given, splits are generated per trial.")
    parser.add_argument("--label_col",   default=None,  type=str,
                        help="Override label column name.")
    parser.add_argument("--smiles_col",  default=None,  type=str,
                        help="Override SMILES column name.")
    parser.add_argument("--keep_neg_mid", default=50,   type=int,
                        help="Max samples from y in [-20, -10) (default: 50).")
    parser.add_argument("--keep_neg",     default=50,   type=int,
                        help="Max samples from y in [-10, 0)  (default: 50).")
    parser.add_argument("--keep_low_pos", default=50,   type=int,
                        help="Max samples from y in [0, 10)   (default: 50).")
    parser.add_argument("--n_trials",    default=50,    type=int)
    parser.add_argument("--study_name",  default=None,  type=str)
    parser.add_argument("--timeout",     default=None,  type=int, help="seconds")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    data_cfg = base_cfg["data"]
    feat_cfg = base_cfg["features"]

    if args.label_col:
        data_cfg["label_col"] = args.label_col
    if args.smiles_col:
        data_cfg["smiles_col"] = args.smiles_col

    csv_mode     = args.csv is not None
    base_run_dir = resolve_run_dir(base_cfg, create_if_missing=True)
    hpo_root     = base_run_dir / "hpo_mlp"
    hpo_root.mkdir(parents=True, exist_ok=True)

    if csv_mode:
        df = pd.read_csv(args.csv)
    else:
        df = pd.read_csv(base_run_dir / "cleaned_dataset.csv")

    label_col  = data_cfg.get("label_col", "pIC50")
    smiles_col = data_cfg["smiles_col"]

    before = len(df)
    df = df.dropna(subset=[smiles_col, label_col]).reset_index(drop=True)
    df = df[df[smiles_col].astype(str).str.strip().ne("nan")].reset_index(drop=True)
    if len(df) < before:
        print(f"[hpo_mlp] dropped {before - len(df)} rows with missing SMILES/label")

    smiles_all = df[smiles_col].astype(str).tolist()
    x_all      = load_features(feat_cfg, smiles_all)
    y_all      = df[label_col].astype(float).values
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

        hd = ([int(d) for d in hidden_str.split("_")] if hidden_str else []) + [4]
        print(f"[arch {arch_label}]  hidden_dims={hd}")
        study.optimize(
            make_objective(base_cfg, all_x, all_y, all_df, arch_dir, hidden_str, csv_mode,
                           keep_neg_mid=args.keep_neg_mid,
                           keep_neg=args.keep_neg,
                           keep_low_pos=args.keep_low_pos),
            n_trials=args.n_trials,
            timeout=args.timeout,
        )

        # ── find best by R² and best by val_loss ──────────────────────────
        best_r2 = study.best_trial
        completed_with_loss = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and "val_loss" in t.user_attrs
        ]
        best_loss = (
            min(completed_with_loss, key=lambda t: t.user_attrs["val_loss"])
            if completed_with_loss else best_r2
        )

        best_r2_dir   = arch_dir / f"trial_{best_r2.number}"
        best_loss_dir = arch_dir / f"trial_{best_loss.number}"

        def _fmt(v):
            return f"{v:.4f}" if v is not None else "N/A"

        print(
            f"  → best R²  : trial {best_r2.number:3d}  "
            f"val_r2={best_r2.value:.4f}  "
            f"val_loss={_fmt(best_r2.user_attrs.get('val_loss'))}"
        )
        print(
            f"  → best loss: trial {best_loss.number:3d}  "
            f"val_r2={best_loss.value:.4f}  "
            f"val_loss={_fmt(best_loss.user_attrs.get('val_loss'))}\n"
        )

        common_kw = dict(
            base_cfg=base_cfg, all_x=all_x, all_y=all_y, all_df=all_df,
            hidden_str=hidden_str, arch_label=arch_label,
            csv_mode=csv_mode,
            keep_neg_mid=args.keep_neg_mid,
            keep_neg=args.keep_neg,
            keep_low_pos=args.keep_low_pos,
            arch_dir=arch_dir,
        )
        scatter_best_trial(trial=best_r2,   trial_dir=best_r2_dir,   tag="r2",   **common_kw)
        scatter_best_trial(trial=best_loss, trial_dir=best_loss_dir, tag="loss", **common_kw)

        all_results.append({
            "arch": arch_label,
            "best_r2": {
                "val_r2":       best_r2.value,
                "val_loss":     best_r2.user_attrs.get("val_loss"),
                "params":       best_r2.params,
                "trial_number": best_r2.number,
                "trial_dir":    str(best_r2_dir),
            },
            "best_loss": {
                "val_r2":       best_loss.value,
                "val_loss":     best_loss.user_attrs.get("val_loss"),
                "params":       best_loss.params,
                "trial_number": best_loss.number,
                "trial_dir":    str(best_loss_dir),
            },
        })

    # ── global best (by val R²) ────────────────────────────────────────────
    all_results.sort(key=lambda r: r["best_r2"]["val_r2"], reverse=True)
    top    = all_results[0]
    top_r2 = top["best_r2"]

    print("=== Overall Best (by val R²) ===")
    print(f"  arch     : {top['arch']}")
    print(f"  val_r2   : {top_r2['val_r2']:.4f}")
    vl = top_r2["val_loss"]
    if vl is not None:
        print(f"  val_loss : {vl:.4f}")
    for k, v in top_r2["params"].items():
        print(f"  {k:20s}: {v}")

    best_ckpt_src = Path(top_r2["trial_dir"]) / "checkpoints" / "best.pt"
    best_ckpt_dst = hpo_root / "best_model.pt"
    if best_ckpt_src.exists():
        shutil.copy2(best_ckpt_src, best_ckpt_dst)
        print(f"Best model  → {best_ckpt_dst}")
    else:
        print(f"[warning] best checkpoint not found: {best_ckpt_src}")

    top_arch_dir = Path(top_r2["trial_dir"]).parent
    scaler_src   = top_arch_dir / "y_scaler_r2.json"
    if scaler_src.exists():
        shutil.copy2(scaler_src, hpo_root / "best_y_scaler.json")
        print(f"Best scaler → {hpo_root / 'best_y_scaler.json'}")

    save_json(hpo_root / "best_params.json", top_r2)
    save_json(hpo_root / "all_results.json", all_results)
    print(f"Saved       → {hpo_root / 'best_params.json'}")

    if csv_mode and "us_seed" in top_r2["params"]:
        best_us_seed    = top_r2["params"]["us_seed"]
        best_split_seed = top_r2["params"]["split_seed"]

        us_idx  = _undersample_indices(all_y[0], seed=best_us_seed,
                                        keep_neg_mid=args.keep_neg_mid,
                                        keep_neg=args.keep_neg,
                                        keep_low_pos=args.keep_low_pos)
        best_df = all_df[0].iloc[us_idx].reset_index(drop=True)

        train_idx, val_idx, test_idx = get_splits(base_cfg, best_df, seed=best_split_seed, csv_mode=True)

        best_df.to_csv(hpo_root / "best_dataset.csv", index=False)
        best_df.iloc[train_idx].reset_index(drop=True).to_csv(hpo_root / "best_train.csv", index=False)
        best_df.iloc[val_idx].reset_index(drop=True).to_csv(hpo_root  / "best_val.csv",   index=False)
        best_df.iloc[test_idx].reset_index(drop=True).to_csv(hpo_root / "best_test.csv",  index=False)
        np.save(hpo_root / "best_train_idx.npy", train_idx)
        np.save(hpo_root / "best_val_idx.npy",   val_idx)
        np.save(hpo_root / "best_test_idx.npy",  test_idx)

        print(
            f"Best dataset → {hpo_root / 'best_dataset.csv'}  "
            f"({len(best_df)} total, train={len(train_idx)}, "
            f"val={len(val_idx)}, test={len(test_idx)}, "
            f"us_seed={best_us_seed}, split_seed={best_split_seed})"
        )


if __name__ == "__main__":
    main()
