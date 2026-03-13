#!/usr/bin/env python
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.features.descriptors import smiles_to_descriptors
from src.features.fingerprints import smiles_to_fp
from src.models.gnn import AttentiveFPModel
from src.models.mlp import FingerprintMLP
from src.screening.latent_opt import optimize_latent_vector, retrieve_nearest
from src.utils.config import parse_config_args
from src.utils.io import load_checkpoint, load_numpy, resolve_run_dir, save_numpy


def _extra_args(parser):
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=10)


def resolve_device(device_cfg: str) -> torch.device:
    d = device_cfg.lower()
    if d == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


def main():
    cfg, args = parse_config_args("Optimize latent vector and retrieve nearest", extra_arg_fn=_extra_args)
    data_cfg, feat_cfg, model_cfg, tr_cfg = (
        cfg["data"],
        cfg["features"],
        cfg["model"],
        cfg["training"],
    )

    run_dir = resolve_run_dir(cfg, create_if_missing=False)
    device = resolve_device(str(tr_cfg.get("device", "auto")))

    df = pd.read_csv(run_dir / "cleaned_dataset.csv")
    train_idx = load_numpy(run_dir / "splits" / "train_idx.npy")
    smiles_all = df[data_cfg["smiles_col"]].astype(str).tolist()

    train_smiles = [smiles_all[i] for i in train_idx]
    z_train = load_numpy(run_dir / "latents_train.npy")

    feature_type = str(feat_cfg.get("type", "fingerprint")).lower()
    model_type = str(model_cfg.get("type", "mlp")).lower()

    if feature_type == "fingerprint":
        if model_type != "mlp":
            raise ValueError("For fingerprint features, model.type must be 'mlp'.")

        x_probe = smiles_to_fp(
            train_smiles[:1],
            bits=int(feat_cfg.get("fp_bits", 2048)),
            radius=int(feat_cfg.get("fp_radius", 2)),
            use_chirality=bool(feat_cfg.get("use_chirality", False)),
        )
        if bool(feat_cfg.get("use_descriptors", False)):
            x_probe = np.concatenate([x_probe, smiles_to_descriptors(train_smiles[:1])], axis=1)

        model = FingerprintMLP(
            input_dim=x_probe.shape[1],
            hidden_dims=list(model_cfg.get("hidden_dims", [512, 256, 128])),
            dropout=float(model_cfg.get("dropout", 0.2)),
            activation=str(model_cfg.get("activation", "relu")),
        )

    elif feature_type == "graph":
        if model_type != "gnn":
            raise ValueError("For graph features, model.type must be 'gnn'.")
        model = AttentiveFPModel(
            in_channels=10,
            edge_dim=6,
            hidden_dim=int(model_cfg.get("gnn_hidden", 200)),
            num_layers=int(model_cfg.get("gnn_layers", 3)),
            dropout=float(model_cfg.get("gnn_dropout", 0.1)),
        )
    else:
        raise ValueError(f"Unsupported features.type: {feature_type}")

    model = model.to(device)
    load_checkpoint(run_dir / "checkpoints" / "best.pt", model=model, map_location=device)

    z_init = np.mean(z_train, axis=0)
    z_opt = optimize_latent_vector(
        model=model,
        z_init=z_init,
        n_steps=int(args.n_steps),
        lr=float(tr_cfg.get("lr", 1e-3)),
    )

    nearest = retrieve_nearest(
        z_opt=z_opt,
        z_train=z_train,
        smiles_train=train_smiles,
        top_k=int(args.top_k),
    )

    out_dir = Path(run_dir) / "latent_opt"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_numpy(out_dir / "z_opt.npy", z_opt)

    nearest_df = pd.DataFrame(nearest, columns=["smiles", "cosine_similarity"])
    nearest_df.to_csv(out_dir / "nearest_train_smiles.csv", index=False)

    print(f"[latent_opt] run_dir={run_dir}")
    print(f"[latent_opt] saved: {out_dir / 'nearest_train_smiles.csv'}")


if __name__ == "__main__":
    main()
