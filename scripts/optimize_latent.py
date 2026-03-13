#!/usr/bin/env python
"""
Latent-space optimization using diffusion models.

Two guidance modes (--guidance_type):
  cfg      : Classifier-Free Guidance — train conditional denoiser, interpolate cond/uncond
  gradient : Gradient Guidance       — train unconditional denoiser, add reg_head gradient each step
  none     : No guidance             — pure unconditional sampling from learned latent distribution

Usage:
  python scripts/optimize_latent.py --config configs/default.yaml \\
      --guidance_type cfg --guidance_scale 3.0 --target_pic50 8.0

  python scripts/optimize_latent.py --config configs/default.yaml \\
      --guidance_type gradient --guidance_scale 3.0
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.features.descriptors import smiles_to_descriptors
from src.features.fingerprints import smiles_to_fp
from src.features.graph import SME_NODE_DIM
from src.models.gnn import AttentiveFPModel, SMERGCNModel
from src.models.mlp import FingerprintMLP
from src.screening.latent_opt import (
    retrieve_nearest,
    sample_cfg,
    sample_gradient_guidance,
    train_diffusion,
    train_diffusion_cfg,
)
from src.utils.config import parse_config_args
from src.utils.io import load_checkpoint, load_numpy, resolve_run_dir
from src.utils.seed import set_seed


def _extra_args(parser):
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Number of latents to sample from diffusion")
    parser.add_argument("--guidance_type", type=str, default="cfg",
                        choices=["cfg", "gradient", "none"],
                        help="cfg: Classifier-Free Guidance | gradient: Gradient Guidance | none: unconditional")
    parser.add_argument("--guidance_scale", type=float, default=3.0,
                        help="CFG weight w or gradient scale s")
    parser.add_argument("--target_pic50", type=float, default=None,
                        help="[cfg only] target pIC50 to condition on (default: 90th percentile of train)")
    parser.add_argument("--diff_epochs", type=int, default=200,
                        help="Epochs to train diffusion model")
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--top_k", type=int, default=50,
                        help="Number of top candidates to save")
    parser.add_argument("--T", type=int, default=1000,
                        help="Diffusion timesteps")


def resolve_device(device_cfg: str) -> torch.device:
    d = device_cfg.lower()
    if d == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


def _extract_reg_head(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "out_layer"):
        return model.out_layer
    if hasattr(model, "reg_head"):
        return model.reg_head
    raise ValueError("Model does not expose out_layer or reg_head.")


def _build_model(cfg, train_smiles):
    feat_cfg = cfg["features"]
    model_cfg = cfg["model"]
    feature_type = str(feat_cfg.get("type", "fingerprint")).lower()
    model_type = str(model_cfg.get("type", "mlp")).lower()

    if feature_type == "fingerprint":
        if model_type != "mlp":
            raise ValueError("fingerprint features require model.type='mlp'.")
        x_probe = smiles_to_fp(
            train_smiles[:1],
            bits=int(feat_cfg.get("fp_bits", 2048)),
            radius=int(feat_cfg.get("fp_radius", 2)),
            use_chirality=bool(feat_cfg.get("use_chirality", False)),
        )
        if bool(feat_cfg.get("use_descriptors", False)):
            x_probe = np.concatenate([x_probe, smiles_to_descriptors(train_smiles[:1])], axis=1)
        return FingerprintMLP(
            input_dim=x_probe.shape[1],
            hidden_dims=list(model_cfg.get("hidden_dims", [512, 256, 128])),
            dropout=float(model_cfg.get("dropout", 0.2)),
            activation=str(model_cfg.get("activation", "relu")),
        )

    if feature_type == "graph":
        if model_type != "gnn":
            raise ValueError("graph features require model.type='gnn'.")
        return AttentiveFPModel(
            in_channels=10,
            edge_dim=6,
            hidden_dim=int(model_cfg.get("gnn_hidden", 200)),
            num_layers=int(model_cfg.get("gnn_layers", 3)),
            dropout=float(model_cfg.get("gnn_dropout", 0.1)),
        )

    if feature_type == "sme_graph":
        if model_type != "sme_rgcn":
            raise ValueError("sme_graph features require model.type='sme_rgcn'.")
        return SMERGCNModel(
            in_feats=SME_NODE_DIM,
            hidden_feats=list(model_cfg.get("sme_hidden_feats", [200, 200])),
            ffn_hidden=int(model_cfg.get("sme_ffn_hidden", 200)),
            rgcn_dropout=float(model_cfg.get("sme_rgcn_dropout", 0.25)),
            ffn_dropout=float(model_cfg.get("sme_ffn_dropout", 0.25)),
        )

    raise ValueError(f"Unsupported features.type: {feature_type}")


def main():
    cfg, args = parse_config_args(
        "Latent-space optimization via diffusion (CFG or gradient guidance)",
        extra_arg_fn=_extra_args,
    )
    data_cfg, tr_cfg = cfg["data"], cfg["training"]
    set_seed(int(tr_cfg.get("seed", 42)))

    run_dir = resolve_run_dir(cfg, create_if_missing=False)
    device = resolve_device(str(tr_cfg.get("device", "auto")))
    device_str = str(device)

    # ---- Load data --------------------------------------------------------
    df = pd.read_csv(run_dir / "cleaned_dataset.csv")
    train_idx = load_numpy(run_dir / "splits" / "train_idx.npy")
    smiles_all = df[data_cfg["smiles_col"]].astype(str).tolist()
    smiles_train = [smiles_all[i] for i in train_idx]
    y_all = df[data_cfg["activity_col"]].astype(float).values
    y_train = y_all[train_idx]

    z_train = load_numpy(run_dir / "latents_train.npy").astype(np.float32)
    if z_train.ndim != 2:
        raise ValueError("latents_train.npy must have shape (N, D).")
    latent_dim = int(z_train.shape[1])

    # ---- Load trained model -----------------------------------------------
    model = _build_model(cfg, smiles_train).to(device)
    load_checkpoint(run_dir / "checkpoints" / "best.pt", model=model, map_location=device)
    model.eval()
    reg_head = _extract_reg_head(model)

    guidance_type = str(args.guidance_type).lower()
    T = int(args.T)
    batch_size = int(tr_cfg.get("batch_size", 256))
    lr = float(tr_cfg.get("lr", 2e-4))

    out_dir = Path(run_dir) / "latent_opt" / guidance_type
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[latent_opt] guidance_type={guidance_type}  guidance_scale={args.guidance_scale}")
    print(f"[latent_opt] n_samples={args.n_samples}  sampler={args.sampler}  T={T}")

    # ---- Train diffusion + sample -----------------------------------------
    if guidance_type == "cfg":
        target_pic50 = args.target_pic50 if args.target_pic50 is not None else float(np.percentile(y_train, 90))
        print(f"[latent_opt] CFG target_pIC50={target_pic50:.3f}  (train 90th pct={np.percentile(y_train,90):.3f})")

        print("[latent_opt] Training conditional diffusion (CFG)...")
        denoiser, stats = train_diffusion_cfg(
            z_train=z_train,
            y_train=y_train,
            latent_dim=latent_dim,
            epochs=int(args.diff_epochs),
            batch_size=batch_size,
            lr=lr,
            T=T,
            device=device_str,
        )
        z_mean, z_std, c_mean, c_std = stats

        print("[latent_opt] Sampling with CFG...")
        z_samples = sample_cfg(
            denoiser=denoiser,
            target_pic50=target_pic50,
            latent_dim=latent_dim,
            n_samples=int(args.n_samples),
            T=T,
            guidance_scale=float(args.guidance_scale),
            sampler=str(args.sampler),
            device=device_str,
        )

        torch.save(
            {
                "state_dict": denoiser.state_dict(),
                "latent_dim": latent_dim,
                "time_dim": denoiser.time_dim,
                "cond_dim": denoiser.cond_dim,
                "z_mean": z_mean, "z_std": z_std,
                "c_mean": c_mean, "c_std": c_std,
                "T": T, "guidance_type": "cfg",
            },
            out_dir / "denoiser.pt",
        )

    elif guidance_type == "gradient":
        print("[latent_opt] Training unconditional diffusion...")
        denoiser, (z_mean, z_std) = train_diffusion(
            z_train=z_train,
            latent_dim=latent_dim,
            epochs=int(args.diff_epochs),
            batch_size=batch_size,
            lr=lr,
            T=T,
            device=device_str,
        )

        print("[latent_opt] Sampling with gradient guidance...")
        z_samples = sample_gradient_guidance(
            denoiser=denoiser,
            reg_head=reg_head,
            latent_dim=latent_dim,
            n_samples=int(args.n_samples),
            T=T,
            guidance_scale=float(args.guidance_scale),
            sampler=str(args.sampler),
            device=device_str,
        )

        torch.save(
            {
                "state_dict": denoiser.state_dict(),
                "latent_dim": latent_dim,
                "time_dim": denoiser.time_dim,
                "z_mean": z_mean, "z_std": z_std,
                "T": T, "guidance_type": "gradient",
            },
            out_dir / "denoiser.pt",
        )

    else:  # none — unconditional sampling
        print("[latent_opt] Training unconditional diffusion (no guidance)...")
        denoiser, (z_mean, z_std) = train_diffusion(
            z_train=z_train,
            latent_dim=latent_dim,
            epochs=int(args.diff_epochs),
            batch_size=batch_size,
            lr=lr,
            T=T,
            device=device_str,
        )

        print("[latent_opt] Sampling (unconditional)...")
        z_samples = sample_gradient_guidance(
            denoiser=denoiser,
            reg_head=reg_head,
            latent_dim=latent_dim,
            n_samples=int(args.n_samples),
            T=T,
            guidance_scale=0.0,  # no guidance
            sampler=str(args.sampler),
            device=device_str,
        )

    # ---- Score + retrieve -------------------------------------------------
    with torch.no_grad():
        z_tensor = torch.tensor(z_samples, dtype=torch.float32, device=device)
        pred_batch = reg_head(z_tensor).view(-1).detach().cpu().numpy()

    order = np.argsort(-pred_batch)
    top_indices = order[: min(int(args.top_k), len(order))]

    rows = []
    for idx in top_indices:
        z_i = z_samples[idx]
        pred_i = float(pred_batch[idx])
        for smi, sim in retrieve_nearest(z_i, z_train, smiles_train, top_k=5):
            rows.append({"smiles": smi, "pred_pIC50": pred_i, "cosine_sim": float(sim)})

    candidates_df = pd.DataFrame(rows)
    if len(candidates_df) > 0:
        candidates_df = (
            candidates_df
            .sort_values(["pred_pIC50", "cosine_sim"], ascending=[False, False])
            .drop_duplicates(subset=["smiles"], keep="first")
            .head(int(args.top_k))
            .reset_index(drop=True)
        )

    np.save(out_dir / "z_samples.npy", z_samples)
    candidates_df.to_csv(out_dir / "top_candidates.csv", index=False)

    best_pred = float(pred_batch.max()) if len(pred_batch) else float("nan")
    print(f"\n[latent_opt] sampled={len(z_samples)}  best_pred_pIC50={best_pred:.4f}")
    print(f"[latent_opt] saved → {out_dir}/")


if __name__ == "__main__":
    main()
