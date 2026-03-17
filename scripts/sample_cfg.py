#!/usr/bin/env python
"""
Sample molecules using a trained CFG diffusion denoiser.

Loads:
  outputs/runs/<exp>/diffusion/denoiser_cfg.pt
  outputs/runs/<exp>/latents_train.npy

Saves to:
  outputs/runs/<exp>/diffusion/cfg_w{guidance_scale}/top_candidates.csv
  outputs/runs/<exp>/diffusion/cfg_w{guidance_scale}/z_samples.npy

Usage:
  python scripts/sample_cfg.py --config /tmp/sme_random_resolved.yaml \\
      --guidance_scale 3.0 --target_pic50 8.0

  # Try multiple guidance scales
  for w in 1.0 2.0 3.0 5.0; do
    python scripts/sample_cfg.py --config /tmp/sme_random_resolved.yaml \\
        --guidance_scale $w --target_pic50 8.0
  done
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.models.diffusion import ConditionalDenoisingMLP
from src.screening.latent_opt import retrieve_nearest, sample_cfg
from src.utils.config import parse_config_args
from src.utils.io import load_numpy, resolve_run_dir
from src.utils.seed import set_seed


def _extra_args(parser):
    parser.add_argument("--guidance_scale", type=float, default=3.0,
                        help="CFG weight w (0=uncond, 1=cond, >1=amplified)")
    parser.add_argument("--target_pic50",   type=float, default=None,
                        help="Target pIC50 in original scale (default: 90th pct of train)")
    parser.add_argument("--n_samples",      type=int,   default=500)
    parser.add_argument("--top_k",          type=int,   default=50)
    parser.add_argument("--sampler",        type=str,   default="ddim",
                        choices=["ddim", "ddpm"])


def main():
    cfg, args = parse_config_args(
        "Sample with CFG diffusion denoiser",
        extra_arg_fn=_extra_args,
    )
    tr_cfg = cfg["training"]
    set_seed(int(tr_cfg.get("seed", 42)))

    run_dir = resolve_run_dir(cfg, create_if_missing=False)
    device = str(tr_cfg.get("device", "auto"))
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # ---- Load saved denoiser ----------------------------------------------
    ckpt_path = Path(run_dir) / "diffusion" / "denoiser_cfg.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Denoiser not found: {ckpt_path}\n"
            f"Run train_diffusion.py first."
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    latent_dim = int(ckpt["latent_dim"])
    T = int(ckpt["T"])

    denoiser = ConditionalDenoisingMLP(
        latent_dim=latent_dim,
        time_dim=int(ckpt["time_dim"]),
        cond_dim=int(ckpt["cond_dim"]),
    )
    denoiser.load_state_dict(ckpt["state_dict"])
    denoiser.set_normalization(
        z_mean=torch.tensor(ckpt["z_mean"]),
        z_std=torch.tensor(ckpt["z_std"]),
        c_mean=torch.tensor([ckpt["c_mean"]]),
        c_std=torch.tensor([ckpt["c_std"]]),
    )

    # ---- Load train latents for nearest-neighbor retrieval ----------------
    df = pd.read_csv(run_dir / "cleaned_dataset.csv")
    train_idx = load_numpy(run_dir / "splits" / "train_idx.npy")
    smiles_all = df[cfg["data"]["smiles_col"]].astype(str).tolist()
    smiles_train = [smiles_all[i] for i in train_idx]
    y_all = df[cfg["data"]["activity_col"]].astype(float).values
    y_train = y_all[train_idx]

    z_train = load_numpy(run_dir / "latents_train.npy").astype(np.float32)

    # ---- Determine target pIC50 -------------------------------------------
    target_pic50 = args.target_pic50
    if target_pic50 is None:
        target_pic50 = float(np.percentile(y_train, 90))
    print(f"[sample_cfg] target_pIC50={target_pic50:.3f}  guidance_scale={args.guidance_scale}")
    print(f"[sample_cfg] n_samples={args.n_samples}  sampler={args.sampler}  T={T}")

    # ---- Sample -----------------------------------------------------------
    z_samples = sample_cfg(
        denoiser=denoiser,
        target_pic50=target_pic50,
        latent_dim=latent_dim,
        n_samples=args.n_samples,
        T=T,
        guidance_scale=args.guidance_scale,
        sampler=args.sampler,
        device=device,
    )

    # ---- Score via reg_head (last linear layer of encoder) ----------------
    # Load encoder model to score sampled latents
    from src.utils.io import load_checkpoint

    # Simpler: score via denoiser's stored normalization + a note
    # We need the original encoder's reg_head — load it
    try:
        from src.utils.io import load_checkpoint
        from src.features.descriptors import smiles_to_descriptors
        from src.features.fingerprints import smiles_to_fp
        from src.features.graph import SME_NODE_DIM
        from src.models.gnn import AttentiveFPModel, SMERGCNModel
        from src.models.mlp import FingerprintMLP

        feat_cfg  = cfg["features"]
        model_cfg = cfg["model"]
        feature_type = str(feat_cfg.get("type", "fingerprint")).lower()

        if feature_type == "fingerprint":
            x_probe = smiles_to_fp(smiles_train[:1],
                bits=int(feat_cfg.get("fp_bits", 2048)),
                radius=int(feat_cfg.get("fp_radius", 2)),
                use_chirality=bool(feat_cfg.get("use_chirality", False)))
            if bool(feat_cfg.get("use_descriptors", False)):
                x_probe = np.concatenate([x_probe, smiles_to_descriptors(smiles_train[:1])], axis=1)
            model = FingerprintMLP(input_dim=x_probe.shape[1],
                hidden_dims=list(model_cfg.get("hidden_dims", [512, 256, 128])),
                dropout=float(model_cfg.get("dropout", 0.2)))
        elif feature_type == "graph":
            model = AttentiveFPModel(in_channels=10, edge_dim=6,
                hidden_dim=int(model_cfg.get("gnn_hidden", 200)),
                num_layers=int(model_cfg.get("gnn_layers", 3)))
        elif feature_type == "sme_graph":
            model = SMERGCNModel(in_feats=SME_NODE_DIM,
                hidden_feats=list(model_cfg.get("sme_hidden_feats", [200, 200])),
                ffn_hidden=int(model_cfg.get("sme_ffn_hidden", 200)))
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

        load_checkpoint(run_dir / "checkpoints" / "best.pt", model=model, map_location=device)
        model.eval()

        reg_head = model.out_layer if hasattr(model, "out_layer") else model.reg_head
        device_t = torch.device(device)
        reg_head = reg_head.to(device_t)
        z_tensor = torch.tensor(z_samples, dtype=torch.float32, device=device_t)
        with torch.no_grad():
            pred_batch = reg_head(z_tensor).view(-1).cpu().numpy()
    except Exception as e:
        print(f"[sample_cfg] Warning: could not load encoder for scoring ({e})")
        print("[sample_cfg] Scores set to 0.")
        pred_batch = np.zeros(len(z_samples))

    # ---- Retrieve nearest train molecules ---------------------------------
    order = np.argsort(-pred_batch)
    top_indices = order[: min(args.top_k, len(order))]

    rows = []
    for idx in top_indices:
        pred_i = float(pred_batch[idx])
        for smi, sim in retrieve_nearest(z_samples[idx], z_train, smiles_train, top_k=5):
            rows.append({"smiles": smi, "pred_pIC50": pred_i, "cosine_sim": float(sim)})

    candidates_df = (
        pd.DataFrame(rows)
        .sort_values(["pred_pIC50", "cosine_sim"], ascending=[False, False])
        .drop_duplicates(subset=["smiles"], keep="first")
        .head(args.top_k)
        .reset_index(drop=True)
    )

    # ---- Save -------------------------------------------------------------
    w_tag = f"cfg_w{args.guidance_scale:.1f}"
    out_dir = Path(run_dir) / "diffusion" / w_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "z_samples.npy", z_samples)
    candidates_df.to_csv(out_dir / "top_candidates.csv", index=False)

    best_pred = float(pred_batch.max()) if len(pred_batch) else float("nan")
    print(f"\n[sample_cfg] best_pred_pIC50={best_pred:.4f}")
    print(f"[sample_cfg] saved → {out_dir}/")
    print(candidates_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
