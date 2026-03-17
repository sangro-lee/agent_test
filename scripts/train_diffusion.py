#!/usr/bin/env python
"""
Train CFG diffusion model on train-set latents.

Saves denoiser to:
  outputs/runs/<exp>/diffusion/denoiser_cfg.pt

Usage:
  python scripts/train_diffusion.py --config /tmp/sme_random_resolved.yaml
  python scripts/train_diffusion.py --config /tmp/sme_random_resolved.yaml \\
      --diff_epochs 500 --batch_size 256 --lr 2e-4
"""
from __future__ import annotations

import torch
import numpy as np
from pathlib import Path

from src.screening.latent_opt import train_diffusion_cfg
from src.utils.config import parse_config_args
from src.utils.io import load_numpy, resolve_run_dir
from src.utils.seed import set_seed


def _extra_args(parser):
    parser.add_argument("--diff_epochs", type=int, default=200)
    parser.add_argument("--batch_size",  type=int, default=256)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--T",           type=int, default=1000)
    parser.add_argument("--p_uncond",    type=float, default=0.15,
                        help="Condition dropout probability during training")


def main():
    cfg, args = parse_config_args(
        "Train CFG diffusion model on train-set latents",
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

    # ---- Load data --------------------------------------------------------
    z_train = load_numpy(run_dir / "latents_train.npy").astype(np.float32)
    y_train = load_numpy(run_dir / "y_train.npy").astype(np.float32)
    latent_dim = int(z_train.shape[1])

    print(f"[train_diffusion] latent_dim={latent_dim}  n_train={len(z_train)}")
    print(f"[train_diffusion] epochs={args.diff_epochs}  T={args.T}  device={device}")

    # ---- Train CFG denoiser -----------------------------------------------
    denoiser, best_state_dict, stats = train_diffusion_cfg(
        z_train=z_train,
        y_train=y_train,
        latent_dim=latent_dim,
        epochs=args.diff_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        T=args.T,
        p_uncond=args.p_uncond,
        device=device,
    )
    z_mean, z_std, c_mean, c_std = stats

    # ---- Save -------------------------------------------------------------
    out_dir = Path(run_dir) / "diffusion" / f"T{args.T}_ep{args.diff_epochs}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _ckpt(state_dict):
        return {
            "state_dict": state_dict,
            "latent_dim": latent_dim,
            "time_dim": denoiser.time_dim,
            "cond_dim": denoiser.cond_dim,
            "z_mean": z_mean,
            "z_std": z_std,
            "c_mean": c_mean,
            "c_std": c_std,
            "T": args.T,
        }

    # best loss checkpoint (used by sample_cfg.py)
    torch.save(_ckpt(best_state_dict), out_dir / "denoiser_cfg.pt")
    # final epoch checkpoint
    torch.save(_ckpt(denoiser.state_dict()), out_dir / "denoiser_cfg_final.pt")

    print(f"\n[train_diffusion] saved → {out_dir}/denoiser_cfg.pt (best loss)")
    print(f"[train_diffusion] saved → {out_dir}/denoiser_cfg_final.pt (final epoch)")


if __name__ == "__main__":
    main()
