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

import os
import time
import datetime
import torch

_n_cpu = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
torch.set_num_threads(_n_cpu)
torch.set_num_interop_threads(_n_cpu)
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
    parser.add_argument("--time_dim",    type=int, default=32)
    parser.add_argument("--cond_dim",    type=int, default=32)
    parser.add_argument("--hidden_dim",  type=int, default=512)
    parser.add_argument("--p_uncond",      type=float, default=0.15,
                        help="Condition dropout probability during training")
    parser.add_argument("--use_vqc",       action="store_true",
                        help="(legacy) Use VQCConditionalDenoiser instead of MLP (requires 8-dim latents)")
    parser.add_argument("--denoiser_type", type=str, default="mlp",
                        choices=["mlp", "mlp_ortho", "unet", "unet_vqc", "vqc_angle", "vqc_angle_delta", "vqc_angle_reupload", "vqc_qubit_cond", "vqc_born_rule"],
                        help="Denoiser architecture: mlp | mlp_ortho | unet | unet_vqc | vqc_angle | vqc_angle_delta | vqc_angle_reupload | vqc_qubit_cond | vqc_born_rule")
    parser.add_argument("--unet_dims",     type=str, default="",
                        help="Comma-separated U-Net hidden dims, e.g. '256,128,64' (default: auto)")
    parser.add_argument("--n_layers",      type=int, default=2,
                        help="Number of VQC layers in U-Net bottleneck (unet_vqc only)")


def main():
    t_start = time.time()
    print(f"[train_diffusion] start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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

    z_val, y_val = None, None
    _val_z = run_dir / "latents_val.npy"
    _val_y = run_dir / "y_val.npy"
    if _val_z.exists() and _val_y.exists():
        z_val = load_numpy(_val_z).astype(np.float32)
        y_val = load_numpy(_val_y).astype(np.float32)

    print(f"[train_diffusion] latent_dim={latent_dim}  n_train={len(z_train)}"
          + (f"  n_val={len(z_val)}" if z_val is not None else ""))
    print(f"[train_diffusion] epochs={args.diff_epochs}  T={args.T}  device={device}")

    # ---- Resolve denoiser type (config < CLI) ----------------------------
    diff_cfg = cfg.get("diffusion", {})
    model_type = "vqc" if args.use_vqc else "mlp"  # legacy flag
    denoiser_type = (
        args.denoiser_type
        if args.denoiser_type != "mlp" or args.use_vqc
        else str(diff_cfg.get("denoiser_type", "mlp"))
    )
    if args.use_vqc:
        denoiser_type = "mlp"  # legacy --use_vqc maps to old VQCConditionalDenoiser
    _cfg_dims = diff_cfg.get("unet_dims", None)
    if args.unet_dims:
        unet_dims = [int(d) for d in args.unet_dims.split(",") if d.strip()] or None
    elif isinstance(_cfg_dims, list):
        unet_dims = [int(d) for d in _cfg_dims] or None
    elif _cfg_dims:
        unet_dims = [int(d) for d in str(_cfg_dims).split(",") if d.strip()] or None
    else:
        unet_dims = None
    n_layers     = args.n_layers if args.n_layers != 2 else int(diff_cfg.get("n_layers", args.n_layers))
    num_blocks   = int(diff_cfg.get("num_blocks", 6))
    initial_cnot  = bool(diff_cfg.get("initial_cnot", False))
    full_encoding = bool(diff_cfg.get("full_encoding", False))
    time_dim     = args.time_dim if args.time_dim != 32 else int(diff_cfg.get("time_dim", args.time_dim))
    cond_dim   = args.cond_dim if args.cond_dim != 32 else int(diff_cfg.get("cond_dim", args.cond_dim))

    print(f"[train_diffusion] denoiser_type={denoiser_type}  unet_dims={unet_dims}")
    print(f"[train_diffusion] time_dim={time_dim}  cond_dim={cond_dim}")

    # ---- Output dir (created before training for live checkpointing) ------
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    out_dir = Path(run_dir) / "diffusion" / date_str / f"T{args.T}_ep{args.diff_epochs}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Static metadata known before training
    _ckpt_meta = {
        "model_type": model_type,
        "denoiser_type": denoiser_type,
        "latent_dim": latent_dim,
        "T": args.T,
        "time_dim": time_dim,
        "cond_dim": cond_dim,
        "hidden_dim": args.hidden_dim,
        "n_layers": n_layers,
        "num_blocks": num_blocks,
        "unet_dims": unet_dims,
        "initial_cnot": initial_cnot,
        "full_encoding": full_encoding,
    }

    # ---- Train CFG denoiser -----------------------------------------------
    denoiser, best_state_dict, stats, history = train_diffusion_cfg(
        z_train=z_train,
        y_train=y_train,
        z_val=z_val,
        y_val=y_val,
        latent_dim=latent_dim,
        epochs=args.diff_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        T=args.T,
        time_dim=time_dim,
        cond_dim=cond_dim,
        hidden_dim=args.hidden_dim,
        p_uncond=args.p_uncond,
        model_type=model_type,
        denoiser_type=denoiser_type,
        n_layers=n_layers,
        num_blocks=num_blocks,
        initial_cnot=initial_cnot,
        full_encoding=full_encoding,
        device=device,
        best_ckpt_path=out_dir / "denoiser_cfg.pt",
        ckpt_meta=_ckpt_meta,
        history_path=out_dir / "loss_history.csv",
    )
    z_mean, z_std, c_mean, c_std = stats

    # ---- Save -------------------------------------------------------------

    def _ckpt(state_dict):
        ckpt = {
            "state_dict": state_dict,
            "model_type": model_type,
            "denoiser_type": denoiser_type,
            "latent_dim": latent_dim,
            "z_mean": z_mean,
            "z_std": z_std,
            "c_mean": c_mean,
            "c_std": c_std,
            "T": args.T,
        }
        if model_type == "vqc":
            ckpt["n_qubits"] = denoiser.n_qubits
            ckpt["n_layers"] = denoiser.n_layers
        elif denoiser_type in ("vqc_angle", "vqc_angle_delta", "vqc_angle_reupload"):
            ckpt["n_layers"]    = denoiser.n_layers
            ckpt["num_blocks"]  = denoiser.num_blocks
            ckpt["use_delta"]    = denoiser.use_delta
            ckpt["use_reupload"] = denoiser.use_reupload
            ckpt["initial_cnot"]  = denoiser.initial_cnot
            ckpt["full_encoding"] = denoiser.full_encoding
            ckpt["time_dim"]     = time_dim
            ckpt["cond_dim"]    = cond_dim
        elif denoiser_type == "vqc_qubit_cond":
            ckpt["n_layers"]   = denoiser.n_layers
            ckpt["num_blocks"] = denoiser.num_blocks
        elif denoiser_type == "vqc_born_rule":
            ckpt["n_layers"]   = denoiser.n_layers
            ckpt["num_blocks"] = denoiser.num_blocks
            ckpt["time_dim"]   = time_dim
            ckpt["cond_dim"]   = cond_dim
        elif denoiser_type in ("unet", "unet_vqc"):
            ckpt["unet_dims"] = denoiser.unet_dims
            ckpt["n_layers"] = denoiser.n_layers
            ckpt["time_dim"] = time_dim
            ckpt["cond_dim"] = cond_dim
        else:  # mlp
            ckpt["time_dim"] = denoiser.time_dim
            ckpt["cond_dim"] = denoiser.cond_dim
            ckpt["hidden_dim"] = args.hidden_dim
        return ckpt

    # best loss checkpoint — re-save with full denoiser-specific metadata
    torch.save(_ckpt(best_state_dict), out_dir / "denoiser_cfg.pt")
    # final epoch checkpoint
    torch.save(_ckpt(denoiser.state_dict()), out_dir / "denoiser_cfg_final.pt")
    # loss_history.csv already written per-epoch during training

    print(f"\n[train_diffusion] saved → {out_dir}/denoiser_cfg.pt (best loss)")
    print(f"[train_diffusion] saved → {out_dir}/denoiser_cfg_final.pt (final epoch)")
    print(f"[train_diffusion] saved → {out_dir}/loss_history.csv")

    elapsed = time.time() - t_start
    print(f"[train_diffusion] end:   {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  "
          f"(elapsed {elapsed/3600:.2f}h)")


if __name__ == "__main__":
    main()
