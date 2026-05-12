#!/usr/bin/env python
"""
Full forward pass VQC encoding analysis.

Loads the complete AngleVQCDenoiser and runs the actual forward pass with
real (z_t, t, c) inputs. Captures the exact input each VQC block receives
via forward hooks — including trained lambda_scales, input_biases, and the
sequential transformation from previous blocks.

Then runs:
  1. Activity discrimination : |Δy| vs (1 - F)
  2. Encoding continuity     : ||Δz_0|| vs (1 - F)

Key differences from analyze_vqc_encoding.py (block-isolated):
  - Trained lambda_scales / input_biases are active (actual input scaling)
  - Each block sees the output of the previous block, not raw z
  - Conditioning (t, c) flows through residuals, changing block k+1's input

Usage:
  python -m src.utils.analyze_vqc_full_forward \\
      --latent_path outputs/runs/.../latents_train.npy \\
      --y_path      outputs/runs/.../y_train.npy \\
      --ckpt_path   outputs/runs/.../diffusion/.../denoiser_cfg.pt \\
      --t_frac 0.0 \\
      --outdir outputs/vqc_full_forward_analysis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from src.models.diffusion import NoiseScheduler
from src.models.vqc_diffusion import AngleVQCDenoiser
from src.utils.analyze_vqc_encoding import (
    sample_pairs,
    scatter_with_trend,
)
from src.utils.expressibility import make_vqc_reupload_qnode


# ── model loading ─────────────────────────────────────────────────────────────

def load_denoiser(ckpt_path: str) -> tuple[AngleVQCDenoiser, dict]:
    """Reconstruct AngleVQCDenoiser from checkpoint and return (model, ckpt)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    denoiser = AngleVQCDenoiser(
        latent_dim   = int(ckpt["latent_dim"]),
        n_layers     = int(ckpt.get("n_layers", 2)),
        num_blocks   = int(ckpt.get("num_blocks", 6)),
        time_dim     = int(ckpt.get("time_dim", 32)),
        cond_dim     = int(ckpt.get("cond_dim", 32)),
        device_type  = "default.qubit",
        use_reupload = True,
        initial_cnot = bool(ckpt.get("initial_cnot", False)),
        full_encoding= bool(ckpt.get("full_encoding", False)),
    )
    denoiser.load_state_dict(ckpt["state_dict"])
    denoiser.set_normalization(
        z_mean = torch.as_tensor(ckpt["z_mean"]),
        z_std  = torch.as_tensor(ckpt["z_std"]),
        c_mean = torch.as_tensor(ckpt.get("c_mean", torch.zeros(1))),
        c_std  = torch.as_tensor(ckpt.get("c_std",  torch.ones(1))),
    )
    denoiser.eval()
    return denoiser, ckpt


# ── noise schedule ────────────────────────────────────────────────────────────

def add_noise(
    z0_norm: np.ndarray,
    t_idx: int,
    T: int,
    seed: int = 0,
) -> np.ndarray:
    """Add DDPM noise to normalized clean latents at integer timestep t_idx."""
    scheduler = NoiseScheduler(T=T)
    ab = float(scheduler.alpha_bars[t_idx])
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(z0_norm.shape).astype(np.float32)
    return np.sqrt(ab) * z0_norm + np.sqrt(1.0 - ab) * eps


# ── hook-based input capture ──────────────────────────────────────────────────

def capture_block_inputs(
    denoiser: AngleVQCDenoiser,
    z_t_norm: np.ndarray,
    t_norm: float,
    c_norm: np.ndarray,
) -> dict[int, np.ndarray]:
    """
    Run one forward pass and return the actual input tensor each VQC block received.

    Returns
    -------
    dict: block_idx → np.ndarray of shape (n_samples, n_layers * n_qubits)
    """
    B = len(z_t_norm)
    captured: dict[int, np.ndarray] = {}
    hooks = []

    for i, block in enumerate(denoiser.vqc_blocks):
        def _make_hook(idx: int):
            def hook(module, args):
                # args[0]: input tensor (B, n_layers * n_qubits)
                captured[idx] = args[0].detach().cpu().numpy()
            return hook
        hooks.append(block.register_forward_pre_hook(_make_hook(i)))

    z_t_t  = torch.tensor(z_t_norm, dtype=torch.float32)
    t_t    = torch.full((B,), t_norm, dtype=torch.float32)
    c_t    = torch.tensor(c_norm, dtype=torch.float32).view(B, 1)

    with torch.no_grad():
        denoiser(z_t_t, t_t, c_t)

    for h in hooks:
        h.remove()

    return captured


# ── quantum state computation from pre-processed inputs ───────────────────────

def compute_states_from_inputs(
    block_inputs: np.ndarray,
    theta: np.ndarray,
    n_qubits: int,
    n_layers: int,
    initial_cnot: bool = False,
) -> np.ndarray:
    """
    Compute statevectors from the actual (pre-processed) VQC block inputs.

    block_inputs : (n_samples, n_layers * n_qubits) — already lambda-scaled + biased
    theta        : flat array of trained variational parameters

    The input is passed DIRECTLY to the circuit (no additional tanh/scaling).
    """
    circuit = make_vqc_reupload_qnode(n_qubits, n_layers, initial_cnot, use_data=True)
    log_every = max(1, len(block_inputs) // 10)
    states = []
    for i, inp in enumerate(block_inputs):
        states.append(np.array(circuit(inp.astype(np.float64), theta)))
        if (i + 1) % log_every == 0:
            print(f"  [states] {i + 1}/{len(block_inputs)}", flush=True)
    return np.stack(states, axis=0)


# ── pairwise analysis (shared with analyze_vqc_encoding) ─────────────────────

def pairwise_analysis(
    states: np.ndarray,
    z0: np.ndarray,
    y: np.ndarray,
    max_pairs: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    idx_i, idx_j = sample_pairs(len(states), max_pairs, seed)
    fidelity = np.array([
        abs(np.vdot(states[i], states[j])) ** 2
        for i, j in zip(idx_i, idx_j)
    ], dtype=np.float64)
    state_dist = 1.0 - fidelity
    delta_y    = np.abs(y[idx_i] - y[idx_j])
    dist_z0    = np.linalg.norm(z0[idx_i] - z0[idx_j], axis=1)
    r_act, p_act = stats.spearmanr(delta_y, state_dist)
    r_cnt, p_cnt = stats.spearmanr(dist_z0,  state_dist)
    return fidelity, state_dist, delta_y, dist_z0, r_act, p_act, r_cnt, p_cnt


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full forward VQC encoding analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--latent_path", required=True,
                        help="latents_train.npy (clean latents z_0)")
    parser.add_argument("--y_path",      required=True,
                        help="y_train.npy  pIC50 (same order as latents)")
    parser.add_argument("--ckpt_path",   required=True,
                        help="AngleVQCDenoiser checkpoint .pt")
    parser.add_argument("--t_frac",      type=float, default=0.0,
                        help="Noise level as fraction of T (0.0 = clean, 0.5 = mid-noise)")
    parser.add_argument("--blocks",      type=str, default="all",
                        help="Comma-separated block indices to analyze, or 'all'")
    parser.add_argument("--n_samples",   type=int, default=None,
                        help="Max molecules to use (default: all)")
    parser.add_argument("--max_pairs",   type=int, default=5000)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--outdir",      type=str,
                        default="outputs/vqc_full_forward_analysis")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[full_forward] loading denoiser from {args.ckpt_path}")
    denoiser, ckpt = load_denoiser(args.ckpt_path)
    T          = int(ckpt["T"])
    n_qubits   = denoiser.latent_dim
    n_layers   = denoiser.n_layers
    num_blocks = denoiser.num_blocks
    initial_cnot = denoiser.initial_cnot

    z_mean = denoiser.z_mean.numpy()
    z_std  = denoiser.z_std.numpy()
    c_mean = float(denoiser.c_mean.item())
    c_std  = float(denoiser.c_std.item())

    block_indices = (
        list(range(num_blocks))
        if args.blocks == "all"
        else [int(b) for b in args.blocks.split(",")]
    )
    print(f"[full_forward] n_qubits={n_qubits}  n_layers={n_layers}"
          f"  num_blocks={num_blocks}  T={T}  t_frac={args.t_frac}")
    print(f"[full_forward] analyzing blocks: {block_indices}")

    # ── Load data ─────────────────────────────────────────────────────────────
    z0_raw = np.load(args.latent_path).astype(np.float32)
    y_raw  = np.load(args.y_path).astype(np.float64).ravel()
    assert len(z0_raw) == len(y_raw), "latent / y length mismatch"

    rng = np.random.default_rng(args.seed)
    if args.n_samples is not None and len(z0_raw) > args.n_samples:
        idx    = rng.choice(len(z0_raw), size=args.n_samples, replace=False)
        z0_raw = z0_raw[idx]
        y_raw  = y_raw[idx]
    n = len(z0_raw)
    print(f"[full_forward] n={n}  y=[{y_raw.min():.2f}, {y_raw.max():.2f}]")

    # ── Normalize z and y (denoiser convention) ───────────────────────────────
    z0_norm = ((z0_raw - z_mean) / (z_std + 1e-8)).astype(np.float32)
    c_norm  = ((y_raw - c_mean) / (c_std + 1e-8)).astype(np.float32)

    # ── Add noise at timestep t ───────────────────────────────────────────────
    t_idx  = max(0, int(args.t_frac * T) - 1)
    t_norm = float(t_idx) / float(T - 1) if T > 1 else 0.0
    if t_idx > 0:
        z_t_norm = add_noise(z0_norm, t_idx, T, seed=args.seed)
        print(f"[full_forward] noise added at t={t_idx}/{T}")
    else:
        z_t_norm = z0_norm.copy()
        print(f"[full_forward] t=0: using clean latents")

    # ── Forward pass: capture actual VQC inputs per block ─────────────────────
    print("[full_forward] running forward pass with hooks ...")
    block_inputs = capture_block_inputs(denoiser, z_t_norm, t_norm, c_norm)
    print(f"[full_forward] captured inputs for blocks: {sorted(block_inputs.keys())}")

    # ── Per-block analysis ────────────────────────────────────────────────────
    all_metrics = []

    for blk in block_indices:
        if blk not in block_inputs:
            print(f"[full_forward] block {blk} not captured — skipping")
            continue

        print(f"\n[full_forward] ── Block {blk} ──────────────────────────────────")
        inp = block_inputs[blk]   # (n, n_layers * n_qubits) — already pre-processed

        # Extract trained theta for this block
        sd  = ckpt["state_dict"]
        key = f"vqc_blocks.{blk}.theta"
        if key not in sd:
            print(f"  theta key '{key}' not found — skipping")
            continue
        theta = sd[key].detach().cpu().numpy().astype(np.float64)
        print(f"  theta shape={theta.shape}  input shape={inp.shape}")

        # Quantum states from actual forward-pass inputs
        print("  computing quantum states ...")
        states = compute_states_from_inputs(inp, theta, n_qubits, n_layers, initial_cnot)

        # Pairwise analysis
        print("  pairwise analysis ...")
        fid, sdist, dy, dz, r_act, p_act, r_cnt, p_cnt = pairwise_analysis(
            states, z0_raw, y_raw, args.max_pairs, args.seed)

        all_metrics.append({
            "block": blk, "n_pairs": len(fid),
            "spearman_r_activity":   r_act, "p_activity":   p_act,
            "spearman_r_continuity": r_cnt, "p_continuity": p_cnt,
        })
        print(f"  Activity   r={r_act:.4f}  p={p_act:.2e}")
        print(f"  Continuity r={r_cnt:.4f}  p={p_cnt:.2e}")

        # Plots
        tag = f"blk{blk}_t{t_idx}"
        scatter_with_trend(
            dy, sdist,
            xlabel="|y_i − y_j|  (ΔpIC50)",
            ylabel="1 − F  (quantum distance)",
            title=(f"Activity Discrimination  block={blk}  t={t_idx}/{T}\n"
                   f"Spearman r={r_act:.4f}  p={p_act:.2e}"),
            out_path=str(out_dir / f"activity_{tag}.png"),
        )
        scatter_with_trend(
            dz, sdist,
            xlabel="‖z_i − z_j‖₂  (latent distance, z_0)",
            ylabel="1 − F  (quantum distance)",
            title=(f"Encoding Continuity  block={blk}  t={t_idx}/{T}\n"
                   f"Spearman r={r_cnt:.4f}  p={p_cnt:.2e}"),
            out_path=str(out_dir / f"continuity_{tag}.png"),
        )

        pd.DataFrame({
            "fidelity": fid, "state_dist": sdist,
            "delta_y": dy,   "dist_z0":    dz,
        }).to_csv(out_dir / f"pairs_{tag}.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    if all_metrics:
        summary = pd.DataFrame(all_metrics)
        summary.to_csv(out_dir / "summary_metrics.csv", index=False)
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"[full_forward summary]  t_frac={args.t_frac}  n={n}")
        print(summary.to_string(index=False))
        print(sep)

    print(f"\n[full_forward] saved to {out_dir}/")


if __name__ == "__main__":
    main()
