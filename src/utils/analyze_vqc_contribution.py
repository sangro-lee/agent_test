#!/usr/bin/env python
"""
VQC contribution analysis.

Three complementary diagnostics on a trained AngleVQCDenoiser:

  1. q_out variance
       Is the VQC circuit actually responsive to different inputs?
       Metric: std( q_out ) per qubit per block (collapse → near 0 → VQC bypassed)

  2. q_out continuity  (classical, no PennyLane statevectors)
       Do similar z_t inputs produce similar q_out vectors?
       Metric: Spearman r( ||Δz_t||, ||Δq_out|| )  — want r > 0

  3. Quantum statevector continuity  (quantum, from actual forward-pass inputs)
       Same captured inputs → compute statevectors → pairwise fidelity.
       Metric: Spearman r( ||Δz_0||, 1 - F )  — want r > 0
       (skip with --skip_sv for speed)

  4. y conditioning sensitivity
       How much does changing y (pIC50) shift the noise prediction?
       Metric: ||ε̂(z_t, t, y_high) - ε̂(z_t, t, y_low)||₂  per sample
       Near zero → model ignores conditioning.

Usage:
  python -m src.utils.analyze_vqc_contribution \\
      --latent_path outputs/runs/.../latents_train.npy \\
      --y_path      outputs/runs/.../y_train.npy \\
      --ckpt_path   outputs/runs/.../diffusion/.../denoiser_cfg.pt \\
      --t_frac 0.0 \\
      --outdir outputs/vqc_contribution_analysis
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
from src.utils.analyze_vqc_encoding import sample_pairs, scatter_with_trend
from src.utils.expressibility import make_vqc_reupload_qnode


# ── model loading ─────────────────────────────────────────────────────────────

def load_denoiser(ckpt_path: str) -> tuple[AngleVQCDenoiser, dict]:
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
        z_mean = ckpt["z_mean"],
        z_std  = ckpt["z_std"],
        c_mean = ckpt.get("c_mean", torch.zeros(1)),
        c_std  = ckpt.get("c_std",  torch.ones(1)),
    )
    denoiser.eval()
    return denoiser, ckpt


# ── noise ─────────────────────────────────────────────────────────────────────

def add_noise(z0_norm: np.ndarray, t_idx: int, T: int, seed: int = 0) -> np.ndarray:
    scheduler = NoiseScheduler(T=T)
    ab = float(scheduler.alpha_bars[t_idx])
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(z0_norm.shape).astype(np.float32)
    return np.sqrt(ab) * z0_norm + np.sqrt(1.0 - ab) * eps


# ── hook capture ──────────────────────────────────────────────────────────────

def capture_hooks(
    denoiser: AngleVQCDenoiser,
    z_t_norm: np.ndarray,
    t_norm: float,
    c_norm: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Run one forward pass and capture per-block:
      block_inputs: inp before VQC  (B, n_layers * n_qubits)  — pre-hook
      block_qouts:  raw PauliZ out  (B, n_qubits)             — post-hook
    """
    B = len(z_t_norm)
    block_inputs: dict[int, np.ndarray] = {}
    block_qouts:  dict[int, np.ndarray] = {}
    hooks = []

    for i, block in enumerate(denoiser.vqc_blocks):
        def _make_pre(idx: int):
            def hook(module, args):
                block_inputs[idx] = args[0].detach().cpu().numpy()
            return hook

        def _make_post(idx: int):
            def hook(module, input, output):
                block_qouts[idx] = output.detach().cpu().float().numpy()
            return hook

        hooks.append(block.register_forward_pre_hook(_make_pre(i)))
        hooks.append(block.register_forward_hook(_make_post(i)))

    z_t_t = torch.tensor(z_t_norm, dtype=torch.float32)
    t_t   = torch.full((B,), t_norm, dtype=torch.float32)
    c_t   = torch.tensor(c_norm, dtype=torch.float32).view(B, 1)

    with torch.no_grad():
        denoiser(z_t_t, t_t, c_t)

    for h in hooks:
        h.remove()

    return block_inputs, block_qouts


# ── y conditioning sensitivity ────────────────────────────────────────────────

def cond_sensitivity(
    denoiser: AngleVQCDenoiser,
    z_t_norm: np.ndarray,
    t_norm: float,
    y_raw: np.ndarray,
    c_mean: float,
    c_std: float,
) -> np.ndarray:
    """||ε̂(z_t, t, y_90pct) - ε̂(z_t, t, y_10pct)||₂ per sample."""
    B = len(z_t_norm)
    y_hi = float(np.percentile(y_raw, 90))
    y_lo = float(np.percentile(y_raw, 10))
    c_hi_norm = (y_hi - c_mean) / (c_std + 1e-8)
    c_lo_norm = (y_lo - c_mean) / (c_std + 1e-8)

    z_t_t = torch.tensor(z_t_norm, dtype=torch.float32)
    t_t   = torch.full((B,), t_norm, dtype=torch.float32)
    c_hi  = torch.full((B, 1), c_hi_norm, dtype=torch.float32)
    c_lo  = torch.full((B, 1), c_lo_norm, dtype=torch.float32)

    with torch.no_grad():
        eps_hi = denoiser(z_t_t, t_t, c_hi).cpu().numpy()
        eps_lo = denoiser(z_t_t, t_t, c_lo).cpu().numpy()

    return np.linalg.norm(eps_hi - eps_lo, axis=1)   # (B,)


# ── quantum statevectors ──────────────────────────────────────────────────────

def compute_statevectors(
    block_inputs: np.ndarray,
    theta: np.ndarray,
    n_qubits: int,
    n_layers: int,
    initial_cnot: bool = False,
) -> np.ndarray:
    """Statevectors from actual forward-pass VQC inputs (no extra preprocessing)."""
    circuit = make_vqc_reupload_qnode(n_qubits, n_layers, initial_cnot, use_data=True)
    log_every = max(1, len(block_inputs) // 10)
    states = []
    for i, inp in enumerate(block_inputs):
        states.append(np.array(circuit(inp.astype(np.float64), theta)))
        if (i + 1) % log_every == 0:
            print(f"  [statevec] {i+1}/{len(block_inputs)}", flush=True)
    return np.stack(states, axis=0)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VQC contribution: q_out, statevec continuity, y sensitivity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--latent_path", required=True,
                        help="latents_train.npy (clean z_0)")
    parser.add_argument("--y_path",      required=True,
                        help="y_train.npy pIC50 values")
    parser.add_argument("--ckpt_path",   required=True,
                        help="AngleVQCDenoiser checkpoint .pt")
    parser.add_argument("--t_frac",      type=float, default=0.0,
                        help="Noise fraction of T (0.0 = clean)")
    parser.add_argument("--blocks",      type=str, default="all",
                        help="Comma-separated block indices or 'all'")
    parser.add_argument("--n_samples",   type=int, default=None,
                        help="Max molecules (default: all)")
    parser.add_argument("--max_pairs",   type=int, default=5000)
    parser.add_argument("--skip_sv",     action="store_true",
                        help="Skip statevector computation (fast classical-only mode)")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--outdir",      type=str,
                        default="outputs/vqc_contribution_analysis")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[vqc_contrib] loading denoiser from {args.ckpt_path}")
    denoiser, ckpt = load_denoiser(args.ckpt_path)
    T            = int(ckpt["T"])
    n_qubits     = denoiser.latent_dim
    n_layers     = denoiser.n_layers
    num_blocks   = denoiser.num_blocks
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
    print(f"[vqc_contrib] n_qubits={n_qubits}  n_layers={n_layers}"
          f"  num_blocks={num_blocks}  T={T}  t_frac={args.t_frac}")
    print(f"[vqc_contrib] analyzing blocks: {block_indices}"
          f"  skip_sv={args.skip_sv}")

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
    print(f"[vqc_contrib] n={n}  y=[{y_raw.min():.2f}, {y_raw.max():.2f}]")

    # ── Normalize ─────────────────────────────────────────────────────────────
    z0_norm = ((z0_raw - z_mean) / (z_std + 1e-8)).astype(np.float32)
    c_norm  = ((y_raw - c_mean) / (c_std + 1e-8)).astype(np.float32)

    # ── Noise ─────────────────────────────────────────────────────────────────
    t_idx  = max(0, int(args.t_frac * T) - 1)
    t_norm = float(t_idx) / float(T - 1) if T > 1 else 0.0
    if t_idx > 0:
        z_t_norm = add_noise(z0_norm, t_idx, T, seed=args.seed)
        print(f"[vqc_contrib] noise at t={t_idx}/{T}")
    else:
        z_t_norm = z0_norm.copy()
        print(f"[vqc_contrib] t=0: using clean latents")

    # ── Hook capture ──────────────────────────────────────────────────────────
    print("[vqc_contrib] running forward pass with hooks ...")
    block_inputs, block_qouts = capture_hooks(denoiser, z_t_norm, t_norm, c_norm)
    print(f"[vqc_contrib] captured blocks: {sorted(block_inputs.keys())}")

    # ── y conditioning sensitivity ────────────────────────────────────────────
    print("[vqc_contrib] computing y conditioning sensitivity ...")
    sens = cond_sensitivity(denoiser, z_t_norm, t_norm, y_raw, c_mean, c_std)
    print(f"[vqc_contrib] ||ε̂_high - ε̂_low||  mean={sens.mean():.4f}  std={sens.std():.4f}")
    np.save(out_dir / f"cond_sensitivity_t{t_idx}.npy", sens)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sens, bins=40, color="steelblue", edgecolor="white")
    ax.set_xlabel("||ε̂(y₉₀) − ε̂(y₁₀)||₂", fontsize=11)
    ax.set_ylabel("count", fontsize=11)
    ax.set_title(f"y Conditioning Sensitivity  t={t_idx}/{T}\n"
                 f"mean={sens.mean():.4f}  std={sens.std():.4f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"cond_sensitivity_t{t_idx}.png", dpi=150)
    plt.close(fig)
    print(f"[vqc_contrib] sensitivity plot saved")

    # ── Shared pairwise index ─────────────────────────────────────────────────
    idx_i, idx_j = sample_pairs(n, args.max_pairs, args.seed)
    dist_zt = np.linalg.norm(
        z_t_norm[idx_i].astype(np.float64) - z_t_norm[idx_j].astype(np.float64),
        axis=1,
    )
    dist_z0 = np.linalg.norm(
        z0_raw[idx_i].astype(np.float64) - z0_raw[idx_j].astype(np.float64),
        axis=1,
    )

    # ── Per-block analysis ────────────────────────────────────────────────────
    all_metrics = []

    for blk in block_indices:
        if blk not in block_qouts:
            print(f"[vqc_contrib] block {blk} not captured — skipping")
            continue

        print(f"\n[vqc_contrib] ── Block {blk} ─────────────────────────────────────────")
        tag = f"blk{blk}_t{t_idx}"

        # ── q_out variance ────────────────────────────────────────────────────
        qout = block_qouts[blk].astype(np.float64)    # (n, n_qubits)
        qout_std = qout.std(axis=0)                    # (n_qubits,)
        qout_std_mean = float(qout_std.mean())
        print(f"  q_out std/qubit: {qout_std.round(4)}  (mean={qout_std_mean:.4f})")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(n_qubits), qout_std, color="steelblue")
        ax.axhline(0.05, color="crimson", linestyle="--", linewidth=1.5, label="0.05")
        ax.set_xlabel("qubit index", fontsize=11)
        ax.set_ylabel("std(q_out) across samples", fontsize=11)
        ax.set_title(f"q_out Variance  block={blk}  t={t_idx}/{T}\n"
                     f"mean std={qout_std_mean:.4f}", fontsize=11)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"qout_variance_{tag}.png", dpi=150)
        plt.close(fig)

        # ── q_out continuity ──────────────────────────────────────────────────
        dist_qout = np.linalg.norm(qout[idx_i] - qout[idx_j], axis=1)
        r_qcont, p_qcont = stats.spearmanr(dist_zt, dist_qout)
        r_qcont, p_qcont = float(r_qcont), float(p_qcont)
        print(f"  q_out continuity  r(||Δz_t||, ||Δq_out||)={r_qcont:.4f}  p={p_qcont:.2e}")

        scatter_with_trend(
            dist_zt, dist_qout,
            xlabel="||z_t,i − z_t,j||₂  (noisy latent distance)",
            ylabel="||q_out,i − q_out,j||₂",
            title=(f"q_out Continuity  block={blk}  t={t_idx}/{T}\n"
                   f"Spearman r={r_qcont:.4f}  p={p_qcont:.2e}"),
            out_path=str(out_dir / f"qout_continuity_{tag}.png"),
        )

        blk_metrics: dict = {
            "block":         blk,
            "n_pairs":       len(idx_i),
            "qout_std_mean": qout_std_mean,
            "r_qout_cont":   r_qcont,
            "p_qout_cont":   p_qcont,
        }

        # ── Quantum statevector continuity ────────────────────────────────────
        if not args.skip_sv:
            inp = block_inputs[blk]   # (n, n_layers*n_qubits) — already processed
            key = f"vqc_blocks.{blk}.theta"
            if key not in ckpt["state_dict"]:
                print(f"  theta key '{key}' not found — skipping statevec")
            else:
                theta = ckpt["state_dict"][key].detach().cpu().numpy().astype(np.float64)
                print(f"  theta shape={theta.shape}  inp shape={inp.shape}")
                print("  computing statevectors ...")
                states = compute_statevectors(inp, theta, n_qubits, n_layers, initial_cnot)

                fidelity = np.array([
                    abs(np.vdot(states[i], states[j])) ** 2
                    for i, j in zip(idx_i, idx_j)
                ], dtype=np.float64)
                state_dist = 1.0 - fidelity
                r_svcont, p_svcont = stats.spearmanr(dist_z0, state_dist)
                r_svcont, p_svcont = float(r_svcont), float(p_svcont)
                print(f"  SV continuity   r(||Δz_0||, 1-F)={r_svcont:.4f}  p={p_svcont:.2e}")

                scatter_with_trend(
                    dist_z0, state_dist,
                    xlabel="||z_0,i − z_0,j||₂  (clean latent distance)",
                    ylabel="1 − F  (quantum distance)",
                    title=(f"Statevector Continuity  block={blk}  t={t_idx}/{T}\n"
                           f"Spearman r={r_svcont:.4f}  p={p_svcont:.2e}"),
                    out_path=str(out_dir / f"sv_continuity_{tag}.png"),
                )

                pd.DataFrame({
                    "dist_z0":    dist_z0,
                    "dist_zt":    dist_zt,
                    "dist_qout":  dist_qout,
                    "fidelity":   fidelity,
                    "state_dist": state_dist,
                }).to_csv(out_dir / f"pairs_{tag}.csv", index=False)

                blk_metrics["r_sv_cont"] = r_svcont
                blk_metrics["p_sv_cont"] = p_svcont

        all_metrics.append(blk_metrics)

    # ── Summary ───────────────────────────────────────────────────────────────
    if all_metrics:
        summary = pd.DataFrame(all_metrics)
        summary.to_csv(out_dir / "summary_metrics.csv", index=False)
        sep = "=" * 64
        print(f"\n{sep}")
        print(f"[vqc_contrib summary]  t_frac={args.t_frac}  n={n}")
        print(summary.to_string(index=False))
        print(f"\ny sensitivity:  mean={sens.mean():.4f}  std={sens.std():.4f}"
              f"  (y_10%={np.percentile(y_raw,10):.2f}, y_90%={np.percentile(y_raw,90):.2f})")
        print(sep)

    print(f"\n[vqc_contrib] saved to {out_dir}/")


if __name__ == "__main__":
    main()
