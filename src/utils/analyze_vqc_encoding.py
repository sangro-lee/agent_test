#!/usr/bin/env python
"""
VQC encoding quality analysis — two pairwise analyses:

  1. Activity discrimination
       Measures whether the VQC maps molecules with different pIC50 to
       distinguishable quantum states.
       Metric: Spearman r( |y_i - y_j|,  1 - F_ij )
       Expected sign: positive (larger pIC50 gap → lower fidelity → larger 1-F)

  2. Encoding continuity
       Measures whether nearby latent vectors stay nearby in quantum state space.
       Metric: Spearman r( ||z_i - z_j||,  1 - F_ij )
       Expected sign: positive (farther in latent → lower fidelity)

Both analyses share the same quantum state computation and the same pairwise
sample, so the runtime cost is paid once.

Usage:
  python -m src.utils.analyze_vqc_encoding \\
      --latent_path outputs/runs/sme_random/latents_train.npy \\
      --y_path      outputs/runs/sme_random/y_train.npy \\
      --ckpt_path   outputs/runs/sme_random/diffusion/.../denoiser_cfg.pt \\
      --n_qubits 4 --n_layers 2 --block_idx 0 \\
      --n_samples 300 --max_pairs 5000 \\
      --outdir outputs/vqc_encoding_analysis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.expressibility import (
    make_vqc_reupload_qnode,
    extract_vqc_theta_from_ckpt,
    _preprocess_z_for_reupload,
    _vqc_reupload_theta_size,
)


# ── quantum state computation ─────────────────────────────────────────────────

def compute_states(
    latents: np.ndarray,
    n_qubits: int,
    n_layers: int,
    theta: np.ndarray,
    initial_cnot: bool = False,
    use_tanh: bool = True,
) -> np.ndarray:
    circuit = make_vqc_reupload_qnode(n_qubits, n_layers, initial_cnot, use_data=True)
    log_every = max(1, len(latents) // 10)
    states = []
    for i, z in enumerate(latents):
        inputs = _preprocess_z_for_reupload(z, n_qubits, n_layers, use_tanh=use_tanh)
        states.append(np.array(circuit(inputs, theta)))
        if (i + 1) % log_every == 0:
            print(f"  [states] {i + 1}/{len(latents)}", flush=True)
    return np.stack(states, axis=0)


# ── pairwise sampling ─────────────────────────────────────────────────────────

def sample_pairs(n: int, max_pairs: int | None, seed: int = 0):
    """Return arrays of indices (idx_i, idx_j) with i < j."""
    total = n * (n - 1) // 2
    rng = np.random.default_rng(seed)
    if max_pairs is None or max_pairs >= total:
        ii, jj = np.tril_indices(n, k=-1)
        return ii, jj
    seen: set[tuple[int, int]] = set()
    pairs_i, pairs_j = [], []
    for _ in range(max_pairs * 20):
        if len(pairs_i) >= max_pairs:
            break
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        pairs_i.append(key[0])
        pairs_j.append(key[1])
    return np.array(pairs_i), np.array(pairs_j)


def compute_pairwise_quantities(
    states: np.ndarray,
    latents: np.ndarray,
    y: np.ndarray,
    max_pairs: int | None,
    seed: int = 0,
):
    """
    For each sampled pair (i, j) compute:
      fidelity  F_ij = |<psi_i|psi_j>|^2
      delta_y   |y_i - y_j|
      dist_z    ||z_i - z_j||_2
    """
    idx_i, idx_j = sample_pairs(len(states), max_pairs, seed)
    print(f"  [pairs] {len(idx_i)} pairs ...", flush=True)

    fidelity = np.array([
        abs(np.vdot(states[i], states[j])) ** 2
        for i, j in zip(idx_i, idx_j)
    ], dtype=np.float64)

    delta_y = np.abs(y[idx_i] - y[idx_j])
    dist_z  = np.linalg.norm(latents[idx_i] - latents[idx_j], axis=1)

    return fidelity, delta_y, dist_z


# ── plotting ──────────────────────────────────────────────────────────────────

def scatter_with_trend(
    x: np.ndarray, y_vals: np.ndarray,
    xlabel: str, ylabel: str, title: str,
    out_path: str,
    n_bins: int = 30,
) -> None:
    """Scatter (alpha-blended) + binned mean ± std trend line."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y_vals, alpha=0.05, s=4, color="steelblue", rasterized=True)

    bin_edges = np.linspace(x.min(), x.max(), n_bins + 1)
    centers, means, stds = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (x >= lo) & (x < hi)
        if mask.sum() < 5:
            continue
        centers.append(0.5 * (lo + hi))
        means.append(y_vals[mask].mean())
        stds.append(y_vals[mask].std())

    centers = np.array(centers)
    means   = np.array(means)
    stds    = np.array(stds)

    ax.plot(centers, means, color="crimson", linewidth=2, label="bin mean")
    ax.fill_between(centers, means - stds, means + stds,
                    color="crimson", alpha=0.2)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[analyze_vqc] plot → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VQC encoding quality: activity discrimination + continuity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--latent_path", required=True)
    parser.add_argument("--y_path",      required=True,
                        help="y_train.npy  pIC50 values (same order as latents)")
    parser.add_argument("--ckpt_path",   required=True,
                        help="Denoiser checkpoint .pt (AngleVQCDenoiser)")
    parser.add_argument("--n_qubits",    type=int, default=4)
    parser.add_argument("--n_layers",    type=int, default=2)
    parser.add_argument("--block_idx",   type=int, default=0)
    parser.add_argument("--initial_cnot", action="store_true")
    parser.add_argument("--no_tanh",     action="store_true")
    parser.add_argument("--n_samples",   type=int, default=300)
    parser.add_argument("--max_pairs",   type=int, default=5000)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--outdir",      type=str,
                        default="outputs/vqc_encoding_analysis")
    args = parser.parse_args()

    out_dir  = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_tanh = not args.no_tanh
    tanh_tag = "tanh" if use_tanh else "no_tanh"

    # ── Load data ─────────────────────────────────────────────────────────────
    latents = np.load(args.latent_path).astype(np.float64)
    y_all   = np.load(args.y_path).astype(np.float64).ravel()
    assert len(latents) == len(y_all), "latent / y length mismatch"

    rng = np.random.default_rng(args.seed)
    if len(latents) > args.n_samples:
        idx     = rng.choice(len(latents), size=args.n_samples, replace=False)
        latents = latents[idx]
        y_all   = y_all[idx]
    n = len(latents)
    print(f"[analyze_vqc] n={n}  latent_dim={latents.shape[1]}"
          f"  y=[{y_all.min():.2f}, {y_all.max():.2f}]")

    # ── Load theta ────────────────────────────────────────────────────────────
    theta    = extract_vqc_theta_from_ckpt(args.ckpt_path, block_idx=args.block_idx)
    expected = _vqc_reupload_theta_size(args.n_qubits, args.n_layers)
    if len(theta) != expected:
        raise ValueError(f"theta size {len(theta)} != expected {expected}")
    print(f"[analyze_vqc] block={args.block_idx}  use_tanh={use_tanh}")

    # ── Quantum states ────────────────────────────────────────────────────────
    print("[analyze_vqc] computing quantum states ...")
    states = compute_states(latents, args.n_qubits, args.n_layers,
                            theta, args.initial_cnot, use_tanh)

    # ── Pairwise quantities ───────────────────────────────────────────────────
    fidelity, delta_y, dist_z = compute_pairwise_quantities(
        states, latents, y_all, args.max_pairs, args.seed)
    state_dist = 1.0 - fidelity

    # ── Correlations ─────────────────────────────────────────────────────────
    r_act, p_act = stats.spearmanr(delta_y, state_dist)
    r_cnt, p_cnt = stats.spearmanr(dist_z,  state_dist)

    sep = "=" * 56
    print(f"\n{sep}")
    print(f"  block={args.block_idx}  {tanh_tag}  n_pairs={len(fidelity)}")
    print(f"  Activity discrimination  r={r_act:.4f}  p={p_act:.2e}"
          f"  (|Δy| vs 1-F, want r>0)")
    print(f"  Encoding continuity      r={r_cnt:.4f}  p={p_cnt:.2e}"
          f"  (||Δz|| vs 1-F, want r>0)")
    print(sep)

    # ── Plots ─────────────────────────────────────────────────────────────────
    scatter_with_trend(
        delta_y, state_dist,
        xlabel="|y_i − y_j|  (ΔpIC50)",
        ylabel="1 − F  (quantum distance)",
        title=(f"Activity Discrimination  block={args.block_idx}  {tanh_tag}\n"
               f"Spearman r={r_act:.4f}  p={p_act:.2e}"),
        out_path=str(out_dir /
                     f"activity_discrimination_blk{args.block_idx}_{tanh_tag}.png"),
    )
    scatter_with_trend(
        dist_z, state_dist,
        xlabel="‖z_i − z_j‖₂  (latent distance)",
        ylabel="1 − F  (quantum distance)",
        title=(f"Encoding Continuity  block={args.block_idx}  {tanh_tag}\n"
               f"Spearman r={r_cnt:.4f}  p={p_cnt:.2e}"),
        out_path=str(out_dir /
                     f"encoding_continuity_blk{args.block_idx}_{tanh_tag}.png"),
    )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    pd.DataFrame({
        "fidelity":   fidelity,
        "state_dist": state_dist,
        "delta_y":    delta_y,
        "dist_z":     dist_z,
    }).to_csv(out_dir / f"pairs_blk{args.block_idx}_{tanh_tag}.csv", index=False)

    pd.DataFrame([{
        "block_idx":             args.block_idx,
        "use_tanh":              use_tanh,
        "n_samples":             n,
        "n_pairs":               len(fidelity),
        "spearman_r_activity":   r_act,
        "p_activity":            p_act,
        "spearman_r_continuity": r_cnt,
        "p_continuity":          p_cnt,
    }]).to_csv(out_dir / f"metrics_blk{args.block_idx}_{tanh_tag}.csv", index=False)

    print(f"\n[analyze_vqc] saved to {out_dir}/")


if __name__ == "__main__":
    main()
