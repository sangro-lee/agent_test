#!/usr/bin/env python
"""
Expressibility analysis for parameterized quantum circuits (PQC) using PennyLane.

Two separate analysis modes — do NOT mix them:

  circuit_only
    Randomize θ only. No data is used.
    Evaluates how well the circuit ARCHITECTURE itself covers the Hilbert space.
    |ψ(θ)⟩ = U_ansatz(θ)|0⟩

  data_dependent / fixed_theta
    Fix θ (trained or random), vary z only.
    Evaluates whether the DATA ENCODING maps distinct latent vectors to
    distinguishable quantum states.  θ is NOT sampled here.
    |ψ(z, θ_fixed)⟩ = U_ansatz(θ_fixed) U_enc(z)|0⟩

  data_dependent / random_theta
    Vary both z_i and θ_i (randomly paired).
    Evaluates the effective state distribution of the FULL VQC, but mixes
    the contribution of data encoding and trainable ansatz.
    |ψ(z_i, θ_i)⟩ = U_ansatz(θ_i) U_enc(z_i)|0⟩

Two circuit types via --vqc_type:

  generic          (default)
    Simple RY+RZ per qubit + CNOT ring per layer.
    Use --encoding_type angle|amplitude for data encoding.

  angle_reupload
    Exact circuit from AngleVQCDenoiser (_make_reupload_vqc_layer):
      every layer: AngleEmbedding(tanh(z)*π, RY) → RZ,RY,RZ per qubit
                   → ring entanglement (RZ,CNOT,RZ,RY,CNOT,RY)
    What is NOT included (on purpose):
      - num_blocks / residual stacking     : classical; doesn't affect quantum state space
      - time_mlp, cond_mlp, cond2wb, norms: classical conditioning; affects how VQC
                                            output is used, not the state itself
    What IS included:
      - n_qubits (= latent_dim), n_layers (circuit depth per block)
      - initial_cnot flag
      - optionally: trained theta from denoiser checkpoint (one block)

Usage:
  # Generic circuit — circuit_only
  python -m src.utils.expressibility \\
      --mode circuit_only --n_qubits 4 --n_layers 2 --n_samples 200

  # angle_reupload — circuit_only (ansatz expressibility, no data)
  python -m src.utils.expressibility \\
      --mode circuit_only --vqc_type angle_reupload \\
      --n_qubits 4 --n_layers 2 --n_samples 200

  # angle_reupload — data_dependent, fixed random theta
  python -m src.utils.expressibility \\
      --mode data_dependent --vqc_type angle_reupload \\
      --latent_path outputs/runs/sme_random/latents_train.npy \\
      --data_mode fixed_theta --n_qubits 4 --n_layers 2

  # angle_reupload — data_dependent, fixed TRAINED theta from checkpoint (block 0)
  python -m src.utils.expressibility \\
      --mode data_dependent --vqc_type angle_reupload \\
      --latent_path outputs/runs/sme_random/latents_train.npy \\
      --ckpt_path outputs/runs/sme_random/diffusion/.../denoiser_cfg.pt \\
      --block_idx 0 --data_mode fixed_theta --n_qubits 4 --n_layers 2
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml


# ── 1. Device ─────────────────────────────────────────────────────────────────

def build_device(n_qubits: int) -> qml.devices.DefaultQubit:
    """Return a PennyLane statevector simulator."""
    return qml.device("default.qubit", wires=n_qubits)


# ── 2. Generic data encoding ──────────────────────────────────────────────────

def data_encoding(z: np.ndarray, n_qubits: int, encoding_type: str) -> None:
    """
    Apply data encoding gates inside a QNode circuit (generic mode).

    ── To replace with your own encoding ──────────────────────────────────────
    Delete or comment out the body below and add your custom PennyLane gates.
    ──────────────────────────────────────────────────────────────────────────
    """
    if encoding_type == "angle":
        for i in range(n_qubits):
            angle = float(z[i]) if i < len(z) else 0.0
            qml.RY(angle, wires=i)

    elif encoding_type == "amplitude":
        dim = 2 ** n_qubits
        vec = np.array(z, dtype=complex)
        if len(vec) < dim:
            vec = np.pad(vec, (0, dim - len(vec)))
        else:
            vec = vec[:dim]
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            vec = np.zeros(dim, dtype=complex); vec[0] = 1.0
        else:
            vec /= norm
        qml.StatePrep(vec, wires=range(n_qubits))

    else:
        raise ValueError(f"Unknown encoding_type: {encoding_type!r}")


# ── 3. Generic ansatz ─────────────────────────────────────────────────────────

def ansatz(params: np.ndarray, n_qubits: int, n_layers: int) -> None:
    """
    Generic trainable ansatz: RY + RZ per qubit, then CNOT ring.
    params shape: (n_layers, n_qubits, 2)

    ── To replace with your own VQC ansatz ────────────────────────────────────
    Delete or comment out the body below.  Adjust sample_random_params() if
    you change the params shape.
    ──────────────────────────────────────────────────────────────────────────
    """
    for l in range(n_layers):
        for q in range(n_qubits):
            qml.RY(params[l, q, 0], wires=q)
            qml.RZ(params[l, q, 1], wires=q)
        for q in range(n_qubits):
            qml.CNOT(wires=[q, (q + 1) % n_qubits])


# ── 4. Generic QNode factory ──────────────────────────────────────────────────

def make_qnode(
    n_qubits: int,
    n_layers: int,
    encoding_type: Optional[str] = None,
    use_data: bool = False,
):
    """
    Build and return a QNode outputting the full statevector (generic mode).

    use_data=False: |ψ(θ)⟩ = U_ansatz(θ)|0⟩           circuit(params)
    use_data=True:  |ψ(z,θ)⟩ = U_ansatz(θ) U_enc(z)|0⟩  circuit(z, params)
    """
    dev = build_device(n_qubits)

    if use_data:
        @qml.qnode(dev, interface="numpy")
        def circuit(z, params):
            data_encoding(z, n_qubits, encoding_type)
            ansatz(params, n_qubits, n_layers)
            return qml.state()
    else:
        @qml.qnode(dev, interface="numpy")
        def circuit(params):
            ansatz(params, n_qubits, n_layers)
            return qml.state()

    return circuit


# ── 5. Generic parameter sampling ─────────────────────────────────────────────

def sample_random_params(
    n_samples: int, n_layers: int, n_qubits: int, seed: int = 0,
) -> np.ndarray:
    """Sample generic ansatz params uniformly from [0, 2π]. Shape: (n_samples, n_layers, n_qubits, 2)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 2 * np.pi, size=(n_samples, n_layers, n_qubits, 2))


# ── 6. Generic circuit-only state sampling ────────────────────────────────────

def sample_states_circuit_only(
    n_qubits: int, n_layers: int, n_samples: int, seed: int = 0,
) -> np.ndarray:
    """
    circuit_only (generic): vary θ only, no data.
    Evaluates expressibility of the circuit ARCHITECTURE.
    """
    circuit = make_qnode(n_qubits, n_layers, use_data=False)
    all_params = sample_random_params(n_samples, n_layers, n_qubits, seed)
    states, log_every = [], max(1, n_samples // 10)
    for i, params in enumerate(all_params):
        states.append(np.array(circuit(params)))
        if (i + 1) % log_every == 0:
            print(f"  [circuit_only] {i + 1}/{n_samples}", flush=True)
    return np.stack(states, axis=0)


# ── 7. Generic data-dependent state sampling ──────────────────────────────────

def sample_states_data_dependent(
    latent_path: str,
    n_qubits: int,
    n_layers: int,
    encoding_type: str,
    data_mode: str,
    n_samples: int,
    trained_params_path: Optional[str] = None,
    seed: int = 0,
) -> np.ndarray:
    """
    data_dependent (generic): loads latent vectors from CSV or NPY.

    fixed_theta  : vary z, fix θ → evaluates DATA ENCODING quality.
    random_theta : vary both z and θ → evaluates FULL VQC distribution.
    """
    p = Path(latent_path)
    if p.suffix == ".npy":
        latents = np.load(latent_path).astype(np.float64)
    elif p.suffix in (".csv", ".tsv"):
        sep = "\t" if p.suffix == ".tsv" else ","
        latents = pd.read_csv(latent_path, sep=sep).values.astype(np.float64)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")

    rng = np.random.default_rng(seed)
    if len(latents) > n_samples:
        latents = latents[rng.choice(len(latents), size=n_samples, replace=False)]
    n = len(latents)
    print(f"  [data_dependent] {n} latent vectors  shape={latents.shape}")

    circuit = make_qnode(n_qubits, n_layers, encoding_type=encoding_type, use_data=True)
    log_every = max(1, n // 10)
    states = []

    if data_mode == "fixed_theta":
        if trained_params_path is not None:
            theta = np.load(trained_params_path).astype(np.float64)
            if theta.shape != (n_layers, n_qubits, 2):
                raise ValueError(f"theta shape {theta.shape} != ({n_layers}, {n_qubits}, 2)")
            print(f"  [fixed_theta] trained θ from {trained_params_path}")
        else:
            theta = rng.uniform(0.0, 2 * np.pi, size=(n_layers, n_qubits, 2))
            print(f"  [fixed_theta] random θ (seed={seed})")
        for i, z in enumerate(latents):
            states.append(np.array(circuit(z, theta)))
            if (i + 1) % log_every == 0:
                print(f"  [fixed_theta] {i + 1}/{n}", flush=True)

    elif data_mode == "random_theta":
        all_params = sample_random_params(n, n_layers, n_qubits, seed)
        for i, (z, params) in enumerate(zip(latents, all_params)):
            states.append(np.array(circuit(z, params)))
            if (i + 1) % log_every == 0:
                print(f"  [random_theta] {i + 1}/{n}", flush=True)

    else:
        raise ValueError(f"Unknown data_mode: {data_mode!r}")

    return np.stack(states, axis=0)


# ══ VQC angle_reupload (actual circuit from AngleVQCDenoiser) ════════════════
#
# Mirrors _make_reupload_vqc_layer in src/models/vqc_diffusion.py,
# but outputs qml.state() instead of PauliZ expectation values.
#
# Circuit per layer l:
#   AngleEmbedding(inputs[l*D:(l+1)*D] * π, rotation="Y")
#   [initial_cnot ring after l==0 if enabled]
#   RZ, RY, RZ  per qubit
#   ring entanglement: RZ(-π/2), CNOT, RZ, RY, CNOT, RY  per pair
#
# What is intentionally NOT included:
#   num_blocks / residual stacking → classical; no shared quantum state across blocks
#   time_mlp, cond_mlp, cond2wb   → classical conditioning; irrelevant to state space
#   output_scales, LayerNorm       → post-processing of expectation values; classical
#   lambda_scales, input_biases    → optionally usable; defaulted to λ=1, bias=0 here
# ─────────────────────────────────────────────────────────────────────────────

def _vqc_reupload_theta_size(n_qubits: int, n_layers: int) -> int:
    """Total number of trainable parameters for one VQC block."""
    params_per_layer = n_qubits * 3 + n_qubits * 3   # rot(RZ,RY,RZ) + entangle(RZ,RY,RY)
    return params_per_layer * n_layers


def _preprocess_z_for_reupload(
    z: np.ndarray, n_qubits: int, n_layers: int,
    lambda_scales: Optional[np.ndarray] = None,
    input_biases: Optional[np.ndarray] = None,
    use_tanh: bool = True,
) -> np.ndarray:
    """
    Prepare circuit inputs from a latent vector z.

    Mirrors AngleVQCDenoiser.forward() with use_reupload=True:
        x_enc = tanh(z)                               # bounded to (-1, 1)
        scaled[l, :] = lambda_scales[l] * x_enc + input_biases[l]   # (n_layers, n_qubits)
        inputs = scaled.flatten()                      # (n_layers * n_qubits,)

    If lambda_scales/input_biases are None, uses λ=1, bias=0 (identity scaling).

    Parameters
    ----------
    lambda_scales : (n_layers, n_qubits) or None
    input_biases  : (n_layers, n_qubits) or None
    use_tanh      : if False, skip tanh and use z directly (for ablation)
    """
    z_bounded = np.tanh(z.astype(np.float64)) if use_tanh else z.astype(np.float64)
    # Take first n_qubits dims (or zero-pad if latent_dim < n_qubits)
    z_q = z_bounded[:n_qubits] if len(z_bounded) >= n_qubits \
        else np.pad(z_bounded, (0, n_qubits - len(z_bounded)))

    if lambda_scales is None:
        scaled = np.tile(z_q, (n_layers, 1))           # (n_layers, n_qubits), λ=1 bias=0
    else:
        scaled = lambda_scales * z_q[None, :] + (input_biases if input_biases is not None
                                                  else np.zeros_like(lambda_scales))
    return scaled.flatten().astype(np.float64)          # (n_layers * n_qubits,)


def make_vqc_reupload_qnode(
    n_qubits: int,
    n_layers: int,
    initial_cnot: bool = False,
    use_data: bool = True,
):
    """
    Build a statevector QNode with the exact vqc_angle_reupload circuit.

    use_data=True  : circuit(inputs, theta)  — inputs: (n_layers*n_qubits,)
    use_data=False : circuit(theta)          — pure variational, no data encoding
    """
    dev = build_device(n_qubits)
    _pi = math.pi

    def _vqc_body(inputs_or_none, theta):
        param_idx = 0
        for l in range(n_layers):
            # ── Data re-uploading: encode at the start of EVERY layer ──
            if inputs_or_none is not None:
                x_l = inputs_or_none[l * n_qubits : (l + 1) * n_qubits]
                qml.AngleEmbedding(x_l * _pi, wires=range(n_qubits), rotation="Y")
            # Fixed CNOT ring once after the first encoding (matches vqc_diffusion.py)
            if initial_cnot and l == 0:
                for j in range(n_qubits):
                    qml.CNOT(wires=[j, (j + 1) % n_qubits])
            # ── Variational rotations: RZ, RY, RZ per qubit ──
            for i in range(n_qubits):
                qml.RZ(theta[param_idx],     wires=i)
                qml.RY(theta[param_idx + 1], wires=i)
                qml.RZ(theta[param_idx + 2], wires=i)
                param_idx += 3
            # ── Ring entanglement: matches vqc_diffusion.py exactly ──
            for p in range(n_qubits):
                nxt = (p + 1) % n_qubits
                qml.RZ(-_pi / 2,             wires=nxt)
                qml.CNOT(wires=[nxt, p])
                qml.RZ(theta[param_idx],     wires=p)
                qml.RY(theta[param_idx + 1], wires=nxt)
                qml.CNOT(wires=[p, nxt])
                qml.RY(theta[param_idx + 2], wires=nxt)
                param_idx += 3

    if use_data:
        @qml.qnode(dev, interface="numpy")
        def circuit(inputs, theta):
            _vqc_body(inputs, theta)
            return qml.state()
    else:
        @qml.qnode(dev, interface="numpy")
        def circuit(theta):
            # circuit_only: no data encoding (inputs = None → AngleEmbedding skipped)
            _vqc_body(None, theta)
            return qml.state()

    return circuit


def sample_vqc_reupload_params(
    n_samples: int, n_qubits: int, n_layers: int, seed: int = 0,
) -> np.ndarray:
    """Sample theta for one vqc_angle_reupload block from [0, 2π]. Shape: (n_samples, theta_size)."""
    theta_size = _vqc_reupload_theta_size(n_qubits, n_layers)
    return np.random.default_rng(seed).uniform(0.0, 2 * np.pi, size=(n_samples, theta_size))


def extract_vqc_theta_from_ckpt(ckpt_path: str, block_idx: int = 0) -> np.ndarray:
    """
    Extract trained theta for one VQC block from an AngleVQCDenoiser checkpoint.

    The state dict key is  vqc_blocks.{block_idx}.theta  (TorchLayer convention).

    Returns
    -------
    theta : 1-D float64 array of shape (theta_size,)
    """
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    key = f"vqc_blocks.{block_idx}.theta"
    if key not in sd:
        available = [k for k in sd if "vqc_blocks" in k]
        raise KeyError(f"Key '{key}' not found in checkpoint. "
                       f"Available vqc_blocks keys: {available[:10]}")
    theta = sd[key].detach().cpu().numpy().astype(np.float64)
    print(f"[expressibility] extracted theta from '{key}'  shape={theta.shape}")
    return theta


def sample_states_vqc_reupload_circuit_only(
    n_qubits: int,
    n_layers: int,
    n_samples: int,
    initial_cnot: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """
    circuit_only for vqc_angle_reupload.

    What is varied    : θ (uniformly random)
    What is NOT varied: no data input (no AngleEmbedding)

    Evaluates the ANSATZ EXPRESSIBILITY of the VQC block:
    how well do the variational rotations + ring entanglement cover
    the Hilbert space when θ is random?
    """
    circuit = make_vqc_reupload_qnode(n_qubits, n_layers, initial_cnot, use_data=False)
    all_theta = sample_vqc_reupload_params(n_samples, n_qubits, n_layers, seed)
    states, log_every = [], max(1, n_samples // 10)
    for i, theta in enumerate(all_theta):
        states.append(np.array(circuit(theta)))
        if (i + 1) % log_every == 0:
            print(f"  [vqc_reupload circuit_only] {i + 1}/{n_samples}", flush=True)
    return np.stack(states, axis=0)


def sample_states_vqc_reupload_data_dependent(
    latent_path: str,
    n_qubits: int,
    n_layers: int,
    data_mode: str,
    n_samples: int,
    initial_cnot: bool = False,
    trained_params_path: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    block_idx: int = 0,
    lambda_scales: Optional[np.ndarray] = None,
    input_biases: Optional[np.ndarray] = None,
    use_tanh: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """
    data_dependent for vqc_angle_reupload.

    Data preprocessing: inputs = tanh(z) × λ + bias, tiled per layer.
    By default λ=1, bias=0 (equivalent to trained lambda_scales=ones, biases=zeros).

    fixed_theta  : vary z → inputs, θ fixed.
                   Evaluates DATA ENCODING: do distinct z remain distinguishable
                   as quantum states after AngleEmbedding?

    random_theta : vary both z and θ.
                   Evaluates FULL CIRCUIT distribution (data + ansatz mixed).

    θ source priority (fixed_theta mode):
      1. --ckpt_path + --block_idx  : extract from denoiser checkpoint
      2. --trained_params_path       : load pre-saved .npy
      3. neither                     : random initialization from seed
    """
    p = Path(latent_path)
    if p.suffix == ".npy":
        latents = np.load(latent_path).astype(np.float64)
    elif p.suffix in (".csv", ".tsv"):
        sep = "\t" if p.suffix == ".tsv" else ","
        latents = pd.read_csv(latent_path, sep=sep).values.astype(np.float64)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")

    rng = np.random.default_rng(seed)
    if len(latents) > n_samples:
        latents = latents[rng.choice(len(latents), size=n_samples, replace=False)]
    n = len(latents)
    print(f"  [vqc_reupload data_dep] {n} latent vectors  shape={latents.shape}")

    circuit = make_vqc_reupload_qnode(n_qubits, n_layers, initial_cnot, use_data=True)
    log_every = max(1, n // 10)
    states = []

    if data_mode == "fixed_theta":
        # Resolve theta source
        if ckpt_path is not None:
            theta = extract_vqc_theta_from_ckpt(ckpt_path, block_idx)
        elif trained_params_path is not None:
            theta = np.load(trained_params_path).astype(np.float64).flatten()
            print(f"  [fixed_theta] θ from {trained_params_path}  size={len(theta)}")
        else:
            theta = rng.uniform(0.0, 2 * np.pi, _vqc_reupload_theta_size(n_qubits, n_layers))
            print(f"  [fixed_theta] random θ (seed={seed})")

        expected = _vqc_reupload_theta_size(n_qubits, n_layers)
        if len(theta) != expected:
            raise ValueError(f"theta size {len(theta)} != expected {expected} "
                             f"(n_qubits={n_qubits}, n_layers={n_layers})")

        for i, z in enumerate(latents):
            inputs = _preprocess_z_for_reupload(z, n_qubits, n_layers,
                                                 lambda_scales, input_biases, use_tanh)
            states.append(np.array(circuit(inputs, theta)))
            if (i + 1) % log_every == 0:
                print(f"  [fixed_theta] {i + 1}/{n}", flush=True)

    elif data_mode == "random_theta":
        all_theta = sample_vqc_reupload_params(n, n_qubits, n_layers, seed)
        for i, (z, theta) in enumerate(zip(latents, all_theta)):
            inputs = _preprocess_z_for_reupload(z, n_qubits, n_layers,
                                                 lambda_scales, input_biases, use_tanh)
            states.append(np.array(circuit(inputs, theta)))
            if (i + 1) % log_every == 0:
                print(f"  [random_theta] {i + 1}/{n}", flush=True)

    else:
        raise ValueError(f"Unknown data_mode: {data_mode!r}")

    return np.stack(states, axis=0)


# ══ Shared analysis functions ════════════════════════════════════════════════

# ── 8. Pairwise fidelities ────────────────────────────────────────────────────

def compute_pairwise_fidelities(
    states: np.ndarray,
    max_pairs: Optional[int] = None,
    seed: int = 0,
) -> np.ndarray:
    """
    Compute pairwise fidelities F_ij = |⟨ψ_i|ψ_j⟩|².

    max_pairs : if set, randomly sample this many unique (i<j) pairs.
                Avoids O(n²) cost for large n_samples.
    """
    n = len(states)
    total_pairs = n * (n - 1) // 2

    if max_pairs is None or max_pairs >= total_pairs:
        fidelities = [
            abs(np.vdot(states[i], states[j])) ** 2
            for i in range(n) for j in range(i + 1, n)
        ]
        return np.array(fidelities, dtype=np.float64)

    rng = np.random.default_rng(seed)
    seen: set[tuple[int, int]] = set()
    fidelities = []
    for _ in range(max_pairs * 20):
        if len(fidelities) >= max_pairs:
            break
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i == j:
            continue
        pair = (min(i, j), max(i, j))
        if pair in seen:
            continue
        seen.add(pair)
        fidelities.append(abs(np.vdot(states[i], states[j])) ** 2)

    return np.array(fidelities, dtype=np.float64)


# ── 9. Histogram ──────────────────────────────────────────────────────────────

def histogram_distribution(
    fidelities: np.ndarray, n_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bin fidelities in [0,1]. Returns (bin_edges, P_circuit) where P_circuit sums to 1."""
    counts, bin_edges = np.histogram(fidelities, bins=n_bins, range=(0.0, 1.0))
    total = counts.sum()
    return bin_edges, counts / total if total > 0 else counts.astype(float)


# ── 10. Haar bin distribution ─────────────────────────────────────────────────

def haar_bin_distribution(bin_edges: np.ndarray, hilbert_dim: int) -> np.ndarray:
    """
    Analytically compute Haar random fidelity probability per bin.

    N = 2^n_qubits:  P_Haar([a,b]) = (1-a)^(N-1) - (1-b)^(N-1)
    """
    N = hilbert_dim
    P_haar = (1.0 - bin_edges[:-1]) ** (N - 1) - (1.0 - bin_edges[1:]) ** (N - 1)
    P_haar = np.clip(P_haar, 0.0, None)
    total = P_haar.sum()
    return P_haar / total if total > 0 else P_haar


# ── 11. Metrics ───────────────────────────────────────────────────────────────

def compute_expressibility_metrics(
    P_circuit: np.ndarray, P_haar: np.ndarray, eps: float = 1e-12,
) -> Tuple[float, float]:
    """
    KL divergence D_KL(P_circuit || P_haar) and TVD.
    Lower → closer to Haar → more expressive.
    """
    p = np.clip(P_circuit, eps, None)
    q = np.clip(P_haar, eps, None)
    kl  = float(np.sum(p * np.log(p / q)))
    tvd = float(0.5 * np.sum(np.abs(P_circuit - P_haar)))
    return kl, tvd


# ── 12. Plot ──────────────────────────────────────────────────────────────────

def plot_histogram(
    bin_edges: np.ndarray, P_circuit: np.ndarray, P_haar: np.ndarray,
    title: str, out_path: str,
) -> None:
    """Bar chart of circuit fidelity vs Haar reference. Saves PNG."""
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = bin_edges[1] - bin_edges[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(bin_centers, P_circuit, width=width * 0.9,
           alpha=0.6, color="steelblue", label="Circuit fidelity")
    ax.plot(bin_centers, P_haar, color="crimson", linewidth=2,
            marker="o", markersize=3, label="Haar (analytic)")
    ax.set_xlabel("Fidelity  F = |⟨ψᵢ|ψⱼ⟩|²", fontsize=11)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[expressibility] plot saved → {out_path}")


# ── 13. Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PQC expressibility analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", required=True,
                        choices=["circuit_only", "data_dependent"])
    parser.add_argument("--vqc_type", default="generic",
                        choices=["generic", "angle_reupload"],
                        help="'generic': simple RY+RZ+CNOT ansatz.  "
                             "'angle_reupload': exact circuit from AngleVQCDenoiser.")

    # ── Data-dependent options ────────────────────────────────────────────────
    parser.add_argument("--latent_path", type=str, default=None,
                        help="Latent vectors (.npy or .csv). Required for data_dependent.")
    parser.add_argument("--data_mode", choices=["fixed_theta", "random_theta"],
                        default="fixed_theta")

    # ── Theta source (fixed_theta mode, mutually prioritized) ─────────────────
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="[angle_reupload] Denoiser checkpoint .pt to extract trained θ.")
    parser.add_argument("--block_idx", type=int, default=0,
                        help="[angle_reupload] Which VQC block's θ to extract (default: 0).")
    parser.add_argument("--all_blocks", action="store_true",
                        help="[angle_reupload] Run all blocks in the checkpoint and compare.")
    parser.add_argument("--num_blocks", type=int, default=6,
                        help="[angle_reupload] Number of VQC blocks (used with --all_blocks).")
    parser.add_argument("--trained_params_path", type=str, default=None,
                        help="Pre-saved θ .npy. For generic: shape (n_layers, n_qubits, 2). "
                             "For angle_reupload: flat array of size theta_size.")

    # ── Generic encoding (generic mode only) ──────────────────────────────────
    parser.add_argument("--encoding_type", choices=["angle", "amplitude"], default="angle",
                        help="Data encoding type (generic mode only).")

    # ── Circuit options ───────────────────────────────────────────────────────
    parser.add_argument("--n_qubits",      type=int,   default=4,
                        help="Number of qubits (= latent_dim for angle_reupload).")
    parser.add_argument("--n_layers",      type=int,   default=2,
                        help="Circuit depth per VQC block (n_layers in AngleVQCDenoiser config).")
    parser.add_argument("--initial_cnot",  action="store_true",
                        help="[angle_reupload] Enable initial CNOT ring after first encoding.")
    parser.add_argument("--no_tanh", action="store_true",
                        help="[angle_reupload] Skip tanh preprocessing (use z directly).")
    parser.add_argument("--n_samples",     type=int,   default=200)
    parser.add_argument("--n_bins",        type=int,   default=75)
    parser.add_argument("--max_pairs",     type=int,   default=None,
                        help="Max random pairs for fidelity. None = all C(n,2).")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--outdir",        type=str,   default="outputs/expressibility")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hilbert_dim = 2 ** args.n_qubits

    # ── Sample states ─────────────────────────────────────────────────────────
    print(f"\n[expressibility] vqc_type={args.vqc_type}  mode={args.mode}"
          + (f"  data_mode={args.data_mode}" if args.mode == "data_dependent" else ""))
    print(f"  n_qubits={args.n_qubits}  n_layers={args.n_layers}  n_samples={args.n_samples}")

    if args.vqc_type == "angle_reupload":
        theta_size = _vqc_reupload_theta_size(args.n_qubits, args.n_layers)
        print(f"  theta_size={theta_size}  initial_cnot={args.initial_cnot}")

        if args.mode == "circuit_only":
            states = sample_states_vqc_reupload_circuit_only(
                n_qubits=args.n_qubits, n_layers=args.n_layers,
                n_samples=args.n_samples, initial_cnot=args.initial_cnot, seed=args.seed,
            )
        else:
            if args.latent_path is None:
                parser.error("--latent_path required for data_dependent mode.")
            print(f"  latent_path={args.latent_path}")
            states = sample_states_vqc_reupload_data_dependent(
                latent_path=args.latent_path,
                n_qubits=args.n_qubits, n_layers=args.n_layers,
                data_mode=args.data_mode, n_samples=args.n_samples,
                initial_cnot=args.initial_cnot,
                trained_params_path=args.trained_params_path,
                ckpt_path=args.ckpt_path, block_idx=args.block_idx,
                use_tanh=not args.no_tanh,
                seed=args.seed,
            )

    else:  # generic
        if args.mode == "circuit_only":
            states = sample_states_circuit_only(
                n_qubits=args.n_qubits, n_layers=args.n_layers,
                n_samples=args.n_samples, seed=args.seed,
            )
        else:
            if args.latent_path is None:
                parser.error("--latent_path required for data_dependent mode.")
            print(f"  encoding={args.encoding_type}  latent_path={args.latent_path}")
            states = sample_states_data_dependent(
                latent_path=args.latent_path,
                n_qubits=args.n_qubits, n_layers=args.n_layers,
                encoding_type=args.encoding_type, data_mode=args.data_mode,
                n_samples=args.n_samples,
                trained_params_path=args.trained_params_path, seed=args.seed,
            )

    # ── Pairwise fidelities ───────────────────────────────────────────────────
    n_states = len(states)
    total_pairs = n_states * (n_states - 1) // 2
    print(f"\n[expressibility] fidelity computation  "
          f"n_states={n_states}  total_pairs={total_pairs}  max_pairs={args.max_pairs}")
    fidelities = compute_pairwise_fidelities(states, max_pairs=args.max_pairs, seed=args.seed)
    print(f"  pairs computed: {len(fidelities)}")

    # ── Distributions & metrics ───────────────────────────────────────────────
    bin_edges, P_circuit = histogram_distribution(fidelities, args.n_bins)
    P_haar = haar_bin_distribution(bin_edges, hilbert_dim)
    kl, tvd = compute_expressibility_metrics(P_circuit, P_haar)

    # ── Summary ───────────────────────────────────────────────────────────────
    sep = "=" * 54
    print(f"\n{sep}")
    print(f"vqc_type       : {args.vqc_type}")
    print(f"Mode           : {args.mode}"
          + (f" / {args.data_mode}" if args.mode == "data_dependent" else ""))
    if args.mode == "data_dependent":
        print(f"latent_path    : {args.latent_path}")
        if args.vqc_type == "angle_reupload":
            src = (args.ckpt_path or args.trained_params_path or f"random (seed={args.seed})")
            print(f"theta source   : {src}"
                  + (f"  block={args.block_idx}" if args.ckpt_path else ""))
    print(f"n_qubits       : {args.n_qubits}")
    print(f"n_layers       : {args.n_layers}")
    if args.vqc_type == "angle_reupload":
        print(f"theta_size     : {_vqc_reupload_theta_size(args.n_qubits, args.n_layers)}")
        print(f"initial_cnot   : {args.initial_cnot}")
    print(f"n_samples      : {n_states}")
    print(f"fidelity pairs : {len(fidelities)}")
    print(f"KL divergence  : {kl:.6f}")
    print(f"TVD            : {tvd:.6f}")
    print(sep)

    # ── Save outputs ──────────────────────────────────────────────────────────
    pd.DataFrame({"fidelity": fidelities}).to_csv(out_dir / "fidelity_values.csv", index=False)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    pd.DataFrame({
        "bin_center": bin_centers,
        "bin_left":   bin_edges[:-1],
        "bin_right":  bin_edges[1:],
        "P_circuit":  P_circuit,
        "P_haar":     P_haar,
    }).to_csv(out_dir / "histogram.csv", index=False)

    meta: dict = {
        "vqc_type":       args.vqc_type,
        "mode":           args.mode,
        "n_qubits":       args.n_qubits,
        "n_layers":       args.n_layers,
        "hilbert_dim":    hilbert_dim,
        "n_samples":      n_states,
        "fidelity_pairs": len(fidelities),
        "kl_divergence":  kl,
        "tvd":            tvd,
    }
    if args.vqc_type == "angle_reupload":
        meta["theta_size"]    = _vqc_reupload_theta_size(args.n_qubits, args.n_layers)
        meta["initial_cnot"]  = args.initial_cnot
        meta["use_tanh"]      = not args.no_tanh
    if args.mode == "data_dependent":
        meta["data_mode"]   = args.data_mode
        meta["latent_path"] = args.latent_path
        if args.vqc_type == "generic":
            meta["encoding_type"] = args.encoding_type
    pd.DataFrame([meta]).to_csv(out_dir / "metrics.csv", index=False)

    # Title for plot
    if args.mode == "circuit_only":
        title = (f"[{args.vqc_type}] Circuit-only Expressibility\n"
                 f"n_qubits={args.n_qubits}, n_layers={args.n_layers}, "
                 f"KL={kl:.4f}, TVD={tvd:.4f}")
    else:
        title = (f"[{args.vqc_type}] Data-dependent ({args.data_mode})\n"
                 f"n_qubits={args.n_qubits}, n_layers={args.n_layers}, "
                 f"KL={kl:.4f}, TVD={tvd:.4f}")
    plot_histogram(bin_edges, P_circuit, P_haar, title=title,
                   out_path=str(out_dir / "fidelity_histogram.png"))

    print(f"\n[expressibility] Saved to {out_dir}/")
    print("  fidelity_values.csv  histogram.csv  metrics.csv  fidelity_histogram.png")


if __name__ == "__main__":
    main()
