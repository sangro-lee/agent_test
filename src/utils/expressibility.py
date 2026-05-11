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

Usage:
  # Circuit-only
  python -m src.utils.expressibility \\
      --mode circuit_only --n_qubits 4 --n_layers 3 --n_samples 200

  # Data-dependent: fixed theta, angle encoding
  python -m src.utils.expressibility \\
      --mode data_dependent \\
      --latent_path outputs/runs/sme_random/latents_train.npy \\
      --data_mode fixed_theta --encoding_type angle \\
      --n_qubits 4 --n_layers 3 --n_samples 200

  # Data-dependent: fixed trained theta
  python -m src.utils.expressibility \\
      --mode data_dependent \\
      --latent_path outputs/runs/sme_random/latents_train.npy \\
      --trained_params_path outputs/runs/sme_random/theta.npy \\
      --data_mode fixed_theta --encoding_type angle \\
      --n_qubits 4 --n_layers 3
"""
from __future__ import annotations

import argparse
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


# ── 2. Data encoding ──────────────────────────────────────────────────────────

def data_encoding(z: np.ndarray, n_qubits: int, encoding_type: str) -> None:
    """
    Apply data encoding gates inside a QNode circuit.

    Parameters
    ----------
    z             : 1-D float array — latent vector for one sample.
    n_qubits      : number of qubits.
    encoding_type : 'angle' or 'amplitude'.

    ── To replace with your own encoding ──────────────────────────────────────
    Delete or comment out the body below and add your custom PennyLane gates.
    This function is called inside a QNode, so just apply gates; no return.

    Example (re-uploading angle encoding):
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.RY(lambda_scale * float(z[i % len(z)]), wires=i)
            qml.BasicEntanglerLayers(weights[l], wires=range(n_qubits))
    ──────────────────────────────────────────────────────────────────────────
    """
    if encoding_type == "angle":
        # RY(z_i) on qubit i.
        # If len(z) > n_qubits, only the first n_qubits dims are used.
        # If len(z) < n_qubits, remaining qubits get angle 0 (no rotation).
        # ── Add amplitude scaling here if your latent values are not in [0, 2π] ──
        for i in range(n_qubits):
            angle = float(z[i]) if i < len(z) else 0.0
            qml.RY(angle, wires=i)

    elif encoding_type == "amplitude":
        # StatePrep encodes z as the quantum amplitude vector.
        # Pads or truncates to 2^n_qubits, then normalizes to unit norm.
        dim = 2 ** n_qubits
        vec = np.array(z, dtype=complex)
        if len(vec) < dim:
            vec = np.pad(vec, (0, dim - len(vec)))
        else:
            vec = vec[:dim]
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            vec = np.zeros(dim, dtype=complex)
            vec[0] = 1.0
        else:
            vec /= norm
        qml.StatePrep(vec, wires=range(n_qubits))

    else:
        raise ValueError(
            f"Unknown encoding_type: {encoding_type!r}. Choose 'angle' or 'amplitude'."
        )


# ── 3. Ansatz ─────────────────────────────────────────────────────────────────

def ansatz(params: np.ndarray, n_qubits: int, n_layers: int) -> None:
    """
    Trainable ansatz: RY + RZ per qubit, then CNOT chain (ring topology).

    params shape: (n_layers, n_qubits, 2)
      params[l, q, 0] → RY angle for layer l, qubit q
      params[l, q, 1] → RZ angle for layer l, qubit q

    ── To replace with your own VQC ansatz ────────────────────────────────────
    Delete or comment out the body below and add your custom parameterized gates.
    If you change the params shape, update sample_random_params() accordingly.

    Example (hardware-efficient ansatz with CRZ):
        for l in range(n_layers):
            for q in range(n_qubits):
                qml.RX(params[l, q, 0], wires=q)
                qml.RY(params[l, q, 1], wires=q)
            for q in range(n_qubits - 1):
                qml.CRZ(params[l, q, 2], wires=[q, q + 1])
    ──────────────────────────────────────────────────────────────────────────
    """
    for l in range(n_layers):
        for q in range(n_qubits):
            qml.RY(params[l, q, 0], wires=q)
            qml.RZ(params[l, q, 1], wires=q)
        # CNOT ring: 0→1, 1→2, ..., (n-1)→0
        for q in range(n_qubits):
            qml.CNOT(wires=[q, (q + 1) % n_qubits])


# ── 4. QNode factory ──────────────────────────────────────────────────────────

def make_qnode(
    n_qubits: int,
    n_layers: int,
    encoding_type: Optional[str] = None,
    use_data: bool = False,
):
    """
    Build and return a QNode that outputs the full statevector.

    use_data=False (circuit_only mode):
        |ψ(θ)⟩ = U_ansatz(θ)|0⟩
        circuit(params) → statevector

    use_data=True (data_dependent mode):
        |ψ(z, θ)⟩ = U_ansatz(θ) U_enc(z)|0⟩
        Encoding is applied first (on |0⟩), then the ansatz.
        circuit(z, params) → statevector
    """
    dev = build_device(n_qubits)

    if use_data:
        @qml.qnode(dev, interface="numpy")
        def circuit(z, params):
            # ── Replace data_encoding body to use your own encoding ──
            data_encoding(z, n_qubits, encoding_type)
            # ── Replace ansatz body to use your own VQC ──
            ansatz(params, n_qubits, n_layers)
            return qml.state()
    else:
        @qml.qnode(dev, interface="numpy")
        def circuit(params):
            # ── Circuit-only: no data encoding ──
            ansatz(params, n_qubits, n_layers)
            return qml.state()

    return circuit


# ── 5. Parameter sampling ─────────────────────────────────────────────────────

def sample_random_params(
    n_samples: int,
    n_layers: int,
    n_qubits: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Sample trainable parameters uniformly from [0, 2π].

    Returns
    -------
    params : (n_samples, n_layers, n_qubits, 2)
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 2 * np.pi, size=(n_samples, n_layers, n_qubits, 2))


# ── 6. Circuit-only state sampling ────────────────────────────────────────────

def sample_states_circuit_only(
    n_qubits: int,
    n_layers: int,
    n_samples: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Sample quantum states for circuit_only expressibility.

    What is varied    : θ (uniformly random)
    What is NOT varied: no data z is used at all

    This evaluates the expressibility of the circuit ARCHITECTURE:
    how uniformly can U_ansatz(θ)|0⟩ cover the Hilbert space as θ varies?

    Returns
    -------
    states : complex array of shape (n_samples, 2^n_qubits)
    """
    circuit = make_qnode(n_qubits, n_layers, use_data=False)
    all_params = sample_random_params(n_samples, n_layers, n_qubits, seed)

    states = []
    log_every = max(1, n_samples // 10)
    for i, params in enumerate(all_params):
        sv = np.array(circuit(params))
        states.append(sv)
        if (i + 1) % log_every == 0:
            print(f"  [circuit_only] {i + 1}/{n_samples}", flush=True)

    return np.stack(states, axis=0)


# ── 7. Data-dependent state sampling ──────────────────────────────────────────

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
    Sample quantum states for data_dependent expressibility.

    Loads latent vectors z from a .npy or .csv file.

    data_mode='fixed_theta'
        What is varied    : z (latent vectors from the dataset)
        What is NOT varied: θ is fixed (trained or one random initialization)
        → Evaluates DATA ENCODING quality: do distinct latent vectors remain
          distinguishable as quantum states?

        If trained_params_path is given, use it as fixed θ.
        If not, draw one random θ from seed (same θ for all z_i).

    data_mode='random_theta'
        What is varied    : both z_i and θ_i (randomly paired)
        → Evaluates the FULL VQC state distribution.
          Mixes data encoding and ansatz effects — harder to interpret
          either contribution in isolation.

    Parameters
    ----------
    latent_path          : .npy (N, d) or .csv file of latent vectors.
    trained_params_path  : .npy file with shape (n_layers, n_qubits, 2).
                           Used as fixed θ in fixed_theta mode; ignored otherwise.

    Returns
    -------
    states : complex array of shape (n_states, 2^n_qubits)
             n_states = min(len(latents), n_samples)
    """
    # Load latent vectors
    p = Path(latent_path)
    if p.suffix == ".npy":
        latents = np.load(latent_path).astype(np.float64)
    elif p.suffix in (".csv", ".tsv"):
        sep = "\t" if p.suffix == ".tsv" else ","
        latents = pd.read_csv(latent_path, sep=sep).values.astype(np.float64)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}. Use .npy or .csv")

    # Subsample if needed
    rng = np.random.default_rng(seed)
    if len(latents) > n_samples:
        idx = rng.choice(len(latents), size=n_samples, replace=False)
        latents = latents[idx]
    n = len(latents)
    print(f"  [data_dependent] loaded {n} latent vectors  shape={latents.shape}")

    circuit = make_qnode(n_qubits, n_layers, encoding_type=encoding_type, use_data=True)
    log_every = max(1, n // 10)
    states = []

    if data_mode == "fixed_theta":
        # Fix θ — varies: z only
        if trained_params_path is not None:
            theta = np.load(trained_params_path).astype(np.float64)
            if theta.shape != (n_layers, n_qubits, 2):
                raise ValueError(
                    f"trained_params shape {theta.shape} != expected "
                    f"({n_layers}, {n_qubits}, 2)"
                )
            print(f"  [fixed_theta] using trained θ from {trained_params_path}")
        else:
            theta = rng.uniform(0.0, 2 * np.pi, size=(n_layers, n_qubits, 2))
            print(f"  [fixed_theta] using random θ (seed={seed})")

        for i, z in enumerate(latents):
            sv = np.array(circuit(z, theta))
            states.append(sv)
            if (i + 1) % log_every == 0:
                print(f"  [fixed_theta] {i + 1}/{n}", flush=True)

    elif data_mode == "random_theta":
        # Vary both z_i and θ_i — each sample gets an independent θ
        all_params = sample_random_params(n, n_layers, n_qubits, seed)
        for i, (z, params) in enumerate(zip(latents, all_params)):
            sv = np.array(circuit(z, params))
            states.append(sv)
            if (i + 1) % log_every == 0:
                print(f"  [random_theta] {i + 1}/{n}", flush=True)

    else:
        raise ValueError(
            f"Unknown data_mode: {data_mode!r}. Choose 'fixed_theta' or 'random_theta'."
        )

    return np.stack(states, axis=0)


# ── 8. Pairwise fidelities ────────────────────────────────────────────────────

def compute_pairwise_fidelities(
    states: np.ndarray,
    max_pairs: Optional[int] = None,
    seed: int = 0,
) -> np.ndarray:
    """
    Compute pairwise fidelities F_ij = |⟨ψ_i|ψ_j⟩|².

    Parameters
    ----------
    states    : complex array of shape (n_states, hilbert_dim).
    max_pairs : if set, randomly sample this many unique (i < j) pairs instead
                of computing all C(n, 2). Use this to avoid O(n²) cost for
                large n_samples.

    Returns
    -------
    fidelities : 1-D float array of length ≤ max_pairs (or C(n,2) if None).
    """
    n = len(states)
    total_pairs = n * (n - 1) // 2

    if max_pairs is None or max_pairs >= total_pairs:
        # All pairs
        fidelities = []
        for i in range(n):
            for j in range(i + 1, n):
                fidelities.append(abs(np.vdot(states[i], states[j])) ** 2)
        return np.array(fidelities, dtype=np.float64)

    # Random pair sampling (without replacement)
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, int]] = set()
    fidelities = []
    max_attempts = max_pairs * 20  # guard against degenerate small pools

    for _ in range(max_attempts):
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
    fidelities: np.ndarray,
    n_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin fidelities in [0, 1] and return normalized probabilities.

    Returns
    -------
    bin_edges : (n_bins + 1,)
    P_circuit : (n_bins,) probabilities summing to 1
    """
    counts, bin_edges = np.histogram(fidelities, bins=n_bins, range=(0.0, 1.0))
    total = counts.sum()
    P_circuit = counts / total if total > 0 else counts.astype(float)
    return bin_edges, P_circuit


# ── 10. Haar bin distribution ─────────────────────────────────────────────────

def haar_bin_distribution(bin_edges: np.ndarray, hilbert_dim: int) -> np.ndarray:
    """
    Analytically compute Haar random fidelity probability per bin.

    For Hilbert space dimension N = 2^n_qubits:
        PDF:  P_Haar(F) = (N - 1)(1 - F)^(N - 2)
        CDF:  CDF(F)    = 1 - (1 - F)^(N - 1)

    Bin probability for [a, b]:
        P_Haar([a, b]) = (1 - a)^(N-1) - (1 - b)^(N-1)

    Returns
    -------
    P_haar : (n_bins,) normalized probabilities
    """
    N = hilbert_dim
    a = bin_edges[:-1]
    b = bin_edges[1:]
    P_haar = (1.0 - a) ** (N - 1) - (1.0 - b) ** (N - 1)
    P_haar = np.clip(P_haar, 0.0, None)
    total = P_haar.sum()
    if total > 0:
        P_haar /= total
    return P_haar


# ── 11. Metrics ───────────────────────────────────────────────────────────────

def compute_expressibility_metrics(
    P_circuit: np.ndarray,
    P_haar: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """
    Compute KL divergence D_KL(P_circuit || P_haar) and TVD.

    KL  = Σ P_circuit_i · log(P_circuit_i / P_haar_i)
    TVD = 0.5 · Σ |P_circuit_i − P_haar_i|

    Lower values → circuit distribution closer to Haar random → more expressive.
    eps prevents log(0).
    """
    p = np.clip(P_circuit, eps, None)
    q = np.clip(P_haar, eps, None)
    kl = float(np.sum(p * np.log(p / q)))
    tvd = float(0.5 * np.sum(np.abs(P_circuit - P_haar)))
    return kl, tvd


# ── 12. Plot ──────────────────────────────────────────────────────────────────

def plot_histogram(
    bin_edges: np.ndarray,
    P_circuit: np.ndarray,
    P_haar: np.ndarray,
    title: str,
    out_path: str,
) -> None:
    """
    Bar chart of circuit fidelity distribution overlaid with Haar reference line.
    Saves PNG to out_path.
    """
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = bin_edges[1] - bin_edges[0]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        bin_centers, P_circuit, width=width * 0.9,
        alpha=0.6, color="steelblue", label="Circuit fidelity",
    )
    ax.plot(
        bin_centers, P_haar,
        color="crimson", linewidth=2, marker="o", markersize=3,
        label="Haar (analytic)",
    )
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
        description="PQC expressibility analysis (circuit_only | data_dependent).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["circuit_only", "data_dependent"],
        help="Analysis mode.",
    )
    # Data-dependent options
    parser.add_argument(
        "--latent_path", type=str, default=None,
        help="Path to latent vectors (.npy or .csv). Required for data_dependent.",
    )
    parser.add_argument(
        "--trained_params_path", type=str, default=None,
        help="NPY file with fixed θ, shape (n_layers, n_qubits, 2). "
             "Used in fixed_theta mode. If not given, a random θ is initialized.",
    )
    parser.add_argument(
        "--data_mode", choices=["fixed_theta", "random_theta"],
        default="fixed_theta",
        help="fixed_theta: fix θ, vary z.  random_theta: vary both z and θ.",
    )
    parser.add_argument(
        "--encoding_type", choices=["angle", "amplitude"],
        default="angle",
    )
    # Circuit options
    parser.add_argument("--n_qubits",  type=int,   default=4)
    parser.add_argument("--n_layers",  type=int,   default=3)
    parser.add_argument("--n_samples", type=int,   default=200)
    parser.add_argument("--n_bins",    type=int,   default=75)
    parser.add_argument(
        "--max_pairs", type=int, default=None,
        help="Max randomly sampled pairs for fidelity computation. "
             "None = all C(n_samples, 2) pairs.",
    )
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--outdir", type=str, default="outputs/expressibility")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hilbert_dim = 2 ** args.n_qubits

    # ── Sample states ─────────────────────────────────────────────────────────
    if args.mode == "circuit_only":
        print(f"\n[expressibility] Mode: circuit_only")
        print(f"  n_qubits={args.n_qubits}  n_layers={args.n_layers}  "
              f"n_samples={args.n_samples}")
        states = sample_states_circuit_only(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            n_samples=args.n_samples,
            seed=args.seed,
        )
    else:
        if args.latent_path is None:
            parser.error("--latent_path is required for data_dependent mode.")
        print(f"\n[expressibility] Mode: data_dependent  data_mode={args.data_mode}")
        print(f"  encoding={args.encoding_type}  latent_path={args.latent_path}")
        print(f"  n_qubits={args.n_qubits}  n_layers={args.n_layers}  "
              f"n_samples={args.n_samples}")
        states = sample_states_data_dependent(
            latent_path=args.latent_path,
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            encoding_type=args.encoding_type,
            data_mode=args.data_mode,
            n_samples=args.n_samples,
            trained_params_path=args.trained_params_path,
            seed=args.seed,
        )

    # ── Pairwise fidelities ───────────────────────────────────────────────────
    n_states = len(states)
    total_pairs = n_states * (n_states - 1) // 2
    print(f"\n[expressibility] Computing fidelities  "
          f"n_states={n_states}  total_pairs={total_pairs}  "
          f"max_pairs={args.max_pairs}")
    fidelities = compute_pairwise_fidelities(
        states, max_pairs=args.max_pairs, seed=args.seed
    )
    print(f"  pairs computed: {len(fidelities)}")

    # ── Distributions ─────────────────────────────────────────────────────────
    bin_edges, P_circuit = histogram_distribution(fidelities, args.n_bins)
    P_haar = haar_bin_distribution(bin_edges, hilbert_dim)
    kl, tvd = compute_expressibility_metrics(P_circuit, P_haar)

    # ── Summary ───────────────────────────────────────────────────────────────
    sep = "=" * 52
    print(f"\n{sep}")
    if args.mode == "circuit_only":
        print(f"Mode           : circuit_only")
        print(f"n_qubits       : {args.n_qubits}")
        print(f"n_layers       : {args.n_layers}")
        print(f"n_samples      : {n_states}")
        print(f"fidelity pairs : {len(fidelities)}")
    else:
        print(f"Mode           : data_dependent")
        print(f"data_mode      : {args.data_mode}")
        print(f"encoding_type  : {args.encoding_type}")
        print(f"latent_path    : {args.latent_path}")
        print(f"n_qubits       : {args.n_qubits}")
        print(f"n_layers       : {args.n_layers}")
        print(f"n_samples      : {n_states}")
        print(f"fidelity pairs : {len(fidelities)}")
    print(f"KL divergence  : {kl:.6f}")
    print(f"TVD            : {tvd:.6f}")
    print(sep)

    # ── Save outputs ──────────────────────────────────────────────────────────
    pd.DataFrame({"fidelity": fidelities}).to_csv(
        out_dir / "fidelity_values.csv", index=False
    )

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    pd.DataFrame({
        "bin_center": bin_centers,
        "bin_left":   bin_edges[:-1],
        "bin_right":  bin_edges[1:],
        "P_circuit":  P_circuit,
        "P_haar":     P_haar,
    }).to_csv(out_dir / "histogram.csv", index=False)

    meta: dict = {
        "mode":            args.mode,
        "n_qubits":        args.n_qubits,
        "n_layers":        args.n_layers,
        "hilbert_dim":     hilbert_dim,
        "n_samples":       n_states,
        "fidelity_pairs":  len(fidelities),
        "kl_divergence":   kl,
        "tvd":             tvd,
    }
    if args.mode == "data_dependent":
        meta.update({
            "data_mode":     args.data_mode,
            "encoding_type": args.encoding_type,
            "latent_path":   args.latent_path,
        })
    pd.DataFrame([meta]).to_csv(out_dir / "metrics.csv", index=False)

    if args.mode == "circuit_only":
        title = (
            f"Circuit-only Expressibility\n"
            f"n_qubits={args.n_qubits}, n_layers={args.n_layers}, "
            f"KL={kl:.4f}, TVD={tvd:.4f}"
        )
    else:
        title = (
            f"Data-dependent Expressibility ({args.data_mode})\n"
            f"encoding={args.encoding_type}, n_qubits={args.n_qubits}, "
            f"n_layers={args.n_layers}, KL={kl:.4f}, TVD={tvd:.4f}"
        )
    plot_histogram(
        bin_edges, P_circuit, P_haar,
        title=title,
        out_path=str(out_dir / "fidelity_histogram.png"),
    )

    print(f"\n[expressibility] Output saved to {out_dir}/")
    print("  fidelity_values.csv  histogram.csv  metrics.csv  fidelity_histogram.png")


if __name__ == "__main__":
    main()
