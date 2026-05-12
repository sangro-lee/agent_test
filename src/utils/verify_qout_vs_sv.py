#!/usr/bin/env python
"""
Verify that TorchLayer q_out == PauliZ from statevector.

For a few samples per block, compares:
  - q_out captured via post-hook (actual TorchLayer output)
  - PauliZ expectations derived from statevector via make_vqc_reupload_qnode

If these differ, the two circuit implementations are inconsistent.

Usage:
  python -m src.utils.verify_qout_vs_sv \
      --latent_path outputs/runs/.../latents_train.npy \
      --y_path      outputs/runs/.../y_train.npy \
      --ckpt_path   outputs/runs/.../diffusion/.../denoiser_cfg.pt \
      --n_check 5
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.models.vqc_diffusion import AngleVQCDenoiser
from src.utils.analyze_vqc_contribution import load_denoiser, capture_hooks
from src.utils.expressibility import make_vqc_reupload_qnode


def pauli_z_from_state(state: np.ndarray, n_qubits: int) -> np.ndarray:
    """Compute <Z_k> for each qubit from statevector."""
    dim = 2 ** n_qubits
    assert len(state) == dim
    probs = np.abs(state) ** 2
    result = []
    for k in range(n_qubits):
        p0 = sum(probs[j] for j in range(dim) if not (j >> k & 1))
        p1 = sum(probs[j] for j in range(dim) if     (j >> k & 1))
        result.append(p0 - p1)
    return np.array(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_path", required=True)
    parser.add_argument("--y_path",      required=True)
    parser.add_argument("--ckpt_path",   required=True)
    parser.add_argument("--n_check",     type=int, default=5,
                        help="Number of samples to verify per block")
    parser.add_argument("--blocks",      type=str, default="0,5")
    args = parser.parse_args()

    print(f"[verify] loading denoiser ...")
    denoiser, ckpt = load_denoiser(args.ckpt_path)
    n_qubits     = denoiser.latent_dim
    n_layers     = denoiser.n_layers
    initial_cnot = denoiser.initial_cnot

    z_mean = denoiser.z_mean.numpy()
    z_std  = denoiser.z_std.numpy()
    c_mean = float(denoiser.c_mean.item())
    c_std  = float(denoiser.c_std.item())

    block_indices = [int(b) for b in args.blocks.split(",")]

    z0_raw = np.load(args.latent_path).astype(np.float32)[:50]
    y_raw  = np.load(args.y_path).astype(np.float64).ravel()[:50]

    z0_norm = ((z0_raw - z_mean) / (z_std + 1e-8)).astype(np.float32)
    c_norm  = ((y_raw - c_mean) / (c_std + 1e-8)).astype(np.float32)
    t_norm  = 0.0

    print(f"[verify] running forward pass ...")
    block_inputs, block_qouts = capture_hooks(denoiser, z0_norm, t_norm, c_norm)

    circuit = make_vqc_reupload_qnode(n_qubits, n_layers, initial_cnot, use_data=True)

    all_ok = True
    for blk in block_indices:
        if blk not in block_inputs:
            print(f"[verify] block {blk} not captured")
            continue

        inp  = block_inputs[blk]    # (n, n_layers*n_qubits)
        qout = block_qouts[blk]     # (n, n_qubits)
        key  = f"vqc_blocks.{blk}.theta"
        theta = ckpt["state_dict"][key].detach().cpu().numpy().astype(np.float64)

        print(f"\n[verify] ── Block {blk} ─────────────────────────────────")
        max_err = 0.0
        for i in range(min(args.n_check, len(inp))):
            state    = np.array(circuit(inp[i].astype(np.float64), theta))
            pz_sv    = pauli_z_from_state(state, n_qubits)
            pz_hook  = qout[i].astype(np.float64)
            err      = np.abs(pz_sv - pz_hook)
            max_err  = max(max_err, err.max())
            print(f"  sample {i}:")
            print(f"    hook q_out : {pz_hook.round(6)}")
            print(f"    SV PauliZ  : {pz_sv.round(6)}")
            print(f"    abs error  : {err.round(6)}  max={err.max():.2e}")

        status = "OK" if max_err < 1e-4 else "MISMATCH"
        print(f"  → block {blk} max error={max_err:.2e}  [{status}]")
        if max_err >= 1e-4:
            all_ok = False

    print(f"\n[verify] {'ALL CONSISTENT' if all_ok else 'INCONSISTENCY DETECTED'}")


if __name__ == "__main__":
    main()
