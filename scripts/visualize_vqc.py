#!/usr/bin/env python
"""Save VQC circuit diagrams as PNG using actual factory functions.

Generates 4 files:
  outputs/vqc_circuits/vqc_angle_L1.png
  outputs/vqc_circuits/vqc_angle_L2.png
  outputs/vqc_circuits/vqc_reupload_L1.png
  outputs/vqc_circuits/vqc_reupload_L2.png

Usage:
  python scripts/visualize_vqc.py
"""
import os
import sys

import numpy as np
import pennylane as qml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.vqc_diffusion import _make_angle_vqc_layer, _make_reupload_vqc_layer

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "vqc_circuits")
os.makedirs(OUT_DIR, exist_ok=True)

N_QUBITS = 4
PARAMS_PER_LAYER = N_QUBITS * 3 + N_QUBITS * 3  # rot(12) + entangle(12) = 24

for n_layers in (1, 2):
    n_theta = PARAMS_PER_LAYER * n_layers

    # ── vqc_angle ──────────────────────────────────────────────────────
    layer = _make_angle_vqc_layer(N_QUBITS, n_layers, "default.qubit")
    qnode = layer.qnode
    dummy_inputs = np.zeros(N_QUBITS)
    dummy_theta  = np.zeros(n_theta)
    fig, ax = qml.draw_mpl(qnode)(dummy_inputs, dummy_theta)
    fig.suptitle(f"vqc_angle  |  {N_QUBITS} qubits  |  {n_layers} layers  |  {n_theta} params", fontsize=9)
    out_path = os.path.join(OUT_DIR, f"vqc_angle_L{n_layers}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # ── vqc_reupload ────────────────────────────────────────────────────
    layer = _make_reupload_vqc_layer(N_QUBITS, n_layers, "default.qubit")
    qnode = layer.qnode
    dummy_inputs = np.zeros(n_layers * N_QUBITS)
    dummy_theta  = np.zeros(n_theta)
    fig, ax = qml.draw_mpl(qnode)(dummy_inputs, dummy_theta)
    fig.suptitle(f"vqc_angle_reupload  |  {N_QUBITS} qubits  |  {n_layers} layers (re-upload)  |  {n_theta} params", fontsize=9)
    out_path = os.path.join(OUT_DIR, f"vqc_reupload_L{n_layers}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
