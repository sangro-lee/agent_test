#!/usr/bin/env python
"""
Visualize the VQC circuit used in HybridUNetDenoiser bottleneck.

Usage:
  python scripts/visualize_vqc.py                    # default: 5 qubits, 2 layers
  python scripts/visualize_vqc.py --n_qubits 3 --n_layers 1
  python scripts/visualize_vqc.py --save circuit.png
"""
import argparse
import math
import pennylane as qml
import numpy as np


def build_circuit(n_qubits: int, n_layers: int):
    params_per_layer = (n_qubits * 3) + (n_qubits * 3)
    n_params = params_per_layer * n_layers
    _pi = math.pi

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs, theta):
        qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)

        # Initial ring entanglement
        for j in range(n_qubits):
            qml.CNOT(wires=[j, (j + 1) % n_qubits])

        param_idx = 0
        for _layer in range(n_layers):
            # Rotation sub-layer
            for i in range(n_qubits):
                qml.RZ(theta[param_idx],     wires=i)
                qml.RY(theta[param_idx + 1], wires=i)
                qml.RZ(theta[param_idx + 2], wires=i)
                param_idx += 3
            # Entanglement sub-layer (ring)
            for p in range(n_qubits):
                nxt = (p + 1) % n_qubits
                qml.RZ(-_pi / 2,             wires=nxt)
                qml.CNOT(wires=[nxt, p])
                qml.RZ(theta[param_idx],     wires=p)
                qml.RY(theta[param_idx + 1], wires=nxt)
                qml.CNOT(wires=[p, nxt])
                qml.RY(theta[param_idx + 2], wires=nxt)
                param_idx += 3

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit, n_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_qubits", type=int, default=5,
                        help="Number of qubits (bottleneck_dim = 2^n_qubits)")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of VQC parameterized layers")
    parser.add_argument("--save",     type=str, default=None,
                        help="Save figure to file (e.g. circuit.png)")
    args = parser.parse_args()

    circuit, n_params = build_circuit(args.n_qubits, args.n_layers)

    bottleneck_dim = 2 ** args.n_qubits
    inputs = np.ones(bottleneck_dim) / math.sqrt(bottleneck_dim)
    theta  = np.zeros(n_params)

    print(f"n_qubits={args.n_qubits}  n_layers={args.n_layers}")
    print(f"bottleneck_dim={bottleneck_dim}  n_params={n_params}")
    print()

    # Text draw
    print(qml.draw(circuit)(inputs, theta))
    print()

    # Matplotlib draw
    fig, ax = qml.draw_mpl(circuit)(inputs, theta)
    fig.suptitle(
        f"VQC Bottleneck  |  {args.n_qubits} qubits  |  {args.n_layers} layers  "
        f"|  ring entanglement  |  {n_params} params",
        fontsize=10,
    )
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved → {args.save}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
