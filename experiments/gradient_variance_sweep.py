"""Diagnose PQC gradient scaling across qubit counts and circuit depths.

This is a circuit-level diagnostic, not evidence about end-to-end controller
performance.  It uses the same data re-uploading, strongly-entangling layers,
and X/Y/Z measurements as the trained policy while excluding the classical
encoder, readout and skip path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import torch


def evaluate_configuration(n_qubits: int, n_layers: int, samples: int,
                           seed: int) -> dict:
    device = qml.device("default.qubit", wires=n_qubits)
    coefficients = torch.linspace(-1.0, 1.0, 3 * n_qubits, dtype=torch.float64)
    coefficients = coefficients / coefficients.norm()

    @qml.qnode(device, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        for layer in range(n_layers):
            for wire in range(n_qubits):
                qml.RX(inputs[layer, wire] * np.pi, wires=wire)
            qml.StronglyEntanglingLayers(
                weights[layer:layer + 1], wires=range(n_qubits))
        return [observable(wire)
                for wire in range(n_qubits)
                for observable in (lambda w: qml.expval(qml.PauliX(w)),
                                   lambda w: qml.expval(qml.PauliY(w)),
                                   lambda w: qml.expval(qml.PauliZ(w)))]

    generator = torch.Generator().manual_seed(
        seed + 10_000 * n_qubits + 100 * n_layers)
    first_gradients = []
    mean_squared_gradients = []
    gradient_norms = []
    for _ in range(samples):
        inputs = 2.0 * torch.rand(
            n_layers, n_qubits, generator=generator,
            dtype=torch.float64) - 1.0
        weights = (2.0 * np.pi * torch.rand(
            n_layers, n_qubits, 3, generator=generator,
            dtype=torch.float64) - np.pi).requires_grad_(True)
        measurements = torch.stack(circuit(inputs, weights))
        scalar = torch.dot(measurements, coefficients)
        gradient = torch.autograd.grad(scalar, weights)[0].detach().numpy()
        first_gradients.append(float(gradient[0, 0, 0]))
        mean_squared_gradients.append(float(np.mean(gradient ** 2)))
        gradient_norms.append(float(np.linalg.norm(gradient)))
    first = np.asarray(first_gradients)
    squared = np.asarray(mean_squared_gradients)
    norms = np.asarray(gradient_norms)
    return {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "random_initializations": samples,
        "fixed_parameter_gradient_mean": first.mean(),
        "fixed_parameter_gradient_variance": first.var(ddof=1),
        "mean_squared_gradient_mean": squared.mean(),
        "mean_squared_gradient_median": np.median(squared),
        "gradient_norm_mean": norms.mean(),
        "gradient_norm_median": np.median(norms),
        "zero_gradient_fraction": np.mean(norms < 1e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", default="2,3,4,6,8")
    parser.add_argument("--layers", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--out", default="logs/gradient_variance")
    args = parser.parse_args()
    qubits = [int(value) for value in args.qubits.split(",")]
    layers = [int(value) for value in args.layers.split(",")]
    rows = []
    for n_qubits in qubits:
        for n_layers in layers:
            row = evaluate_configuration(
                n_qubits, n_layers, args.samples, args.seed)
            rows.append(row)
            print(n_qubits, n_layers,
                  f"variance={row['fixed_parameter_gradient_variance']:.3e}",
                  flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    protocol = {
        "qubits": qubits,
        "layers": layers,
        "random_initializations_per_configuration": args.samples,
        "seed": args.seed,
        "scalar_cost": "fixed normalized linear combination of all X/Y/Z expectations",
        "scope": "PQC-only diagnostic; no encoder, readout, skip path, or RL loss",
    }
    (out / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
