"""Alternative ansatz families for the re-uploading policy.

Each layer re-uploads the projected features with RX, then applies the
trainable block of one family (PennyLane versions of the Qiskit circuits):
- strongly_entangling: Rot per qubit + CNOT ring (the original policy)
- real_amplitudes: RY per qubit + reverse-linear CNOT chain
- efficient_su2: RY, RZ per qubit + reverse-linear CNOT chain
- two_local: RY per qubit + CZ between every pair
"""
import numpy as np
import pennylane as qml
import pytest
import torch

from models.vqc import QuantumPolicy, VariationalQuantumCircuit

FAMILIES = [("strongly_entangling", 3), ("real_amplitudes", 1),
            ("efficient_su2", 2), ("two_local", 1)]


@pytest.mark.parametrize("ansatz, per_qubit", FAMILIES)
def test_policy_builds_and_outputs_bounded_actions(ansatz, per_qubit):
    policy = QuantumPolicy(state_dim=32, n_qubits=3, n_layers=4, ansatz=ansatz, seed=0)
    assert tuple(policy.weights.shape) == (4, 3, per_qubit)
    actions = policy(torch.randn(5, 32))
    assert actions.shape == (5, 2)
    assert torch.all((actions > 0) & (actions < 1))


@pytest.mark.parametrize("ansatz, per_qubit", FAMILIES)
def test_gradients_reach_the_ansatz_weights(ansatz, per_qubit):
    policy = QuantumPolicy(state_dim=32, n_qubits=3, n_layers=4, ansatz=ansatz, seed=0)
    policy(torch.randn(4, 32)).sum().backward()
    assert policy.weights.grad is not None
    assert policy.weights.grad.abs().sum() > 0


def _gate_counts(ansatz):
    vqc = VariationalQuantumCircuit(n_qubits=3, n_layers=2, ansatz=ansatz)
    inputs = np.zeros((2, 3))
    weights = np.ones((2, 3, dict(FAMILIES)[ansatz])) * 0.3
    return qml.specs(vqc.qnode)(inputs, weights)["resources"].gate_types


def test_real_amplitudes_uses_ry_and_a_cnot_chain():
    gates = _gate_counts("real_amplitudes")
    assert gates["RY"] == 6 and gates["CNOT"] == 4 and "Rot" not in gates and "RZ" not in gates


def test_efficient_su2_uses_ry_rz_and_a_cnot_chain():
    gates = _gate_counts("efficient_su2")
    assert gates["RY"] == 6 and gates["RZ"] == 6 and gates["CNOT"] == 4


def test_two_local_uses_full_cz_entanglement():
    gates = _gate_counts("two_local")
    assert gates["RY"] == 6 and gates["CZ"] == 6 and "CNOT" not in gates


def test_unknown_ansatz_is_rejected():
    with pytest.raises(ValueError):
        VariationalQuantumCircuit(n_qubits=3, n_layers=2, ansatz="hardware_efficient")


@pytest.mark.parametrize("ansatz, per_qubit", FAMILIES)
def test_initial_weights_match_the_ansatz(ansatz, per_qubit):
    vqc = VariationalQuantumCircuit(n_qubits=3, n_layers=4, ansatz=ansatz)
    assert tuple(vqc.get_initial_weights(seed=0).shape) == (4, 3, per_qubit)
