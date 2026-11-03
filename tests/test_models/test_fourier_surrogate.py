"""Classical Fourier-feature control with the circuit's frequency set.

Each of the 12 projected angles (4 layers x 3 qubits) enters the circuit once,
so the circuit output is a trigonometric polynomial with frequencies in
{-1, 0, 1} per angle. The surrogate keeps the projections, input scaling and
skip branch, uses D fixed frequency vectors from that set, and reads out
cos/sin features linearly. Its readout (4D + 2 parameters) matches the
circuit's 36 angles plus its 9 -> 2 readout (56 parameters).
"""
import torch

from models.vqc import FourierSurrogatePolicy, QuantumPolicy


def _surrogate(**kwargs):
    return FourierSurrogatePolicy(state_dim=32, n_qubits=3, n_layers=4, seed=0, **kwargs)


def test_frequencies_come_from_the_circuit_frequency_set():
    freqs = _surrogate().frequencies
    assert freqs.shape == (14, 12)
    assert set(freqs.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert torch.all(freqs.abs().sum(dim=1) > 0)
    rows = [tuple(r) for r in freqs.tolist()]
    assert len(set(rows)) == len(rows)
    assert not any(tuple(-x for x in r) in set(rows) for r in rows)


def test_frequencies_do_not_depend_on_the_training_seed():
    a = FourierSurrogatePolicy(state_dim=32, n_qubits=3, n_layers=4, seed=0)
    b = FourierSurrogatePolicy(state_dim=32, n_qubits=3, n_layers=4, seed=7)
    assert torch.equal(a.frequencies, b.frequencies)


def test_readout_budget_matches_the_circuit_and_its_readout():
    surrogate = _surrogate()
    quantum = QuantumPolicy(state_dim=32, n_qubits=3, n_layers=4, seed=0)
    circuit_budget = quantum.weights.numel() + sum(
        p.numel() for p in quantum.quantum_residual.parameters())
    readout = sum(p.numel() for p in surrogate.quantum_residual.parameters())
    assert circuit_budget == 56
    assert abs(readout - circuit_budget) <= 2


def test_outputs_are_bounded_and_trainable():
    surrogate = _surrogate()
    actions = surrogate(torch.randn(6, 32))
    assert actions.shape == (6, 2)
    assert torch.all((actions > 0) & (actions < 1))
    actions.sum().backward()
    assert surrogate.layer_projections[0][0].weight.grad.abs().sum() > 0
    assert surrogate.input_scaling.grad.abs().sum() > 0
