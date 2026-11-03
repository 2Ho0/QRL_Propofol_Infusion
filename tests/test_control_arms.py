import numpy as np
import pytest
import torch

from agents.baseline_agents import PIDController
from agents.classical_agent import MLPActor
from models.vqc import QuantumDisabledPolicy, QuantumPolicy


def test_classical_prior_matches_quantum_initial_action_contract():
    actor = MLPActor(
        input_dim=32, output_dim=2, hidden_dims=[256, 128, 64],
        dropout=0.0, action_prior=[0.5, 0.075])
    action = actor(torch.zeros(4, 32)).detach()
    expected = torch.tensor([0.5, 0.075]).repeat(4, 1)
    assert torch.allclose(action, expected, atol=1e-6)


def test_quantum_disabled_is_actor_parameter_matched():
    quantum = QuantumPolicy(
        state_dim=32, n_qubits=3, n_layers=4, action_dim=2, seed=0)
    disabled = QuantumDisabledPolicy(
        state_dim=32, n_qubits=3, n_layers=4, action_dim=2, seed=0)
    assert sum(p.numel() for p in quantum.parameters()) == 6930
    assert sum(p.numel() for p in disabled.parameters()) == 6930
    assert not any("vqc" in name for name, _ in disabled.named_parameters())
    assert disabled(torch.zeros(3, 32)).shape == (3, 2)


def test_noentanglement_preserves_trainable_angle_count():
    full = QuantumPolicy(
        state_dim=32, n_qubits=3, n_layers=4, action_dim=2,
        entanglement=True, seed=0)
    noentangle = QuantumPolicy(
        state_dim=32, n_qubits=3, n_layers=4, action_dim=2,
        entanglement=False, seed=0)
    assert full.weights.shape == noentangle.weights.shape == (4, 3, 3)
    assert sum(p.numel() for p in full.parameters()) == sum(
        p.numel() for p in noentangle.parameters())
    assert noentangle(torch.zeros(2, 32)).shape == (2, 2)


def test_pid_baselines_are_explicit_hyperparameters():
    pid = PIDController(base_ppf=7.5, base_rftn=0.2)
    pid.reset()
    action = pid.select_action(np.zeros(13, dtype=np.float32))
    assert action == pytest.approx([7.5, 0.2])
