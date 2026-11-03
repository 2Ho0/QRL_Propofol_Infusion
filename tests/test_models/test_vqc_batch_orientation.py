"""Batched circuit outputs must keep the batch on axis 0 for every batch size.

The vmapped QNode returns [n_observables, batch]. The bridge transposed it only
when the first axis differed from the batch size, so a batch of exactly
3 * n_qubits samples (9 here) came back with batch and observables swapped.
"""
import torch

from models.quantum_critic import HybridQuantumTwinCritic
from models.vqc import QuantumPolicy


def test_policy_batch_of_nine_matches_single_samples():
    policy = QuantumPolicy(state_dim=32, n_qubits=3, n_layers=2, seed=0)
    x = torch.randn(9, 32)
    with torch.no_grad():
        batch = policy(x)
        single = torch.stack([policy(x[i]) for i in range(9)])
    assert torch.allclose(batch, single, atol=1e-5)


def test_critic_batch_of_nine_matches_single_samples():
    critic = HybridQuantumTwinCritic(state_dim=8, action_dim=2, hidden_dims=[16],
                                     n_qubits=3, n_layers=2, seed=0)
    s, a = torch.randn(9, 8), torch.rand(9, 2)
    with torch.no_grad():
        batch = critic.q1(s, a)
        single = torch.cat([critic.q1(s[i:i + 1], a[i:i + 1]) for i in range(9)])
    assert torch.allclose(batch, single, atol=1e-5)
