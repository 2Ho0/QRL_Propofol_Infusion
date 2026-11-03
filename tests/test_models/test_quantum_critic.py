"""Hybrid quantum twin critic.

Each Q head is the classical critic used everywhere else plus the linear
readout of a re-uploading VQC on projections of (state, action), so the arm
differs from the classical critic only by the quantum branch.
"""
import torch

from models.quantum_critic import HybridQuantumTwinCritic


def _critic(seed=0):
    torch.manual_seed(seed)
    return HybridQuantumTwinCritic(state_dim=32, action_dim=2, hidden_dims=[64, 64],
                                   n_qubits=3, n_layers=2, seed=seed)


def test_interface_matches_the_twin_critic():
    critic = _critic()
    s, a = torch.randn(5, 32), torch.rand(5, 2)
    q1, q2 = critic(s, a)
    assert q1.shape == (5, 1) and q2.shape == (5, 1)
    assert torch.allclose(critic.q1(s, a), q1)
    assert torch.allclose(critic.q_min(s, a), torch.min(q1, q2))


def test_action_gradient_flows_through_both_branches():
    critic = _critic()
    s = torch.randn(4, 32)
    a = torch.rand(4, 2, requires_grad=True)
    critic.q1(s, a).sum().backward()
    assert a.grad.abs().sum() > 0
    head = critic.critic1
    assert head.quantum_readout.weight.grad.abs().sum() > 0
    assert head.weights.grad.abs().sum() > 0


def test_quantum_branch_starts_as_a_small_residual():
    critic = _critic()
    s, a = torch.randn(64, 32), torch.rand(64, 2)
    head = critic.critic1
    residual = head(s, a) - head.classical(s, a)
    assert residual.abs().max() < 0.5


def test_target_copy_loads_the_online_weights():
    online, target = _critic(0), _critic(1)
    target.load_state_dict(online.state_dict())
    s, a = torch.randn(3, 32), torch.rand(3, 2)
    assert torch.allclose(online.q_min(s, a), target.q_min(s, a))
