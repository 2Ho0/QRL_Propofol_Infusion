"""Quantum-NoSkip ablation.

The actor sums a parallel classical MLP with the circuit readout, so it is not a
pure quantum policy head. This switch removes that path so the circuit's own
contribution can be measured.
"""
import torch

from models.vqc import QuantumPolicy

STATE_DIM = 16


def _count(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _policy(skip=True):
    return QuantumPolicy(state_dim=STATE_DIM, n_qubits=3, n_layers=2, seed=0,
                         use_classical_skip=skip)


def test_default_keeps_the_classical_skip():
    assert _policy().classical_skip is not None


def test_noskip_has_no_classical_skip_module():
    assert _policy(skip=False).classical_skip is None


def test_noskip_has_strictly_fewer_parameters():
    assert _count(_policy(skip=False)) < _count(_policy())


def test_noskip_still_emits_actions_in_the_unit_interval():
    a = _policy(skip=False)(torch.zeros(4, STATE_DIM))
    assert a.shape == (4, 2)
    assert torch.all(a >= 0.0) and torch.all(a <= 1.0)


def test_noskip_action_depends_on_the_circuit_weights():
    """With the skip removed the circuit must drive the output; if it does not,
    the ablation would be measuring nothing."""
    p = _policy(skip=False)
    x = torch.ones(1, STATE_DIM)
    before = p(x).detach().clone()
    with torch.no_grad():
        p.weights += 0.7
    after = p(x).detach()
    assert not torch.allclose(before, after, atol=1e-6)


def test_default_path_is_unchanged_by_the_refactor():
    """The full actor must still combine both branches before the sigmoid."""
    p = _policy(skip=True)
    x = torch.ones(2, STATE_DIM)
    with torch.no_grad():
        full = p(x).clone()
        for param in p.classical_skip.parameters():
            param.zero_()
        zeroed = p(x)
    # Zeroing the skip changes the action, so it genuinely contributes.
    assert not torch.allclose(full, zeroed, atol=1e-6)


def test_action_prior_sets_the_initial_normalized_dose():
    prior = [0.5, 0.075]
    policy = QuantumPolicy(
        state_dim=STATE_DIM,
        n_qubits=3,
        n_layers=2,
        seed=0,
        action_prior=prior,
        residual_init_scale=0.0,
    )
    action = policy(torch.zeros(3, STATE_DIM))
    expected = torch.tensor(prior).expand_as(action)
    assert torch.allclose(action, expected, atol=1e-6)
