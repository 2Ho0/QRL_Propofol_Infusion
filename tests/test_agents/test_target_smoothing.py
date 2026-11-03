"""TD3 target policy smoothing (Fujimoto et al. 2018).

Actions live in normalised [0, 1] units, so the TD3 defaults of sigma 0.2 and
clip 0.5 on [-1, 1] become 0.1 and 0.25 here.
"""
import copy

import pytest
import torch
import yaml

from agents.classical_agent import ClassicalDDPGAgent
from agents.td3_utils import smoothed_target_actions


class _ConstantActor(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, states):
        return torch.full((states.shape[0], 2), self.value)


def test_zero_noise_returns_the_target_actor_output():
    states = torch.zeros(64, 32)
    actions = smoothed_target_actions(_ConstantActor(0.4), states, noise_std=0.0, noise_clip=0.25)
    assert torch.equal(actions, torch.full((64, 2), 0.4))


def test_noise_is_clipped_and_actions_stay_in_range():
    torch.manual_seed(0)
    states = torch.zeros(4096, 32)
    actions = smoothed_target_actions(_ConstantActor(0.95), states, noise_std=0.1, noise_clip=0.25)
    assert actions.min() >= 0.0 and actions.max() <= 1.0
    assert (actions - 0.95).abs().max() <= 0.25 + 1e-7
    assert (actions < 0.95).any() and (actions == 1.0).any()


def test_smoothing_is_reproducible_under_a_torch_seed():
    states = torch.zeros(16, 32)
    torch.manual_seed(7)
    first = smoothed_target_actions(_ConstantActor(0.5), states, noise_std=0.1, noise_clip=0.25)
    torch.manual_seed(7)
    second = smoothed_target_actions(_ConstantActor(0.5), states, noise_std=0.1, noise_clip=0.25)
    assert torch.equal(first, second)


def test_agent_reads_smoothing_settings_from_config():
    with open("config/hyperparameters.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config = copy.deepcopy(config)
    config["training"]["target_policy_noise"] = 0.1
    config["training"]["target_noise_clip"] = 0.25
    agent = ClassicalDDPGAgent(state_dim=13, action_dim=2, config=config,
                               encoder_type="none", seed=0)
    assert agent.target_policy_noise == pytest.approx(0.1)
    assert agent.target_noise_clip == pytest.approx(0.25)
