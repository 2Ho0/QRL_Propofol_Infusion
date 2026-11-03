"""Safety penalty and decision-level potential shaping (sim v2).

v1 had a safety penalty that jumped at BIS 20, 30 and 70, and it added
0.99*Phi(s') - Phi(s) on every 10-s step while the critic saw the undiscounted
sum of six steps per 60-s decision. That sum is not potential-based at the
decision level.
"""
import numpy as np
import pytest

from environment.dual_drug_env import DualDrugEnv, safety_penalty
from models.pharmacokinetics import PatientParameters

ACTION = np.array([8.0, 0.2])


@pytest.mark.parametrize("bis", [20.0, 30.0, 40.0, 60.0, 70.0])
def test_safety_penalty_is_continuous_at_breakpoints(bis):
    assert safety_penalty(bis - 1e-9) == pytest.approx(safety_penalty(bis + 1e-9), abs=1e-6)


@pytest.mark.parametrize("bis, expected", [
    (50.0, 0.0), (40.0, 0.0), (60.0, 0.0), (30.0, -0.5), (20.0, -5.0),
    (0.0, -20.0), (70.0, -0.2), (100.0, -1.0),
])
def test_safety_penalty_keeps_v1_anchor_values(bis, expected):
    assert safety_penalty(bis) == pytest.approx(expected)


def test_safety_penalty_never_rewards_moving_away_from_the_band():
    low = np.array([safety_penalty(b) for b in np.linspace(0.0, 40.0, 4001)])
    high = np.array([safety_penalty(b) for b in np.linspace(60.0, 100.0, 4001)])
    assert np.all(np.diff(low) >= 0.0)
    assert np.all(np.diff(high) <= 0.0)


def _env(shaping_interval_steps, max_steps=60, seed=3):
    patient = PatientParameters(age=50, weight=70, height=170, gender="male")
    return DualDrugEnv(patient=patient, seed=seed, dt=10.0 / 60.0,
                       max_steps=max_steps,
                       shaping_interval_steps=shaping_interval_steps)


def test_decision_level_shaping_adds_one_potential_difference_per_decision():
    env = _env(shaping_interval_steps=6)
    phi_start = env._potential_function(env.bis)
    rewards, bases, shapings = [], [], []
    for _ in range(6):
        _, reward, _, _, info = env.step(ACTION)
        rewards.append(reward)
        bases.append(info["reward_base"])
        shapings.append(info["reward_shaping"])
    assert shapings[:5] == [0.0] * 5
    expected = sum(bases) + env.gamma * env._potential_function(env.bis) - phi_start
    assert sum(rewards) == pytest.approx(expected, abs=1e-9)


def test_second_decision_starts_from_the_first_decision_end():
    env = _env(shaping_interval_steps=6)
    for _ in range(6):
        env.step(ACTION)
    phi_boundary = env._potential_function(env.bis)
    shaping = 0.0
    for _ in range(6):
        _, _, _, _, info = env.step(ACTION)
        shaping += info["reward_shaping"]
    assert shaping == pytest.approx(
        env.gamma * env._potential_function(env.bis) - phi_boundary, abs=1e-9)


def test_terminal_decision_closes_with_zero_potential():
    env = _env(shaping_interval_steps=6, max_steps=6)
    phi_start = env._potential_function(env.bis)
    total_shaping = 0.0
    truncated = False
    for _ in range(6):
        _, _, _, truncated, info = env.step(ACTION)
        total_shaping += info["reward_shaping"]
    assert truncated
    assert total_shaping == pytest.approx(-phi_start, abs=1e-9)


def test_interval_of_one_reproduces_per_step_shaping():
    env = _env(shaping_interval_steps=1)
    previous_bis = env.bis
    for _ in range(4):
        _, _, _, _, info = env.step(ACTION)
        expected = (env.gamma * env._potential_function(env.bis)
                    - env._potential_function(previous_bis))
        assert info["reward_shaping"] == pytest.approx(expected, abs=1e-12)
        previous_bis = env.bis
