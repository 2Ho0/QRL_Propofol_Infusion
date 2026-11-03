"""Residual-on-PID mode of DualDrugEnv (v3, D14) and the reward weights (D15).

The tuned PID sets the base infusion once per control decision; the agent's
action, read as a fraction of the action range, adds a bounded residual
(a / a_max - 0.5) * 2 * R. A policy that always outputs the middle of the
range must therefore reproduce the tuned PID.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from agents.baseline_agents import PIDController
from environment.dual_drug_env import DualDrugEnv, dose_cost
from environment.variability import get_variability_spec
from models.pharmacokinetics import PatientParameters

PID = json.loads(Path("logs/revision_v2/pid_tuning/selected_pid.json").read_text())["parameters"]
PATIENT = PatientParameters(age=55, weight=62.0, height=165, gender="female")
SPEC = get_variability_spec("literature")
PER_DECISION = 6            # 10-s observations, 60-s decisions
MIDDLE = np.array([6.0, 1.0])


def _env(**kwargs):
    return DualDrugEnv(patient=PATIENT, seed=123, dt=10 / 60, max_steps=600,
                       shaping_interval_steps=PER_DECISION, variability=SPEC, **kwargs)


def _rollout(env, policy, decisions=40):
    state, _ = env.reset()
    bis, doses = [], []
    for _ in range(decisions):
        action = policy(state)
        for _ in range(PER_DECISION):
            state, _, _, _, info = env.step(action)
            bis.append(env.bis)
            doses.append(info["applied_action"])
    return np.array(bis), np.array(doses)


def test_zero_residual_reproduces_the_tuned_pid():
    pid = PIDController(**PID)
    pid.reset()
    bis_pid, dose_pid = _rollout(_env(), pid.select_action)
    bis_res, dose_res = _rollout(_env(residual_pid=PID), lambda state: MIDDLE)
    np.testing.assert_allclose(dose_res, dose_pid, rtol=0, atol=1e-5)
    np.testing.assert_allclose(bis_res, bis_pid, rtol=0, atol=1e-5)


@pytest.mark.parametrize("action, sign", [(np.array([12.0, 2.0]), 1.0),
                                          (np.array([0.0, 0.0]), -1.0)])
def test_the_action_range_maps_to_plus_or_minus_the_residual_range(action, sign):
    env = _env(residual_pid=PID, residual_range=(4.0, 0.05))
    state, _ = env.reset()
    pid = PIDController(**PID)
    base = np.asarray(pid.select_action(state), dtype=float)
    _, _, _, _, info = env.step(action)
    expected = np.clip(base + sign * np.array([4.0, 0.05]), [0.0, 0.0], [12.0, 2.0])
    np.testing.assert_allclose(info["applied_action"], expected, atol=1e-6)


def test_the_pid_base_is_held_for_one_control_decision():
    env = _env(residual_pid=PID)
    env.reset()
    applied = []
    for _ in range(20 * PER_DECISION):
        _, _, _, _, info = env.step(MIDDLE)
        applied.append(info["applied_action"])
    blocks = np.array(applied).reshape(20, PER_DECISION, 2)
    assert np.all(blocks == blocks[:, :1, :])                # constant within a decision
    assert len({tuple(block[0]) for block in blocks}) > 1    # recomputed between decisions


def test_the_applied_dose_stays_inside_the_action_space():
    env = _env(residual_pid=PID, residual_range=(20.0, 3.0))
    env.reset()
    _, _, _, _, info = env.step(np.array([12.0, 2.0]))
    np.testing.assert_array_equal(info["applied_action"], [12.0, 2.0])
    _, _, _, _, info = env.step(np.array([0.0, 0.0]))
    np.testing.assert_array_equal(info["applied_action"], [0.0, 0.0])


def test_without_a_pid_the_action_is_applied_as_commanded():
    env = _env()
    env.reset()
    _, _, _, _, info = env.step(np.array([15.0, -1.0]))
    np.testing.assert_array_equal(info["applied_action"], [12.0, 0.0])


@pytest.mark.parametrize("bad", [(4.0, 0.0), (4.0, -0.05), (4.0,)])
def test_residual_range_needs_two_positive_values(bad):
    with pytest.raises(ValueError, match="residual_range"):
        _env(residual_pid=PID, residual_range=bad)


def test_default_reward_weights_are_the_v2_weights():
    env = _env()
    assert (env.reward_w3, env.reward_w_smooth) == (0.1, 2.0)


def test_reward_weights_scale_the_dose_cost_and_the_smoothness_penalty():
    action = np.array([6.0, 0.15])
    rewards = []
    for w3, w_smooth in ((0.1, 2.0), (0.03, 0.5)):
        env = _env(reward_w3=w3, reward_w_smooth=w_smooth)
        env.reset()
        _, _, _, _, info = env.step(action)
        rewards.append(info["reward_base"])
    smooth = -((6.0 / 12.0) ** 2 + (0.15 / 2.0) ** 2)
    expected = (0.1 - 0.03) * dose_cost(6.0, 0.15) + (2.0 - 0.5) * smooth
    assert rewards[0] - rewards[1] == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("bad", [(4.0, float("nan")), (4.0, float("inf"))])
def test_residual_range_must_be_finite(bad):
    with pytest.raises(ValueError, match="residual_range"):
        _env(residual_pid=PID, residual_range=bad)


@pytest.mark.parametrize("action", [np.array([[6.0, 1.0]]), np.array([6.0, 1.0, 0.0])])
def test_residual_mode_refuses_an_action_of_the_wrong_shape(action):
    env = _env(residual_pid=PID)
    env.reset()
    with pytest.raises(ValueError, match="shape"):
        env.step(action)


def test_an_action_beyond_the_range_adds_at_most_the_residual_range():
    applied = []
    for action in (np.array([12.0, 2.0]), np.array([24.0, 4.0])):
        env = _env(residual_pid=PID, residual_range=(4.0, 0.05))
        env.reset()
        for _ in range(3 * PER_DECISION):
            _, _, _, _, info = env.step(action)
        applied.append(info["applied_action"])
    np.testing.assert_array_equal(applied[1], applied[0])
