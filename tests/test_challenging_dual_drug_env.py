import numpy as np

from environment.challenging_dual_drug_env import ChallengingDualDrugEnv
from models.pharmacokinetics import PatientParameters


def _patient():
    return PatientParameters(age=45, weight=70, height=170, gender="male")


def test_scenario_is_deterministic_and_patient_seeded():
    first = ChallengingDualDrugEnv(_patient(), scenario_seed=123)
    second = ChallengingDualDrugEnv(_patient(), scenario_seed=123)
    other = ChallengingDualDrugEnv(_patient(), scenario_seed=124)

    assert first.events == second.events
    np.testing.assert_allclose(first._sensor_noise, second._sensor_noise)
    assert first.pd_factors == second.pd_factors
    assert first.events != other.events


def test_phase_targets_are_exposed_in_the_observation():
    env = ChallengingDualDrugEnv(_patient(), scenario_seed=123)
    state, _ = env.reset()
    assert env.target_bis == 55.0
    assert np.isclose(state[0], 55.0 - env.bis)

    action = np.array([8.0, 0.1], dtype=np.float32)
    for _ in range(20):
        state, _, _, _, info = env.step(action)
    assert info["active_target"] == 50.0
    assert np.isclose(state[0], 50.0 - info["observed_bis"])


def test_locked_ranges_and_latent_outcome_are_available():
    env = ChallengingDualDrugEnv(_patient(), scenario_seed=123)
    _, info = env.reset()
    assert 0.80 <= info["pd_factors"]["c50_ppf"] <= 1.25
    assert 0.75 <= info["pd_factors"]["c50_rftn"] <= 1.30
    assert 0.90 <= info["pd_factors"]["gamma"] <= 1.10

    _, _, _, _, info = env.step(np.array([8.0, 0.1], dtype=np.float32))
    assert 0.0 <= info["true_bis"] <= 100.0
    assert 0.0 <= info["observed_bis"] <= 100.0
    assert info["scenario_seed"] == 123


def test_remifentanil_attenuation_is_monotone():
    env = ChallengingDualDrugEnv(_patient(), scenario_seed=123)
    raw = 12.0
    low_ce = raw / (1.0 + 1.0 / 4.0)
    high_ce = raw / (1.0 + 8.0 / 4.0)
    assert high_ce < low_ce < raw

