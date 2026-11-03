import numpy as np
import pandas as pd

from data.vitaldb_loader import VitalDBLoader
from environment.dual_drug_env import DualDrugEnv


def _segment(n=20):
    return pd.DataFrame({
        "BIS": np.linspace(60.0, 50.0, n),
        "PPF20_RATE": np.full(n, 6.0),
        "PPF20_CE": np.full(n, 2.0),
        "RFTN_RATE": np.full(n, 0.12),
        "RFTN_CE": np.full(n, 1.5),
        "age": np.full(n, 55.0),
        "sex": np.ones(n),
        "bmi": np.full(n, 24.0),
    })


def test_dense_state_features_have_physical_time_units():
    states = VitalDBLoader._dual_drug_state_segment(
        _segment(), observation_interval_seconds=10)

    # Six observations are one minute.  A constant 6 mg/kg/h infusion gives
    # 0.1 mg/kg over that minute; 0.12 ug/kg/min gives 0.12 ug/kg.
    np.testing.assert_allclose(states[5, 6], 0.1, rtol=1e-6)
    np.testing.assert_allclose(states[5, 7], 0.12, rtol=1e-6)
    # BIS derivative is expressed per minute, not per array row.
    expected = (_segment().BIS.iloc[1] - _segment().BIS.iloc[0]) * 6.0
    np.testing.assert_allclose(states[1, 3], expected, rtol=1e-6)


def test_environment_windows_follow_dt_and_match_offline_interaction():
    env = DualDrugEnv(dt=1 / 6, max_steps=12, bis_noise_std=0.0, seed=0)
    env.reset()
    assert env.ppf_accumulation.maxlen == 6
    assert env.rftn_accumulation.maxlen == 6
    assert env.bis_window.maxlen == 19

    observation, *_ = env.step(np.array([6.0, 0.12], dtype=np.float32))
    np.testing.assert_allclose(
        observation[9], observation[1] * observation[2], rtol=1e-6)
