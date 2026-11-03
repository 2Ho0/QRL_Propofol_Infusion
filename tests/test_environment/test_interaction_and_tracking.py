"""Tests for the BIS interaction model and the tracking reward.

1. sim v2 uses the published Bouillon et al. (2004) BIS response surface:
   BIS = E0 / (1 + U^gamma) with the additive interaction
   U = Ce_ppf / 4.47 + Ce_rftn / 19.3 and E0 = 97.4 (TCI-concentration
   estimates, Table 8). v1 used 98 (1 + x)^-1.43 with remifentanil as a pure
   potentiator, which is not the form whose C50 values were cited.
2. The tracking reward was flat at +1.0 inside |e_BIS| <= 5, so the objective
   could not distinguish BIS 50.0 from BIS 54.9 while MDAPE separates them by
   9.8 points.
"""
import numpy as np
import pytest

from environment.dual_drug_env import DualDrugEnv
from environment.patient_simulator import PatientParameters, PatientSimulator

TARGET = 50.0
E0 = 97.4


@pytest.fixture
def sim():
    return PatientSimulator(PatientParameters(age=45, weight=70, height=175,
                                              gender="male"))


# --- 1. interaction model -------------------------------------------------

def test_baseline_without_drugs_is_bouillon_e0(sim):
    assert sim._compute_bis(ce_ppf=0.0, ce_rftn=0.0) == pytest.approx(E0)


def test_propofol_c50_halves_the_baseline(sim):
    assert sim._compute_bis(4.47, 0.0) == pytest.approx(E0 / 2.0)


def test_remifentanil_c50_halves_the_baseline(sim):
    """Bouillon 2004 found an additive interaction for BIS."""
    assert sim._compute_bis(0.0, 19.3) == pytest.approx(E0 / 2.0)


def test_interaction_is_additive(sim):
    assert sim._compute_bis(4.47 / 2.0, 19.3 / 2.0) == pytest.approx(E0 / 2.0)


def test_sigmoid_emax_slope(sim):
    assert sim._compute_bis(2.0 * 4.47, 0.0) == pytest.approx(E0 / (1.0 + 2.0 ** 1.43))


def test_propofol_alone_still_produces_hypnosis(sim):
    assert sim._compute_bis(5.0, 0.0) < 50.0
    assert sim._compute_bis(0.5, 0.0) > sim._compute_bis(3.0, 0.0)


def test_remifentanil_potentiates_propofol(sim):
    """Adding remifentanil to a fixed propofol level must deepen hypnosis."""
    without = sim._compute_bis(3.0, 0.0)
    with_remi = sim._compute_bis(3.0, 4.0)
    assert with_remi < without


def test_potentiation_is_monotone_in_remifentanil(sim):
    values = [sim._compute_bis(3.0, cr) for cr in (0.0, 2.0, 4.0, 8.0, 16.0)]
    assert all(values[i] > values[i + 1] for i in range(len(values) - 1))


def test_bis_stays_in_range(sim):
    for cp in (0.0, 1.0, 3.0, 10.0):
        for cr in (0.0, 4.0, 20.0):
            assert 0.0 <= sim._compute_bis(cp, cr) <= 100.0


# --- 2. tracking reward ---------------------------------------------------

def _r_track(e_bis: float) -> float:
    """The tracking term as implemented, isolated for testing."""
    return float(np.exp(-((e_bis / 5.0) ** 2)) - 0.5 * min(e_bis / 50.0, 1.0))


def test_tracking_reward_is_strictly_decreasing_in_error():
    errors = [0.0, 1.0, 2.5, 4.9, 5.0, 7.5, 10.0, 20.0, 40.0, 60.0]
    r = [_r_track(e) for e in errors]
    assert all(r[i] > r[i + 1] for i in range(len(r) - 1))


def test_tracking_reward_separates_inside_the_old_flat_band():
    """The defect: BIS 50.0 and 54.9 previously scored identically."""
    assert _r_track(0.0) - _r_track(4.9) > 0.5


def test_tracking_reward_is_maximal_at_target():
    assert _r_track(0.0) == pytest.approx(1.0)


def test_env_reward_rewards_tighter_tracking():
    """End-to-end through the live reward path."""
    env = DualDrugEnv(patient=PatientParameters(age=45, weight=70, height=175,
                                                gender="male"),
                      seed=0, reward_type='potential')
    env.reset()

    def reward_at(bis):
        env.bis = bis
        env.prev_action_ppf, env.prev_action_rftn = 4.0, 0.1
        return env._reward_potential_based(4.0, 0.1, prev_bis=bis, done=False)

    assert reward_at(TARGET) > reward_at(TARGET + 4.9) > reward_at(TARGET + 9.0)


def test_initial_bis_inverse_round_trips():
    env = DualDrugEnv(patient=PatientParameters(age=45, weight=70, height=175,
                                                gender="male"), seed=0)
    ce_ppf, ce_rftn = env._compute_target_concentrations(50.0)
    assert env.simulator._compute_bis(ce_ppf, ce_rftn) == pytest.approx(50.0, abs=1e-9)


def test_bis_decreases_in_each_concentration(sim):
    grid = np.linspace(0.0, 12.0, 49)
    for ce_rftn in (0.0, 4.0, 15.0):
        values = [sim._compute_bis(cp, ce_rftn) for cp in grid]
        assert np.all(np.diff(values) < 0)
    for ce_ppf in (0.0, 2.0, 5.0):
        values = [sim._compute_bis(ce_ppf, cr) for cr in np.linspace(0.0, 60.0, 49)]
        assert np.all(np.diff(values) < 0)
