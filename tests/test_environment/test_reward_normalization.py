"""Reward normalisation regression tests.

History. v0 divided remifentanil by 0.3, which made max-propofol/zero-
remifentanil the reward optimum. A later version
divided each drug by its action ceiling (12 mg/kg/h, 2 ug/kg/min) instead.
Under the Bouillon 2004 additive BIS surface of sim v2 that made
remifentanil about 7.7 times cheaper than propofol per clinical dose, and
the seed-0 pilot classical policy replaced propofol with remifentanil.
sim v2 (plan amendment D12) prices each drug by its median rate in the
VitalDB training split; the function itself is tested in test_dose_cost.py.
"""
import pytest

from environment.dual_drug_env import DualDrugEnv
from environment.patient_simulator import PatientParameters

PPF_REFERENCE = 6.27    # mg/kg/h, VitalDB training-split median
RFTN_REFERENCE = 0.135  # ug/kg/min, VitalDB training-split median
DOSE_COST_SCALE = 0.295
W3 = 0.1                # dose-efficiency weight


@pytest.fixture
def env():
    patient = PatientParameters(age=45, weight=70, height=175, gender="male")
    e = DualDrugEnv(patient=patient, seed=0, reward_type='potential')
    e.reset()
    return e


def _reward_at(env, ppf, rftn, bis=50.0):
    """Reward with tracking, safety and shaping held constant and smoothness zeroed.

    Setting prev_action == current action makes r_smooth exactly 0, and
    prev_bis == bis makes the shaping term identical across calls, so any
    difference between two calls comes only from R_eff.
    """
    env.bis = bis
    env.prev_action_ppf = ppf
    env.prev_action_rftn = rftn
    return env._reward_potential_based(ppf, rftn, prev_bis=bis, done=False)


def test_each_drug_at_its_clinical_reference_rate_costs_the_same(env):
    """A typical maintenance dose of either drug must cost the same.

    With ceiling normalisation propofol at 6.27 cost w3*0.52 while
    remifentanil at 0.135 cost w3*0.07.
    """
    ppf_only = _reward_at(env, PPF_REFERENCE, 0.0)
    rftn_only = _reward_at(env, 0.0, RFTN_REFERENCE)
    assert ppf_only == pytest.approx(rftn_only, abs=1e-9)


@pytest.mark.parametrize("ppf, rftn", [(PPF_REFERENCE, 0.0), (0.0, RFTN_REFERENCE)])
def test_one_reference_dose_costs_the_dose_cost_scale(env, ppf, rftn):
    at_zero = _reward_at(env, 0.0, 0.0)
    at_reference = _reward_at(env, ppf, rftn)
    assert (at_zero - at_reference) == pytest.approx(W3 * DOSE_COST_SCALE, abs=1e-9)
