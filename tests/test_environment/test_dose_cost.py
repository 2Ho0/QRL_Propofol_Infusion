"""Dose cost priced at clinical maintenance rates (sim v2).

Under the Bouillon 2004 additive BIS surface remifentanil alone lowers BIS.
v1 normalised each drug by its action ceiling (12 mg/kg/h, 2 ug/kg/min), which
made remifentanil about 7.7 times cheaper than propofol at clinical doses,
and the seed-0 pilot classical policy settled on 1.52 mg/kg/h propofol with
0.58 ug/kg/min remifentanil. sim v2 prices each drug by its median rate in
the VitalDB training split.
"""
import numpy as np
import pytest

from environment.dual_drug_env import DualDrugEnv, dose_cost
from models.pharmacokinetics import PatientParameters

PPF_REFERENCE, RFTN_REFERENCE = 6.27, 0.135   # mg/kg/h, ug/kg/min
W_EFF = 0.1


def test_each_drug_costs_the_same_at_its_reference_rate():
    assert dose_cost(PPF_REFERENCE, 0.0) == pytest.approx(dose_cost(0.0, RFTN_REFERENCE))


def test_cost_at_reference_rates_keeps_the_v1_magnitude():
    v1 = -(PPF_REFERENCE / 12.0 + RFTN_REFERENCE / 2.0)
    assert dose_cost(PPF_REFERENCE, RFTN_REFERENCE) == pytest.approx(v1, rel=2e-3)


def test_remifentanil_substitution_costs_more_than_the_clinical_mix():
    # Seed-0 pilot medians: classical_large (substitution) vs quantum_full.
    assert dose_cost(1.52, 0.577) < dose_cost(6.75, 0.215)


def test_zero_infusion_costs_nothing():
    assert dose_cost(0.0, 0.0) == 0.0


def _base_reward(env, ppf, remi):
    env.prev_action_ppf, env.prev_action_rftn = ppf, remi   # no smoothness term
    env._reward_potential_based(ppf, remi, prev_bis=env.bis)
    return env.last_reward_components["base"]


def test_step_charges_the_dose_cost_of_the_clipped_action(monkeypatch):
    import environment.dual_drug_env as module
    charged = []
    real = module.dose_cost
    monkeypatch.setattr(module, "dose_cost",
                        lambda ppf, remi: charged.append((ppf, remi)) or real(ppf, remi))
    patient = PatientParameters(age=50, weight=70, height=170, gender="male")
    env = DualDrugEnv(patient=patient, seed=0)
    env.step(np.array([25.0, 0.7]))   # propofol above its 12 mg/kg/h ceiling
    assert len(charged) == 1
    assert charged[0] == pytest.approx((12.0, 0.7))


@pytest.mark.parametrize("ppf, remi", [
    (PPF_REFERENCE, 0.0), (0.0, RFTN_REFERENCE), (6.75, 0.215), (1.52, 0.577),
])
def test_environment_reward_charges_the_dose_cost(ppf, remi):
    patient = PatientParameters(age=50, weight=70, height=170, gender="male")
    env = DualDrugEnv(patient=patient, seed=0)
    charged = _base_reward(env, ppf, remi) - _base_reward(env, 0.0, 0.0)
    assert charged == pytest.approx(W_EFF * dose_cost(ppf, remi), abs=1e-12)
