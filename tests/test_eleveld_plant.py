"""Eleveld plant for the model-mismatch evaluation (experiments/eleveld_plant.py)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

import eleveld_plant as ep  # noqa: E402
import environment.dual_drug_env as env_module  # noqa: E402
from environment.patient_simulator import PatientSimulator  # noqa: E402
from environment.variability import LITERATURE_SPEC  # noqa: E402
from models.pharmacokinetics.base import PatientParameters  # noqa: E402

KEYS = ("V1", "V2", "V3", "Cl1", "Cl2", "Cl3", "ke0")
# Oracle 1: the Eleveld propofol code of the Python Anesthesia Simulator (PAS), with
# opioids present and arterial ke0; it agrees with the R package tci.
PROPOFOL_ORACLE = [
    ((35, 70, 170, "male"), (6.283078, 25.501315, 168.210976, 1.619285, 1.830921, 0.771329, 0.146000)),
    ((60, 55, 158, "female"), (5.772932, 13.554830, 79.488031, 1.476606, 1.119512, 0.446116, 0.155073)),
    ((72, 82, 175, "male"), (6.595835, 16.752257, 112.951280, 1.640401, 1.306534, 0.582650, 0.140338)),
]
# Oracle 2: a literal transcription of the NONMEM $PK block of Eleveld et al. 2017
# (Supplemental Digital Content 2), with ke0 from its PD control file.
REMIFENTANIL_ORACLE = [
    ((35, 70, 170, "male"), (5.807263, 8.821301, 5.025120, 2.576048, 1.716547, 0.124446, 1.090570)),
    ((30, 60, 160, "female"), (4.283183, 9.202994, 5.476005, 2.918415, 2.606362, 0.136462, 1.259835)),
    ((70, 85, 178, "male"), (5.545734, 9.122214, 1.308290, 2.568562, 1.449768, 0.037357, 0.397222)),
    ((55, 52, 155, "female"), (3.305225, 5.824695, 2.719270, 1.902180, 1.247528, 0.070275, 0.612369)),
]


def patient(age, weight, height, sex):
    return PatientParameters(age=age, weight=weight, height=height, gender=sex)


@pytest.mark.parametrize("covariates, expected", PROPOFOL_ORACLE)
def test_propofol_matches_the_pas_implementation(covariates, expected):
    params = ep.eleveld_propofol(patient(*covariates))
    assert [getattr(params, k) for k in KEYS] == pytest.approx(expected, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("covariates, expected", REMIFENTANIL_ORACLE)
def test_remifentanil_matches_the_published_nonmem_code(covariates, expected):
    params = ep.eleveld_remifentanil(patient(*covariates))
    assert [getattr(params, k) for k in KEYS] == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_rate_constants_follow_volumes_and_clearances():
    for params in (ep.eleveld_propofol(patient(50, 75, 172, "male")),
                   ep.eleveld_remifentanil(patient(50, 75, 172, "male"))):
        assert params.k10 == pytest.approx(params.Cl1 / params.V1)
        assert params.k12 == pytest.approx(params.Cl2 / params.V1)
        assert params.k21 == pytest.approx(params.Cl2 / params.V2)
        assert params.k13 == pytest.approx(params.Cl3 / params.V1)
        assert params.k31 == pytest.approx(params.Cl3 / params.V3)


def test_variability_uses_the_published_variances_and_keeps_the_bis_model():
    spec = ep.ELEVELD_SPEC
    assert spec.propofol_pk["V1"] == pytest.approx(math.sqrt(0.610))
    assert spec.propofol_pk["ke0"] == pytest.approx(math.sqrt(0.702))
    assert spec.remifentanil_pk["V3"] == pytest.approx(math.sqrt(0.809706))
    assert spec.remifentanil_pk["ke0"] == pytest.approx(math.sqrt(0.946705))
    assert dict(spec.pd) == dict(LITERATURE_SPEC.pd)
    assert spec.truncation_sd == LITERATURE_SPEC.truncation_sd


def test_typical_plant_is_eleveld_while_observer_and_bis_model_stay_as_trained():
    pat = patient(50, 75, 172, "male")
    trained = PatientSimulator(patient=pat, seed=11, variability=LITERATURE_SPEC)
    mismatched = ep.make_simulator_class("typical")(patient=pat, seed=11, variability=LITERATURE_SPEC)
    assert mismatched.ppf_params == ep.eleveld_propofol(pat)
    assert mismatched.rftn_params == ep.eleveld_remifentanil(pat)
    assert mismatched.ppf_model.params == trained.ppf_model.params      # Schnider observer
    assert mismatched.rftn_model.params == trained.rftn_model.params    # Minto observer
    assert mismatched.drug_interaction_params == trained.drug_interaction_params


def test_variable_plant_is_reproducible_and_bounded():
    pat = patient(50, 75, 172, "female")
    cls = ep.make_simulator_class("iiv")
    first = cls(patient=pat, seed=11, variability=LITERATURE_SPEC)
    again = cls(patient=pat, seed=11, variability=LITERATURE_SPEC)
    other = cls(patient=pat, seed=12, variability=LITERATURE_SPEC)
    assert first.ppf_params == again.ppf_params
    assert first.ppf_params != other.ppf_params
    typical = ep.eleveld_propofol(pat)
    bound = math.exp(2 * ep.ELEVELD_SPEC.propofol_pk["V1"])
    assert 1 / bound <= first.ppf_params.V1 / typical.V1 <= bound


def test_mismatched_plant_reaches_lower_bis_under_infusion():
    sim = ep.make_simulator_class("typical")(patient=patient(50, 75, 172, "male"), seed=3,
                                             variability=LITERATURE_SPEC)
    for _ in range(60):  # 10 minutes of 8 mg/kg/h propofol and 0.15 ug/kg/min remifentanil
        sim.step(8 * 1000 / 60, dt=10.0, rftn_rate=0.15 * 1000)
    bis = sim.get_bis(add_noise=False)
    assert np.isfinite(bis) and bis < 70
    assert sim.observer_state_ppf[3] > 0 and sim.state_ppf[3] > 0
    assert sim.state_ppf[3] != pytest.approx(sim.observer_state_ppf[3])


def test_install_swaps_the_simulator_used_by_the_environment():
    original = env_module.PatientSimulator
    try:
        ep.install("typical")
        assert env_module.PatientSimulator is not original
        assert issubclass(env_module.PatientSimulator, PatientSimulator)
        ep.install(None)
        assert env_module.PatientSimulator is original
    finally:
        env_module.PatientSimulator = original
