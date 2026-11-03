"""Inter-individual variability in the plant and a TCI-style observer.

The plant (true patient) gets log-normal random effects on PK and PD. The
policy state receives the effect-site concentrations predicted by the
covariate-only population model, as a TCI pump displays them. The spreads used
here are test values; the literature values live in ``variability.py``.
"""
import numpy as np
import pytest

from environment.dual_drug_env import DualDrugEnv
from environment.patient_simulator import PatientSimulator
from environment.variability import (
    VariabilitySpec,
    individualize_pk,
    sample_patient_effects,
)
from models.pharmacokinetics import PatientParameters
from models.pharmacokinetics.schnider_model import SchniderModel

PK = ("V1", "V2", "V3", "Cl1", "Cl2", "Cl3", "ke0")
SPEC = VariabilitySpec(
    propofol_pk={k: 0.3 for k in PK},
    remifentanil_pk={k: 0.3 for k in PK},
    pd={"c50_ppf": 0.2, "c50_rftn": 0.5, "gamma": 0.3, "e0": 0.05},
)
ZERO = VariabilitySpec(
    propofol_pk={k: 0.0 for k in PK},
    remifentanil_pk={k: 0.0 for k in PK},
    pd={"c50_ppf": 0.0, "c50_rftn": 0.0, "gamma": 0.0, "e0": 0.0},
)
PATIENT = PatientParameters(age=55, weight=68, height=166, gender="female")


def test_effects_are_reproducible_for_a_seed():
    assert sample_patient_effects(SPEC, 42) == sample_patient_effects(SPEC, 42)


def test_effects_change_with_the_seed():
    assert sample_patient_effects(SPEC, 42) != sample_patient_effects(SPEC, 43)


def test_log_effects_have_the_specified_spread():
    logs = np.log([sample_patient_effects(SPEC, s).pd["c50_rftn"] for s in range(20000)])
    assert abs(logs.mean()) < 0.02
    assert logs.std() == pytest.approx(0.5, rel=0.03)


def test_zero_spread_gives_unit_multipliers():
    effects = sample_patient_effects(ZERO, 7)
    for group in (effects.propofol_pk, effects.remifentanil_pk, effects.pd):
        assert all(value == 1.0 for value in group.values())


def test_individualized_pk_scales_volumes_and_clearances():
    base = SchniderModel(patient=PATIENT).params
    factors = {"V1": 2.0, "V2": 1.0, "V3": 1.0, "Cl1": 1.5, "Cl2": 1.0, "Cl3": 1.0, "ke0": 1.2}
    ind = individualize_pk(base, factors)
    assert ind.V1 == pytest.approx(2.0 * base.V1)
    assert ind.k10 * ind.V1 == pytest.approx(1.5 * base.k10 * base.V1)
    assert ind.k21 == pytest.approx(base.k21)          # Cl2 / V2 unchanged
    assert ind.k12 == pytest.approx(base.k12 / 2.0)    # Cl2 / V1 with V1 doubled
    assert ind.ke0 == pytest.approx(1.2 * base.ke0)


def _infuse(sim, minutes=60):
    for _ in range(minutes * 6):
        sim.step(100.0, dt=10.0, rftn_rate=150.0)


def test_observer_matches_plant_without_variability():
    sim = PatientSimulator(patient=PATIENT, dt=10.0, noise_std=2.0, seed=5, variability=ZERO)
    _infuse(sim)
    assert np.allclose(sim.observer_state_ppf, sim.state_ppf, atol=1e-9)
    assert np.allclose(sim.observer_state_rftn, sim.state_rftn, atol=1e-9)


def test_observer_differs_from_plant_with_variability():
    differing = 0
    for seed in range(10):
        sim = PatientSimulator(patient=PATIENT, dt=10.0, noise_std=2.0, seed=seed, variability=SPEC)
        _infuse(sim, minutes=30)
        differing += abs(sim.observer_state_ppf[3] - sim.state_ppf[3]) > 1e-3
    assert differing >= 8


def test_no_infusion_gives_the_individual_baseline():
    sim = PatientSimulator(patient=PATIENT, dt=10.0, noise_std=2.0, seed=3, variability=SPEC)
    for _ in range(10):
        sim.step(0.0, dt=10.0, rftn_rate=0.0)
    assert sim.drug_interaction_params.bis0 != pytest.approx(98.0)
    assert sim.get_bis(add_noise=False) == pytest.approx(sim.drug_interaction_params.bis0)
    assert np.all(sim.observer_state_ppf == 0.0) and np.all(sim.state_ppf == 0.0)


def test_variability_does_not_change_the_measurement_noise_stream():
    # Measured BIS is clipped at 100, so compare the noise generator itself.
    plain = PatientSimulator(patient=PATIENT, dt=10.0, noise_std=2.0, seed=9)
    varied = PatientSimulator(patient=PATIENT, dt=10.0, noise_std=2.0, seed=9, variability=SPEC)
    assert plain.rng.bit_generator.state == varied.rng.bit_generator.state
    for _ in range(20):
        plain.step(50.0, dt=10.0)
        varied.step(50.0, dt=10.0)
    assert plain.rng.bit_generator.state == varied.rng.bit_generator.state


def test_env_state_uses_predicted_concentrations():
    env = DualDrugEnv(patient=PATIENT, seed=4, dt=10.0 / 60.0, variability=SPEC,
                      shaping_interval_steps=6)
    for _ in range(60):
        obs, _, _, _, info = env.step(np.array([8.0, 0.2]))
    assert obs[1] == pytest.approx(env.simulator.observer_state_ppf[3], rel=1e-6)
    assert obs[2] == pytest.approx(env.simulator.observer_state_rftn[3], rel=1e-6)
    assert info["ce_propofol"] == pytest.approx(env.simulator.state_ppf[3])
    assert info["ce_propofol_predicted"] == pytest.approx(env.simulator.observer_state_ppf[3])


# --- literature values and truncation --------------------------------------
from environment.variability import LITERATURE_SPEC, get_variability_spec  # noqa: E402


def test_literature_spec_uses_published_log_scale_sds():
    # Schnider 1998 Table 2 and Minto 1997 Table 3 (%CV = 100 * omega),
    # Bouillon 2004 Table 8 (CV is the SD in the log domain, TCI concentrations).
    assert LITERATURE_SPEC.propofol_pk["V3"] == pytest.approx(0.1435)
    assert LITERATURE_SPEC.propofol_pk["Cl1"] == pytest.approx(0.1005)
    assert LITERATURE_SPEC.remifentanil_pk["V1"] == pytest.approx(0.26)
    assert LITERATURE_SPEC.remifentanil_pk["Cl3"] == pytest.approx(0.41)
    assert LITERATURE_SPEC.pd["c50_ppf"] == pytest.approx(0.182)
    assert LITERATURE_SPEC.pd["c50_rftn"] == pytest.approx(0.888)
    assert LITERATURE_SPEC.pd["gamma"] == pytest.approx(0.304)
    assert LITERATURE_SPEC.pd["e0"] == 0.0


def test_truncation_bounds_every_random_effect():
    spec = VariabilitySpec(propofol_pk=SPEC.propofol_pk, remifentanil_pk=SPEC.remifentanil_pk,
                           pd=SPEC.pd, truncation_sd=2.0)
    logs = np.log([sample_patient_effects(spec, s).pd["c50_rftn"] for s in range(5000)])
    assert np.abs(logs).max() <= 2.0 * 0.5 + 1e-12
    # SD of a standard normal truncated at +/-2 is 0.8796.
    assert logs.std() == pytest.approx(0.5 * 0.8796, rel=0.04)


def test_variability_spec_lookup():
    assert get_variability_spec("none") is None
    assert get_variability_spec("literature") is LITERATURE_SPEC
    with pytest.raises(ValueError):
        get_variability_spec("v1")
