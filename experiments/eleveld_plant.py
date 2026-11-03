"""Eleveld plant for the model-mismatch evaluation.

The policies were trained on a plant that follows the Schnider (propofol) and
Minto (remifentanil) models. Here the simulated patient follows the Eleveld
models instead:
- propofol: Eleveld et al., Br J Anaesth 2018;120:942-59, with opioids present
  and the arterial effect-site rate. The equations and estimates agree with two
  public implementations, the Python Anesthesia Simulator (PAS) and the R
  package tci (tests/test_eleveld_plant.py checks the PAS values).
- remifentanil: Eleveld et al., Anesthesiology 2017;126:1005-18, transcribed
  from the NONMEM control files of its Supplemental Digital Content 2. PAS and
  tci differ from that code (PAS: V2ref 8.882 and no sex effect on V2; tci: V3
  scaled by V2ref), so they were not used for this drug.
Everything else is unchanged: the Bouillon BIS surface and its random effects,
the monitor noise, and the observer in the state, which integrates the
Schnider/Minto population models as a TCI pump would. The controllers
therefore act on concentration predictions from the model they were trained on
while the patient follows a different model.

Two plants are offered:
- "typical": the Eleveld typical values for the patient's covariates;
- "iiv": the same with the published interindividual variability of the
  Eleveld models (log-normal, truncated at +/-2 SD as in training). The latent
  draws are those of the training-type plant for the same seed, because
  sample_patient_effects draws one standard normal per key in a fixed order.
"""
from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from math import exp, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import environment.dual_drug_env as env_module  # noqa: E402
from environment.patient_simulator import PatientSimulator  # noqa: E402
from environment.variability import (  # noqa: E402
    LITERATURE_SPEC, VariabilitySpec, individualize_pk, sample_patient_effects)

_ORIGINAL_SIMULATOR = env_module.PatientSimulator
MODES = ("typical", "iiv")


@dataclass(frozen=True)
class PlantPK:
    """Three-compartment parameters in the units of the simulator (L, L/min, 1/min)."""
    V1: float
    V2: float
    V3: float
    Cl1: float
    Cl2: float
    Cl3: float
    ke0: float
    k10: float
    k12: float
    k21: float
    k13: float
    k31: float


def _plant(v1, v2, v3, cl, q2, q3, ke0) -> PlantPK:
    return PlantPK(V1=v1, V2=v2, V3=v3, Cl1=cl, Cl2=q2, Cl3=q3, ke0=ke0,
                   k10=cl / v1, k12=q2 / v1, k21=q2 / v2, k13=q3 / v1, k31=q3 / v3)


def _sigmoid(x, e50, slope):
    return x ** slope / (x ** slope + e50 ** slope)


def _fat_free_mass(male: bool, age, weight, height_cm):
    """Al-Sallami fat-free mass (kg), as used by both Eleveld models."""
    bmi = weight / (height_cm / 100.0) ** 2
    if male:
        return (0.88 + 0.12 / (1 + (age / 13.4) ** -12.7)) * 9270 * weight / (6680 + 216 * bmi)
    return (1.11 - 0.11 / (1 + (age / 7.1) ** -1.1)) * 9270 * weight / (8780 + 244 * bmi)


def _fat_free_mass_nonmem(male: bool, age, weight, height_cm):
    """Al-Sallami fat-free mass with the rounded constants of the Eleveld 2017 NONMEM code."""
    ht2 = (height_cm / 100.0) ** 2
    if male:
        maturation = 0.88 + (1 - 0.88) / (1 + (age / 13.4) ** -12.7)
        return maturation * 42.92 * ht2 * weight / (30.93 * ht2 + weight)
    maturation = 1.11 + (1 - 1.11) / (1 + (age / 7.1) ** -1.1)
    return maturation * 37.99 * ht2 * weight / (35.98 * ht2 + weight)


def _covariates(patient):
    return patient.age, patient.weight, patient.height, patient.gender == "male"


# Eleveld 2018 propofol estimates (theta numbering of the paper).
_PROPOFOL = {1: 6.2830780766822, 2: 25.5013145036879, 3: 272.8166615043603,
             4: 1.7895836588902, 5: 1.7500983738779, 6: 1.1085424008536,
             8: 42.2760190602615, 9: 9.0548452392807, 10: -0.015633, 11: -0.00285709,
             12: 33.5531248778544, 13: -0.0138166, 14: 68.2767978846832,
             15: 2.1002218877899, 16: 1.3042680471360}
_PROPOFOL_KE0_ARTERIAL = 0.146


def eleveld_propofol(patient, opioids: bool = True) -> PlantPK:
    """Eleveld 2018 propofol PK and arterial ke0 for the patient's covariates."""
    th = _PROPOFOL
    age, weight, height, male = _covariates(patient)
    age_ref, weight_ref = 35.0, 70.0

    def aging(x):
        return exp(x * (age - age_ref))

    def opioid(x):
        return exp(x * age) if opioids else 1.0

    def central(w):
        return _sigmoid(w, th[12], 1)

    pma_weeks, pma_ref_weeks = age * 52 + 40, age_ref * 52 + 40
    cl_maturation = _sigmoid(pma_weeks, th[8], th[9]) / _sigmoid(pma_ref_weeks, th[8], th[9])
    q3_maturation = _sigmoid(pma_weeks, th[14], 1)
    q3_maturation_ref = _sigmoid(pma_ref_weeks, th[14], 1)
    ffm_ratio = _fat_free_mass(male, age, weight, height) / _fat_free_mass(True, age_ref, weight_ref, 170.0)

    v1 = th[1] * central(weight) / central(weight_ref)
    v2 = th[2] * weight / weight_ref * aging(th[10])
    v3 = th[3] * ffm_ratio * opioid(th[13])
    cl = (th[4] if male else th[15]) * (weight / weight_ref) ** 0.75 * cl_maturation * opioid(th[11])
    q2 = th[5] * (v2 / th[2]) ** 0.75 * (1 + th[16] * (1 - q3_maturation))
    q3 = th[6] * (v3 / th[3]) ** 0.75 * q3_maturation / q3_maturation_ref
    ke0 = _PROPOFOL_KE0_ARTERIAL * (weight / 70.0) ** -0.25
    return _plant(v1, v2, v3, cl, q2, q3, ke0)


# Eleveld 2017 remifentanil estimates (THETA of the NONMEM PK control file).
_REMI = {1: 1.759110, 2: 2.177170, 3: 1.614450, 4: 0.9462570, 5: 0.5403150, 6: -2.083880,
         7: 2.878980, 8: -5.544810e-03, 9: -3.269850e-03, 10: -3.151350e-02,
         11: 4.704050e-01, 12: -2.604960e-02}
_REMI_KE0, _REMI_KE0_AGING = 1.090570, -2.885600e-02  # PD control file


def eleveld_remifentanil(patient) -> PlantPK:
    """Eleveld 2017 remifentanil PK and ke0 for the patient's covariates."""
    th = _REMI
    age, weight, height, male = _covariates(patient)
    maturation = _sigmoid(weight, th[7], 2) / _sigmoid(70.0, th[7], 2)
    # The NONMEM reference FFMR has no maturation factor (42.92 * 1.7^2 * 70 / (30.93 * 1.7^2 + 70)).
    size = _fat_free_mass_nonmem(male, age, weight, height) / (42.92 * 1.7 ** 2 * 70 / (30.93 * 1.7 ** 2 + 70))
    k_v1 = exp(th[8] * (age - 35))          # also Q2 and Q3
    k_v2 = exp(th[9] * (age - 35))          # also CL
    k_v3 = exp(th[10] * (age - 35))
    sex = 1.0 if male else 1 + _sigmoid(age, 12, 6) * (1 - _sigmoid(age, 45, 6)) * th[11]
    v1 = exp(th[1]) * size * k_v1
    v2 = exp(th[2]) * size * k_v2 * sex
    v3 = exp(th[3]) * size * k_v3 * exp(th[12] * (weight - 70))
    cl = exp(th[4]) * size ** 0.75 * k_v2 * sex * maturation
    q2 = exp(th[5]) * (v2 / exp(th[2])) ** 0.75 * k_v1 * sex
    q3 = exp(th[6]) * (v3 / exp(th[3])) ** 0.75 * k_v1
    ke0 = _REMI_KE0 * exp(_REMI_KE0_AGING * (age - 35))
    return _plant(v1, v2, v3, cl, q2, q3, ke0)


# Published interindividual variances (omega^2) as log-scale SDs. Propofol: Eleveld
# 2018 (PK and ke0); remifentanil: Eleveld 2017 SDC 2 ($OMEGA of the PK and PD files).
ELEVELD_SPEC = VariabilitySpec(
    propofol_pk={"V1": sqrt(0.610), "V2": sqrt(0.565), "V3": sqrt(0.597), "Cl1": sqrt(0.265),
                 "Cl2": sqrt(0.346), "Cl3": sqrt(0.209), "ke0": sqrt(0.702)},
    remifentanil_pk={"V1": sqrt(0.103576), "V2": sqrt(0.115366), "V3": sqrt(0.809706),
                     "Cl1": sqrt(0.0197392), "Cl2": sqrt(0.0546616), "Cl3": sqrt(0.285485),
                     "ke0": sqrt(0.946705)},
    pd=LITERATURE_SPEC.pd,
    source="Eleveld 2018 (propofol), Eleveld 2017 SDC 2 (remifentanil); BIS: Bouillon 2004; +/-2 SD",
    truncation_sd=LITERATURE_SPEC.truncation_sd,
)


def make_simulator_class(mode: str):
    """A PatientSimulator whose plant follows the Eleveld models ("typical" or "iiv")."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    signature = inspect.signature(PatientSimulator.__init__)

    class EleveldPlantSimulator(PatientSimulator):
        plant_mode = mode

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            propofol = eleveld_propofol(self.patient)
            remifentanil = eleveld_remifentanil(self.patient)
            if mode == "iiv":
                seed = signature.bind(self, *args, **kwargs).arguments.get("seed")
                if seed is None:
                    raise ValueError("the variable Eleveld plant needs a seed")
                effects = sample_patient_effects(ELEVELD_SPEC, seed)
                propofol = individualize_pk(propofol, effects.propofol_pk)
                remifentanil = individualize_pk(remifentanil, effects.remifentanil_pk)
            self.ppf_params, self.rftn_params = propofol, remifentanil

    return EleveldPlantSimulator


def install(mode):
    """Make DualDrugEnv build Eleveld plants (mode) or the trained-type plant again (None)."""
    env_module.PatientSimulator = _ORIGINAL_SIMULATOR if mode is None else make_simulator_class(mode)
