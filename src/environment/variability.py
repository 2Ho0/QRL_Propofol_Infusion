"""Inter-individual variability for the sim v2 plant (Reviewer 2, comment 2).

Each simulated patient gets log-normal random effects, multiplier = exp(eta)
with eta ~ N(0, omega^2), on the PK volumes, clearances and ke0 of both drugs
and on the BIS parameters. The policy never sees these effects: its state
carries the effect-site concentrations of the covariate-only population model,
which is what a TCI pump would display.

Random effects depend only on the environment seed, so all arms and training
seeds see identical patients for the same evaluation seed (common random
numbers), and they come from their own stream so the BIS measurement noise is
unchanged.
"""
from dataclasses import dataclass, replace
from typing import Mapping, Optional

import numpy as np
from scipy.stats import truncnorm

PK_KEYS = ("V1", "V2", "V3", "Cl1", "Cl2", "Cl3", "ke0")
PD_KEYS = ("c50_ppf", "c50_rftn", "gamma", "e0")

# Spawn key separating the random-effect stream from the noise stream.
_EFFECT_STREAM = 1729


@dataclass(frozen=True)
class VariabilitySpec:
    """Standard deviations omega of the log-scale random effects.

    ``truncation_sd`` bounds every eta to +/- that many omegas (None: no
    truncation), so that a single draw cannot produce an implausible patient.
    """
    propofol_pk: Mapping[str, float]
    remifentanil_pk: Mapping[str, float]
    pd: Mapping[str, float]
    source: str = ""
    truncation_sd: Optional[float] = None


@dataclass(frozen=True)
class PatientEffects:
    """Multipliers exp(eta) for one simulated patient."""
    propofol_pk: Mapping[str, float]
    remifentanil_pk: Mapping[str, float]
    pd: Mapping[str, float]


def sample_patient_effects(spec: VariabilitySpec, seed: int) -> PatientEffects:
    """Draw one patient's random effects from ``spec``.

    One standard normal is drawn for every key in a fixed order, then scaled
    by its omega, so a patient keeps its latent effects when an omega changes.
    """
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), _EFFECT_STREAM]))
    bound = spec.truncation_sd

    def draw(omegas: Mapping[str, float], keys) -> dict:
        unknown = set(omegas) - set(keys)
        if unknown:
            raise KeyError(f"unknown variability keys: {sorted(unknown)}")
        if bound is None:
            z = rng.standard_normal(len(keys))
        else:
            z = truncnorm.rvs(-bound, bound, size=len(keys), random_state=rng)
        return {key: float(np.exp(float(omegas.get(key, 0.0)) * z_k))
                for key, z_k in zip(keys, z)}

    return PatientEffects(
        propofol_pk=draw(spec.propofol_pk, PK_KEYS),
        remifentanil_pk=draw(spec.remifentanil_pk, PK_KEYS),
        pd=draw(spec.pd, PD_KEYS),
    )


def individualize_pk(params, factors: Mapping[str, float]):
    """Scale volumes, clearances and ke0 of a Schnider or Minto parameter set.

    Clearances are recovered from the rate constants (Cl1 = k10 V1,
    Cl2 = k12 V1, Cl3 = k13 V1), scaled, and turned back into rate constants
    for the scaled volumes.
    """
    v1, v2, v3 = params.V1 * factors["V1"], params.V2 * factors["V2"], params.V3 * factors["V3"]
    cl1 = params.k10 * params.V1 * factors["Cl1"]
    cl2 = params.k12 * params.V1 * factors["Cl2"]
    cl3 = params.k13 * params.V1 * factors["Cl3"]
    updates = dict(
        V1=v1, V2=v2, V3=v3,
        k10=cl1 / v1, k12=cl2 / v1, k21=cl2 / v2, k13=cl3 / v1, k31=cl3 / v3,
        ke0=params.ke0 * factors["ke0"],
    )
    if hasattr(params, "Cl1"):
        updates.update(Cl1=cl1, Cl2=cl2, Cl3=cl3)
    return replace(params, **updates)


# Published inter-individual variability, as log-scale SDs (omega).
# Schnider et al. 1998 Table 2 and Minto et al. 1997 Table 3 report %CV,
# defined by Minto as the square root of the variance of eta, so
# omega = %CV / 100 ("<1%" entries are taken as 0.01). Neither paper gives a
# variability for the ke0 of the model used here (Schnider 0.456 min^-1 from
# Schnider 1999; Minto's age-dependent ke0 from Table 5), so ke0 has none.
# Bouillon et al. 2004 Table 8 reports the CV as the SD in the log domain; the
# TCI-concentration column is used because the plant's concentrations come
# from the same Schnider/Minto models, and the baseline has no variability.
# Draws are truncated at +/-2 SD: with omega(C50_remi) = 0.888 an untruncated
# draw yields remifentanil C50 values below 2 ng/ml.
LITERATURE_SPEC = VariabilitySpec(
    propofol_pk={"V1": 0.0404, "V2": 0.01, "V3": 0.1435,
                 "Cl1": 0.1005, "Cl2": 0.01, "Cl3": 0.1179, "ke0": 0.0},
    remifentanil_pk={"V1": 0.26, "V2": 0.29, "V3": 0.66,
                     "Cl1": 0.14, "Cl2": 0.36, "Cl3": 0.41, "ke0": 0.0},
    pd={"c50_ppf": 0.182, "c50_rftn": 0.888, "gamma": 0.304, "e0": 0.0},
    source="Schnider 1998, Minto 1997, Bouillon 2004 (TCI concentrations); +/-2 SD",
    truncation_sd=2.0,
)


def get_variability_spec(name: str) -> Optional[VariabilitySpec]:
    """Look up a named variability setting ('literature' or 'none')."""
    if name == "literature":
        return LITERATURE_SPEC
    if name == "none":
        return None
    raise ValueError(f"unknown variability setting: {name!r}")
