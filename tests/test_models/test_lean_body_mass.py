"""James lean body mass and the PK parameters that depend on it (sim v2).

The v1 code evaluated ``weight / height_m**2`` (a BMI) instead of
``(weight / height_cm)**2``, which pushed every subject onto the lower clamp
(0.3 W for Schnider, 20 kg for Minto). These tests pin the published formula.
"""
import pytest

from models.pharmacokinetics.base import PatientParameters, calculate_lean_body_mass
from models.pharmacokinetics.minto_model import MintoModel
from models.pharmacokinetics.schnider_model import SchniderModel

# Median female test subject of the locked cohort.
FEMALE = dict(age=56.39, weight=61.7, height=162.0, gender="female")


def test_james_lbm_female_reference():
    # 1.07*61.7 - 148*(61.7/162)**2
    assert calculate_lean_body_mass(61.7, 162.0, "female") == pytest.approx(44.5504, abs=1e-3)


def test_james_lbm_male_reference():
    # 1.1*80 - 128*(80/180)**2
    assert calculate_lean_body_mass(80.0, 180.0, "male") == pytest.approx(62.7160, abs=1e-3)


def test_patient_parameters_use_james_lbm():
    patient = PatientParameters(**FEMALE)
    assert patient.lbm == pytest.approx(44.5504, abs=1e-3)


def test_lbm_clamped_to_fraction_of_weight_at_extreme_bmi():
    # 1.07*150 - 148*(150/150)**2 = 12.5 kg, below 0.3 W = 45 kg.
    assert calculate_lean_body_mass(150.0, 150.0, "female") == pytest.approx(45.0)


def test_schnider_clearance_uses_james_lbm():
    params = SchniderModel(patient=PatientParameters(**FEMALE)).params
    # Cl1 = 1.89 + 0.0456(W-77) - 0.0681(LBM-59) + 0.0264(H-177)
    assert params.k10 * params.V1 == pytest.approx(1.7803, abs=2e-3)


def test_minto_central_volume_uses_james_lbm():
    params = MintoModel(patient=PatientParameters(**FEMALE)).params
    # V1 = 5.1 - 0.0201(age-40) + 0.072(LBM-55)
    assert params.lbm == pytest.approx(44.5504, abs=1e-3)
    assert params.V1 == pytest.approx(4.0182, abs=2e-3)


def test_minto_volume_depends_on_sex():
    male = MintoModel(patient=PatientParameters(age=50, weight=70, height=170, gender="male")).params
    female = MintoModel(patient=PatientParameters(age=50, weight=70, height=170, gender="female")).params
    assert male.V1 != pytest.approx(female.V1, abs=1e-3)


@pytest.mark.parametrize("age, expected", [(40.0, 0.595), (70.0, 0.385), (25.0, 0.700)])
def test_minto_ke0_is_age_dependent(age, expected):
    # Minto 1997: ke0 = 0.595 - 0.007 (age - 40) min^-1 (R2 minor 6).
    params = MintoModel(patient=PatientParameters(age=age, weight=70, height=170, gender="male")).params
    assert params.ke0 == pytest.approx(expected, abs=1e-9)


def test_lbm_stays_physiological_at_bmi_45():
    # Female 162 cm, 118 kg (BMI 45): past the James peak but still positive.
    lbm = calculate_lean_body_mass(118.0, 162.0, "female")
    assert 0.3 * 118.0 <= lbm <= 118.0
    assert lbm == pytest.approx(1.07 * 118.0 - 148 * (118.0 / 162.0) ** 2, abs=1e-6)
