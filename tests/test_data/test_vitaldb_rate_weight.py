"""Infusion rates must be per kg of the patient's recorded body weight.

vitaldb.load_case returns NaN for Clinical/weight, so every cache entry was
converted from pump mL/h with the 70-kg fallback, and _apply_clinical_info
then replaced the demographics without converting the rates again. All 3,000
dataset cases carried rates per 70 kg rather than per kg of the patient, while
the simulator multiplies actions by the real weight (independent review,
docs/agent_logs/2026-10-20_0230_review_dose_cost.md).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.vitaldb_loader import VitalDBLoader

CACHE = Path(__file__).resolve().parents[2] / "data" / "vitaldb_cache"
CASE = 7


def _frame():
    # 0.14 ug/kg/min at 70 kg from remifentanil 20 ug/mL is 29.4 mL/h.
    return pd.DataFrame({
        "PPF20_RATE": [7.0, 3.5], "RFTN_RATE": [0.14, 0.07],
        "RFTN20_RATE": [29.4, 14.7], "PPF20_CE": [3.0, 3.1],
        "weight": [70.0, 70.0], "height": [170.0, 170.0], "age": [50.0, 50.0],
        "sex": [1.0, 1.0], "bmi": [24.2, 24.2],
    })


@pytest.fixture
def loader(tmp_path, monkeypatch):
    clinical = pd.DataFrame(
        {"subjectid": [1], "age": [40.0], "sex": ["F"], "height": [160.0],
         "weight": [50.0]},
        index=pd.Index([CASE], name="caseid"))
    instance = VitalDBLoader(cache_dir=str(tmp_path))
    monkeypatch.setattr(instance, "load_clinical_info", lambda: clinical)
    return instance


def test_rates_are_per_kg_of_the_clinical_weight(loader):
    frame = loader._apply_clinical_info(_frame(), CASE)
    assert frame["weight"].tolist() == [50.0, 50.0]
    np.testing.assert_allclose(frame["PPF20_RATE"], [9.8, 4.9])
    np.testing.assert_allclose(frame["RFTN_RATE"], [0.196, 0.098])


def test_applying_the_clinical_record_twice_changes_nothing(loader):
    once = loader._apply_clinical_info(_frame(), CASE)
    twice = loader._apply_clinical_info(once.copy(), CASE)
    pd.testing.assert_frame_equal(once, twice)


def test_pump_volume_rates_and_concentrations_are_unchanged(loader):
    frame = loader._apply_clinical_info(_frame(), CASE)
    np.testing.assert_allclose(frame["RFTN20_RATE"], [29.4, 14.7])
    np.testing.assert_allclose(frame["PPF20_CE"], [3.0, 3.1])


def test_downloaded_case_rates_are_per_kg_of_the_clinical_weight(loader, monkeypatch):
    import vitaldb
    tracks = np.full((3, 12), np.nan)   # Clinical/* columns 8-11 come back NaN
    tracks[:, 0] = 50.0                 # BIS
    tracks[:, 1] = 21.0                 # propofol 20 mg/mL, mL/h
    tracks[:, 4] = 15.0                 # remifentanil 20 ug/mL, mL/h
    monkeypatch.setattr(vitaldb, "load_case", lambda *args, **kwargs: tracks)
    frame = loader.load_case(CASE)
    assert frame["weight"].iloc[0] == 50.0
    np.testing.assert_allclose(frame["PPF20_RATE"], 21.0 * 20.0 / 50.0)
    np.testing.assert_allclose(frame["RFTN_RATE"], 15.0 * 20.0 / (50.0 * 60.0))


@pytest.mark.skipif(not (CACHE / "case_9.pkl").exists(), reason="VitalDB cache not present")
def test_cached_case_rates_match_the_pump_volume_at_the_clinical_weight():
    real = VitalDBLoader(cache_dir=str(CACHE))
    frame = real.load_case(9)
    weight = float(real.load_clinical_info().loc[9, "weight"])
    assert weight != 70.0
    used = frame["RFTN20_RATE"].gt(0) & frame["RFTN_RATE"].gt(0)
    assert used.sum() > 100
    expected = frame.loc[used, "RFTN20_RATE"] * 20.0 / (weight * 60.0)
    np.testing.assert_allclose(frame.loc[used, "RFTN_RATE"], expected, rtol=1e-9)


@pytest.mark.skipif(not (CACHE / "case_9.pkl").exists(), reason="VitalDB cache not present")
def test_control_data_can_be_rebuilt_for_a_locked_case_list():
    real = VitalDBLoader(cache_dir=str(CACHE))
    data = real.prepare_dual_drug_control_data(n_cases=2, caseids=[9, 14])
    assert set(np.unique(data["caseids"]).tolist()) == {9, 14}
