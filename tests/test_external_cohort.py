"""External (NHANES) cohort of the cross-dataset evaluation (experiments/evaluate_external_cohort.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import evaluate_external_cohort as ext  # noqa: E402


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "SEQN": np.arange(130000, 130000 + n, dtype=float),
        "RIAGENDR": rng.choice([1.0, 2.0], n),
        "RIDAGEYR": rng.integers(5, 81, n).astype(float),
        "WTMEC2YR": rng.uniform(5e3, 1e5, n),
        "BMXWT": rng.uniform(30, 160, n),
        "BMXHT": rng.uniform(140, 200, n),
    })


def test_cohort_keeps_only_eligible_adults():
    frame = _frame()
    frame.loc[0, "BMXWT"] = np.nan
    cohort = ext.select_cohort(frame, n=50, seed=1)
    bmi = cohort.weight / (cohort.height / 100) ** 2
    assert len(cohort) == 50 and cohort.caseid.is_unique
    assert cohort.age.between(18, 80).all()
    assert bmi.between(ext.BMI_RANGE[0], ext.BMI_RANGE[1]).all()
    assert 130000 not in set(cohort.caseid)
    assert set(cohort.sex) <= {"M", "F"}


def test_cohort_is_deterministic_and_depends_on_seed():
    frame = _frame()
    a = ext.select_cohort(frame, n=50, seed=1)
    pd.testing.assert_frame_equal(a, ext.select_cohort(frame.sample(frac=1, random_state=3), n=50, seed=1))
    assert set(a.caseid) != set(ext.select_cohort(frame, n=50, seed=2).caseid)


def test_cohort_sampling_follows_the_exam_weights():
    frame = _frame(n=2000)
    eligible = ext.eligible(frame)
    heavy = eligible.SEQN.iloc[:20]
    frame.loc[frame.SEQN.isin(heavy), "WTMEC2YR"] = 1e7
    cohort = ext.select_cohort(frame, n=40, seed=5)
    assert cohort.caseid.isin(heavy.astype(int)).sum() >= 15


def test_cohort_refuses_too_few_eligible():
    with pytest.raises(ValueError, match="eligible"):
        ext.select_cohort(_frame(n=30), n=500, seed=1)


def test_cohort_ids_do_not_collide_with_vitaldb_case_ids():
    cohort = ext.select_cohort(_frame(), n=50, seed=1)
    assert (cohort.caseid > 10_000).all()
