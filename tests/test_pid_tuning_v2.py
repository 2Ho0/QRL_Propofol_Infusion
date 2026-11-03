"""PID tuning must score candidates on the same plant as the learned arms."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from evaluation.per_patient_eval import patient_from_record  # noqa: E402
from tune_pid_patient_safe import DEFAULT, _score  # noqa: E402

COHORT = [(1234, patient_from_record(
    {"age": 61.0, "sex": "F", "height": 158.0, "weight": 57.0}))]


def test_score_uses_the_requested_variability():
    plain = _score(DEFAULT, COHORT, 10, 60, "none")
    varied = _score(DEFAULT, COHORT, 10, 60, "literature")
    assert np.isfinite(plain["objective"]) and np.isfinite(varied["objective"])
    assert plain["mean_mdape"] != varied["mean_mdape"]
