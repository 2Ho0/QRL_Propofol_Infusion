"""Noise-free BIS metrics next to the measured ones (sim v2).

MDAPE is computed on the measured, noisy BIS. The contract keeps the
measured-BIS metrics and adds the same metrics on the noise-free BIS.
"""
import csv

import numpy as np
import pytest

from evaluation.per_patient_eval import evaluate_patient
from evaluation.results_contract import PatientResult, RESULT_COLUMNS, append_results
from models.pharmacokinetics import PatientParameters


class _ConstantAgent:
    sequence_length = 1

    def select_action(self, state, deterministic=True):
        return np.array([6.0, 0.15])


def test_contract_has_noise_free_metric_columns():
    for column in ("mdpe_true", "mdape_true", "wobble_true", "divergence_true"):
        assert column in RESULT_COLUMNS


def test_evaluate_patient_reports_noise_free_metrics():
    patient = PatientParameters(age=50, weight=70, height=170, gender="male")
    metrics, bis, _, _ = evaluate_patient(
        _ConstantAgent(), patient, seed=11, steps=30,
        observation_interval_seconds=10, control_interval_seconds=60,
        variability=None)
    true_trace = np.asarray(metrics["bis_true_trace"])
    assert true_trace.shape == bis.shape
    # Measured BIS = noise-free BIS + N(0, 2^2), clipped to [0, 100].
    residual = bis - true_trace
    assert 1.5 < residual.std() < 2.5
    assert np.isfinite(metrics["mdape_true"])
    assert metrics["mdape_true"] != pytest.approx(metrics["mdape"], abs=1e-9)


def test_append_refuses_a_file_with_a_different_header(tmp_path):
    path = tmp_path / "results.csv"
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerow(RESULT_COLUMNS[:-4])
    row = PatientResult(method="m", patient_id=1, seed=0, mdpe=0.0, mdape=0.0,
                        wobble=0.0, divergence=0.0, tit_45_55=0.0, tit_40_60=0.0,
                        n_bis_lt30=0, n_bis_gt70=0, mean_ppf=0.0, mean_remi=0.0)
    with pytest.raises(ValueError, match="header"):
        append_results([row], path)


def test_evaluation_requires_an_explicit_plant():
    """Unported scripts must fail loudly instead of evaluating a v1 plant."""
    patient = PatientParameters(age=50, weight=70, height=170, gender="male")
    with pytest.raises(TypeError):
        evaluate_patient(_ConstantAgent(), patient, seed=1, steps=2,
                         observation_interval_seconds=10, control_interval_seconds=60)


def test_append_to_an_empty_file_writes_the_header(tmp_path):
    path = tmp_path / "results.csv"
    path.write_text("")
    row = PatientResult(method="m", patient_id=1, seed=0, mdpe=0.0, mdape=0.0,
                        wobble=0.0, divergence=0.0, tit_45_55=0.0, tit_40_60=0.0,
                        n_bis_lt30=0, n_bis_gt70=0, mean_ppf=0.0, mean_remi=0.0)
    append_results([row], path)
    with path.open(newline="") as stream:
        header = next(csv.reader(stream))
    assert header == RESULT_COLUMNS
