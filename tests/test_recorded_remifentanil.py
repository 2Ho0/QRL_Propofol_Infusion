"""Recorded-remifentanil plant of the single-drug baseline (experiments/recorded_remifentanil.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

import recorded_remifentanil as rr  # noqa: E402
import evaluation.per_patient_eval as per_patient_eval  # noqa: E402
from environment.dual_drug_env import DualDrugEnv, RFTN_REFERENCE_RATE  # noqa: E402

PID = {"kp_ppf": 0.35, "ki_ppf": 0.04, "kd_ppf": 0.28, "kp_rftn": 0.08,
       "ki_rftn": 0.008, "kd_rftn": 0.007, "base_ppf": 8.6, "base_rftn": 0.07}


def _frame(ppf, remi):
    return pd.DataFrame({"PPF20_RATE": np.asarray(ppf, dtype=float),
                         "RFTN_RATE": np.asarray(remi, dtype=float)})


def _schedule(rates, hold):
    return rr.RemifentanilSchedule(rates=tuple(rates), hold=hold)


def _patient(schedule, caseid=7):
    return rr.RecordedRemifentanilPatient(age=50, weight=60, height=165, gender="female",
                                          caseid=caseid, remifentanil=schedule)


def _env(schedule=None, residual=False, **kwargs):
    settings = dict(dt=1 / 6, max_steps=60, shaping_interval_steps=6, seed=3)
    if residual:
        settings.update(residual_pid=PID, residual_range=(4.0, 0.05))
    settings.update(kwargs)
    patient = None if schedule is None else _patient(schedule)
    return rr.RecordedRemifentanilEnv(patient=patient, **settings)


def test_schedule_starts_with_the_first_drug_and_fills_pump_gaps_forward():
    # 10-s grid: nothing for 3 samples, propofol starts at sample 3, remifentanil
    # at sample 4 with a missing sample (NaN) that the pump held at 0.2.
    ppf = [0, np.nan, 0] + [5.0] * 15
    remi = [np.nan, 0, 0, 0, 0.1, 0.2, np.nan, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0, 0]
    schedule = rr.schedule_from_case(_frame(ppf, remi))
    # Minute 0 = samples 3..8 (0, 0.1, 0.2, 0.2 filled, 0.2, 0.3); minute 1 = samples
    # 9..14 (0.3 x 6); the record ends at sample 15 (last positive remifentanil).
    assert schedule.rates == pytest.approx((1.0 / 6, 0.3, 0.4))
    assert schedule.hold == pytest.approx(0.3)   # median of the positive minute rates


def test_schedule_keeps_the_zeros_the_pump_recorded():
    # VitalDBLoader.load_case merges the tracks as where(RFTN20 > 0, RFTN20, RFTN50),
    # so a recorded RFTN20 of 0 in a case without RFTN50 becomes NaN in RFTN_RATE.
    # That is a pause, not a missing sample; only samples absent from both raw
    # tracks are missing.
    nan = np.nan
    r20 = [nan, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nan, 6.0, 6.0]
    frame = pd.DataFrame({
        "PPF20_RATE": [0.0] + [5.0] * 11,
        "RFTN20_RATE": r20,
        "RFTN50_RATE": [nan] * 12,
        "RFTN_RATE": [nan, 0.3, 0.3, nan, nan, nan, nan, nan, nan, nan, 0.3, 0.3],
    })
    schedule = rr.schedule_from_case(frame)
    # Samples 1..11: minute 0 = 0.3, 0.3, 0, 0, 0, 0; minute 1 = 0, 0, then the
    # truly missing sample 9 takes the previous value (0), then 0.3, 0.3.
    assert schedule.rates == pytest.approx((0.1, 0.12))


def test_schedule_needs_some_remifentanil():
    with pytest.raises(ValueError, match="remifentanil"):
        rr.schedule_from_case(_frame([5.0] * 12, [0.0] * 12))


def test_schedule_holds_its_rate_after_the_record():
    schedule = _schedule([0.5, 0.1], hold=0.2)
    assert [schedule.rate_at(m) for m in (0.0, 0.99, 1.0, 1.5, 2.0, 300.0)] == \
        pytest.approx([0.5, 0.5, 0.1, 0.1, 0.2, 0.2])


def test_schedule_round_trips_through_json(tmp_path):
    schedules = {16: _schedule([0.5, 0.1], 0.2), 19: _schedule([0.3], 0.3)}
    path = tmp_path / "schedules.json"
    rr.save_schedules(schedules, path, {"rule": "test"})
    assert rr.load_schedules(path) == schedules


def test_plain_environment_applies_the_recorded_rate_not_the_command():
    env = _env(_schedule([0.5, 0.1], hold=0.2))
    env.reset()
    applied = []
    for _ in range(18):                       # three minutes of 10-s steps
        _, _, _, _, info = env.step(np.array([6.0, 1.7]))
        applied.append(info["applied_action"])
    applied = np.asarray(applied)
    assert np.allclose(applied[:, 0], 6.0)
    assert np.allclose(applied[:, 1], [0.5] * 6 + [0.1] * 6 + [0.2] * 6)
    assert env.prev_action_rftn == pytest.approx(0.2)   # the agent observes the record


def test_residual_environment_keeps_the_propofol_residual_and_replaces_remifentanil():
    schedule = _schedule([0.5, 0.1], hold=0.2)
    recorded = _env(schedule, residual=True)
    reference = DualDrugEnv(patient=_patient(schedule), dt=1 / 6, max_steps=60,
                            shaping_interval_steps=6, seed=3, residual_pid=PID,
                            residual_range=(4.0, 0.05))
    recorded.reset()
    reference.reset()
    action = np.array([9.0, 0.3])             # fractions 0.75 and 0.15 of the action range
    _, _, _, _, got = recorded.step(action)
    _, _, _, _, expected = reference.step(action)
    # First decision: same PID base and residual for propofol, record for remifentanil.
    assert got["applied_action"][0] == pytest.approx(expected["applied_action"][0])
    assert got["applied_action"][1] == pytest.approx(0.5)
    rates = [recorded.step(action)[4]["applied_action"][1] for _ in range(11)]
    assert rates == pytest.approx([0.5] * 5 + [0.1] * 6)


def test_recorded_rate_is_clipped_to_the_action_range():
    env = _env(_schedule([3.5], hold=3.5))
    env.reset()
    assert env.step(np.array([6.0, 0.0]))[4]["applied_action"][1] == pytest.approx(2.0)


def test_patient_without_a_record_is_refused():
    plain = rr.RecordedRemifentanilPatient(age=50, weight=60, height=165, gender="female")
    with pytest.raises(ValueError, match="remifentanil record"):
        rr.RecordedRemifentanilEnv(patient=plain, dt=1 / 6, max_steps=6, seed=0)


def test_default_patient_gets_the_reference_rate():
    # stage2_online_finetuning builds one sanity-check environment without a patient.
    env = _env(None)
    env.reset()
    assert env.step(np.array([6.0, 1.0]))[4]["applied_action"][1] == pytest.approx(
        RFTN_REFERENCE_RATE)


def test_install_makes_the_per_patient_evaluation_use_the_record():
    class Constant:
        def select_action(self, state, deterministic=True):
            return np.array([6.0, 1.9])

    patient = _patient(_schedule([0.5, 0.1], hold=0.2))
    try:
        rr.install()
        assert per_patient_eval.DualDrugEnv is rr.RecordedRemifentanilEnv
        _, _, ppf, remi = per_patient_eval.evaluate_patient(
            Constant(), patient, seed=1, steps=3, observation_interval_seconds=10,
            control_interval_seconds=60, variability=None)
    finally:
        rr.restore()
    assert per_patient_eval.DualDrugEnv is DualDrugEnv
    assert np.allclose(ppf, 6.0)
    assert np.allclose(remi, [0.5] * 6 + [0.1] * 6 + [0.2] * 6)


def test_recorded_patient_from_record_attaches_the_case_schedule():
    schedules = {16: _schedule([0.5], 0.5)}
    rec = {"caseid": 16, "age": 61, "sex": "F", "height": 160.0, "weight": 55.0, "subjectid": 3}
    patient = rr.recorded_patient_from_record(rec, schedules)
    assert (patient.caseid, patient.age, patient.weight, patient.gender) == (16, 61, 55.0, "female")
    assert patient.remifentanil == schedules[16]
    with pytest.raises(KeyError):
        rr.recorded_patient_from_record({**rec, "caseid": 99}, schedules)
