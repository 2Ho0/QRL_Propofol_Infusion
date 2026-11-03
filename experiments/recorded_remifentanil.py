"""Recorded remifentanil for the single-drug (propofol-only) baseline.

In Yun et al. (Comput. Biol. Med. 2023), the controller sets propofol only,
and remifentanil follows the anesthesiologist's manual infusion from the
clinical record. This module gives the simulator that structure: every patient
carries the remifentanil rate recorded in its own VitalDB case, and the
environment applies that rate whatever the policy commands for remifentanil.
Propofol is unchanged (in the v3 residual mode, tuned-PID propofol plus the
policy's residual).

Rule for the recorded rate (fixed before any run):
1. the 10-s grid of VitalDBLoader.load_case (ug/kg/min of recorded weight);
2. simulator t = 0 is the first sample with propofol or remifentanil above 0;
3. the record ends at the last sample with remifentanil above 0;
4. missing samples in between take the previous value (a pump holds its rate);
   a sample is missing only if both raw tracks (RFTN20, RFTN50) lack it, and a
   recorded 0 in either track is a pause, not a missing sample;
5. one-minute means give the schedule;
6. after the record, the median of the positive one-minute rates is held;
7. the environment clips the applied rate to its action range (0-2 ug/kg/min).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import evaluation.per_patient_eval as per_patient_eval  # noqa: E402
from environment.dual_drug_env import DualDrugEnv, RFTN_REFERENCE_RATE  # noqa: E402
from models.pharmacokinetics.base import PatientParameters  # noqa: E402

SAMPLES_PER_MINUTE = 6   # 10-s grid


@dataclass(frozen=True)
class RemifentanilSchedule:
    rates: Tuple[float, ...]   # ug/kg/min, one per minute from the start of the anesthetic
    hold: float               # rate after the record

    def rate_at(self, minute: float) -> float:
        index = int(np.floor(minute + 1e-9))
        return self.rates[index] if index < len(self.rates) else self.hold


def schedule_from_case(frame: pd.DataFrame) -> RemifentanilSchedule:
    """The recorded remifentanil schedule of one case (rules 2-6)."""
    ppf = frame["PPF20_RATE"].to_numpy(dtype=float)
    remi = frame["RFTN_RATE"].to_numpy(dtype=float).copy()
    # load_case merges where(RFTN20 > 0, RFTN20, RFTN50), so a recorded RFTN20
    # of 0 without an RFTN50 track becomes NaN. A zero in either raw track is a
    # recorded pause; only samples absent from both tracks are missing.
    raw = [frame[c].to_numpy(dtype=float) for c in ("RFTN20_RATE", "RFTN50_RATE")
           if c in frame.columns]
    if raw:
        remi[np.isnan(remi) & np.any([r == 0 for r in raw], axis=0)] = 0.0
    started = np.flatnonzero((np.nan_to_num(ppf) > 0) | (np.nan_to_num(remi) > 0))
    given = np.flatnonzero(np.nan_to_num(remi) > 0)
    if not len(given):
        raise ValueError("the case has no recorded remifentanil infusion")
    window = pd.Series(remi[started[0]:given[-1] + 1]).ffill().fillna(0.0).to_numpy()
    minutes = [float(window[i:i + SAMPLES_PER_MINUTE].mean())
               for i in range(0, len(window), SAMPLES_PER_MINUTE)]
    positive = [rate for rate in minutes if rate > 0]
    return RemifentanilSchedule(rates=tuple(minutes), hold=float(np.median(positive)))


def build_schedules(caseids: Iterable[int], loader) -> Dict[int, RemifentanilSchedule]:
    schedules = {}
    for caseid in caseids:
        frame = loader.load_case(int(caseid))
        if frame is None:
            raise ValueError(f"case {caseid} could not be loaded")
        schedules[int(caseid)] = schedule_from_case(frame)
    return schedules


def save_schedules(schedules: Mapping[int, RemifentanilSchedule], path: Path,
                   metadata: Dict) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; the schedules are already locked")
    path.write_text(json.dumps({
        "metadata": metadata,
        "schedules": {str(c): {"rates": list(s.rates), "hold": s.hold}
                      for c, s in sorted(schedules.items())}}), encoding="utf-8")


def load_schedules(path: Path) -> Dict[int, RemifentanilSchedule]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(c): RemifentanilSchedule(rates=tuple(s["rates"]), hold=s["hold"])
            for c, s in document["schedules"].items()}


@dataclass
class RecordedRemifentanilPatient(PatientParameters):
    caseid: int = -1
    remifentanil: Optional[RemifentanilSchedule] = None


def recorded_patient_from_record(rec: Mapping, schedules: Mapping[int, RemifentanilSchedule]
                                 ) -> RecordedRemifentanilPatient:
    """patient_from_record plus the case's remifentanil schedule (KeyError if none)."""
    base = per_patient_eval.patient_from_record(dict(rec))
    caseid = int(rec["caseid"])
    return RecordedRemifentanilPatient(age=base.age, weight=base.weight, height=base.height,
                                       gender=base.gender, caseid=caseid,
                                       remifentanil=schedules[caseid])


class RecordedRemifentanilEnv(DualDrugEnv):
    """DualDrugEnv whose remifentanil follows the patient's clinical record."""

    def __init__(self, patient=None, **kwargs):
        if patient is None:
            # Only the sanity-check environment of stage2_online_finetuning has
            # no patient; it gets the training-split median rate.
            schedule = RemifentanilSchedule(rates=(), hold=RFTN_REFERENCE_RATE)
        else:
            schedule = getattr(patient, "remifentanil", None)
            if schedule is None:
                raise ValueError("the patient carries no remifentanil record")
        self.remifentanil_schedule = schedule
        super().__init__(patient=patient, **kwargs)

    def _recorded_rate(self) -> float:
        return self.remifentanil_schedule.rate_at(self.current_step * self.dt)

    def step(self, action):
        action = np.array(action, dtype=float).ravel()
        if self.residual_pid is None:
            action[1] = self._recorded_rate()
        return super().step(action)

    def _residual_action(self, action) -> np.ndarray:
        applied = super()._residual_action(action)
        applied[1] = self._recorded_rate()
        return applied


_ORIGINAL_ENV = per_patient_eval.DualDrugEnv


def install() -> None:
    """Make the per-patient evaluation build the recorded-remifentanil environment."""
    per_patient_eval.DualDrugEnv = RecordedRemifentanilEnv


def restore() -> None:
    per_patient_eval.DualDrugEnv = _ORIGINAL_ENV
