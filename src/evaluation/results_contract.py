"""Schema for per-patient evaluation results.

One row per (method, patient_id, seed) rollout. All aggregation - averaging
seeds within a patient, then comparing methods across patients - happens in the
analysis step, never in the training scripts, so the paper's tables are a
deterministic function of this file.

This replaces the previous output, which was 1000 rows keyed by episode with no
patient id and no seed, and therefore could not support the per-patient
statistics the protocol specifies.
"""
import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import List, Union


@dataclass
class PatientResult:
    """Metrics for one method on one patient under one seed."""

    method: str
    patient_id: int
    seed: int
    mdpe: float          # median performance error, %
    mdape: float         # median absolute performance error, %
    wobble: float        # intra-patient variability, %
    divergence: float    # drift, %/min
    tit_45_55: float     # time in BIS [45, 55], %
    tit_40_60: float     # time in BIS [40, 60], %
    n_bis_lt30: int      # steps with BIS < 30
    n_bis_gt70: int      # steps with BIS > 70
    mean_ppf: float      # mean propofol, mg/kg/h
    mean_remi: float     # mean remifentanil, ug/kg/min
    # The same Varvel metrics on the noise-free BIS (sim v2). The columns
    # above are computed on the measured BIS, which carries N(0, 2^2) noise.
    mdpe_true: float = float("nan")
    mdape_true: float = float("nan")
    wobble_true: float = float("nan")
    divergence_true: float = float("nan")


RESULT_COLUMNS: List[str] = [f.name for f in fields(PatientResult)]


def write_results(rows: List[PatientResult], path: Union[str, Path]) -> None:
    """Write results to CSV, rejecting duplicate (method, patient_id, seed).

    Args:
        rows: Results to write.
        path: Destination CSV; parent directories are created.

    Raises:
        ValueError: If two rows share the same method, patient and seed. A
            duplicate means a bug upstream, and silently averaging it later
            would hide the error.
    """
    keys = [(r.method, r.patient_id, r.seed) for r in rows]
    if len(set(keys)) != len(keys):
        seen, dupes = set(), set()
        for k in keys:
            (dupes if k in seen else seen).add(k)
        raise ValueError(f"duplicate (method, patient_id, seed): {sorted(dupes)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def append_results(rows: List[PatientResult], path: Union[str, Path]) -> None:
    """Append rows to an existing contract file, writing a header if new.

    Used when arms are evaluated one at a time so a long run accumulates
    results incrementally rather than only at the end.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # An empty file has no header yet, so treat it as new.
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
        if header and header != RESULT_COLUMNS:
            raise ValueError(
                f"{path} has a different header than the current contract; "
                "write v2 results to a new file instead of appending")
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if not exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
