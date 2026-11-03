"""Run evaluation units as GPU subprocesses under the campaign's stall watchdog.

The GPU is shared with another project, and CUDA host-device synchronisation
hung about once per GPU-hour during the campaigns (2026-10-19/20). Each unit
therefore runs in its own process. A unit whose console log stops growing for
``stall_seconds`` is killed, its folder is kept aside as ``*.stalled-N`` and it
is retried (run_revision_campaign._run_once, move_aside). Finished units are
cached, so a stopped run resumes where it left off.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_revision_campaign import _run_once, move_aside, run_phase  # noqa: E402


@dataclass(frozen=True)
class Unit:
    label: str
    command: tuple
    out: Path            # folder holding results.csv and console.log
    methods: frozenset
    n_patients: int


def unit_complete(unit: Unit) -> bool:
    result = unit.out / "results.csv"
    if not result.exists():
        return False
    try:
        frame = pd.read_csv(result)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return False
    return (set(frame.method) == set(unit.methods)
            and all((frame.method == m).sum() == unit.n_patients for m in unit.methods)
            and not frame.duplicated(["method", "patient_id", "seed"]).any()
            and not frame.isna().any().any())


def run_unit(unit: Unit, env: Dict[str, str], stall_seconds: float = 600.0,
             poll_seconds: float = 30.0, max_attempts: int = 4) -> None:
    if unit_complete(unit):
        print(f"{unit.label}: cached", flush=True)
        return
    if unit.out.exists():
        kept = move_aside(unit.out, "interrupted", timeout_seconds=0)
        print(f"{unit.label}: kept an interrupted attempt in {kept.name}", flush=True)
    for attempt in range(1, max_attempts + 1):
        unit.out.mkdir(parents=True, exist_ok=True)
        print(f"{unit.label}: starting (attempt {attempt})", flush=True)
        status = _run_once(unit.command, env, unit.out / "console.log", stall_seconds, poll_seconds)
        if status == "ok" and unit_complete(unit):
            print(f"{unit.label}: complete", flush=True)
            return
        if status == "stalled" and attempt < max_attempts:
            kept = move_aside(unit.out, "stalled")
            print(f"{unit.label}: stalled; kept in {kept.name}, retrying", flush=True)
            continue
        raise RuntimeError(f"{unit.label} {'failed' if status == 'ok' else status}; "
                           f"inspect {unit.out / 'console.log'}")


def gpu_env(device: str = "0") -> Dict[str, str]:
    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
                "CUDA_VISIBLE_DEVICES": device, "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    return env


def run_units(units: List[Unit], jobs: int, stall_seconds: float = 600.0,
              max_attempts: int = 4) -> List[str]:
    env = gpu_env()
    return run_phase(units, lambda u: run_unit(u, env, stall_seconds, 30.0, max_attempts), jobs)


_KEPT_ATTEMPT = re.compile(r"\.(stalled|interrupted)-\d+$")


def unit_results(units_dir: Path) -> List[Path]:
    """results.csv of every unit folder, without attempts kept as *.stalled-N / *.interrupted-N.

    Condition names may contain dots (depolarizing_0.001), so only the kept-attempt
    suffix is excluded (2026-10-27: a dot filter dropped 60 of 110 noise units).
    """
    return sorted(p for p in Path(units_dir).glob("*/results.csv")
                  if not _KEPT_ATTEMPT.search(p.parent.name))


def merge(results: Iterable[Path], target: Path) -> pd.DataFrame:
    """Concatenate result files into ``target``; refuse duplicated rows."""
    frames = [pd.read_csv(p) for p in results if p.exists() and p.stat().st_size > 0]
    merged = pd.concat(frames, ignore_index=True)
    duplicated = merged.duplicated(["method", "patient_id", "seed"])
    if duplicated.any():
        raise RuntimeError(f"{int(duplicated.sum())} duplicated rows while merging into {target}")
    merged.to_csv(target, index=False)
    return merged


def progress_printer(total: int):
    """A progress wrapper for evaluate_cohort that writes one line per patient (watchdog)."""
    def wrap(items):
        for index, item in enumerate(items, 1):
            yield item
            print(f"patient {index}/{total}", flush=True)
    return wrap
