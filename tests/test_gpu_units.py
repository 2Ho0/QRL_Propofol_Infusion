"""GPU unit runner of the evaluation studies (experiments/gpu_units.py)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import gpu_units  # noqa: E402


def _results(folder: Path, method: str, patients, seed: int = 0) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"method": method, "patient_id": list(patients), "seed": seed,
                          "mdape": 4.0})
    path = folder / "results.csv"
    frame.to_csv(path, index=False)
    return path


def _unit(folder: Path, method="qrl", n=3):
    return gpu_units.Unit("u", (sys.executable, "-c", "pass"), folder, frozenset({method}), n)


def test_unit_is_complete_only_with_every_patient(tmp_path):
    folder = tmp_path / "unit"
    assert not gpu_units.unit_complete(_unit(folder))
    _results(folder, "qrl", [1, 2])
    assert not gpu_units.unit_complete(_unit(folder))
    _results(folder, "qrl", [1, 2, 3])
    assert gpu_units.unit_complete(_unit(folder))
    _results(folder, "other", [1, 2, 3])
    assert not gpu_units.unit_complete(_unit(folder))


def test_finished_unit_is_cached_without_running(tmp_path, capsys):
    folder = tmp_path / "unit"
    _results(folder, "qrl", [1, 2, 3])
    gpu_units.run_unit(_unit(folder), env={}, stall_seconds=5, poll_seconds=0.1)
    assert "cached" in capsys.readouterr().out


def test_unit_that_writes_nothing_fails(tmp_path):
    with pytest.raises(RuntimeError, match="failed"):
        gpu_units.run_unit(_unit(tmp_path / "unit"), env=dict(gpu_units.gpu_env()),
                           stall_seconds=60, poll_seconds=0.1, max_attempts=1)


def test_merge_combines_units_and_refuses_duplicates(tmp_path):
    a = _results(tmp_path / "a", "qrl", [1, 2], seed=0)
    b = _results(tmp_path / "b", "qrl", [1, 2], seed=1)
    merged = gpu_units.merge([a, b, tmp_path / "missing.csv"], tmp_path / "all.csv")
    assert len(merged) == 4 and (tmp_path / "all.csv").exists()
    with pytest.raises(RuntimeError, match="duplicated"):
        gpu_units.merge([a, a], tmp_path / "dup.csv")


def test_unit_results_skip_kept_attempts_but_not_dotted_names(tmp_path):
    keep = [_results(tmp_path / "depolarizing_0.001_seed0", "depolarizing_0.001", [1]),
            _results(tmp_path / "qrl_seed1", "qrl", [1])]
    _results(tmp_path / "qrl_seed1.stalled-1", "qrl", [1])
    _results(tmp_path / "readout_0.01_seed2.interrupted-2", "readout_0.01", [1])
    assert gpu_units.unit_results(tmp_path) == sorted(keep)


def test_progress_printer_writes_one_line_per_patient(capsys):
    items = list(gpu_units.progress_printer(3)(["x", "y", "z"]))
    assert items == ["x", "y", "z"]
    assert capsys.readouterr().out.splitlines() == ["patient 1/3", "patient 2/3", "patient 3/3"]
