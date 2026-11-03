"""Guards against overwriting earlier outputs and mixing plants (review MED 2, 3)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from run_pilot import ensure_fresh_output  # noqa: E402
from run_revision_campaign import check_pid_protocol  # noqa: E402
from tune_pid_patient_safe import ensure_fresh_output as ensure_fresh_pid_output  # noqa: E402

V2_PID = {"variability": "literature", "observation_interval_seconds": 10,
          "control_interval_seconds": 60, "split_seed": 1}


@pytest.mark.parametrize("name", ["results.csv", "patient_split.json", "run_config.json"])
def test_run_pilot_refuses_a_directory_with_previous_outputs(tmp_path, name):
    (tmp_path / name).write_text("{}")
    with pytest.raises(FileExistsError):
        ensure_fresh_output(tmp_path)


def test_run_pilot_accepts_a_campaign_task_directory(tmp_path):
    (tmp_path / "code_fingerprint.txt").write_text("abc\n")
    (tmp_path / "console.log").write_text("COMMAND: ...\n")
    ensure_fresh_output(tmp_path)


def test_pid_tuning_refuses_to_overwrite_a_protocol(tmp_path):
    (tmp_path / "protocol.json").write_text("{}")
    with pytest.raises(FileExistsError):
        ensure_fresh_pid_output(tmp_path)


@pytest.mark.parametrize("key, value", [("variability", "none"),
                                        ("observation_interval_seconds", 60),
                                        ("split_seed", 0)])
def test_campaign_rejects_pid_tuned_on_another_plant(tmp_path, key, value):
    (tmp_path / "protocol.json").write_text(json.dumps({**V2_PID, key: value}))
    with pytest.raises(RuntimeError):
        check_pid_protocol(tmp_path / "selected_pid.json")


def test_campaign_accepts_pid_tuned_on_sim_v2(tmp_path):
    (tmp_path / "protocol.json").write_text(json.dumps(V2_PID))
    check_pid_protocol(tmp_path / "selected_pid.json")
