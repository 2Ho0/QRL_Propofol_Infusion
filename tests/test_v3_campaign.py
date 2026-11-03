"""Task layout and protocol lock of the v3 campaign runner (D14-D17)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

import run_revision_campaign  # noqa: E402
from run_revision_campaign import BASELINES, Task, code_fingerprint, run_task  # noqa: E402
from run_v3_campaign import (  # noqa: E402
    V3_FINGERPRINT_FILES, V3_PHASES, aux_action_prior, build_v3_tasks,
    campaign_fingerprint_files, v3_protocol)

ROOT = Path("logs/revision_v3/campaign_test")
PID = Path("logs/revision_v2/pid_tuning/selected_pid.json")


def _tasks(phases=tuple(V3_PHASES), prior="0.5,0.5"):
    return build_v3_tasks(ROOT, phases, PID, prior)


def _arg(command, flag):
    return command[command.index(flag) + 1]


def test_110_tasks_with_ten_seeds_per_arm():
    assert len(_tasks()) == 110
    assert len(_tasks(["primary"])) == 40


def test_baselines_run_once_in_the_seed0_quantum_task():
    with_baselines = [t for t in _tasks() if "--skip_baselines" not in t.command]
    assert [t.label for t in with_baselines] == ["primary/quantum_full/seed0"]
    assert _arg(with_baselines[0].command, "--pid_params") == str(PID)
    assert set(BASELINES) <= with_baselines[0].methods


def test_every_task_trains_the_v3_residual_online_only():
    for task in _tasks():
        c = task.command
        assert "--skip_offline" in c
        assert _arg(c, "--online_bc_weight") == "0"
        assert _arg(c, "--online_episodes") == "150"
        assert _arg(c, "--residual_pid") == str(PID)
        assert _arg(c, "--reward_w3") == "0.03"
        assert _arg(c, "--reward_w_smooth") == "0.5"
        assert _arg(c, "--eval_split") == "test"
        assert _arg(c, "--max_test_patients") == "600"
        assert _arg(c, "--variability") == "literature"
        assert _arg(c, "--split_seed") == "1"


@pytest.mark.parametrize("prior, aux_prior", [("0.5,0.5", "0.5,0.5"),
                                              ("0.5,0.075", "0.5,0.429167")])
def test_quantum_and_classical_arms_start_from_the_same_output(prior, aux_prior):
    for task in _tasks(prior=prior):
        expected = aux_prior if task.label.startswith("aux_remi03/") else prior
        assert _arg(task.command, "--classical_action_prior") == expected
        assert _arg(task.command, "--quantum_action_prior") == expected


@pytest.mark.parametrize("prior", ["0.5,0.5", "0.5,0.075"])
def test_the_auxiliary_arms_start_from_the_same_dose_as_the_main_arms(prior):
    """The prior is a fraction of the residual range (independent review, MED)."""
    def initial_residual(fraction, half_range):
        return (fraction - 0.5) * 2.0 * half_range

    main = [float(x) for x in prior.split(",")]
    aux = [float(x) for x in aux_action_prior(prior).split(",")]
    for m, a, main_range, aux_range in zip(main, aux, (4.0, 0.05), (4.0, 0.3)):
        assert initial_residual(a, aux_range) == pytest.approx(
            initial_residual(m, main_range), abs=1e-5)


def test_the_task_fingerprint_covers_the_pid_parameters():
    files = campaign_fingerprint_files(PID.resolve())
    assert "logs/revision_v2/pid_tuning/selected_pid.json" in files
    assert set(V3_FINGERPRINT_FILES) <= set(files)


def test_an_unvalidated_prior_is_refused():
    with pytest.raises(ValueError, match="prior"):
        _tasks(prior="0.5,0.2")


def test_only_the_auxiliary_phase_widens_the_remifentanil_residual():
    for task in _tasks():
        aux = task.label.startswith("aux_remi03/")
        assert _arg(task.command, "--residual_range") == ("4.0,0.3" if aux else "4.0,0.05")
        assert all(m.endswith("_remi03") for m in task.methods) == aux


def test_output_directories_are_unique():
    outs = [t.result.parent for t in _tasks()]
    assert len(outs) == len(set(outs))


def test_the_v3_fingerprint_covers_the_v3_runner():
    assert "experiments/run_v3_campaign.py" in V3_FINGERPRINT_FILES
    assert code_fingerprint(V3_FINGERPRINT_FILES) != code_fingerprint()


def test_run_task_checks_the_fingerprint_of_the_given_files(tmp_path):
    task = Task(label="primary/quantum_full/seed0",
                command=(sys.executable, "-c", "raise SystemExit(3)"),
                result=tmp_path / "seed_0" / "results.csv",
                methods=frozenset({"quantum_full"}))
    fingerprint = code_fingerprint(V3_FINGERPRINT_FILES)
    with pytest.raises(RuntimeError, match="code changed"):
        run_task(task, env=None, by_label={}, fingerprint=fingerprint)
    with pytest.raises(RuntimeError, match="failed"):
        run_task(task, env=None, by_label={}, fingerprint=fingerprint,
                 fingerprint_files=V3_FINGERPRINT_FILES)


def test_the_locked_protocol_refuses_a_different_prior(tmp_path, monkeypatch):
    manifest = json.loads(run_revision_campaign.MANIFEST.read_text())
    monkeypatch.setattr(run_revision_campaign, "file_sha256",
                        lambda path: manifest["dataset_sha256"])
    path = tmp_path / "locked_protocol.json"
    run_revision_campaign.write_locked_protocol(path, v3_protocol(PID, "0.5,0.5"), "abc", False)
    with pytest.raises(RuntimeError, match="locked protocol changed"):
        run_revision_campaign.write_locked_protocol(
            path, v3_protocol(PID, "0.5,0.075"), "abc", False)


def test_tasks_run_seed_by_seed_so_that_pairs_finish_together():
    labels = [t.label for t in _tasks(["primary"])]
    assert labels == [f"primary/{arm}/seed{seed}"
                      for seed in range(10) for arm in V3_PHASES["primary"]]


def test_every_task_uses_the_validated_actor_learning_rates():
    """Per family, chosen on the validation subjects."""
    for task in _tasks():
        assert _arg(task.command, "--classical_actor_lr") == "0.0001"
        assert _arg(task.command, "--quantum_actor_lr") == "0.001"


def test_the_protocol_records_the_actor_learning_rates():
    rates = v3_protocol(PID, "0.5,0.5")["training"]["actor_learning_rates"]
    assert rates["mlp"] == 0.0001 and rates["quantum_family"] == 0.001
