"""Task layout of the campaign runner (experiments/run_revision_campaign.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from run_revision_campaign import PHASES, SEEDS, build_tasks  # noqa: E402

ROOT = Path("logs/revision_v2/campaign_test")
PID = Path("logs/revision_v2/pid_tuning/selected_pid.json")


def _tasks(phases):
    return build_tasks(ROOT, phases, PID)


def _arg(command, flag):
    return command[command.index(flag) + 1]


def test_ten_seeds_per_primary_arm():
    tasks = _tasks(["primary"])
    assert SEEDS == tuple(range(10))
    assert len(tasks) == 3 * 10
    assert {t.methods for t in tasks if t.label.startswith("primary/quantum_full")} == {
        frozenset({"quantum_full"}), frozenset({"quantum_full", "pid", "pid_tuned", "tci", "guideline"})}


def test_baselines_run_exactly_once():
    tasks = _tasks(list(PHASES))
    with_baselines = [t for t in tasks if "--skip_baselines" not in t.command]
    assert len(with_baselines) == 1
    assert _arg(with_baselines[0].command, "--pid_params") == str(PID)


def test_every_task_uses_the_v2_protocol_on_the_test_split():
    for task in _tasks(list(PHASES)):
        c = task.command
        assert _arg(c, "--variability") == "literature"
        assert _arg(c, "--online_patients") == "training"
        assert _arg(c, "--smoothing_beta") == "0.0"
        assert _arg(c, "--eval_split") == "test"
        assert _arg(c, "--max_test_patients") == "600"
        assert _arg(c, "--split_seed") == "1"


def test_td3bc_tasks_use_the_standard_settings_and_a_suffix():
    tasks = _tasks(["td3bc"])
    assert len(tasks) == 2 * 10
    for task in tasks:
        assert _arg(task.command, "--online_rl_scale") == "2.5"
        assert _arg(task.command, "--remi_bc_weight") == "1.0"
        assert _arg(task.command, "--method_suffix") == "_td3bc"
        assert all(m.endswith("_td3bc") for m in task.methods)


def test_offline_only_resumes_the_primary_checkpoint():
    for task in _tasks(["ablation"]):
        if "offline_only" in task.label:
            seed = _arg(task.command, "--seed")
            assert _arg(task.command, "--online_episodes") == "0"
            assert _arg(task.command, "--resume_stage1").endswith(
                str(Path(f"primary/quantum_full/seed_{seed}/quantum_full/stage1/best_val.pt")))
            assert task.depends_on == f"primary/quantum_full/seed{seed}"


def test_output_directories_are_unique():
    outs = [t.result.parent for t in _tasks(list(PHASES))]
    assert len(outs) == len(set(outs))


def test_a_task_refuses_to_start_after_the_code_changed(tmp_path):
    import pytest
    from run_revision_campaign import Task, run_task
    task = Task(label="primary/quantum_full/seed0", command=("python", "-c", "pass"),
                result=tmp_path / "seed_0" / "results.csv",
                methods=frozenset({"quantum_full"}))
    with pytest.raises(RuntimeError, match="code changed"):
        run_task(task, env={}, by_label={}, fingerprint="not-the-current-code")
    assert not (tmp_path / "seed_0").exists()


def _fake_task(tmp_path, code):
    from run_revision_campaign import Task
    return Task(label="primary/quantum_full/seed0",
                command=(sys.executable, "-c", code),
                result=tmp_path / "seed_0" / "results.csv",
                methods=frozenset({"quantum_full"}))


def test_a_silent_task_is_killed_kept_and_retried(tmp_path):
    import pytest
    from run_revision_campaign import code_fingerprint, run_task
    task = _fake_task(tmp_path, "import time; time.sleep(60)")
    with pytest.raises(RuntimeError, match="stalled"):
        run_task(task, env=None, by_label={}, fingerprint=code_fingerprint(),
                 stall_seconds=2, poll_seconds=0.5, max_attempts=2)
    assert (tmp_path / "seed_0.stalled-1" / "console.log").exists()
    assert (tmp_path / "seed_0" / "console.log").exists()


def test_a_crashing_task_is_not_retried(tmp_path):
    import pytest
    from run_revision_campaign import code_fingerprint, run_task
    task = _fake_task(tmp_path, "raise SystemExit(1)")
    with pytest.raises(RuntimeError, match="failed"):
        run_task(task, env=None, by_label={}, fingerprint=code_fingerprint(),
                 stall_seconds=30, poll_seconds=0.5, max_attempts=3)
    assert not list(tmp_path.glob("seed_0.stalled-*"))


def test_a_stalled_attempt_is_kept_once_its_log_is_released(tmp_path):
    # A process killed inside a CUDA call can take a while to exit and holds
    # console.log (its stdout) until then; Windows refuses to rename the
    # folder meanwhile (validation pilot, 2026-10-20).
    import threading
    from run_revision_campaign import move_aside
    out = tmp_path / "seed_0"
    out.mkdir()
    log = open(out / "console.log", "w")
    threading.Timer(1.0, log.close).start()
    kept = move_aside(out, "stalled", retry_seconds=0.2, timeout_seconds=20)
    assert kept == tmp_path / "seed_0.stalled-1"
    assert (kept / "console.log").exists()
    assert not out.exists()


def test_moving_aside_picks_the_next_free_name(tmp_path):
    from run_revision_campaign import move_aside
    (tmp_path / "seed_0.stalled-1").mkdir()
    out = tmp_path / "seed_0"
    out.mkdir()
    assert move_aside(out, "stalled") == tmp_path / "seed_0.stalled-2"


def test_a_folder_that_stays_locked_fails_the_task_not_the_runner(tmp_path):
    import os
    import pytest
    from run_revision_campaign import move_aside
    if os.name != "nt":
        pytest.skip("only Windows refuses to rename a folder with an open file")
    out = tmp_path / "seed_0"
    out.mkdir()
    with open(out / "console.log", "w"):
        with pytest.raises(RuntimeError, match="could not move"):
            move_aside(out, "stalled", retry_seconds=0.1, timeout_seconds=0.5)
    assert out.exists()


def test_an_interrupted_attempt_is_moved_aside_before_the_task_reruns(tmp_path):
    import pytest
    from run_revision_campaign import code_fingerprint, run_task
    out = tmp_path / "seed_0"
    out.mkdir()
    (out / "run_config.json").write_text("{}")
    task = _fake_task(tmp_path, "pass")   # exits 0 without results.csv
    with pytest.raises(RuntimeError, match="failed"):
        run_task(task, env=None, by_label={}, fingerprint=code_fingerprint(),
                 stall_seconds=30, poll_seconds=0.5, max_attempts=1)
    assert (tmp_path / "seed_0.interrupted-1" / "run_config.json").exists()
    assert not (out / "run_config.json").exists()


def test_an_unexpected_error_in_one_task_does_not_stop_the_phase():
    from run_revision_campaign import run_phase
    done = []

    def run(task):
        if task == "bad":
            raise OSError("disk full")
        done.append(task)

    failures = run_phase(["a", "bad", "b"], run, jobs=2)
    assert sorted(done) == ["a", "b"]
    assert len(failures) == 1 and "disk full" in failures[0]


def test_an_unreadable_result_counts_as_incomplete(tmp_path):
    from run_revision_campaign import complete
    task = _fake_task(tmp_path, "pass")
    (tmp_path / "seed_0").mkdir()
    task.result.write_text("")   # a results.csv cut off before its header
    assert complete(task) is False


def test_a_completed_task_is_neither_moved_nor_rerun(tmp_path):
    import pandas as pd
    from run_revision_campaign import N_TEST, code_fingerprint, run_task
    task = _fake_task(tmp_path, "raise SystemExit('must not run')")
    (tmp_path / "seed_0").mkdir()
    pd.DataFrame({"method": ["quantum_full"] * N_TEST, "patient_id": range(N_TEST),
                  "seed": [0] * N_TEST, "mdape": [5.0] * N_TEST}).to_csv(task.result, index=False)
    run_task(task, env=None, by_label={}, fingerprint=code_fingerprint())
    assert not list(tmp_path.glob("seed_0.*"))
    assert task.result.exists()


def test_a_folder_still_in_use_is_left_alone_and_fails_fast(tmp_path):
    # A task process of an earlier runner may still be finishing; waiting for
    # its log to be released and then moving the folder would discard a
    # result that was about to complete (independent review, 2026-10-20).
    import os
    import time
    import pytest
    from run_revision_campaign import code_fingerprint, run_task
    if os.name != "nt":
        pytest.skip("only Windows refuses to rename a folder with an open file")
    out = tmp_path / "seed_0"
    out.mkdir()
    task = _fake_task(tmp_path, "pass")
    with open(out / "console.log", "a"):
        started = time.time()
        with pytest.raises(RuntimeError, match="in use"):
            run_task(task, env=None, by_label={}, fingerprint=code_fingerprint())
        assert time.time() - started < 30
    assert out.exists()
    assert not list(tmp_path.glob("seed_0.interrupted-*"))


def test_a_second_runner_on_the_same_root_is_refused(tmp_path):
    import pytest
    from run_revision_campaign import acquire_runner_lock
    first = acquire_runner_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="another campaign runner"):
            acquire_runner_lock(tmp_path)
    finally:
        first.close()
    acquire_runner_lock(tmp_path).close()   # free again once the first runner is gone


def test_a_failed_task_is_reported_with_its_label():
    from run_revision_campaign import Task, run_phase
    task = Task(label="primary/quantum_full/seed3", command=(),
                result=Path("unused") / "results.csv", methods=frozenset())

    def run(_task):
        raise OSError("disk full")

    assert run_phase([task], run, jobs=1) == ["primary/quantum_full/seed3: OSError: disk full"]
