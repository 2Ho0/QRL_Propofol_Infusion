"""Run the sim v2 revision campaign for JBHI-05743-2026 (10 seeds, all arms).

Plan: docs/superpowers/plans/2026-10-19-jbhi-revision-rerun.md. Each task
trains and evaluates one arm for one seed through run_pilot.py on the locked
2,100/300/600 subject split and writes its own results.csv, so tasks run in
parallel and resume individually. Phases run in the order given; the
offline-only ablation reuses the stage-1 checkpoint of the primary QRL run of
the same seed.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from environment.variability import LITERATURE_SPEC  # noqa: E402

# Rates per kg of recorded body weight (rebuilt 2026-10-20, same 3,000 cases
# and split; docs/experiments/2026-10-20_0215_dataset_weight_fix.md).
DATA = ROOT / "data/offline_dataset/vitaldb_obs10_ctrl60_t60_subject3000_pkvalid_wtfix.pkl"
MANIFEST = ROOT / "logs/revision_v2/dataset_wtfix/dataset_manifest.json"
DEFAULT_OUT = ROOT / "logs/revision_v2/campaign"
DEFAULT_PID = ROOT / "logs/revision_v2/pid_tuning/selected_pid.json"
SEEDS = tuple(range(10))
N_TEST = 600
BASELINES = ("pid", "pid_tuned", "tci", "guideline")

# Phase -> arms, in execution order.
PHASES: Dict[str, tuple] = {
    "primary": ("quantum_full", "classical_large", "classical_matched"),
    "core": ("quantum_disabled", "quantum_noskip", "quantum_noentangle",
             "quantum_2layer"),
    "reviewer": ("quantum_critic", "ansatz_realamplitudes",
                 "ansatz_efficientsu2", "ansatz_twolocal", "fourier_surrogate"),
    "td3bc": ("quantum_full", "classical_large"),
    "ablation": ("offline_only", "online_only"),
    "exploratory": ("quantum_2qubit", "quantum_4qubit", "quantum_8layer",
                    "quantum_noencoder"),
}

FINGERPRINT_FILES = (
    "config/hyperparameters.yaml",
    "experiments/run_pilot.py",
    "experiments/train_hybrid.py",
    "experiments/run_revision_campaign.py",
)


@dataclass(frozen=True)
class Task:
    label: str
    command: tuple
    result: Path
    methods: frozenset
    depends_on: Optional[str] = None


def _set(command: List[str], flag: str, value: Optional[str] = None) -> List[str]:
    """Set ``flag`` to ``value`` (or add a bare flag) without duplicating it."""
    command = list(command)
    if flag in command:
        if value is not None:
            command[command.index(flag) + 1] = value
    else:
        command += [flag] if value is None else [flag, value]
    return command


def base_command(seed: int, out: Path) -> List[str]:
    return [
        sys.executable, str(ROOT / "experiments/run_pilot.py"),
        "--data_path", str(DATA),
        "--clean_temporal_protocol",
        "--n_cases", "3000",
        "--observation_interval_seconds", "10",
        "--control_interval_seconds", "60",
        "--sequence_length", "60",
        "--offline_epochs", "15",
        "--online_episodes", "30",
        "--warmup_episodes", "5",
        "--online_eval_interval", "10",
        "--n_online_eval_patients", "30",
        "--batch_size", "128",
        "--num_workers", "0",
        "--bc_weight", "1.0",
        "--online_bc_weight", "1.0",
        "--online_rl_scale", "0.05",
        "--remi_bc_weight", "4.0",
        "--smoothing_beta", "0.0",
        "--classical_actor_lr", "0.0001",
        "--classical_dropout", "0.0",
        "--classical_action_prior", "0.5,0.075",
        "--seed", str(seed),
        "--split_seed", "1",
        "--evaluation_seed", "0",
        "--online_validation_seed", "1000",
        "--eval_split", "test",
        "--max_test_patients", str(N_TEST),
        "--variability", "literature",
        "--online_patients", "training",
        "--skip_baselines",
        "--out", str(out),
    ]


def build_tasks(root: Path, phases: Iterable[str], pid_params: Path) -> List[Task]:
    tasks: List[Task] = []
    for phase in phases:
        for arm in PHASES[phase]:
            for seed in SEEDS:
                out = root / phase / arm / f"seed_{seed}"
                command = base_command(seed, out)
                methods = {arm}
                depends_on = None
                if phase == "primary" and arm == "quantum_full" and seed == 0:
                    command.remove("--skip_baselines")
                    command = _set(command, "--pid_params", str(pid_params))
                    methods |= set(BASELINES)
                if phase == "td3bc":
                    command = _set(command, "--online_rl_scale", "2.5")
                    command = _set(command, "--remi_bc_weight", "1.0")
                    command = _set(command, "--method_suffix", "_td3bc")
                    methods = {f"{arm}_td3bc"}
                    trained_arm = arm
                elif phase == "ablation":
                    trained_arm = "quantum_full"
                    command = _set(command, "--method_suffix", f"_{arm}")
                    methods = {f"quantum_full_{arm}"}
                    if arm == "offline_only":
                        checkpoint = (root / "primary" / "quantum_full" / f"seed_{seed}"
                                      / "quantum_full" / "stage1" / "best_val.pt")
                        command = _set(command, "--online_episodes", "0")
                        command = _set(command, "--resume_stage1", str(checkpoint))
                        depends_on = f"primary/quantum_full/seed{seed}"
                    else:
                        command = _set(command, "--skip_offline")
                        command = _set(command, "--online_bc_weight", "0")
                else:
                    trained_arm = arm
                command = _set(command, "--arms", trained_arm)
                tasks.append(Task(
                    label=f"{phase}/{arm}/seed{seed}", command=tuple(command),
                    result=out / "results.csv", methods=frozenset(methods),
                    depends_on=depends_on))
    return tasks


def complete(task: Task) -> bool:
    if not task.result.exists():
        return False
    try:
        frame = pd.read_csv(task.result)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        # A write that was cut off; run_task keeps the folder and reruns.
        return False
    return (set(frame.method) == set(task.methods)
            and all((frame.method == m).sum() == N_TEST for m in task.methods)
            and not frame.duplicated(["method", "patient_id", "seed"]).any()
            and not frame.isna().any().any())


def code_fingerprint(files: Iterable[str] = FINGERPRINT_FILES) -> str:
    paths = sorted((ROOT / "src").rglob("*.py")) + [ROOT / p for p in files]
    digest = hashlib.sha256()
    for path in paths:
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _kill_tree(process: subprocess.Popen) -> None:
    """Kill a task and its children (the venv python.exe is a launcher)."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       capture_output=True, check=False)
    else:
        process.kill()
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        pass


def _run_once(command, env, log: Path, stall_seconds: float,
              poll_seconds: float) -> str:
    """Run ``command`` once; return 'ok', 'failed' or 'stalled'.

    A command whose log stops growing for ``stall_seconds`` is killed. On
    2026-10-19 two GPU pilots hung (95 min and 2.7 h) inside CUDA
    host-device synchronisation while the GPU was shared with another
    project.
    """
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n===== attempt {datetime.now(timezone.utc).isoformat()} =====\n")
        stream.write("COMMAND: " + subprocess.list2cmdline(list(command)) + "\n\n")
        stream.flush()
        process = subprocess.Popen(list(command), cwd=ROOT, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT)
        last_size, last_change = -1, time.time()
        while True:
            try:
                code = process.wait(timeout=poll_seconds)
                return "ok" if code == 0 else "failed"
            except subprocess.TimeoutExpired:
                size = log.stat().st_size
                if size != last_size:
                    last_size, last_change = size, time.time()
                elif time.time() - last_change > stall_seconds:
                    _kill_tree(process)
                    stream.write(f"\n===== killed: no output for {stall_seconds:.0f} s =====\n")
                    return "stalled"


def move_aside(out: Path, tag: str, retry_seconds: float = 10.0,
               timeout_seconds: float = 600.0) -> Path:
    """Rename ``out`` to the first free ``<out>.<tag>-N`` and return that path.

    Windows refuses to rename a folder while any process holds a file in it
    open. A task killed inside a CUDA call can take minutes to exit and keeps
    console.log (its stdout) open until then, so the rename is retried
    (validation pilot, 2026-10-20). A folder that stays locked fails the task
    with RuntimeError instead of stopping the runner.
    """
    deadline = time.time() + timeout_seconds
    while True:
        n = 1
        while out.with_name(f"{out.name}.{tag}-{n}").exists():
            n += 1
        kept = out.with_name(f"{out.name}.{tag}-{n}")
        try:
            out.rename(kept)
            return kept
        except OSError as error:
            if time.time() >= deadline:
                raise RuntimeError(f"could not move {out} aside: {error}") from error
            time.sleep(retry_seconds)


def run_task(task: Task, env: Dict[str, str], by_label: Dict[str, Task],
             fingerprint: str, stall_seconds: float = 15 * 60,
             poll_seconds: float = 30.0, max_attempts: int = 3,
             fingerprint_files: Iterable[str] = FINGERPRINT_FILES) -> None:
    if complete(task):
        print(f"{task.label}: cached", flush=True)
        return
    if task.depends_on and not complete(by_label[task.depends_on]):
        raise RuntimeError(f"{task.label} needs {task.depends_on} first")
    # Each task imports the code afresh, so check it has not changed since
    # the protocol was locked (e.g. an edit to src/ during the campaign).
    if code_fingerprint(fingerprint_files) != fingerprint:
        raise RuntimeError(
            f"{task.label}: code changed since the campaign started; "
            "restart the runner after reviewing the change")
    out = task.result.parent
    if out.exists():
        # An earlier runner stopped inside this task; keep what it wrote. Its
        # task process may still be alive (it holds console.log as stdout), so
        # never wait for the folder: a result about to complete would be moved
        # aside and recomputed (independent review, 2026-10-20).
        try:
            kept = move_aside(out, "interrupted", timeout_seconds=0)
        except RuntimeError as error:
            raise RuntimeError(
                f"{task.label}: {out} is in use; is a task process of an earlier "
                "runner still running?") from error
        print(f"{task.label}: kept an interrupted attempt in {kept.name}", flush=True)
    started = time.time()
    for attempt in range(1, max_attempts + 1):
        out.mkdir(parents=True, exist_ok=True)
        (out / "code_fingerprint.txt").write_text(fingerprint + "\n", encoding="utf-8")
        log = out / "console.log"
        print(f"{task.label}: starting (attempt {attempt})", flush=True)
        status = _run_once(task.command, env, log, stall_seconds, poll_seconds)
        minutes = (time.time() - started) / 60
        if status == "ok" and complete(task):
            print(f"{task.label}: complete in {minutes:.1f} min", flush=True)
            return
        if status == "stalled" and attempt < max_attempts:
            # Keep the stalled attempt for inspection and start clean.
            kept = move_aside(out, "stalled")
            print(f"{task.label}: stalled; kept in {kept.name}, retrying", flush=True)
            continue
        status = "failed" if status == "ok" else status
        raise RuntimeError(f"{task.label} {status} after {minutes:.1f} min; inspect {log}")


def run_phase(tasks: List, run, jobs: int) -> List[str]:
    """Run ``run(task)`` for every task with ``jobs`` workers; return the failures.

    Any exception fails only its own task, so the other tasks of the phase
    still run.
    """
    failures = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(run, task) for task in tasks]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:  # noqa: BLE001 - reported, phase continues
                label = getattr(tasks[futures.index(future)], "label",
                                tasks[futures.index(future)])
                failures.append(f"{label}: {type(error).__name__}: {error}")
                print(f"FAILED: {failures[-1]}", flush=True)
                print("".join(traceback.format_exception(error)), flush=True)
    return failures


def acquire_runner_lock(root: Path):
    """Hold ``root/runner.lock`` for this runner's lifetime; refuse a second runner.

    The operating system releases the lock when the runner exits, however it
    exits. Return the open handle; the lock lasts while it stays open.
    """
    handle = open(root / "runner.lock", "a+")
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        raise RuntimeError(f"another campaign runner holds {root / 'runner.lock'}") from error
    return handle


def check_pid_protocol(pid_params: Path) -> None:
    """The tuned PID must come from the same plant and protocol as the arms."""
    protocol = json.loads((Path(pid_params).parent / "protocol.json").read_text())
    expected = {"variability": "literature", "observation_interval_seconds": 10,
                "control_interval_seconds": 60, "split_seed": 1}
    wrong = {k: protocol.get(k) for k, v in expected.items() if protocol.get(k) != v}
    if wrong:
        raise RuntimeError(f"PID was tuned under a different protocol: {wrong}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lock_protocol(root: Path, phases: List[str], pid_params: Path,
                  fingerprint: str, allow_code_change: bool) -> None:
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("split_seed") != 1:
        raise RuntimeError("dataset manifest must lock split seed 1")
    check_pid_protocol(pid_params)
    protocol = {
        "status": "locked_before_campaign",
        "plan": "docs/superpowers/plans/2026-10-19-jbhi-revision-rerun.md",
        "dataset_sha256": manifest["dataset_sha256"],
        "split": {"train": 2100, "validation": 300, "test": 600, "split_seed": 1},
        "seeds": list(SEEDS),
        "phases": {name: list(PHASES[name]) for name in PHASES},
        "variability": dataclasses.asdict(LITERATURE_SPEC),
        "pid_params_path": str(pid_params),
        "pid_params_sha256": hashlib.sha256(pid_params.read_bytes()).hexdigest(),
        "primary_analysis": {
            "metric": "mean test MDAPE per training run (measured BIS)",
            "contrasts": ["quantum_full vs classical_matched",
                          "quantum_full vs quantum_disabled"],
            "statistics": "per-run values, mean difference with 95% bootstrap-"
                          "over-runs CI, Cohen's d, TOST with margin "
                          "+/-1.0 MDAPE percentage point (alpha 0.05)",
        },
        "test_outcomes_used_for_selection": False,
    }
    write_locked_protocol(root / "locked_protocol.json", protocol, fingerprint,
                          allow_code_change)


def write_locked_protocol(path: Path, protocol: Dict, fingerprint: str,
                          allow_code_change: bool) -> None:
    """Write ``protocol`` once; later runs may change only the recorded code fingerprint."""
    protocol = dict(protocol, code_fingerprint=fingerprint)
    if path.exists():
        existing = json.loads(path.read_text())
        changed = [k for k in protocol if k != "code_fingerprint"
                   and existing.get(k) != protocol[k]]
        if changed:
            raise RuntimeError(f"locked protocol changed at {changed}")
        if existing.get("code_fingerprint") != fingerprint:
            if not allow_code_change:
                raise RuntimeError(
                    "code changed since the protocol was locked; rerun with "
                    "--allow_code_change only for changes that cannot affect "
                    "results, and the change is recorded")
            existing.setdefault("code_amendments", []).append({
                "utc": datetime.now(timezone.utc).isoformat(),
                "from": existing["code_fingerprint"], "to": fingerprint})
            existing["code_fingerprint"] = fingerprint
            path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        return
    manifest = json.loads(MANIFEST.read_text())
    if file_sha256(DATA) != manifest["dataset_sha256"]:
        raise RuntimeError("dataset file does not match its manifest hash")
    protocol["locked_utc"] = datetime.now(timezone.utc).isoformat()
    protocol["git_head"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
        text=True, check=True).stdout.strip()
    path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phases", default=",".join(PHASES),
                        help=f"comma-separated subset of {list(PHASES)}")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pid_params", type=Path, default=DEFAULT_PID)
    parser.add_argument("--cuda_visible_devices", default="0")
    parser.add_argument("--allow_code_change", action="store_true")
    parser.add_argument("--stall_minutes", type=float, default=15.0,
                        help="kill and retry a task whose log stops growing this long")
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    unknown = set(phases) - set(PHASES)
    if unknown:
        raise ValueError(f"unknown phases: {sorted(unknown)}")
    root = args.out.resolve()
    tasks = build_tasks(root, phases, args.pid_params.resolve())
    if args.dry_run:
        for task in tasks:
            print(task.label, subprocess.list2cmdline(task.command))
        return
    if not DATA.exists() or not args.pid_params.exists():
        raise FileNotFoundError("dataset or tuned PID parameters missing")
    root.mkdir(parents=True, exist_ok=True)
    runner_lock = acquire_runner_lock(root)  # noqa: F841 - held until exit
    fingerprint = code_fingerprint()
    lock_protocol(root, phases, args.pid_params.resolve(), fingerprint,
                  args.allow_code_change)

    env = os.environ.copy()
    env.update({
        "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
        "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
    })
    by_label = {task.label: task for task in build_tasks(root, list(PHASES),
                                                         args.pid_params.resolve())}
    for phase in phases:
        phase_tasks = [t for t in tasks if t.label.startswith(f"{phase}/")]
        print(f"=== phase {phase}: {len(phase_tasks)} tasks ===", flush=True)
        failures = run_phase(
            phase_tasks,
            lambda t: run_task(t, env, by_label, fingerprint,
                               args.stall_minutes * 60, 30.0, args.max_attempts),
            args.jobs)
        if failures:
            for failure in failures:
                print(failure, flush=True)
            raise SystemExit(f"{len(failures)} task(s) failed in phase {phase}")


if __name__ == "__main__":
    main()
