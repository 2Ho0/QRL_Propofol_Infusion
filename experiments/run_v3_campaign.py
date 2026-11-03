"""Run the v3 campaign for JBHI-05743-2026: quantum residual RL on the tuned PID.

Plan: docs/superpowers/plans/2026-10-21-v3-residual-qrl.md (D14-D17). Every
learned arm trains online only (TD3, 150 episodes, no imitation) as a bounded
residual on the tuned PID and is evaluated on the 600 test patients of the
locked split. Phases run in order: primary arms, the auxiliary wider
remifentanil residual, then the reviewer-requested arms. The baselines (PID,
tuned PID, TCI, guideline) run once, in the plain environment, inside the
seed-0 quantum_full task.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

from environment.variability import LITERATURE_SPEC  # noqa: E402
from run_revision_campaign import (  # noqa: E402
    BASELINES, DEFAULT_PID, FINGERPRINT_FILES, MANIFEST, SEEDS, Task, _set,
    acquire_runner_lock, base_command, check_pid_protocol, code_fingerprint,
    run_phase, run_task, write_locked_protocol)

DEFAULT_OUT = ROOT / "logs/revision_v3/campaign"
RESIDUAL_MAIN = "4.0,0.05"
RESIDUAL_AUX = "4.0,0.3"
# D17: the validation gate (plan Task 8) picks one; every learned arm uses it.
ACTION_PRIORS = ("0.5,0.5", "0.5,0.075")
# Actor learning rates per family, chosen on validation patients
# (docs/experiments/2026-10-21_1600_v3_actor_lr_check.md).
MLP_ACTOR_LR = "0.0001"
QUANTUM_ACTOR_LR = "0.001"

V3_PHASES: Dict[str, tuple] = {
    "primary": ("quantum_full", "classical_matched", "quantum_disabled",
                "classical_large"),
    "aux_remi03": ("quantum_full", "classical_matched"),
    "reviewer": ("ansatz_realamplitudes", "ansatz_efficientsu2", "ansatz_twolocal",
                 "quantum_critic", "fourier_surrogate"),
}
V3_FINGERPRINT_FILES = FINGERPRINT_FILES + ("experiments/run_v3_campaign.py",)


def campaign_fingerprint_files(pid_params: Path) -> tuple:
    """Files checked before every task: the code and the tuned PID it builds on."""
    return V3_FINGERPRINT_FILES + (Path(pid_params).resolve().relative_to(ROOT).as_posix(),)


def aux_action_prior(action_prior: str) -> str:
    """Prior of the wider-remifentanil arms that starts them at the main arms' dose (D17).

    The prior is a fraction of the residual range, so the same fraction would
    start the auxiliary arms six times further from the PID (independent
    review, 2026-10-21).
    """
    main = [float(x) for x in RESIDUAL_MAIN.split(",")]
    aux = [float(x) for x in RESIDUAL_AUX.split(",")]
    prior = [float(x) for x in action_prior.split(",")]
    return ",".join(f"{0.5 + (p - 0.5) * m / a:.6g}" for p, m, a in zip(prior, main, aux))


def v3_command(seed: int, out: Path, pid_params: Path, action_prior: str,
               residual_range: str = RESIDUAL_MAIN) -> List[str]:
    """The v2 protocol command with the v3 method (D14-D17)."""
    command = base_command(seed, out)
    for flag, value in (("--online_episodes", "150"), ("--online_bc_weight", "0"),
                        ("--online_rl_scale", "0.5"),
                        ("--classical_action_prior", action_prior),
                        ("--quantum_action_prior", action_prior),
                        ("--classical_actor_lr", MLP_ACTOR_LR),
                        ("--quantum_actor_lr", QUANTUM_ACTOR_LR),
                        ("--residual_pid", str(pid_params)),
                        ("--residual_range", residual_range),
                        ("--reward_w3", "0.03"), ("--reward_w_smooth", "0.5")):
        command = _set(command, flag, value)
    return _set(command, "--skip_offline")


def build_v3_tasks(root: Path, phases: Iterable[str], pid_params: Path,
                   action_prior: str) -> List[Task]:
    if action_prior not in ACTION_PRIORS:
        raise ValueError(f"action prior must be one of {ACTION_PRIORS}, got {action_prior}")
    tasks: List[Task] = []
    # Seed by seed within a phase, so that paired runs finish together.
    for phase in phases:
        for seed in SEEDS:
            for arm in V3_PHASES[phase]:
                out = root / phase / arm / f"seed_{seed}"
                aux = phase == "aux_remi03"
                command = v3_command(seed, out, pid_params,
                                     aux_action_prior(action_prior) if aux else action_prior,
                                     RESIDUAL_AUX if aux else RESIDUAL_MAIN)
                methods = {f"{arm}_remi03"} if aux else {arm}
                if aux:
                    command = _set(command, "--method_suffix", "_remi03")
                if phase == "primary" and arm == "quantum_full" and seed == 0:
                    command.remove("--skip_baselines")
                    command = _set(command, "--pid_params", str(pid_params))
                    methods |= set(BASELINES)
                command = _set(command, "--arms", arm)
                tasks.append(Task(label=f"{phase}/{arm}/seed{seed}",
                                  command=tuple(command), result=out / "results.csv",
                                  methods=frozenset(methods)))
    return tasks


def v3_protocol(pid_params: Path, action_prior: str) -> Dict:
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("split_seed") != 1:
        raise RuntimeError("dataset manifest must lock split seed 1")
    check_pid_protocol(pid_params)
    return {
        "status": "locked_before_campaign",
        "plan": "docs/superpowers/plans/2026-10-21-v3-residual-qrl.md",
        "method": "bounded residual on the tuned PID, online TD3 only (D14-D17)",
        "dataset_sha256": manifest["dataset_sha256"],
        "split": {"train": 2100, "validation": 300, "test": 600, "split_seed": 1},
        "seeds": list(SEEDS),
        "phases": {name: list(arms) for name, arms in V3_PHASES.items()},
        "residual": {
            "pid_params_path": str(pid_params),
            "pid_params_sha256": hashlib.sha256(Path(pid_params).read_bytes()).hexdigest(),
            "range_main": RESIDUAL_MAIN, "range_aux": RESIDUAL_AUX,
            "base_update": "once per 60-s control decision"},
        "reward": {"dose_cost_weight": 0.03, "smoothness_weight": 0.5},
        "training": {
            "stage1": "none", "online_episodes": 150, "online_bc_weight": 0.0,
            "actor_loss": "TD3 (-Q); --online_rl_scale has no effect without BC",
            "action_prior_all_learned_arms": action_prior,
            "action_prior_aux_remi03": aux_action_prior(action_prior),
            "actor_learning_rates": {
                "mlp": float(MLP_ACTOR_LR), "quantum_family": float(QUANTUM_ACTOR_LR),
                "selected_on": "validation patients, "
                               "docs/experiments/2026-10-21_1600_v3_actor_lr_check.md"}},
        "variability": dataclasses.asdict(LITERATURE_SPEC),
        "primary_analysis": {
            "unit": "one training run: mean MDAPE (measured BIS) over the 600 test patients",
            "P1": "quantum_full vs pid_tuned: run means minus the deterministic PID "
                  "mean; mean difference, 95% bootstrap CI over runs, exact Wilcoxon",
            "P2": "quantum_full vs classical_matched, paired by seed (D10 statistics)",
            "P3": "quantum_full vs quantum_disabled, paired by seed (D10 statistics)",
            "multiplicity": "Holm over the three primary Wilcoxon p-values (alpha 0.05)",
            "script": "experiments/analyze_v3_campaign.py"},
        "test_outcomes_used_for_selection": False,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phases", default=",".join(V3_PHASES))
    parser.add_argument("--action_prior", required=True, choices=ACTION_PRIORS,
                        help="initial actor output of every learned arm (gate, D17)")
    parser.add_argument("--gate_record", type=Path, required=True,
                        help="experiment record of the validation gate")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pid_params", type=Path, default=DEFAULT_PID)
    parser.add_argument("--cuda_visible_devices", default="0")
    parser.add_argument("--allow_code_change", action="store_true")
    parser.add_argument("--stall_minutes", type=float, default=15.0)
    parser.add_argument("--max_attempts", type=int, default=6)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    unknown = set(phases) - set(V3_PHASES)
    if unknown:
        raise ValueError(f"unknown phases: {sorted(unknown)}")
    root = args.out.resolve()
    pid_params = args.pid_params.resolve()
    tasks = build_v3_tasks(root, phases, pid_params, args.action_prior)
    if args.dry_run:
        for task in tasks:
            print(task.label, subprocess.list2cmdline(task.command))
        return
    if not args.gate_record.exists():
        raise FileNotFoundError(f"validation gate record missing: {args.gate_record}")
    root.mkdir(parents=True, exist_ok=True)
    runner_lock = acquire_runner_lock(root)  # noqa: F841 - held until exit
    fingerprint_files = campaign_fingerprint_files(pid_params)
    fingerprint = code_fingerprint(fingerprint_files)
    protocol = v3_protocol(pid_params, args.action_prior)
    protocol["validation_gate"] = args.gate_record.as_posix()
    write_locked_protocol(root / "locked_protocol.json", protocol, fingerprint,
                          args.allow_code_change)

    env = os.environ.copy()
    env.update({
        # Unbuffered output keeps the last lines of a task that is killed
        # after a GPU hang (validation gate, 2026-10-21).
        "PYTHONUNBUFFERED": "1",
        "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
        "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
    })
    for phase in phases:
        phase_tasks = [t for t in tasks if t.label.startswith(f"{phase}/")]
        print(f"=== phase {phase}: {len(phase_tasks)} tasks ===", flush=True)
        failures = run_phase(
            phase_tasks,
            lambda t: run_task(t, env, {}, fingerprint, args.stall_minutes * 60, 30.0,
                               args.max_attempts, fingerprint_files),
            args.jobs)
        if failures:
            for failure in failures:
                print(failure, flush=True)
            raise SystemExit(f"{len(failures)} task(s) failed in phase {phase}")


if __name__ == "__main__":
    main()
