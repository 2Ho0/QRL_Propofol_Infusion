"""Inference latency, memory and update time of the v3 arms (compute cost).

Every measurement runs in its own process, one after another, so that the arms
are measured on the same idle machine and their memory is isolated.

- prepare: the seed-0 best checkpoint of the arm controls one test patient for
  the 200-min test episode in its training environment (run_config). The
  60-step state windows of the 200 decisions and the transitions are saved.
- infer: CPU, one thread (an infusion pump is an embedded CPU target). Times the
  deployed computation of one 60-s decision, the encoder and then the actor, on
  the saved windows, and samples the working set of the process. The circuit
  is simulated on the CPU, so the quantum figures are simulation costs, not
  hardware run times. The tuned PID, which the residual adds to, is timed as a
  reference.
- update: one TD3 gradient update (minibatch 64, actor every second update) on
  the devices of the campaign (networks on the GPU, circuits on the CPU), from a
  replay buffer holding the saved transitions; time per update and peak memory.

    python experiments/profile_v3_compute.py --out logs/revision_v3/compute_profile_<stamp>
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

# Every worker imports the same modules (the circuit stack included), so the
# working set after the imports is the same baseline for every arm.
from agents.baseline_agents import PIDController  # noqa: E402
from data.vitaldb_loader import VitalDBLoader  # noqa: E402
from environment.dual_drug_env import DualDrugEnv  # noqa: E402
from environment.variability import LITERATURE_SPEC  # noqa: E402
from evaluate_noise_v2 import training_config  # noqa: E402
from evaluation.per_patient_eval import patient_from_record  # noqa: E402
from run_pilot import (  # noqa: E402
    TRAINED_ARMS, build_agent, env_kwargs_from_run_config, run_config_for_checkpoint)

ARMS = {  # arm -> campaign phase
    "classical_matched": "primary", "classical_large": "primary",
    "quantum_disabled": "primary", "quantum_full": "primary",
    "fourier_surrogate": "reviewer", "ansatz_realamplitudes": "reviewer",
    "ansatz_efficientsu2": "reviewer", "ansatz_twolocal": "reviewer",
    "quantum_critic": "reviewer"}
CAMPAIGN = ROOT / "logs/revision_v3/campaign"
CHECKPOINT = "{root}/{phase}/{arm}/seed_{seed}/{arm}/stage2/best_mdape.pt"
PID_PARAMS = ROOT / "logs/revision_v2/pid_tuning/selected_pid.json"
DECISIONS = 200            # 200-min test episode, one decision per 60 s
OBS_PER_DECISION = 6       # 10-s observations


def _mb(value: float) -> float:
    return value / 1024 ** 2


def _working_set() -> float:
    return _mb(psutil.Process().memory_info().rss)


def _checkpoint(arm: str, seed: int = 0) -> Path:
    return Path(CHECKPOINT.format(root=CAMPAIGN, phase=ARMS[arm], arm=arm, seed=seed))


def _load_agent(arm: str, device: str):
    checkpoint = _checkpoint(arm)
    agent = build_agent(arm, TRAINED_ARMS[arm], training_config(), torch.device(device), 0)
    agent.load(str(checkpoint))
    return agent, checkpoint


def _test_patient():
    split = json.loads((CAMPAIGN / "primary/quantum_full/seed_0/patient_split.json").read_text())
    caseid = int(split["evaluated_caseids"][0])
    clinical = VitalDBLoader(cache_dir=str(ROOT / "data/vitaldb_cache")).load_clinical_info()
    return caseid, patient_from_record(clinical.loc[caseid].to_dict())


def prepare(arm: str, work: Path) -> dict:
    """One test episode of the trained arm; save its windows and transitions."""
    torch.set_num_threads(1)
    agent, checkpoint = _load_agent(arm, "cpu")
    env_kwargs = env_kwargs_from_run_config(run_config_for_checkpoint(checkpoint))
    caseid, patient = _test_patient()
    env = DualDrugEnv(patient=patient, seed=caseid, dt=10 / 60.0,
                      max_steps=DECISIONS * OBS_PER_DECISION,
                      shaping_interval_steps=OBS_PER_DECISION,
                      variability=LITERATURE_SPEC, **env_kwargs)
    agent.encoder.eval()
    agent.actor.eval()
    length = int(agent.sequence_length)
    state, _ = env.reset()
    history = deque([np.asarray(state, np.float32)], maxlen=length)

    def window():
        observed = list(history)
        return np.asarray([observed[0]] * (length - len(observed)) + observed, np.float32)

    record = {k: [] for k in ("windows", "actions", "rewards", "next_windows", "dones")}
    for _ in range(DECISIONS):
        current = window()
        action = agent.select_action(state, deterministic=True, state_sequence=current)
        total, done = 0.0, False
        for _ in range(OBS_PER_DECISION):
            state, reward, terminated, truncated, _ = env.step(action)
            history.append(np.asarray(state, np.float32))
            total += float(reward)
            done = bool(terminated or truncated)
            if done:
                break
        record["windows"].append(current)
        record["actions"].append([action[0] / 12.0, action[1] / 2.0])
        record["rewards"].append(total)
        record["next_windows"].append(window())
        record["dones"].append(done)
        if done:
            break
    np.savez(work / f"{arm}_transitions.npz", **{k: np.asarray(v) for k, v in record.items()})
    return {"arm": arm, "caseid": caseid, "decisions": len(record["dones"]),
            "checkpoint": str(checkpoint.relative_to(ROOT))}


def _deployed_bytes(*modules) -> int:
    return int(sum(t.numel() * t.element_size()
                   for m in modules for t in list(m.parameters()) + list(m.buffers())))


def infer(arm: str, work: Path, steps: int, warmup: int) -> dict:
    """Latency of one decision of the deployed path on one CPU thread."""
    torch.set_num_threads(1)
    windows = np.load(work / f"{arm}_transitions.npz")["windows"]
    before = _working_set()
    if arm == "pid_tuned":
        pid = PIDController(**json.loads(PID_PARAMS.read_text())["parameters"])
        step = lambda w: pid.select_action(w[-1], deterministic=True)  # noqa: E731
        param_bytes = 0
    else:
        agent, _ = _load_agent(arm, "cpu")
        encoder, actor = agent.encoder.eval(), agent.actor.eval()
        param_bytes = _deployed_bytes(encoder, actor)
        # Only the deployed path stays: critics, targets, optimizers and the
        # replay buffer are training objects.
        del agent
        gc.collect()

        def step(w):
            with torch.no_grad():
                z = encoder(torch.from_numpy(w).unsqueeze(0))
                z = z[0] if isinstance(z, tuple) else z
                z = z[:, -1, :] if z.dim() == 3 else z
                return actor(z).numpy()[0] * np.array([12.0, 2.0])
    loaded = _working_set()
    for i in range(warmup):
        step(windows[i % len(windows)])
    samples, peak = [], loaded
    for i in range(steps):
        w = windows[i % len(windows)]
        start = time.perf_counter()
        step(w)
        samples.append((time.perf_counter() - start) * 1000.0)
        if i % 100 == 0:
            peak = max(peak, _working_set())
    s = np.asarray(samples)
    return {"arm": arm, "mean_ms": float(s.mean()), "median_ms": float(np.median(s)),
            "p95_ms": float(np.percentile(s, 95)), "max_ms": float(s.max()),
            "deployed_param_kb": param_bytes / 1024, "ws_imports_mb": before,
            "ws_loaded_mb": loaded, "ws_inference_max_mb": max(peak, _working_set()),
            "ws_increment_mb": max(peak, _working_set()) - before, "steps": steps}


def update(arm: str, work: Path, steps: int, warmup: int) -> dict:
    """Time of one TD3 gradient update on the training devices."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(work / f"{arm}_transitions.npz")
    agent, _ = _load_agent(arm, device)
    for i in range(len(data["dones"])):
        agent.replay_buffer.push(data["windows"][i], data["actions"][i], float(data["rewards"][i]),
                                 data["next_windows"][i], bool(data["dones"][i]))
    agent.encoder.train()
    agent.actor.train()

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    for _ in range(warmup):
        agent.update()
    sync()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(steps):
        start = time.perf_counter()
        metrics = agent.update()
        sync()
        samples.append((time.perf_counter() - start) * 1000.0)
        if not metrics:
            raise RuntimeError(f"{arm}: update returned no metrics (buffer not ready?)")
    s = np.asarray(samples)
    return {"arm": arm, "device": device, "update_mean_ms": float(s.mean()),
            "update_median_ms": float(np.median(s)), "update_p95_ms": float(np.percentile(s, 95)),
            "process_peak_ws_mb": _mb(psutil.Process().memory_info().peak_wset)
            if hasattr(psutil.Process().memory_info(), "peak_wset") else float("nan"),
            "gpu_peak_allocated_mb": _mb(torch.cuda.max_memory_allocated()) if device == "cuda"
            else 0.0, "steps": steps}


def _worker_env(mode: str) -> dict:
    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
    if mode == "update":   # as in the campaign (gpu_units.gpu_env)
        env.update({"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    else:                  # one CPU thread for torch and for the XLA circuit
        env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"})
    return env


def _run_worker(mode: str, arm: str, work: Path, extra: list) -> dict:
    target = work / f"{mode}_{arm}.json"
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", mode, "--arm", arm,
               "--work", str(work)] + extra + ["--result", str(target)]
    with open(work / f"{mode}_{arm}.log", "w", encoding="utf-8") as log:
        code = subprocess.run(command, env=_worker_env(mode), stdout=log, stderr=subprocess.STDOUT,
                              cwd=str(ROOT)).returncode
    if code != 0 or not target.exists():
        raise RuntimeError(f"{mode} {arm} failed (exit {code}); see {work / f'{mode}_{arm}.log'}")
    return json.loads(target.read_text())


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--infer_steps", type=int, default=2000)
    parser.add_argument("--infer_warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--update_steps", type=int, default=200)
    parser.add_argument("--update_warmup", type=int, default=10)
    parser.add_argument("--worker", choices=("prepare", "infer", "update"))
    parser.add_argument("--arm")
    parser.add_argument("--work", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args(argv)

    if args.worker:
        if args.worker == "prepare":
            result = prepare(args.arm, args.work)
        elif args.worker == "infer":
            result = infer(args.arm, args.work, args.infer_steps, args.infer_warmup)
        else:
            result = update(args.arm, args.work, args.update_steps, args.update_warmup)
        args.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return

    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose a new folder")
    work = args.out / "workers"
    work.mkdir(parents=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise SystemExit(f"unknown arms: {sorted(unknown)}")
    host = {"cpu": psutil.cpu_count(logical=False), "logical_cpus": psutil.cpu_count(),
            "cpu_load_percent_at_start": psutil.cpu_percent(interval=2.0),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__, "python": sys.version.split()[0]}
    (args.out / "protocol.json").write_text(json.dumps({
        "arms": arms, "checkpoint": CHECKPOINT, "seed": 0, "decisions": DECISIONS,
        "infer_steps": args.infer_steps, "infer_warmup": args.infer_warmup,
        "repeats": args.repeats, "update_steps": args.update_steps,
        "update_warmup": args.update_warmup, "host": host}, indent=2), encoding="utf-8")
    infer_rows, update_rows = [], []
    for arm in arms:
        print(f"{arm}: prepare", flush=True)
        _run_worker("prepare", arm, work, [])
    # The PID reference uses the windows of the QRL episode (or the first arm's).
    source = "quantum_full" if "quantum_full" in arms else arms[0]
    pid_source = np.load(work / f"{source}_transitions.npz")
    np.savez(work / "pid_tuned_transitions.npz", **dict(pid_source))
    for repeat in range(args.repeats):
        for arm in arms + ["pid_tuned"]:
            row = _run_worker("infer", arm, work, ["--infer_steps", str(args.infer_steps),
                                                    "--infer_warmup", str(args.infer_warmup)])
            row["repeat"] = repeat
            infer_rows.append(row)
            print(f"{arm} infer {repeat}: median {row['median_ms']:.3f} ms, "
                  f"+{row['ws_increment_mb']:.1f} MB", flush=True)
    for arm in arms:
        row = _run_worker("update", arm, work, ["--update_steps", str(args.update_steps),
                                                 "--update_warmup", str(args.update_warmup)])
        update_rows.append(row)
        print(f"{arm} update: mean {row['update_mean_ms']:.1f} ms", flush=True)
    inference = pd.DataFrame(infer_rows)
    inference.to_csv(args.out / "inference_runs.csv", index=False)
    pd.DataFrame(update_rows).to_csv(args.out / "update.csv", index=False)
    summary = inference.groupby("arm", sort=False).agg(
        median_ms=("median_ms", "median"), p95_ms=("p95_ms", "median"),
        mean_ms=("mean_ms", "median"), deployed_param_kb=("deployed_param_kb", "first"),
        ws_increment_mb=("ws_increment_mb", "median"),
        ws_inference_max_mb=("ws_inference_max_mb", "median")).reset_index()
    summary = summary.merge(pd.DataFrame(update_rows)[
        ["arm", "update_mean_ms", "update_median_ms", "process_peak_ws_mb",
         "gpu_peak_allocated_mb"]], on="arm", how="left")
    summary.to_csv(args.out / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
