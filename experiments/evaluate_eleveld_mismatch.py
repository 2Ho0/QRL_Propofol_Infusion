"""Model-mismatch evaluation: trained v3 controllers on an Eleveld plant.

Each trained policy of the v3 campaign (best checkpoint, residual environment
restored from its run_config) and each baseline is run again on the same 600
test patients, evaluation seed and BIS random effects as in the campaign, but
the plant follows the Eleveld models (experiments/eleveld_plant.py) while the
state keeps the Schnider/Minto predictions. Conditions:
- trained: the training-type plant; reproduces the campaign (consistency check)
- eleveld_typical: Eleveld typical PK for each patient's covariates
- eleveld_iiv: Eleveld PK with its published interindividual variability

Every (condition, method, seed) is one unit, run as a GPU subprocess under the
stall watchdog of experiments/gpu_units.py (the policy runs on the GPU; the
simulator and the circuit, JAX on Windows, run on the CPU). Units already in
``<out>/<condition>/results.csv`` (the CPU run of 2026-10-27) are kept.
``<out>/<condition>/results_all.csv`` merges everything.

    python experiments/evaluate_eleveld_mismatch.py --out logs/revision_v3/eleveld_mismatch_20261027
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import eleveld_plant  # noqa: E402
import gpu_units  # noqa: E402
from agents.baseline_agents import ClinicalGuidelineAgent, PIDController  # noqa: E402
from agents.tci_agent import TCIAgent  # noqa: E402
from data.vitaldb_loader import VitalDBLoader  # noqa: E402
from environment.variability import LITERATURE_SPEC  # noqa: E402
from evaluate_noise_v2 import training_config  # noqa: E402
from evaluation.per_patient_eval import evaluate_cohort, patient_from_record  # noqa: E402
from evaluation.results_contract import append_results  # noqa: E402
from run_pilot import (  # noqa: E402
    TRAINED_ARMS, build_agent, env_kwargs_from_run_config, run_config_for_checkpoint)

CONDITIONS = {"trained": None, "eleveld_typical": "typical", "eleveld_iiv": "iiv"}
ARMS = ("quantum_full", "classical_matched", "quantum_disabled", "classical_large")
BASELINES = ("pid_tuned", "pid", "tci", "guideline")
CHECKPOINT = "{root}/{phase}/{arm}/seed_{seed}/{arm}/stage2/best_mdape.pt"
PID_PARAMS = ROOT / "logs/revision_v2/pid_tuning/selected_pid.json"


def _baseline(name):
    if name == "pid_tuned":
        return PIDController(**json.loads(PID_PARAMS.read_text())["parameters"])
    if name == "pid":
        return PIDController()
    if name == "tci":
        return TCIAgent(control_interval_seconds=60)
    if name == "guideline":
        return ClinicalGuidelineAgent()
    raise ValueError(name)


def _patient_ids(root: str, max_patients: int):
    split = json.loads(Path(f"{root}/primary/quantum_full/seed_0/patient_split.json").read_text())
    ids = [int(c) for c in split["evaluated_caseids"][:max_patients]]
    if len(ids) != max_patients:
        raise RuntimeError("not enough evaluated patients in the split file")
    return ids


def evaluate_unit(condition, method, seed, ids, root, phase, device="cpu", progress=None):
    """Evaluate one (condition, method, seed) on the cohort; return its rows."""
    torch.set_num_threads(1)
    eleveld_plant.install(CONDITIONS[condition])
    clinical = VitalDBLoader(cache_dir=str(ROOT / "data/vitaldb_cache")).load_clinical_info()
    cohort = [(c, patient_from_record(clinical.loc[c].to_dict())) for c in ids]
    if method in BASELINES:
        agent, env_kwargs = _baseline(method), None   # baselines act on the plant directly
    else:
        checkpoint = CHECKPOINT.format(root=root, phase=phase, arm=method, seed=seed)
        agent = build_agent(method, TRAINED_ARMS[method], training_config(), torch.device(device), seed)
        agent.load(checkpoint)
        env_kwargs = env_kwargs_from_run_config(run_config_for_checkpoint(checkpoint))
    return evaluate_cohort(agent, method, cohort, seed=seed, environment_seed=0,
                           smoothing_beta=0.0, observation_interval_seconds=10,
                           control_interval_seconds=60, variability=LITERATURE_SPEC,
                           env_kwargs=env_kwargs, progress=progress)


def _done(path: Path, n_patients: int) -> set:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    counts = pd.read_csv(path).groupby(["method", "seed"]).size()
    return {key for key, n in counts.items() if n == n_patients}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="logs/revision_v3/campaign")
    parser.add_argument("--phase", default="primary")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--baselines", default=",".join(BASELINES))
    parser.add_argument("--seeds", default=",".join(map(str, range(10))))
    parser.add_argument("--conditions", default="eleveld_typical,eleveld_iiv")
    parser.add_argument("--max_patients", type=int, default=600)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--one", default=None, help="CONDITION:METHOD:SEED, run one unit in-process")
    parser.add_argument("--unit_out", default=None, help="folder of the --one unit")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    ids = _patient_ids(args.root, args.max_patients)

    if args.one:  # one unit, called by the runner below
        condition, method, seed = args.one.split(":")
        rows = evaluate_unit(condition, method, int(seed), ids, args.root, args.phase, args.device,
                             progress=gpu_units.progress_printer(len(ids)))
        append_results(rows, Path(args.unit_out) / "results.csv")
        print(f"{condition} {method} seed {seed}: mean MDAPE {np.mean([r.mdape for r in rows]):.3f}",
              flush=True)
        return

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"unknown conditions: {sorted(unknown)}")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(json.dumps({
        "conditions": {c: CONDITIONS[c] for c in conditions}, "arms": arms, "baselines": baselines,
        "seeds": seeds, "patients": len(ids), "root": args.root, "phase": args.phase,
        "plant": eleveld_plant.__doc__.strip().splitlines()[0],
        "eleveld_variability": eleveld_plant.ELEVELD_SPEC.source, "device": args.device,
        "environment_seed": 0, "observation_interval_seconds": 10,
        "control_interval_seconds": 60}, indent=2), encoding="utf-8")

    units = []
    for condition in conditions:
        earlier = _done(out / condition / "results.csv", len(ids))   # CPU run, kept
        wanted = [(b, 0) for b in baselines] + [(a, s) for s in seeds for a in arms]
        for method, seed in wanted:
            if (method, seed) in earlier:
                continue
            folder = out / condition / "units" / f"{method}_seed{seed}"
            command = (sys.executable, str(Path(__file__).resolve()), "--one",
                       f"{condition}:{method}:{seed}", "--unit_out", str(folder.resolve()),
                       "--device", args.device, "--root", args.root, "--phase", args.phase,
                       "--max_patients", str(len(ids)))
            units.append(gpu_units.Unit(f"{condition}/{method}/seed{seed}", command, folder,
                                        frozenset({method}), len(ids)))
    print(f"{len(units)} units to run on {args.device}", flush=True)
    failures = gpu_units.run_units(units, args.jobs)
    for condition in conditions:
        sources = [out / condition / "results.csv"] + gpu_units.unit_results(out / condition / "units")
        merged = gpu_units.merge(sources, out / condition / "results_all.csv")
        print(f"{condition}: {merged.groupby(['method', 'seed']).ngroups} method-seed pairs merged",
              flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} unit(s) failed: {failures}")


if __name__ == "__main__":
    main()
