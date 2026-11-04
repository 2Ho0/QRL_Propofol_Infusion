"""BIS and infusion time courses of the trained v3 controllers on the test patients.

Runs the evaluation of experiments/evaluate_eleveld_mismatch.py again (same
600 test patients, checkpoints, environment seed 0 + case id, residual
environment from each run_config) and also stores the 10-s traces of measured
BIS and of the applied propofol and remifentanil rates. Conditions:
- trained: the Schnider/Minto plant of training; the per-patient metrics must
  equal the campaign results (consistency check)
- eleveld_iiv: Eleveld PK with its published interindividual variability; the
  metrics must equal the model-mismatch evaluation

Every (condition, method, seed) is one GPU subprocess under the stall watchdog
of experiments/gpu_units.py. Each unit folder holds traces.npz (written first)
and results.csv.

    python experiments/evaluate_timecourse.py --out logs/revision_v3/timecourse_20261104
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import eleveld_plant  # noqa: E402
import gpu_units  # noqa: E402
from data.vitaldb_loader import VitalDBLoader  # noqa: E402
from environment.variability import LITERATURE_SPEC  # noqa: E402
from evaluate_eleveld_mismatch import (  # noqa: E402
    BASELINES, CHECKPOINT, CONDITIONS, _baseline, _patient_ids)
from evaluate_noise_v2 import training_config  # noqa: E402
from evaluation.per_patient_eval import evaluate_patient, patient_from_record  # noqa: E402
from evaluation.results_contract import PatientResult, append_results  # noqa: E402
from run_pilot import (  # noqa: E402
    TRAINED_ARMS, build_agent, env_kwargs_from_run_config, run_config_for_checkpoint)

ENVIRONMENT_SEED = 0          # common random numbers, as in the campaign


def evaluate_unit(condition, method, seed, ids, root, phase, device, unit_out, progress=None):
    """Evaluate one (condition, method, seed); write traces.npz, then results.csv."""
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

    rows, traces = [], {"bis": [], "ppf": [], "remi": []}
    for caseid, patient in (cohort if progress is None else progress(cohort)):
        m, bis, ppf, rftn = evaluate_patient(
            agent, patient, seed=ENVIRONMENT_SEED + int(caseid), smoothing_beta=0.0,
            observation_interval_seconds=10, control_interval_seconds=60,
            variability=LITERATURE_SPEC, env_kwargs=env_kwargs)
        traces["bis"].append(bis)
        traces["ppf"].append(ppf)
        traces["remi"].append(rftn)
        rows.append(PatientResult(   # the same fields as evaluation.per_patient_eval.evaluate_cohort
            method=method, patient_id=int(caseid), seed=int(seed),
            mdpe=m["mdpe"], mdape=m["mdape"], wobble=m["wobble"], divergence=m["divergence"],
            tit_45_55=float(100.0 * np.mean((bis >= 45) & (bis <= 55))),
            tit_40_60=float(100.0 * np.mean((bis >= 40) & (bis <= 60))),
            n_bis_lt30=int(np.sum(bis < 30)), n_bis_gt70=int(np.sum(bis > 70)),
            mean_ppf=float(np.mean(ppf)), mean_remi=float(np.mean(rftn)),
            mdpe_true=m["mdpe_true"], mdape_true=m["mdape_true"],
            wobble_true=m["wobble_true"], divergence_true=m["divergence_true"]))

    lengths = {len(trace) for values in traces.values() for trace in values}
    if len(lengths) != 1:
        raise RuntimeError(f"traces of unequal length: {sorted(lengths)}")
    unit_out = Path(unit_out)
    unit_out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(unit_out / "traces.npz", caseid=np.asarray(ids, dtype=np.int64),
                        **{k: np.asarray(v, dtype=np.float32) for k, v in traces.items()})
    append_results(rows, unit_out / "results.csv")
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="logs/revision_v3/campaign")
    parser.add_argument("--phase", default="primary")
    parser.add_argument("--arms", default="quantum_full")
    parser.add_argument("--baselines", default="pid_tuned")
    parser.add_argument("--seeds", default=",".join(map(str, range(10))))
    parser.add_argument("--conditions", default="trained,eleveld_iiv")
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
                             args.unit_out, progress=gpu_units.progress_printer(len(ids)))
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
        "purpose": "BIS and infusion time courses (traces.npz per unit)",
        "conditions": {c: CONDITIONS[c] for c in conditions}, "arms": arms, "baselines": baselines,
        "seeds": seeds, "patients": len(ids), "root": args.root, "phase": args.phase,
        "device": args.device, "environment_seed": ENVIRONMENT_SEED,
        "observation_interval_seconds": 10, "control_interval_seconds": 60}, indent=2),
        encoding="utf-8")

    units = []
    for condition in conditions:
        wanted = [(b, 0) for b in baselines] + [(a, s) for s in seeds for a in arms]
        for method, seed in wanted:
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
        merged = gpu_units.merge(gpu_units.unit_results(out / condition / "units"),
                                 out / condition / "results_all.csv")
        print(f"{condition}: {merged.groupby(['method', 'seed']).ngroups} method-seed pairs merged",
              flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} unit(s) failed: {failures}")


if __name__ == "__main__":
    main()
