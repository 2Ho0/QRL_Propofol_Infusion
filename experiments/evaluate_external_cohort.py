"""Cross-dataset evaluation: trained v3 controllers on an external population.

In the v3 method, VitalDB enters the controllers only through the covariates
(age, sex, weight, height) of the simulated training patients. The external
cohort is drawn from NHANES 2021-2023 (U.S. adults, CDC; files DEMO_L and BMX_L),
a population of another country with a larger body size. Every trained policy of
the v3 campaign (best checkpoint, residual environment of its run_config) and
every baseline is run on these patients, with the evaluation seed and random
effects of the campaign:
- external_trained: the training-type plant (Schnider/Minto with IIV)
- external_eleveld_iiv: the Eleveld plant with IIV (experiments/eleveld_plant.py)

The cohort is locked (``--lock``) before any rollout: adults aged 18-80 years
with measured weight and height and a BMI of 15-35 kg/m2 (above about 36 in
women, the James lean body mass of the Schnider and Minto models decreases with
weight), 600 participants drawn without replacement with probability
proportional to the MEC examination weight. NHANES respondent numbers (SEQN)
serve as case ids; they do not overlap the VitalDB case ids.

    python experiments/evaluate_external_cohort.py --lock --out logs/revision_v3/external_cohort_<stamp>
    python experiments/evaluate_external_cohort.py --out logs/revision_v3/external_cohort_<stamp>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import eleveld_plant  # noqa: E402
import gpu_units  # noqa: E402
from environment.variability import LITERATURE_SPEC  # noqa: E402
from evaluate_eleveld_mismatch import ARMS, BASELINES, CHECKPOINT, _baseline  # noqa: E402
from evaluate_noise_v2 import training_config  # noqa: E402
from evaluation.per_patient_eval import evaluate_cohort, patient_from_record  # noqa: E402
from evaluation.results_contract import append_results  # noqa: E402
from run_pilot import (  # noqa: E402
    TRAINED_ARMS, build_agent, env_kwargs_from_run_config, run_config_for_checkpoint)

CONDITIONS = {"external_trained": None, "external_eleveld_iiv": "iiv"}
NHANES = ROOT / "data/external/nhanes_2021_2023"
AGE_RANGE = (18, 80)
BMI_RANGE = (15.0, 35.0)
VITALDB_MAX_CASEID = 10_000
COHORT_SEED = 20261027


def load_nhanes(folder: Path = NHANES) -> pd.DataFrame:
    demo = pd.read_sas(folder / "DEMO_L.xpt")[["SEQN", "RIAGENDR", "RIDAGEYR", "WTMEC2YR"]]
    body = pd.read_sas(folder / "BMX_L.xpt")[["SEQN", "BMXWT", "BMXHT"]]
    return demo.merge(body, on="SEQN", how="inner")


def eligible(frame: pd.DataFrame) -> pd.DataFrame:
    """Adults in the age and BMI ranges with weight, height, sex and exam weight, by SEQN."""
    frame = frame.dropna(subset=["SEQN", "RIAGENDR", "RIDAGEYR", "WTMEC2YR", "BMXWT", "BMXHT"])
    bmi = frame.BMXWT / (frame.BMXHT / 100.0) ** 2
    keep = (frame.RIDAGEYR.between(*AGE_RANGE) & bmi.between(*BMI_RANGE)
            & frame.RIAGENDR.isin([1.0, 2.0]) & (frame.WTMEC2YR > 0))
    return frame[keep].sort_values("SEQN").reset_index(drop=True)


def select_cohort(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Draw n eligible participants, weighted by the exam weight; one row per patient."""
    pool = eligible(frame)
    if len(pool) < n:
        raise ValueError(f"only {len(pool)} eligible participants for a cohort of {n}")
    if (pool.SEQN <= VITALDB_MAX_CASEID).any():
        raise ValueError("NHANES SEQN would collide with VitalDB case ids")
    weights = pool.WTMEC2YR.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    chosen = pool.iloc[np.sort(rng.choice(len(pool), size=n, replace=False, p=weights / weights.sum()))]
    return pd.DataFrame({
        "caseid": chosen.SEQN.astype(int).to_numpy(),
        "age": chosen.RIDAGEYR.astype(float).to_numpy(),
        "sex": np.where(chosen.RIAGENDR.to_numpy() == 1.0, "M", "F"),
        "weight": chosen.BMXWT.astype(float).to_numpy(),
        "height": chosen.BMXHT.astype(float).to_numpy(),
    }).sort_values("caseid").reset_index(drop=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock(out: Path, n: int, seed: int) -> Path:
    """Write the cohort manifest; no controller is run."""
    target = out / "cohort.json"
    if target.exists():
        raise SystemExit(f"{target} exists; the cohort is already locked")
    raw = load_nhanes()
    cohort = select_cohort(raw, n, seed)
    out.mkdir(parents=True, exist_ok=True)
    records = cohort.to_dict(orient="records")
    target.write_text(json.dumps({
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "NHANES August 2021-August 2023 (CDC), DEMO_L.xpt and BMX_L.xpt",
        "files_sha256": {p.name: _sha256(p) for p in sorted(NHANES.glob("*.xpt"))},
        "criteria": {"age": AGE_RANGE, "bmi": BMI_RANGE, "measured weight and height": True},
        "sampling": "without replacement, probability proportional to WTMEC2YR",
        "seed": seed, "eligible": int(len(eligible(raw))), "n": n,
        "caseids_sha256": hashlib.sha256(json.dumps(cohort.caseid.tolist()).encode()).hexdigest(),
        "outcomes_evaluated_during_lock": False,
        "patients": records}, indent=1), encoding="utf-8")
    return target


def read_cohort(out: Path):
    records = json.loads((out / "cohort.json").read_text(encoding="utf-8"))["patients"]
    return [(int(r["caseid"]), patient_from_record(r)) for r in records]


def evaluate_unit(condition, method, seed, cohort, root, phase, device="cpu", progress=None):
    """Evaluate one (condition, method, seed) on the external cohort; return its rows."""
    torch.set_num_threads(1)
    eleveld_plant.install(CONDITIONS[condition])
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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lock", action="store_true", help="lock the cohort and stop")
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--root", default="logs/revision_v3/campaign")
    parser.add_argument("--phase", default="primary")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--baselines", default=",".join(BASELINES))
    parser.add_argument("--seeds", default=",".join(map(str, range(10))))
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--one", default=None, help="CONDITION:METHOD:SEED, run one unit in-process")
    parser.add_argument("--unit_out", default=None, help="folder of the --one unit")
    args = parser.parse_args(argv)

    if args.lock:
        print(f"cohort locked in {lock(args.out, args.n, COHORT_SEED)}", flush=True)
        return
    cohort = read_cohort(args.out)
    if args.one:  # one unit, called by the runner below
        condition, method, seed = args.one.split(":")
        rows = evaluate_unit(condition, method, int(seed), cohort, args.root, args.phase,
                             args.device, progress=gpu_units.progress_printer(len(cohort)))
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
    (args.out / "protocol.json").write_text(json.dumps({
        "conditions": {c: CONDITIONS[c] for c in conditions}, "arms": arms, "baselines": baselines,
        "seeds": seeds, "patients": len(cohort), "root": args.root, "phase": args.phase,
        "cohort": "cohort.json", "device": args.device, "environment_seed": 0,
        "variability": "LITERATURE_SPEC (trained plant); Eleveld IIV for external_eleveld_iiv",
        "observation_interval_seconds": 10, "control_interval_seconds": 60}, indent=2),
        encoding="utf-8")

    units = []
    for condition in conditions:
        wanted = [(b, 0) for b in baselines] + [(a, s) for s in seeds for a in arms]
        for method, seed in wanted:
            folder = args.out / condition / "units" / f"{method}_seed{seed}"
            command = (sys.executable, str(Path(__file__).resolve()), "--out", str(args.out.resolve()),
                       "--one", f"{condition}:{method}:{seed}", "--unit_out", str(folder.resolve()),
                       "--device", args.device, "--root", args.root, "--phase", args.phase)
            units.append(gpu_units.Unit(f"{condition}/{method}/seed{seed}", command, folder,
                                        frozenset({method}), len(cohort)))
    print(f"{len(units)} units to run on {args.device}", flush=True)
    failures = gpu_units.run_units(units, args.jobs)
    for condition in conditions:
        merged = gpu_units.merge(gpu_units.unit_results(args.out / condition / "units"),
                                 args.out / condition / "results_all.csv")
        print(f"{condition}: {merged.groupby(['method', 'seed']).ngroups} method-seed pairs merged",
              flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} unit(s) failed: {failures}")


if __name__ == "__main__":
    main()
