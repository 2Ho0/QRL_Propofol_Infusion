"""Tune PID only on the 120-patient development cohort, then test on 480.

The split is the fixed 3,000-case, patient-level 70/10/20 split.  The first
120 test IDs were already used by the earlier pilot, so they are explicitly
treated as a development set.  The remaining 480 IDs are never read by the
search objective and are evaluated exactly once after selecting the PID.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.baseline_agents import PIDController  # noqa: E402
from data.patient_splits import split_by_patient  # noqa: E402
from data.vitaldb_loader import VitalDBLoader  # noqa: E402
from evaluation.per_patient_eval import (  # noqa: E402
    evaluate_cohort,
    patient_from_record,
)
from evaluation.results_contract import append_results  # noqa: E402
from environment.variability import get_variability_spec  # noqa: E402


N_CASES = 3000
N_DEV = 120
N_FINAL = 480
PARAMETERS = (
    "kp_ppf", "ki_ppf", "kd_ppf", "kp_rftn", "ki_rftn", "kd_rftn",
    "base_ppf", "base_rftn",
)
LOWER = np.array([0.02, 0.0, 0.0, 0.001, 0.0, 0.0, 2.0, 0.01])
UPPER = np.array([0.60, 0.04, 0.40, 0.10, 0.02, 0.05, 10.0, 0.40])
DEFAULT = np.array([0.25, 0.005, 0.10, 0.02, 0.001, 0.005, 6.0, 0.1])


def ensure_fresh_output(out: Path) -> None:
    """Refuse to overwrite an earlier tuning run."""
    if (Path(out) / "protocol.json").exists():
        raise FileExistsError(f"{out} already holds a PID tuning run")


def _controller(values: Sequence[float]) -> PIDController:
    return PIDController(**dict(zip(PARAMETERS, map(float, values))))


def _score(values: Sequence[float], cohort,
           observation_interval_seconds: int = 60,
           control_interval_seconds: int = 60,
           variability: str = "none") -> Dict[str, float]:
    rows = evaluate_cohort(
        _controller(values), "pid_candidate", cohort, seed=0,
        environment_seed=0,
        observation_interval_seconds=observation_interval_seconds,
        control_interval_seconds=control_interval_seconds,
        variability=get_variability_spec(variability))
    frame = pd.DataFrame([row.__dict__ for row in rows])
    observations_per_episode = (
        200.0 * control_interval_seconds / observation_interval_seconds)
    unsafe_low_rate = (
        frame["n_bis_lt30"].sum() / (observations_per_episode * len(frame)))
    awake_rate = (
        frame["n_bis_gt70"].sum() / (observations_per_episode * len(frame)))
    # MDAPE is the prespecified primary objective.  Large penalties prevent a
    # candidate from improving it by accepting clinically extreme trajectories.
    objective = (frame["mdape"].mean() + 100.0 * unsafe_low_rate
                 + 10.0 * awake_rate)
    result = dict(zip(PARAMETERS, map(float, values)))
    result.update({
        "objective": float(objective),
        "mean_mdape": float(frame["mdape"].mean()),
        "median_mdape": float(frame["mdape"].median()),
        "mean_tit_45_55": float(frame["tit_45_55"].mean()),
        "mean_tit_40_60": float(frame["tit_40_60"].mean()),
        "unsafe_low_rate": float(unsafe_low_rate),
        "awake_rate": float(awake_rate),
    })
    return result


def _run_candidates(candidates: Iterable[np.ndarray], cohort, jobs: int,
                    observation_interval_seconds: int = 60,
                    control_interval_seconds: int = 60,
                    variability: str = "none"):
    results: List[Dict[str, float]] = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(
            _score, row, cohort, observation_interval_seconds,
            control_interval_seconds, variability): i
                   for i, row in enumerate(candidates)}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"candidate {futures[future]:03d}: objective "
                  f"{result['objective']:.3f}, MDAPE {result['mean_mdape']:.3f}",
                  flush=True)
    return results


def _cohorts(clean_manifest: str = None):
    if clean_manifest:
        manifest = json.loads(Path(clean_manifest).read_text())
        validation_ids = manifest["partitions"]["validation"]["caseids"]
        test_ids = manifest["partitions"]["test"]["caseids"]
        clinical = VitalDBLoader(
            cache_dir="data/vitaldb_cache").load_clinical_info()
        make = lambda ids: [
            (int(caseid), patient_from_record(clinical.loc[int(caseid)].to_dict()))
            for caseid in ids]
        return make(validation_ids), make(test_ids), np.asarray(
            validation_ids), np.asarray(test_ids)
    loader = VitalDBLoader(cache_dir="data/vitaldb_cache")
    *_, caseids = loader.prepare_dual_drug_training_data(
        n_cases=N_CASES, sampling_interval=60)
    _, _, test = split_by_patient(caseids, ratios=(0.7, 0.1, 0.2), seed=0)
    test_ids = np.unique(caseids[test])
    if len(test_ids) != N_DEV + N_FINAL:
        raise RuntimeError(f"expected 600 held-out patients, got {len(test_ids)}")
    clinical = loader.load_clinical_info()
    make = lambda ids: [
        (int(caseid), patient_from_record(clinical.loc[int(caseid)].to_dict()))
        for caseid in ids]
    return (make(test_ids[:N_DEV]), make(test_ids[N_DEV:]),
            test_ids[:N_DEV], test_ids[N_DEV:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", type=int, default=64)
    parser.add_argument("--local", type=int, default=32)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--out", required=True,
                        help="new output directory (never an earlier tuning run)")
    parser.add_argument("--clean_manifest", default=None)
    parser.add_argument("--observation_interval_seconds", type=int, default=60)
    parser.add_argument("--control_interval_seconds", type=int, default=60)
    parser.add_argument("--variability", choices=("none", "literature"),
                        default="none",
                        help="sim v2 uses 'literature'")
    parser.add_argument("--skip_final", action="store_true",
                        help="select on the development cohort only; the "
                             "held-out cohort is evaluated later with all arms")
    args = parser.parse_args()

    out = Path(args.out)
    ensure_fresh_output(out)
    out.mkdir(parents=True, exist_ok=True)
    dev, final, dev_ids, final_ids = _cohorts(args.clean_manifest)
    clean_split_seed = (
        int(json.loads(Path(args.clean_manifest).read_text())["split_seed"])
        if args.clean_manifest else 0)
    manifest = {
        "n_cases": N_CASES,
        "split_seed": clean_split_seed,
        "environment_seed": 0,
        "observation_interval_seconds": args.observation_interval_seconds,
        "control_interval_seconds": args.control_interval_seconds,
        "objective": "mean MDAPE + 100*BIS<30 rate + 10*BIS>70 rate",
        "variability": args.variability,
        "development_caseids": list(map(int, dev_ids)),
        "untouched_final_caseids": list(map(int, final_ids)),
        "bounds": {name: [float(lo), float(hi)] for name, lo, hi in
                   zip(PARAMETERS, LOWER, UPPER)},
    }
    (out / "protocol.json").write_text(json.dumps(manifest, indent=2))

    coarse_unit = qmc.LatinHypercube(d=len(PARAMETERS), seed=20260811).random(
        args.coarse)
    coarse = qmc.scale(coarse_unit, LOWER, UPPER)
    coarse = np.vstack([DEFAULT, coarse])
    results = _run_candidates(
        coarse, dev, args.jobs, args.observation_interval_seconds,
        args.control_interval_seconds, args.variability)
    coarse_frame = pd.DataFrame(results).sort_values("objective")
    coarse_frame.to_csv(out / "coarse_search.csv", index=False)

    rng = np.random.default_rng(20260812)
    elite = coarse_frame.head(min(8, len(coarse_frame)))[list(PARAMETERS)].to_numpy()
    span = UPPER - LOWER
    local = []
    for i in range(args.local):
        centre = elite[i % len(elite)]
        local.append(np.clip(centre + rng.normal(0.0, 0.08, len(PARAMETERS)) * span,
                             LOWER, UPPER))
    local_results = _run_candidates(
        local, dev, args.jobs, args.observation_interval_seconds,
        args.control_interval_seconds, args.variability)
    all_results = pd.concat(
        [coarse_frame, pd.DataFrame(local_results)], ignore_index=True
    ).sort_values("objective")
    all_results.to_csv(out / "all_candidates.csv", index=False)

    best = all_results.iloc[0]
    selected = {name: float(best[name]) for name in PARAMETERS}
    selection = {"parameters": selected,
                 "development_metrics": {
                     name: float(best[name]) for name in (
                         "objective", "mean_mdape", "median_mdape",
                         "mean_tit_45_55", "mean_tit_40_60",
                         "unsafe_low_rate", "awake_rate")}}
    (out / "selected_pid.json").write_text(json.dumps(selection, indent=2))
    if args.skip_final:
        print("SELECTED", json.dumps(selection, indent=2))
        return

    rows = evaluate_cohort(_controller([selected[p] for p in PARAMETERS]),
                           "pid_tuned", final, seed=0, environment_seed=0,
                           observation_interval_seconds=(
                               args.observation_interval_seconds),
                           control_interval_seconds=(
                               args.control_interval_seconds),
                           variability=get_variability_spec(args.variability))
    append_results(rows, out / "untouched_final_results.csv")
    final_frame = pd.DataFrame([row.__dict__ for row in rows])
    summary = final_frame.groupby("method").agg(
        n=("patient_id", "nunique"), mean_mdape=("mdape", "mean"),
        median_mdape=("mdape", "median"),
        mean_tit_45_55=("tit_45_55", "mean"),
        mean_tit_40_60=("tit_40_60", "mean"),
        total_bis_lt30=("n_bis_lt30", "sum"),
        total_bis_gt70=("n_bis_gt70", "sum"),
    ).reset_index()
    summary.to_csv(out / "untouched_final_summary.csv", index=False)
    print("\nSELECTED", json.dumps(selection, indent=2))
    print("\nUNTOUCHED FINAL\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
