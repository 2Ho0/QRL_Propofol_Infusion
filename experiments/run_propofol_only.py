"""Single-drug baseline: Classical-Matched with recorded remifentanil.

The controller sets propofol only; remifentanil follows the clinical record.
This runner retrains the v3 Classical-Matched arm with that structure and
nothing else changed: for every seed, run_pilot.main is called with the
arguments of the campaign's Classical-Matched run (only --out and
--method_suffix _ppfonly differ), while the training and evaluation
environments apply each patient's recorded remifentanil
(experiments/recorded_remifentanil.py). Propofol stays tuned-PID propofol plus
the policy's residual. After training, each run is checked against its
campaign run: the same training, checkpoint and test patients and the same
network sizes. The tuned PID is also evaluated with recorded remifentanil
(pid_tuned_ppfonly), after a check that this evaluation path reproduces the
campaign's pid_tuned values.

    python experiments/run_propofol_only.py --lock --out logs/revision_v3/propofol_only_<stamp>
    python experiments/run_propofol_only.py --out logs/revision_v3/propofol_only_<stamp> --jobs 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import gpu_units  # noqa: E402
import recorded_remifentanil as rr  # noqa: E402
import run_pilot  # noqa: E402
from agents.baseline_agents import PIDController  # noqa: E402
from data.vitaldb_loader import VitalDBLoader  # noqa: E402
from environment.variability import get_variability_spec  # noqa: E402
from evaluation.per_patient_eval import evaluate_cohort, patient_from_record  # noqa: E402
from evaluation.results_contract import append_results  # noqa: E402

CAMPAIGN = ROOT / "logs/revision_v3/campaign"
CACHE = ROOT / "data/vitaldb_cache"
ARM = "classical_matched"
SUFFIX = "_ppfonly"
METHOD = ARM + SUFFIX
BASELINE = "pid_tuned_ppfonly"
SEEDS = range(10)
N_PATIENTS = 600
SCHEDULES = "remifentanil_schedules.json"
SPLIT_KEYS = ("train_caseids", "val_caseids", "test_caseids", "evaluated_caseids",
              "online_caseids", "checkpoint_caseids")
REPRODUCTION_PATIENTS = 5


def campaign_run(campaign: Path, seed: int, arm: str = ARM) -> Path:
    return Path(campaign) / "primary" / arm / f"seed_{seed}"


def campaign_argv(run_config: Dict, out: Path) -> List[str]:
    """run_pilot arguments of a campaign run, with a new --out and the _ppfonly suffix."""
    parser = run_pilot.build_parser()
    wanted = dict(run_config, out=str(out), method_suffix=SUFFIX)
    argv: List[str] = []
    for action in parser._actions:
        if not action.option_strings or action.dest == "help":
            continue
        value = wanted[action.dest]
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                argv.append(action.option_strings[0])
        elif value is not None:
            argv += [action.option_strings[0], str(value)]
    parsed = vars(parser.parse_args(argv))
    if parsed != wanted:
        diff = {k: (parsed.get(k), wanted.get(k))
                for k in set(parsed) | set(wanted) if parsed.get(k) != wanted.get(k)}
        raise ValueError(f"campaign arguments do not round-trip: {diff}")
    return argv


def verify_against_campaign(unit_out: Path, run: Path) -> None:
    """Same patients and network sizes as the campaign run, or RuntimeError."""
    read = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))  # noqa: E731
    unit_split, run_split = read(unit_out / "patient_split.json"), read(run / "patient_split.json")
    unit_sizes, run_sizes = read(unit_out / "timings.json")[METHOD], read(run / "timings.json")[ARM]
    differences = [key for key in SPLIT_KEYS if unit_split[key] != run_split[key]]
    differences += [key for key in ("actor_params", "critic_params")
                    if unit_sizes[key] != run_sizes[key]]
    if differences:
        raise RuntimeError(f"{unit_out} differs from its campaign run {run}: {differences}")
    (Path(unit_out) / "verification.json").write_text(json.dumps({
        "campaign_run": str(run), "identical": True, "checked": list(SPLIT_KEYS)
        + ["actor_params", "critic_params"],
        # The checkpoints only work in this environment; run_config.json does not say so.
        "environment": rr.RecordedRemifentanilEnv.__name__}, indent=2), encoding="utf-8")


def unverified(units: Iterable[gpu_units.Unit]) -> List[str]:
    """Training units without verification.json (a failed check leaves results.csv behind)."""
    return [u.label for u in units if u.label != BASELINE
            and not (u.out / "verification.json").exists()]


def write_hash(root: Path) -> None:
    digest = hashlib.sha256((Path(root) / SCHEDULES).read_bytes()).hexdigest()
    (Path(root) / (SCHEDULES + ".sha256")).write_text(digest, encoding="utf-8")


def load_locked_schedules(root: Path):
    """The locked schedules, after checking the file against the sha256 written at lock time."""
    path = Path(root) / SCHEDULES
    expected = (Path(root) / (SCHEDULES + ".sha256")).read_text(encoding="utf-8").strip()
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"{path} does not match its locked sha256")
    return rr.load_schedules(path)


class _ClinicalWithCaseid(VitalDBLoader):
    """The clinical table with its caseid index repeated as a column, so records name their case."""

    def load_clinical_info(self) -> pd.DataFrame:
        table = super().load_clinical_info()
        return table.assign(caseid=table.index)


_RUN_PILOT_ORIGINALS = (run_pilot.VitalDBLoader, run_pilot.patient_from_record,
                        run_pilot.DualDrugEnv)


def install_into_run_pilot(schedules) -> None:
    """Patients carry their record, and training and evaluation apply it."""
    run_pilot.VitalDBLoader = _ClinicalWithCaseid
    run_pilot.patient_from_record = lambda rec: rr.recorded_patient_from_record(rec, schedules)
    run_pilot.DualDrugEnv = rr.RecordedRemifentanilEnv
    rr.install()


def restore_run_pilot() -> None:
    run_pilot.VitalDBLoader, run_pilot.patient_from_record, run_pilot.DualDrugEnv = \
        _RUN_PILOT_ORIGINALS
    rr.restore()


def build_units(root: Path, seeds: Iterable[int] = SEEDS,
                campaign: Path = CAMPAIGN) -> List[gpu_units.Unit]:
    root = Path(root)
    script = str(Path(__file__).resolve())

    def unit(label, folder, method, one):
        command = (sys.executable, script, "--out", str(root.resolve()),
                   "--campaign", str(Path(campaign).resolve()), "--one", one)
        return gpu_units.Unit(label, command, root / "units" / folder, frozenset({method}),
                              N_PATIENTS)
    return ([unit(BASELINE, BASELINE, BASELINE, "baseline")]
            + [unit(f"{METHOD}/seed{s}", f"{METHOD}_seed{s}", METHOD, f"train:{s}")
               for s in seeds])


def lock(root: Path, campaign: Path) -> Path:
    """Compute and lock the recorded schedules of every split case; no controller is run."""
    target = Path(root) / SCHEDULES
    if target.exists():
        raise SystemExit(f"{target} exists; the schedules are already locked")
    splits = [json.loads((campaign_run(campaign, s) / "patient_split.json").read_text())
              for s in SEEDS]
    for key in ("train_caseids", "val_caseids", "test_caseids"):
        if any(split[key] != splits[0][key] for split in splits):
            raise RuntimeError(f"campaign seeds disagree on {key}")
    caseids = sorted(splits[0]["train_caseids"] + splits[0]["val_caseids"]
                     + splits[0]["test_caseids"])
    schedules = rr.build_schedules(caseids, VitalDBLoader(cache_dir=str(CACHE)))
    minutes = np.array([len(s.rates) for s in schedules.values()])
    rates = np.concatenate([s.rates for s in schedules.values()])
    test = [schedules[c] for c in splits[0]["evaluated_caseids"]]
    evaluated = np.array([[s.rate_at(m) for m in range(200)] for s in test])
    Path(root).mkdir(parents=True, exist_ok=True)
    rr.save_schedules(schedules, target, {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "rule": rr.__doc__.split("Rule for the recorded rate")[1].strip(),
        "source": f"VitalDBLoader.load_case, cache {CACHE.relative_to(ROOT).as_posix()}",
        "n_cases": len(schedules),
        "caseids_sha256": hashlib.sha256(json.dumps(caseids).encode()).hexdigest(),
        "outcomes_evaluated_during_lock": False,
        "recorded_minutes": {"median": float(np.median(minutes)),
                             "q25": float(np.percentile(minutes, 25)),
                             "q75": float(np.percentile(minutes, 75)),
                             "share_below_200": float(np.mean(minutes < 200)),
                             "share_below_360": float(np.mean(minutes < 360))},
        "hold_rate_median": float(np.median([s.hold for s in schedules.values()])),
        "share_of_recorded_minutes_above_2": float(np.mean(rates > 2.0)),
        "test_first_200_min": {"mean_rate": float(evaluated.mean()),
                               "share_above_2": float(np.mean(evaluated > 2.0))}})
    write_hash(root)
    return target


def _cohort(caseids, schedules):
    clinical = VitalDBLoader(cache_dir=str(CACHE)).load_clinical_info()
    return [(int(c), rr.recorded_patient_from_record(
        {**clinical.loc[int(c)].to_dict(), "caseid": int(c)}, schedules)) for c in caseids]


def baseline_unit(root: Path, unit_out: Path, campaign: Path) -> None:
    """pid_tuned with recorded remifentanil, after reproducing the campaign's pid_tuned."""
    run = campaign_run(campaign, 0, "quantum_full")   # the run that evaluated the baselines
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    split = json.loads((run / "patient_split.json").read_text(encoding="utf-8"))
    tuned = json.loads(Path(config["pid_params"]).read_text(encoding="utf-8"))["parameters"]
    schedules = load_locked_schedules(root)
    cohort = _cohort(split["evaluated_caseids"], schedules)
    settings = dict(seed=config["seed"], environment_seed=config["evaluation_seed"],
                    variability=get_variability_spec(config["variability"]),
                    observation_interval_seconds=config["observation_interval_seconds"],
                    control_interval_seconds=config["control_interval_seconds"])

    rr.restore()
    check = cohort[:REPRODUCTION_PATIENTS]
    rows = evaluate_cohort(PIDController(**tuned), "pid_tuned", check, **settings)
    campaign_rows = pd.read_csv(CAMPAIGN / "analysis/results_raw.csv")
    campaign_rows = campaign_rows[campaign_rows.method == "pid_tuned"].set_index("patient_id")
    for row in rows:
        expected = campaign_rows.loc[row.patient_id, "mdape"]
        if not np.isclose(row.mdape, expected, rtol=0, atol=1e-9):
            raise RuntimeError(f"pid_tuned of case {row.patient_id} gives {row.mdape}, "
                               f"campaign {expected}: the evaluation path differs")
    print(f"pid_tuned reproduced on {len(rows)} patients", flush=True)

    rr.install()
    rows = evaluate_cohort(PIDController(**tuned), BASELINE, cohort, **settings,
                           progress=gpu_units.progress_printer(len(cohort)))
    append_results(rows, Path(unit_out) / "results.csv")
    print(f"{BASELINE}: mean MDAPE {np.mean([r.mdape for r in rows]):.3f}", flush=True)


def train_unit(root: Path, unit_out: Path, seed: int, campaign: Path) -> None:
    run = campaign_run(campaign, seed)
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    argv = campaign_argv(config, Path(unit_out).resolve())
    install_into_run_pilot(load_locked_schedules(root))
    sys.argv = [str(Path(run_pilot.__file__).resolve()), *argv]
    run_pilot.main()
    verify_against_campaign(Path(unit_out), run)
    print(f"{METHOD} seed {seed}: verified against {run}", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lock", action="store_true", help="lock the recorded schedules and stop")
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--stall_minutes", type=float, default=15.0)
    parser.add_argument("--one", default=None, help="'baseline' or 'train:SEED', run in-process")
    args = parser.parse_args(argv)

    if args.lock:
        print(f"schedules locked in {lock(args.out, args.campaign)}", flush=True)
        return
    if not (args.out / SCHEDULES).exists():
        raise SystemExit(f"lock the schedules first: {args.out / SCHEDULES} is missing")
    if args.one:
        units = {u.command[-1]: u for u in build_units(args.out, campaign=args.campaign)}
        unit = units[args.one]
        if args.one == "baseline":
            baseline_unit(args.out, unit.out, args.campaign)
        else:
            train_unit(args.out, unit.out, int(args.one.split(":")[1]), args.campaign)
        return

    seeds = [int(s) for s in args.seeds.split(",")]
    units = build_units(args.out, seeds, args.campaign)
    print(f"{len(units)} units", flush=True)
    failures = gpu_units.run_units(units, args.jobs, stall_seconds=args.stall_minutes * 60)
    missing = unverified(units)
    if missing:
        raise SystemExit(f"not merged; runs not verified against the campaign: {missing}"
                         + (f"; failed: {failures}" if failures else ""))
    merged = gpu_units.merge(gpu_units.unit_results(args.out / "units"),
                             args.out / "results_all.csv")
    print(f"{merged.groupby(['method', 'seed']).ngroups} method-seed pairs merged", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} unit(s) failed: {failures}")


if __name__ == "__main__":
    main()
