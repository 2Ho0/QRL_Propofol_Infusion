"""Summaries of the v3 evaluation-only studies, with the locked analysis functions.

- Eleveld mismatch: the pre-registered contrasts (analyze_v3_campaign.analyse) on
  each Eleveld plant, and each method's change from the training-type plant
  (campaign analysis) on the same patients and seeds.
- Noise: each condition against the noise-free (analytic) evaluation, paired by
  training seed (campaign_statistics.paired_contrast, TOST margin 1.0),
  and tuned PID on the same 120 patients for reference.

    python experiments/summarize_v3_evaluations.py --out logs/revision_v3/eval_summary_<stamp>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_v3_campaign as locked  # noqa: E402
from campaign_statistics import paired_contrast, run_summary  # noqa: E402

ELEVELD = "logs/revision_v3/eleveld_mismatch_gpu_20261027"
NOISE = "logs/revision_v3/noise_study_gpu_20261027"
CAMPAIGN = "logs/revision_v3/campaign"
METHODS = ("quantum_full", "classical_matched", "quantum_disabled", "classical_large",
           "pid_tuned", "pid", "tci", "guideline")


def eleveld(out: Path, n_boot: int) -> None:
    trained = pd.read_csv(f"{CAMPAIGN}/analysis/run_summary.csv")
    trained = trained[trained.method.isin(METHODS)]
    tables, contrasts = [], []
    for condition in ("eleveld_typical", "eleveld_iiv"):
        raw = pd.read_csv(f"{ELEVELD}/{condition}/results_all.csv")
        summary = run_summary(raw)
        merged = summary.merge(trained, on=["method", "seed"], suffixes=("", "_trained"))
        merged["change_mdape"] = merged.mean_mdape - merged.mean_mdape_trained
        table = merged.groupby("method").agg(
            runs=("seed", "nunique"), mean_mdape_trained=("mean_mdape_trained", "mean"),
            mean_mdape=("mean_mdape", "mean"), sd_mdape=("mean_mdape", "std"),
            change=("change_mdape", "mean"), mean_mdape_true=("mean_mdape_true", "mean"),
            tit_45_55=("mean_tit_45_55", "mean"), bis_lt30=("bis_lt30_rate", "mean"),
            bis_gt70=("bis_gt70_rate", "mean"), ppf=("mean_ppf", "mean"),
            remi=("mean_remi", "mean")).reset_index()
        table.insert(0, "condition", condition)
        tables.append(table)
        result = locked.analyse(raw, n_boot=n_boot)
        result.insert(0, "condition", condition)
        contrasts.append(result)
        # Does QRL lose more than tuned PID? Per-seed change of QRL minus PID's change.
        pid_change = float(merged.loc[merged.method == "pid_tuned", "change_mdape"].iloc[0])
        for arm in ("quantum_full", "classical_matched", "quantum_disabled", "classical_large"):
            d = merged.loc[merged.method == arm, "change_mdape"].to_numpy() - pid_change
            boot = np.random.default_rng(20261027).choice(d, size=(n_boot, len(d))).mean(axis=1)
            contrasts.append(pd.DataFrame([{
                "condition": condition, "role": "change_vs_pid", "kind": "one_sample",
                "a": arm, "b": "pid_tuned", "mean_difference": float(d.mean()),
                "ci95_low": float(np.percentile(boot, 2.5)), "ci95_high": float(np.percentile(boot, 97.5)),
                "runs_better": int((d < 0).sum()), "n_pairs": len(d)}]))
    pd.concat(tables).to_csv(out / "eleveld_methods.csv", index=False)
    pd.concat(contrasts, ignore_index=True).to_csv(out / "eleveld_contrasts.csv", index=False)


def noise(out: Path, n_boot: int) -> None:
    raw = pd.read_csv(f"{NOISE}/results_all.csv")
    summary = run_summary(raw)
    rows = []
    for condition in sorted(set(summary.method) - {"analytic"}):
        rows.append({"condition": condition, **paired_contrast(
            summary[summary.method.isin([condition, "analytic"])], condition, "analytic",
            n_boot=n_boot)})
    contrasts = pd.DataFrame(rows)
    table = summary.groupby("method").agg(
        runs=("seed", "nunique"), patients=("n_patients", "min"),
        mean_mdape=("mean_mdape", "mean"), sd_mdape=("mean_mdape", "std"),
        tit_45_55=("mean_tit_45_55", "mean"), bis_lt30=("bis_lt30_rate", "mean"),
        bis_gt70=("bis_gt70_rate", "mean")).reset_index().rename(columns={"method": "condition"})
    # Tuned PID on the same patients (campaign results, deterministic).
    ids = json.loads(Path(f"{NOISE}/protocol.json").read_text())["patients"]
    campaign = pd.read_csv(f"{CAMPAIGN}/primary/quantum_full/seed_0/results.csv")
    pid = run_summary(campaign[(campaign.method == "pid_tuned") & campaign.patient_id.isin(ids)]).iloc[0]
    table = pd.concat([table, pd.DataFrame([{
        "condition": "pid_tuned (same patients)", "runs": 1, "patients": int(pid.n_patients),
        "mean_mdape": pid.mean_mdape, "sd_mdape": np.nan, "tit_45_55": pid.mean_tit_45_55,
        "bis_lt30": pid.bis_lt30_rate, "bis_gt70": pid.bis_gt70_rate}])], ignore_index=True)
    table.to_csv(out / "noise_conditions.csv", index=False)
    contrasts.to_csv(out / "noise_contrasts.csv", index=False)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n_boot", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose a new folder")
    args.out.mkdir(parents=True)
    eleveld(args.out, args.n_boot)
    noise(args.out, args.n_boot)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        for name in ("eleveld_methods", "noise_conditions"):
            print(f"== {name}")
            print(pd.read_csv(args.out / f"{name}.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
