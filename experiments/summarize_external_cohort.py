"""Summary of the cross-dataset evaluation (NHANES cohort), with the locked analysis functions.

For each condition: the pre-registered contrasts (analyze_v3_campaign.analyse),
each method's change from the VitalDB test cohort on the same plant and seed
(external_trained vs the campaign, external_eleveld_iiv vs the Eleveld IIV
study), and each learned arm's change minus that of tuned PID.

    python experiments/summarize_external_cohort.py --out logs/revision_v3/external_cohort_20261027/summary
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_v3_campaign as locked  # noqa: E402
from campaign_statistics import run_summary  # noqa: E402

EXTERNAL = "logs/revision_v3/external_cohort_20261027"
REFERENCE = {  # condition -> VitalDB test results on the same plant
    "external_trained": "logs/revision_v3/campaign/analysis/run_summary.csv",
    "external_eleveld_iiv": "logs/revision_v3/eleveld_mismatch_gpu_20261027/eleveld_iiv/results_all.csv"}
METHODS = ("quantum_full", "classical_matched", "quantum_disabled", "classical_large",
           "pid_tuned", "pid", "tci", "guideline")
ARMS = METHODS[:4]


def _reference(condition: str) -> pd.DataFrame:
    path = REFERENCE[condition]
    frame = pd.read_csv(path)
    summary = frame if path.endswith("run_summary.csv") else run_summary(frame)
    return summary[summary.method.isin(METHODS)]


def summarise(external: str, n_boot: int):
    tables, contrasts = [], []
    for condition in REFERENCE:
        raw = pd.read_csv(f"{external}/{condition}/results_all.csv")
        summary = run_summary(raw)
        merged = summary.merge(_reference(condition), on=["method", "seed"],
                               suffixes=("", "_vitaldb"))
        merged["change_mdape"] = merged.mean_mdape - merged.mean_mdape_vitaldb
        table = merged.groupby("method").agg(
            runs=("seed", "nunique"), mean_mdape_vitaldb=("mean_mdape_vitaldb", "mean"),
            mean_mdape=("mean_mdape", "mean"), sd_mdape=("mean_mdape", "std"),
            change=("change_mdape", "mean"), mean_mdape_true=("mean_mdape_true", "mean"),
            tit_45_55=("mean_tit_45_55", "mean"), bis_lt30=("bis_lt30_rate", "mean"),
            bis_lt30_vitaldb=("bis_lt30_rate_vitaldb", "mean"), bis_gt70=("bis_gt70_rate", "mean"),
            ppf=("mean_ppf", "mean"), remi=("mean_remi", "mean")).reset_index()
        table.insert(0, "condition", condition)
        tables.append(table)
        result = locked.analyse(raw, n_boot=n_boot)
        result.insert(0, "condition", condition)
        contrasts.append(result)
        pid_change = float(merged.loc[merged.method == "pid_tuned", "change_mdape"].iloc[0])
        for arm in ARMS:
            d = merged.loc[merged.method == arm, "change_mdape"].to_numpy() - pid_change
            boot = np.random.default_rng(20261027).choice(d, size=(n_boot, len(d))).mean(axis=1)
            contrasts.append(pd.DataFrame([{
                "condition": condition, "role": "change_vs_pid", "kind": "one_sample",
                "a": arm, "b": "pid_tuned", "mean_difference": float(d.mean()),
                "ci95_low": float(np.percentile(boot, 2.5)),
                "ci95_high": float(np.percentile(boot, 97.5)),
                "runs_better": int((d < 0).sum()), "n_pairs": len(d)}]))
    return pd.concat(tables, ignore_index=True), pd.concat(contrasts, ignore_index=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external", default=EXTERNAL)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n_boot", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose a new folder")
    args.out.mkdir(parents=True)
    table, contrasts = summarise(args.external, args.n_boot)
    table.to_csv(args.out / "external_methods.csv", index=False)
    contrasts.to_csv(args.out / "external_contrasts.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(table.round(4).to_string(index=False))
        columns = ["condition", "role", "a", "b", "mean_difference", "ci95_low", "ci95_high",
                   "runs_better", "holm_p", "tost_p", "equivalent"]
        print(contrasts.reindex(columns=columns).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
