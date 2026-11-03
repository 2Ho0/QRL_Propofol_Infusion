"""Pre-registered analysis of the v3 campaign (written before any v3 test result).

The unit of
analysis is one training run: its mean MDAPE (measured BIS) over the 600 test
patients. Primary contrasts:
  P1 quantum_full vs pid_tuned. The tuned PID is deterministic and evaluated
     once on the same patients and simulator seeds, so every run is compared
     with that single value: mean difference, 95% bootstrap CI over runs,
     exact Wilcoxon signed-rank p and the number of runs better than PID.
  P2 quantum_full vs classical_matched and P3 quantum_full vs
     quantum_disabled, paired by training seed with the run-level statistics.
The three primary Wilcoxon p-values are Holm-adjusted once all three exist
with the complete design (10 runs of 600 patients each); until then every row
says it is incomplete. A pair with a missing seed is compared on the seeds both
methods have. The same three contrasts on noise-free BIS are reported
without adjustment. Everything else is secondary; patient-level comparisons
are descriptive.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from campaign_statistics import (  # noqa: E402
    load_campaign, paired_contrast, patient_level, run_summary)

REFERENCE = "pid_tuned"
EXPECTED_RUNS = 10
EXPECTED_PATIENTS = 600
PRIMARY_ONE_SAMPLE: List[Tuple[str, str]] = [("quantum_full", REFERENCE)]
PRIMARY_PAIRED: List[Tuple[str, str]] = [
    ("quantum_full", "classical_matched"),
    ("quantum_full", "quantum_disabled"),
]
SECONDARY_ONE_SAMPLE: List[Tuple[str, str]] = [
    ("classical_matched", REFERENCE), ("classical_large", REFERENCE),
    ("quantum_disabled", REFERENCE), ("quantum_full_remi03", REFERENCE),
    ("classical_matched_remi03", REFERENCE),
    ("quantum_full", "pid"), ("quantum_full", "tci"), ("quantum_full", "guideline"),
]
SECONDARY_PAIRED: List[Tuple[str, str]] = [
    ("quantum_full", "classical_large"),
    ("quantum_full_remi03", "classical_matched_remi03"),
    ("quantum_full", "fourier_surrogate"),
    ("quantum_critic", "classical_matched"),
    ("ansatz_realamplitudes", "quantum_full"),
    ("ansatz_efficientsu2", "quantum_full"),
    ("ansatz_twolocal", "quantum_full"),
]


def one_sample_contrast(summary: pd.DataFrame, a: str, reference: str,
                        n_boot: int = 10000, metric: str = "mean_mdape",
                        seed: int = 20261021) -> Dict:
    """Runs of ``a`` against a deterministic method evaluated once."""
    ref = summary[summary.method == reference][metric]
    if len(ref) != 1:
        raise ValueError(f"{reference} must be evaluated exactly once ({len(ref)} runs)")
    xa = summary[summary.method == a].set_index("seed")[metric].sort_index()
    if len(xa) < 2:
        raise ValueError(f"{a} needs at least two runs")
    reference_value = float(ref.iloc[0])
    diff = xa.to_numpy() - reference_value
    n = len(diff)
    boot = np.random.default_rng(seed).choice(diff, size=(n_boot, n)).mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    wilcoxon_p = (float(stats.wilcoxon(diff, method="exact").pvalue)
                  if np.any(diff != 0) else 1.0)
    return {
        "a": a, "b": reference, "metric": metric, "n_pairs": n, "n_runs": n,
        "mean_a": float(xa.mean()), "mean_b": reference_value,
        "mean_difference": float(diff.mean()),
        "sd_difference": float(diff.std(ddof=1)),
        "ci95_low": float(low), "ci95_high": float(high),
        "wilcoxon_exact_p": wilcoxon_p, "runs_better": int((diff < 0).sum()),
        "per_seed_differences": json.dumps([round(float(x), 6) for x in diff]),
    }


def holm(pvalues: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    if not np.all(np.isfinite(p)):
        raise ValueError(f"Holm needs finite p-values, got {p.tolist()}")
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[index]))
        adjusted[index] = running
    return adjusted.tolist()


def _paired_on_common_seeds(summary: pd.DataFrame, a: str, b: str, n_boot: int,
                            metric: str):
    """D10 paired contrast on the seeds both methods have; None below two pairs."""
    seeds = (set(summary[summary.method == a].seed)
             & set(summary[summary.method == b].seed))
    if len(seeds) < 2:
        return None
    subset = summary[summary.seed.isin(seeds) & summary.method.isin([a, b])]
    return paired_contrast(subset, a, b, n_boot=n_boot, metric=metric)


def _design(summary: pd.DataFrame, a: str, b: str, kind: str, n_pairs: int,
            expected_runs: int, expected_patients: int) -> Dict:
    """Patients per run and whether the contrast has the complete design."""
    runs = summary[summary.method.isin([a, b])]
    n_patients_min = int(runs.n_patients.min())
    reference_runs = int((summary.method == b).sum())
    complete = (n_pairs == expected_runs and n_patients_min == expected_patients
                and (kind == "paired" or reference_runs == 1))
    return {"n_patients_min": n_patients_min, "complete": bool(complete)}


def analyse(raw: pd.DataFrame, n_boot: int = 10000,
            expected_runs: int = EXPECTED_RUNS,
            expected_patients: int = EXPECTED_PATIENTS) -> pd.DataFrame:
    """Every available contrast, primary first; Holm only for the complete primaries."""
    summary = run_summary(raw)
    methods = set(summary.method)
    rows = []
    families = [("primary", PRIMARY_ONE_SAMPLE, PRIMARY_PAIRED, "mean_mdape")]
    if "mean_mdape_true" in summary.columns:
        families.append(("primary_true_bis", PRIMARY_ONE_SAMPLE, PRIMARY_PAIRED,
                         "mean_mdape_true"))
    families.append(("secondary", SECONDARY_ONE_SAMPLE, SECONDARY_PAIRED, "mean_mdape"))
    for role, one_sample, paired, metric in families:
        for kind, pairs in (("one_sample", one_sample), ("paired", paired)):
            for a, b in pairs:
                if a not in methods or b not in methods:
                    continue
                result = (one_sample_contrast(summary, a, b, n_boot=n_boot, metric=metric)
                          if kind == "one_sample" else
                          _paired_on_common_seeds(summary, a, b, n_boot, metric))
                if result is None:
                    continue
                rows.append({"role": role, "kind": kind, **result,
                             **_design(summary, a, b, kind, result["n_pairs"],
                                       expected_runs, expected_patients)})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["holm_p"] = np.nan
    primary = table.role == "primary"
    if (primary.sum() == len(PRIMARY_ONE_SAMPLE) + len(PRIMARY_PAIRED)
            and table.loc[primary, "complete"].all()):
        table.loc[primary, "holm_p"] = holm(table.loc[primary, "wilcoxon_exact_p"].tolist())
    return table


def method_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Per method: means over runs, with run-level median doses and noise-free MDAPE."""
    runs = run_summary(raw).merge(
        raw.groupby(["method", "seed"]).agg(
            median_ppf=("mean_ppf", "median"),
            median_remi=("mean_remi", "median")).reset_index(),
        on=["method", "seed"])
    aggregations = dict(
        n_runs=("seed", "nunique"),
        mean_mdape=("mean_mdape", "mean"), sd_mdape=("mean_mdape", "std"),
        mean_tit_45_55=("mean_tit_45_55", "mean"),
        bis_lt30_rate=("bis_lt30_rate", "mean"),
        bis_gt70_rate=("bis_gt70_rate", "mean"),
        mean_ppf=("mean_ppf", "mean"), mean_remi=("mean_remi", "mean"),
        median_ppf=("median_ppf", "mean"), median_remi=("median_remi", "mean"),
    )
    if "mean_mdape_true" in runs.columns:
        aggregations["mean_mdape_true"] = ("mean_mdape_true", "mean")
    return runs.groupby("method").agg(**aggregations).reset_index()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("logs/revision_v3/campaign"))
    parser.add_argument("--n_boot", type=int, default=10000)
    args = parser.parse_args(argv)
    out = args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    raw = load_campaign(args.root)
    raw.to_csv(out / "results_raw.csv", index=False)
    run_summary(raw).to_csv(out / "run_summary.csv", index=False)
    method_summary(raw).to_csv(out / "method_summary.csv", index=False)

    table = analyse(raw, n_boot=args.n_boot)
    table.to_csv(out / "contrasts.csv", index=False)
    methods = set(raw.method)
    patient_rows = [patient_level(raw, a, b)
                    for a, b in PRIMARY_ONE_SAMPLE + PRIMARY_PAIRED
                    if a in methods and b in methods]
    pd.DataFrame(patient_rows).to_csv(out / "patient_level_primary.csv", index=False)
    if table.empty:
        print("no complete contrasts yet")
        return
    if not table.loc[table.role == "primary", "complete"].all():
        print("INCOMPLETE DESIGN: interim numbers, no Holm adjustment")
    print(table[["role", "kind", "a", "b", "n_pairs", "complete", "mean_difference",
                 "ci95_low", "ci95_high", "wilcoxon_exact_p", "holm_p"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
