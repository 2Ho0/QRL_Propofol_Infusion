"""Pre-specified run-level statistics of the training campaigns.

Written on 2026-10-19, before any campaign result existed. The unit of
analysis is one training run: for each method and training seed the 600 test
patients are averaged, and contrasts pair runs that share a training seed
(same simulator-stage patients, same evaluation patients). For each contrast
the script reports the per-seed differences, their mean with a 95% bootstrap
CI over the paired runs, Cohen's d_z, the exact Wilcoxon signed-rank p and a
TOST equivalence test with a margin of +/-1.0 MDAPE percentage point.
Patient-level comparisons treat the fitted models as fixed and are reported
only as secondary descriptions.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

OBSERVATIONS_PER_EPISODE = 1200          # 200 min at 10-s observations
MARGIN = 1.0                             # MDAPE percentage points
PRIMARY_CONTRASTS: List[Tuple[str, str]] = [
    ("quantum_full", "classical_matched"),
    ("quantum_full", "quantum_disabled"),
]
SECONDARY_CONTRASTS: List[Tuple[str, str]] = [
    ("quantum_full", "classical_large"),
    ("quantum_full", "fourier_surrogate"),
    ("quantum_full", "quantum_noskip"),
    ("quantum_full", "quantum_noentangle"),
    ("quantum_full", "quantum_2layer"),
    ("quantum_critic", "classical_matched"),
    ("ansatz_realamplitudes", "quantum_full"),
    ("ansatz_efficientsu2", "quantum_full"),
    ("ansatz_twolocal", "quantum_full"),
    ("quantum_full", "quantum_full_td3bc"),
    ("classical_large", "classical_large_td3bc"),
    ("quantum_full", "quantum_full_offline_only"),
    ("quantum_full", "quantum_full_online_only"),
]


def load_campaign(root: Path) -> pd.DataFrame:
    """Concatenate every task's results.csv, refusing duplicate rows.

    Only ``seed_<k>`` folders count; ``seed_<k>.stalled-N`` and
    ``seed_<k>.interrupted-N`` keep abandoned attempts.
    """
    frames = []
    for path in sorted(root.glob("*/*/seed_*/results.csv")):
        if not re.fullmatch(r"seed_\d+", path.parent.name):
            continue
        frame = pd.read_csv(path)
        frame["source"] = path.relative_to(root).parent.as_posix()
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no results under {root}")
    raw = pd.concat(frames, ignore_index=True)
    duplicated = raw.duplicated(["method", "patient_id", "seed"])
    if duplicated.any():
        raise ValueError(f"{int(duplicated.sum())} duplicate (method, patient, seed) rows")
    return raw


def run_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (method, training seed), averaging over test patients."""
    def rate(column):
        return (column, lambda x: x.sum() / (OBSERVATIONS_PER_EPISODE * len(x)))

    aggregations = dict(
        mean_mdape=("mdape", "mean"),
        median_mdape=("mdape", "median"),
        mean_tit_45_55=("tit_45_55", "mean"),
        mean_tit_40_60=("tit_40_60", "mean"),
        bis_lt30_rate=rate("n_bis_lt30"),
        bis_gt70_rate=rate("n_bis_gt70"),
        mean_ppf=("mean_ppf", "mean"),
        mean_remi=("mean_remi", "mean"),
        n_patients=("patient_id", "nunique"),
    )
    for column in ("mdpe", "wobble", "divergence", "mdape_true", "mdpe_true",
                   "wobble_true", "divergence_true"):
        if column in raw.columns:
            aggregations[f"mean_{column}"] = (column, "mean")
    return raw.groupby(["method", "seed"]).agg(**aggregations).reset_index()


def paired_contrast(summary: pd.DataFrame, a: str, b: str,
                    margin: float = MARGIN, n_boot: int = 10000,
                    metric: str = "mean_mdape", seed: int = 20261019) -> Dict:
    """Run-level contrast a - b, paired by training seed."""
    xa = summary[summary.method == a].set_index("seed")[metric]
    xb = summary[summary.method == b].set_index("seed")[metric]
    if set(xa.index) != set(xb.index) or len(xa) < 2:
        raise ValueError(f"{a} and {b} need the same training seeds "
                         f"({sorted(xa.index)} vs {sorted(xb.index)})")
    seeds = sorted(xa.index)
    diff = xa.loc[seeds].to_numpy() - xb.loc[seeds].to_numpy()
    n = len(diff)
    mean, sd = float(diff.mean()), float(diff.std(ddof=1))
    boot = np.random.default_rng(seed).choice(diff, size=(n_boot, n)).mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    if sd > 0:
        se = sd / np.sqrt(n)
        p_lower = 1.0 - stats.t.cdf((mean + margin) / se, n - 1)   # H0: diff <= -margin
        p_upper = stats.t.cdf((mean - margin) / se, n - 1)         # H0: diff >= +margin
        tost_p = float(max(p_lower, p_upper))
        d_z = mean / sd
    else:
        tost_p = 0.0 if abs(mean) < margin else 1.0
        d_z = float("nan")
    if np.any(diff != 0):
        wilcoxon_p = float(stats.wilcoxon(diff, method="exact").pvalue)
    else:
        wilcoxon_p = 1.0
    return {
        "a": a, "b": b, "metric": metric, "n_pairs": n,
        "mean_a": float(xa.mean()), "mean_b": float(xb.mean()),
        "mean_difference": mean, "sd_difference": sd,
        "ci95_low": float(low), "ci95_high": float(high),
        "cohens_dz": d_z, "wilcoxon_exact_p": wilcoxon_p,
        "tost_margin": margin, "tost_p": tost_p, "equivalent": tost_p < 0.05,
        "per_seed_differences": json.dumps([round(float(x), 6) for x in diff]),
    }


def patient_level(raw: pd.DataFrame, a: str, b: str) -> Dict:
    """Secondary: patients averaged over seeds; models treated as fixed."""
    pa = raw[raw.method == a].groupby("patient_id").mdape.mean()
    pb = raw[raw.method == b].groupby("patient_id").mdape.mean()
    common = pa.index.intersection(pb.index)
    diff = pa.loc[common] - pb.loc[common]
    return {"a": a, "b": b, "n_patients": len(common),
            "mean_difference": float(diff.mean()),
            "wilcoxon_p": float(stats.wilcoxon(diff).pvalue) if len(common) else float("nan")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("logs/revision_v2/campaign"))
    parser.add_argument("--n_boot", type=int, default=10000)
    args = parser.parse_args()
    out = args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    raw = load_campaign(args.root)
    raw.to_csv(out / "results_raw.csv", index=False)
    summary = run_summary(raw)
    summary.to_csv(out / "run_summary.csv", index=False)
    summary.groupby("method").agg(
        n_runs=("seed", "nunique"),
        mean_mdape=("mean_mdape", "mean"), sd_mdape=("mean_mdape", "std"),
        mean_tit_45_55=("mean_tit_45_55", "mean"),
        bis_lt30_rate=("bis_lt30_rate", "mean"),
        bis_gt70_rate=("bis_gt70_rate", "mean"),
        mean_ppf=("mean_ppf", "mean"), mean_remi=("mean_remi", "mean"),
    ).reset_index().to_csv(out / "method_summary.csv", index=False)

    methods = set(summary.method)
    rows, patient_rows = [], []
    for role, contrasts in (("primary", PRIMARY_CONTRASTS),
                            ("secondary", SECONDARY_CONTRASTS)):
        for a, b in contrasts:
            if a in methods and b in methods:
                rows.append({"role": role,
                             **paired_contrast(summary, a, b, n_boot=args.n_boot)})
                patient_rows.append({"role": role, **patient_level(raw, a, b)})
    pd.DataFrame(rows).to_csv(out / "run_level_contrasts.csv", index=False)
    pd.DataFrame(patient_rows).to_csv(out / "patient_level_secondary.csv", index=False)
    print(pd.DataFrame(rows)[["role", "a", "b", "n_pairs", "mean_difference",
                              "ci95_low", "ci95_high", "tost_p"]].to_string(index=False)
          if rows else "no complete contrasts yet")


if __name__ == "__main__":
    main()
