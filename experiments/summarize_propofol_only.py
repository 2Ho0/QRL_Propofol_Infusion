"""Pre-specified analysis of the single-drug (propofol-only) baseline.

The analysis was fixed before any result. Unit of analysis: one training run's mean MDAPE
(measured BIS) over the 600 test patients. Classical-Matched, QRL and tuned-PID
values are the v3 campaign's, unchanged.
  D1 classical_matched_ppfonly - classical_matched, paired by seed: the
     contribution of dual-drug control (positive = single drug worse).
  D2 quantum_full - classical_matched_ppfonly, paired by seed: the proposed
     method against the single-drug controller.
D1 and D2 use the locked run-level statistics (paired_contrast) and are
Holm-adjusted together; the same contrasts on noise-free BIS are unadjusted.

    python experiments/summarize_propofol_only.py --out logs/revision_v3/propofol_only_<stamp>/summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from campaign_statistics import paired_contrast, run_summary  # noqa: E402
from analyze_v3_campaign import holm, method_summary, one_sample_contrast  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STUDY = ROOT / "logs/revision_v3/propofol_only_20261027"
CAMPAIGN_RAW = ROOT / "logs/revision_v3/campaign/analysis/results_raw.csv"
SINGLE = "classical_matched_ppfonly"
SINGLE_PID = "pid_tuned_ppfonly"
CAMPAIGN_METHODS = ("classical_matched", "quantum_full", "pid_tuned")
PRIMARY = (("D1", SINGLE, "classical_matched"), ("D2", "quantum_full", SINGLE))
SEEDS = range(10)


def _load(root: Path, campaign_raw: Path) -> pd.DataFrame:
    missing = [f"seed{s}" for s in SEEDS
               if not (root / "units" / f"{SINGLE}_seed{s}" / "verification.json").exists()]
    if missing:
        raise RuntimeError(f"runs not verified against the campaign: {missing}")
    new = pd.read_csv(root / "results_all.csv")
    campaign = pd.read_csv(campaign_raw)
    campaign = campaign[campaign.method.isin(CAMPAIGN_METHODS)].drop(columns="source")
    raw = pd.concat([new, campaign], ignore_index=True)
    if raw.duplicated(["method", "patient_id", "seed"]).any():
        raise RuntimeError("duplicated (method, patient, seed) rows")
    return raw


def verdict(d1: Dict) -> Dict:
    """The first matching row of the pre-specified wording table."""
    if d1["ci95_low"] > 0 and d1["holm_p"] < 0.05:
        row = "dual_drug_better"
    elif d1["tost_p"] < 0.05:
        row = "equivalent"
    elif d1["ci95_high"] < 0:
        row = "single_drug_better"
    else:
        row = "inconclusive"
    return {"row": row, "single_drug_better_ci": bool(d1["ci95_high"] < 0),
            "dual_drug_better_ci": bool(d1["ci95_low"] > 0)}


def summarise(root: Path, campaign_raw: Path, n_boot: int = 10000
              ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    raw = _load(Path(root), Path(campaign_raw))
    summary = run_summary(raw)
    rows = []
    for role, metric in (("primary", "mean_mdape"), ("true_bis", "mean_mdape_true")):
        for label, a, b in PRIMARY:
            rows.append({"role": role, "label": label, "kind": "paired",
                         **paired_contrast(summary, a, b, n_boot=n_boot, metric=metric)})
    for a, b in ((SINGLE, "pid_tuned"), (SINGLE, SINGLE_PID)):
        # Not in the pre-registered list (record section 2): exploratory.
        rows.append({"role": "exploratory", "label": f"{a}-{b}", "kind": "one_sample",
                     **one_sample_contrast(summary, a, b, n_boot=n_boot)})
    pid = summary.set_index("method").mean_mdape
    rows.append({"role": "secondary", "label": "pid_single-pid_dual", "kind": "single_value",
                 "a": SINGLE_PID, "b": "pid_tuned", "metric": "mean_mdape",
                 "mean_difference": float(pid[SINGLE_PID] - pid["pid_tuned"])})
    contrasts = pd.DataFrame(rows)
    contrasts["holm_p"] = float("nan")
    primary = contrasts.role == "primary"
    contrasts.loc[primary, "holm_p"] = holm(contrasts.loc[primary, "wilcoxon_exact_p"].tolist())
    d1 = contrasts[primary & (contrasts.label == "D1")].iloc[0].to_dict()
    return method_summary(raw), contrasts, verdict(d1)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=STUDY)
    parser.add_argument("--campaign_raw", type=Path, default=CAMPAIGN_RAW)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n_boot", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose a new folder")
    methods, contrasts, result = summarise(args.root, args.campaign_raw, args.n_boot)
    args.out.mkdir(parents=True)
    methods.to_csv(args.out / "methods.csv", index=False)
    contrasts.to_csv(args.out / "contrasts.csv", index=False)
    (args.out / "verdict.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(methods.round(4).to_string(index=False))
        columns = ["role", "label", "a", "b", "n_pairs", "mean_difference", "ci95_low",
                   "ci95_high", "wilcoxon_exact_p", "holm_p", "tost_p", "equivalent"]
        print(contrasts.reindex(columns=columns).round(4).to_string(index=False))
    print(f"verdict: {result}")


if __name__ == "__main__":
    main()
