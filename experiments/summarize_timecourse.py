"""Summaries of the time courses written by experiments/evaluate_timecourse.py.

1. Consistency: the per-patient metrics of every unit must equal the reported
   evaluations (campaign results for the training-type plant, the
   model-mismatch evaluation for the Eleveld plant with variability).
2. quantiles.csv: 25th, 50th and 75th percentiles of measured BIS and of the
   applied propofol and remifentanil rates at every 10-s observation, over the
   600 test patients (pooled over the ten training runs for learned arms).
3. induction.csv: per-method induction measures, median (IQR) over patients:
   time to the first measured BIS <= 60, minimum BIS in the first 20 min, and
   the mean rates in the first 2 min.

    python experiments/summarize_timecourse.py --out logs/revision_v3/timecourse_20261104
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REFERENCES = {
    "trained": ROOT / "logs/revision_v3/campaign/analysis/results_raw.csv",
    "eleveld_iiv": ROOT / "logs/revision_v3/eleveld_mismatch_gpu_20261027/eleveld_iiv/results_all.csv",
}
METRICS = ["mdpe", "mdape", "wobble", "divergence", "tit_45_55", "tit_40_60", "n_bis_lt30",
           "n_bis_gt70", "mean_ppf", "mean_remi", "mdape_true"]
OBSERVATION_MIN = 10.0 / 60.0


def check_consistency(out: Path) -> pd.DataFrame:
    rows = []
    for condition, reference in REFERENCES.items():
        new = pd.read_csv(out / condition / "results_all.csv")
        ref = pd.read_csv(reference)
        merged = new.merge(ref, on=["method", "patient_id", "seed"], suffixes=("", "_ref"))
        for (method, seed), group in merged.groupby(["method", "seed"]):
            n_new = int(((new.method == method) & (new.seed == seed)).sum())
            diff = max(float(np.max(np.abs(group[m] - group[f"{m}_ref"]))) for m in METRICS)
            rows.append({"condition": condition, "method": method, "seed": seed, "patients": n_new,
                         "matched": len(group), "max_abs_difference": diff})
    table = pd.DataFrame(rows)
    table.to_csv(out / "summary" / "consistency.csv", index=False)
    return table


def load_traces(out: Path, condition: str, method: str):
    files = sorted((out / condition / "units").glob(f"{method}_seed*/traces.npz"))
    if not files:
        raise FileNotFoundError(f"no traces for {condition}/{method}")
    data = [np.load(f) for f in files]
    return {key: np.concatenate([d[key] for d in data]) for key in ("bis", "ppf", "remi")}, len(files)


def quantiles(traces) -> pd.DataFrame:
    n = traces["bis"].shape[1]
    frame = {"time_min": (np.arange(n) + 1) * OBSERVATION_MIN}
    for key, values in traces.items():
        for q in (25, 50, 75):
            frame[f"{key}_q{q}"] = np.percentile(values, q, axis=0)
    bis = traces["bis"]
    frame["bis_in_45_55"] = 100.0 * ((bis >= 45) & (bis <= 55)).mean(axis=0)   # % of episodes
    return pd.DataFrame(frame)


def induction(traces) -> dict:
    bis = traces["bis"]
    below = bis <= 60.0
    reached = below.any(axis=1)
    t60 = np.where(reached, (below.argmax(axis=1) + 1) * OBSERVATION_MIN, np.nan)
    first20 = int(round(20.0 / OBSERVATION_MIN))
    first2 = int(round(2.0 / OBSERVATION_MIN))
    measures = {"time_to_bis60_min": t60, "min_bis_first20": bis[:, :first20].min(axis=1),
                "ppf_first2": traces["ppf"][:, :first2].mean(axis=1),
                "remi_first2": traces["remi"][:, :first2].mean(axis=1)}
    maintenance = bis[:, first20:]
    row = {"episodes": len(bis), "reached_bis60": float(reached.mean()),
           "in_45_55_after20_pct": 100.0 * float(((maintenance >= 45) & (maintenance <= 55)).mean()),
           "in_40_60_after20_pct": 100.0 * float(((maintenance >= 40) & (maintenance <= 60)).mean()),
           "remi_zero_after20_pct": 100.0 * float((traces["remi"][:, first20:] <= 0.0).mean())}
    for name, values in measures.items():
        q25, q50, q75 = np.nanpercentile(values, [25, 50, 75])
        row.update({f"{name}_median": q50, f"{name}_q25": q25, f"{name}_q75": q75})
    return row


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--methods", default="quantum_full,pid_tuned")
    args = parser.parse_args(argv)
    out = Path(args.out)
    (out / "summary").mkdir(parents=True, exist_ok=True)

    consistency = check_consistency(out)
    print(consistency.to_string(index=False))
    if (consistency.max_abs_difference > 1e-9).any() or (consistency.matched != consistency.patients).any():
        raise SystemExit("time-course metrics differ from the reported evaluations")

    quantile_frames, induction_rows = [], []
    for condition in REFERENCES:
        for method in args.methods.split(","):
            traces, runs = load_traces(out, condition, method)
            frame = quantiles(traces)
            frame.insert(0, "method", method)
            frame.insert(0, "condition", condition)
            quantile_frames.append(frame)
            induction_rows.append({"condition": condition, "method": method, "runs": runs,
                                   **induction(traces)})
    pd.concat(quantile_frames).to_csv(out / "summary" / "quantiles.csv", index=False)
    table = pd.DataFrame(induction_rows)
    table.to_csv(out / "summary" / "induction.csv", index=False)
    print(table.T.to_string())


if __name__ == "__main__":
    main()
