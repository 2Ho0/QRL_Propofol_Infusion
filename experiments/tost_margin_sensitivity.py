"""Smallest equivalence margin at which the locked TOST still holds (post hoc).

The locked
analysis tests equivalence of paired contrasts with a paired t TOST at a margin
of +/-1.0 MDAPE point (campaign_statistics.paired_contrast). For each
paired contrast in the locked contrasts.csv, this script reports the smallest
margin at which the same test gives p < 0.05:
    delta_min = |mean difference| + t(0.95, n - 1) * SD / sqrt(n).
The locked results are read, not changed.

    python experiments/tost_margin_sensitivity.py --out logs/tost_margin
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from campaign_statistics import paired_contrast  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONTRASTS = ROOT / "logs/revision_v3/campaign/analysis/contrasts.csv"


def tost_p(diff: np.ndarray, margin: float) -> float:
    """TOST p of the locked analysis, for per-seed differences diff."""
    summary = pd.DataFrame({"method": ["a"] * len(diff) + ["b"] * len(diff),
                            "seed": list(range(len(diff))) * 2,
                            "mean_mdape": np.concatenate([diff, np.zeros(len(diff))])})
    return paired_contrast(summary, "a", "b", margin=margin)["tost_p"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    contrasts = pd.read_csv(CONTRASTS)
    rows = []
    for _, c in contrasts[contrasts.kind == "paired"].iterrows():
        diff = np.array(json.loads(c.per_seed_differences))
        n = len(diff)
        delta = abs(diff.mean()) + stats.t.ppf(0.95, n - 1) * diff.std(ddof=1) / np.sqrt(n)
        locked = tost_p(diff, 1.0)
        # The stored differences are rounded to 1e-6, so allow a small relative error.
        if not np.isclose(locked, c.tost_p, rtol=1e-2, atol=1e-12):
            raise RuntimeError(f"{c.a} - {c.b}: TOST p {locked} != locked {c.tost_p}")
        below, above = tost_p(diff, delta * (1 - 1e-6)), tost_p(diff, delta * (1 + 1e-6))
        if not below >= 0.05 > above:
            raise RuntimeError(f"{c.a} - {c.b}: margin {delta} is not the 0.05 boundary")
        rows.append({"role": c.role, "a": c.a, "b": c.b, "metric": c.metric, "n": n,
                     "mean_difference": diff.mean(), "sd_difference": diff.std(ddof=1),
                     "ci95_low": c.ci95_low, "ci95_high": c.ci95_high,
                     "tost_p_margin_1": c.tost_p, "min_margin": delta,
                     "min_margin_fraction_of_locked": delta / 1.0})
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "min_margins.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
