"""Pre-specified analysis of the single-drug baseline (experiments/summarize_propofol_only.py)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import summarize_propofol_only as spo  # noqa: E402

COLUMNS = ["mdpe", "wobble", "divergence", "tit_45_55", "tit_40_60", "n_bis_lt30",
           "n_bis_gt70", "mean_ppf", "mean_remi", "mdpe_true", "mdape_true", "wobble_true",
           "divergence_true"]


def _rows(method, seeds, level, patients=4, spread=0.05):
    rng = np.random.default_rng(sum(map(ord, method)))
    rows = []
    for seed in seeds:
        shift = level + spread * rng.standard_normal()
        for patient in range(patients):
            row = {"method": method, "patient_id": 100 + patient, "seed": seed,
                   "mdape": shift + 0.01 * patient}
            row.update({c: 1.0 for c in COLUMNS})
            row["mdape_true"] = shift - 1.0
            rows.append(row)
    return rows


def _write(tmp_path, ppfonly_level=5.0):
    root = tmp_path / "study"
    seeds = range(10)
    new = pd.DataFrame(_rows("classical_matched_ppfonly", seeds, ppfonly_level)
                       + _rows("pid_tuned_ppfonly", [0], 5.5))
    root.mkdir()
    new.to_csv(root / "results_all.csv", index=False)
    for seed in seeds:
        unit = root / "units" / f"classical_matched_ppfonly_seed{seed}"
        unit.mkdir(parents=True)
        (unit / "verification.json").write_text(json.dumps({"identical": True}))
    campaign = pd.DataFrame(_rows("classical_matched", seeds, 4.1) + _rows("quantum_full", seeds, 4.1)
                            + _rows("pid_tuned", [0], 4.14) + _rows("tci", [0], 13.0))
    campaign["source"] = "campaign"
    campaign.to_csv(tmp_path / "campaign.csv", index=False)
    return root, tmp_path / "campaign.csv"


def test_summary_uses_the_campaign_arms_and_the_new_runs(tmp_path):
    root, campaign = _write(tmp_path)
    methods, contrasts, verdict = spo.summarise(root, campaign, n_boot=2000)
    assert set(methods.method) == {"classical_matched_ppfonly", "pid_tuned_ppfonly",
                                   "classical_matched", "quantum_full", "pid_tuned"}
    d1 = contrasts[(contrasts.label == "D1") & (contrasts.metric == "mean_mdape")].iloc[0]
    assert (d1.a, d1.b, d1.n_pairs) == ("classical_matched_ppfonly", "classical_matched", 10)
    assert d1.mean_difference == pytest.approx(0.9, abs=0.1)
    primary = contrasts[contrasts.role == "primary"]
    assert list(primary.label) == ["D1", "D2"] and primary.holm_p.notna().all()
    assert contrasts[contrasts.role == "true_bis"].holm_p.isna().all()
    assert verdict["row"] == "dual_drug_better"


def test_verdict_follows_the_pre_specified_rows():
    def d1(low, high, holm, tost):
        return {"ci95_low": low, "ci95_high": high, "holm_p": holm, "tost_p": tost}
    assert spo.verdict(d1(0.2, 0.6, 0.01, 0.001))["row"] == "dual_drug_better"
    assert spo.verdict(d1(-0.1, 0.3, 0.30, 0.001))["row"] == "equivalent"
    assert spo.verdict(d1(-0.6, -0.2, 0.01, 0.001))["row"] == "equivalent"
    assert spo.verdict(d1(-1.8, -1.2, 0.01, 0.9))["row"] == "single_drug_better"
    assert spo.verdict(d1(-0.5, 1.5, 0.30, 0.4))["row"] == "inconclusive"
    assert spo.verdict(d1(-0.6, -0.2, 0.01, 0.001))["single_drug_better_ci"] is True


def test_summary_refuses_runs_that_were_not_verified(tmp_path):
    root, campaign = _write(tmp_path)
    (root / "units" / "classical_matched_ppfonly_seed3" / "verification.json").unlink()
    with pytest.raises(RuntimeError, match="seed3"):
        spo.summarise(root, campaign, n_boot=200)


def test_main_refuses_an_existing_output(tmp_path):
    root, campaign = _write(tmp_path)
    (tmp_path / "out").mkdir()
    with pytest.raises(SystemExit, match="exists"):
        spo.main(["--root", str(root), "--campaign_raw", str(campaign),
                  "--out", str(tmp_path / "out")])
