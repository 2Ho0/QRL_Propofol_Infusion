"""Pre-registered v3 analysis (P1-P3 with Holm, secondary contrasts)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from campaign_statistics import run_summary  # noqa: E402
from analyze_v3_campaign import (  # noqa: E402
    analyse, holm, main, method_summary, one_sample_contrast)

QF = np.array([4.05, 4.10, 3.98, 4.12, 4.07, 4.01, 4.09, 4.03, 4.11, 4.00])


def _results(method, seed_means, n_patients=40, spread=0.3, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    rows = []
    for s, mean in enumerate(seed_means):
        for p in range(n_patients):
            rows.append({"method": method, "patient_id": p, "seed": s,
                         "mdape": mean + spread * rng.standard_normal(),
                         "tit_45_55": 90.0, "tit_40_60": 98.0, "n_bis_lt30": 0,
                         "n_bis_gt70": 1, "mean_ppf": 8.0, "mean_remi": 0.12,
                         "mdape_true": mean})
    return pd.DataFrame(rows)


def _pid(value=4.18):
    return _results("pid_tuned", [value], spread=0.0)


def test_one_sample_contrast_compares_every_run_with_the_single_pid_value():
    raw = pd.concat([_results("quantum_full", QF, spread=0.0), _pid()])
    result = one_sample_contrast(run_summary(raw), "quantum_full", "pid_tuned", n_boot=4000)
    assert result["n_runs"] == 10
    assert result["mean_difference"] == pytest.approx(QF.mean() - 4.18)
    assert result["ci95_high"] < 0
    assert result["runs_better"] == 10
    assert result["wilcoxon_exact_p"] == pytest.approx(2 / 2 ** 10)


def test_a_reference_evaluated_more_than_once_is_refused():
    raw = pd.concat([_results("quantum_full", QF), _results("pid_tuned", [4.18, 4.18])])
    with pytest.raises(ValueError, match="exactly once"):
        one_sample_contrast(run_summary(raw), "quantum_full", "pid_tuned")


def test_holm_adjustment():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_primary_rows_come_first_and_carry_holm_p():
    raw = pd.concat([
        _results("quantum_full", QF), _pid(),
        _results("classical_matched", QF + 0.2, rng_seed=1),
        _results("quantum_disabled", QF + 0.1, rng_seed=2),
        _results("classical_large", QF + 0.3, rng_seed=3)])
    table = analyse(raw, n_boot=2000, expected_patients=40)
    assert list(table.role[:3]) == ["primary"] * 3
    assert table.loc[table.role == "primary", "holm_p"].notna().all()
    assert table.loc[table.role == "secondary", "holm_p"].isna().all()


def test_holm_is_withheld_until_all_primary_contrasts_exist():
    raw = pd.concat([_results("quantum_full", QF), _pid()])
    table = analyse(raw, n_boot=2000)
    assert table.holm_p.isna().all()


def test_main_writes_the_analysis_tables(tmp_path):
    for method, means in (("quantum_full", QF), ("classical_matched", QF + 0.2)):
        for s, mean in enumerate(means):
            folder = tmp_path / "primary" / method / f"seed_{s}"
            folder.mkdir(parents=True)
            frame = _results(method, [mean], rng_seed=s)
            frame["seed"] = s
            if method == "quantum_full" and s == 0:
                frame = pd.concat([frame, _pid()])
            frame.to_csv(folder / "results.csv", index=False)
    main(["--root", str(tmp_path), "--n_boot", "500"])
    contrasts = pd.read_csv(tmp_path / "analysis" / "contrasts.csv")
    primary = contrasts[contrasts.role == "primary"]
    assert set(zip(primary.a, primary.b)) == {("quantum_full", "pid_tuned"),
                                              ("quantum_full", "classical_matched")}



def _full_design(cm_seeds=10):
    return pd.concat([
        _results("quantum_full", QF), _pid(),
        _results("classical_matched", (QF + 0.2)[:cm_seeds], rng_seed=1),
        _results("quantum_disabled", QF + 0.1, rng_seed=2)])


def test_holm_waits_for_the_complete_design():
    table = analyse(_full_design(), n_boot=1000)        # 40 patients, not 600
    assert not table.complete.any()
    assert table.holm_p.isna().all()


def test_a_missing_seed_drops_only_that_pair():
    table = analyse(_full_design(cm_seeds=9), n_boot=1000, expected_patients=40)
    row = table[(table.a == "quantum_full") & (table.b == "classical_matched")
                & (table.role == "primary")].iloc[0]
    assert row.n_pairs == 9 and not row.complete
    other = table[(table.b == "quantum_disabled") & (table.role == "primary")].iloc[0]
    assert other.n_pairs == 10 and other.complete
    assert table.loc[table.role == "primary", "holm_p"].isna().all()


def test_holm_refuses_nan_p_values():
    with pytest.raises(ValueError, match="finite"):
        holm([0.01, float("nan")])


def test_noise_free_contrasts_accompany_the_primary_ones():
    table = analyse(_full_design(), n_boot=1000, expected_patients=40)
    true_bis = table[table.role == "primary_true_bis"]
    assert len(true_bis) == 3
    assert set(true_bis.metric) == {"mean_mdape_true"}
    assert true_bis.holm_p.isna().all()


def test_method_summary_reports_median_doses_and_noise_free_mdape():
    summary = method_summary(_full_design()).set_index("method")
    assert summary.loc["quantum_full", "median_ppf"] == pytest.approx(8.0)
    assert summary.loc["quantum_full", "median_remi"] == pytest.approx(0.12)
    assert summary.loc["quantum_full", "mean_mdape_true"] == pytest.approx(QF.mean())
    assert summary.loc["quantum_full", "n_runs"] == 10
