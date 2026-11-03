"""Pre-specified run-level statistics of the training campaigns (experiments/campaign_statistics.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from campaign_statistics import paired_contrast, run_summary  # noqa: E402


def _results(method, seed_means, n_patients=50, spread=0.3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s, mean in enumerate(seed_means):
        for p in range(n_patients):
            rows.append({"method": method, "patient_id": p, "seed": s,
                         "mdape": mean + spread * rng.standard_normal(),
                         "tit_45_55": 90.0, "tit_40_60": 98.0,
                         "n_bis_lt30": 0, "n_bis_gt70": 2,
                         "mean_ppf": 6.0, "mean_remi": 0.1, "mdape_true": mean})
    return pd.DataFrame(rows)


def test_run_summary_has_one_row_per_method_and_seed():
    raw = pd.concat([_results("a", np.full(10, 4.0)), _results("b", np.full(10, 5.0))])
    summary = run_summary(raw)
    assert len(summary) == 20
    assert summary.groupby("method").mean_mdape.mean().round(1).to_dict() == {"a": 4.0, "b": 5.0}


def test_clear_difference_is_detected_and_not_equivalent():
    a = np.array([4.0, 4.2, 3.9, 4.1, 4.3, 3.8, 4.0, 4.2, 3.9, 4.1])
    raw = pd.concat([_results("a", a), _results("b", a + 2.0)])
    result = paired_contrast(run_summary(raw), "a", "b", margin=1.0, n_boot=4000)
    assert result["n_pairs"] == 10
    assert result["mean_difference"] == pytest.approx(-2.0, abs=0.05)
    assert result["ci95_high"] < 0
    assert result["wilcoxon_exact_p"] == pytest.approx(2 / 1024, rel=1e-6)
    assert not result["equivalent"]


def test_tiny_difference_within_margin_is_equivalent():
    rng = np.random.default_rng(1)
    a = 4.0 + 0.2 * rng.standard_normal(10)
    raw = pd.concat([_results("a", a), _results("b", a + 0.05 + 0.1 * rng.standard_normal(10))])
    result = paired_contrast(run_summary(raw), "a", "b", margin=1.0, n_boot=4000)
    assert result["equivalent"]
    assert result["tost_p"] < 0.05


def test_contrast_refuses_unpaired_seeds():
    raw = pd.concat([_results("a", np.full(10, 4.0)), _results("b", np.full(9, 4.0))])
    with pytest.raises(ValueError):
        paired_contrast(run_summary(raw), "a", "b", margin=1.0)


def test_moved_aside_attempts_are_not_loaded(tmp_path):
    # The runner keeps stalled and interrupted attempts as seed_k.stalled-N and
    # seed_k.interrupted-N; only seed_k holds the completed result.
    import pandas as pd
    from campaign_statistics import load_campaign
    rows = pd.DataFrame({"method": ["quantum_full"] * 2, "patient_id": [1, 2],
                         "seed": [0, 0], "mdape": [5.0, 6.0]})
    for name in ("seed_0", "seed_0.stalled-1", "seed_0.interrupted-1"):
        folder = tmp_path / "primary" / "quantum_full" / name
        folder.mkdir(parents=True)
        rows.to_csv(folder / "results.csv", index=False)
    loaded = load_campaign(tmp_path)
    assert len(loaded) == 2
    assert set(loaded.source) == {"primary/quantum_full/seed_0"}
