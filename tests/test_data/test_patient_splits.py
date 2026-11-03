"""Patient-level split tests."""
import numpy as np
import pytest

from data.patient_splits import split_by_patient


def test_no_patient_appears_in_more_than_one_split():
    caseids = np.repeat([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 20)
    tr, va, te = split_by_patient(caseids, ratios=(0.7, 0.1, 0.2), seed=42)
    s_tr, s_va, s_te = (set(caseids[i]) for i in (tr, va, te))
    assert s_tr.isdisjoint(s_va)
    assert s_tr.isdisjoint(s_te)
    assert s_va.isdisjoint(s_te)


def test_every_transition_is_assigned_exactly_once():
    caseids = np.repeat(np.arange(10), 20)
    tr, va, te = split_by_patient(caseids, seed=42)
    allocated = np.concatenate([tr, va, te])
    assert np.array_equal(np.sort(allocated), np.arange(len(caseids)))


def test_split_is_by_patient_count_not_transition_count():
    """Patient 0 holds 1000 transitions and the rest hold 10. A transition-level
    split would put patient 0 alone in train; a patient-level split must give
    7/1/2 patients regardless of their lengths."""
    caseids = np.concatenate([np.zeros(1000, dtype=int),
                              np.repeat(np.arange(1, 10), 10)])
    tr, va, te = split_by_patient(caseids, ratios=(0.7, 0.1, 0.2), seed=0)
    assert len(set(caseids[tr])) == 7
    assert len(set(caseids[va])) == 1
    assert len(set(caseids[te])) == 2


def test_split_is_deterministic_given_a_seed():
    caseids = np.repeat(np.arange(10), 5)
    for a, b in zip(split_by_patient(caseids, seed=7),
                    split_by_patient(caseids, seed=7)):
        assert np.array_equal(a, b)


def test_different_seeds_give_different_partitions():
    caseids = np.repeat(np.arange(50), 5)
    a = set(caseids[split_by_patient(caseids, seed=0)[2]])
    b = set(caseids[split_by_patient(caseids, seed=1)[2]])
    assert a != b


def test_raises_when_too_few_patients_to_fill_every_split():
    with pytest.raises(ValueError, match="at least 3 patients"):
        split_by_patient(np.repeat([1, 2], 10), seed=0)


def test_raises_when_ratios_do_not_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        split_by_patient(np.repeat(np.arange(5), 4), ratios=(0.5, 0.2, 0.2))


def test_every_split_is_non_empty_even_with_few_patients():
    """Rounding must never starve a split; 3 patients at 70/10/20 is the
    tightest case."""
    caseids = np.repeat([1, 2, 3], 10)
    tr, va, te = split_by_patient(caseids, ratios=(0.7, 0.1, 0.2), seed=0)
    assert len(tr) and len(va) and len(te)


# --- sim v2: simulator-stage patients come from the training subjects ---
from data.patient_splits import sample_online_patients  # noqa: E402


def test_online_patients_are_distinct_training_cases():
    train_ids = np.arange(1000, 3100)
    chosen = sample_online_patients(train_ids, n_episodes=30, seed=0)
    assert len(chosen) == 30
    assert len(set(chosen.tolist())) == 30
    assert set(chosen.tolist()) <= set(train_ids.tolist())


def test_online_patients_depend_on_the_training_seed():
    train_ids = np.arange(1000, 3100)
    first = sample_online_patients(train_ids, n_episodes=30, seed=0)
    again = sample_online_patients(train_ids, n_episodes=30, seed=0)
    other = sample_online_patients(train_ids, n_episodes=30, seed=1)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_online_patients_refuse_more_episodes_than_subjects():
    with pytest.raises(ValueError):
        sample_online_patients(np.arange(10), n_episodes=11, seed=0)
