"""Rebuilding the locked VitalDB cohort must keep the patient-level split.

The 2026-10-20 rate-unit fix changes which 10-s samples pass the rate filters,
so a fresh scan could select different cases and reshuffle every partition.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from prepare_clean_retraining_data import locked_caseids, split_differences  # noqa: E402

LOCKED = {"partitions": {"train": {"caseids": [5, 9]},
                         "validation": {"caseids": [7]},
                         "test": {"caseids": [14]}}}


def test_locked_caseids_cover_every_partition_in_scan_order():
    assert locked_caseids(LOCKED) == [5, 7, 9, 14]


def test_identical_partitions_have_no_differences():
    assert split_differences(LOCKED, LOCKED) == []


def test_a_case_moving_between_partitions_is_reported():
    moved = {"partitions": {"train": {"caseids": [5]},
                            "validation": {"caseids": [7, 9]},
                            "test": {"caseids": [14]}}}
    assert split_differences(moved, LOCKED) == ["train", "validation"]


def test_a_rebuild_uses_the_split_seed_of_the_locked_manifest():
    from prepare_clean_retraining_data import resolve_split_seed
    assert resolve_split_seed(None, {"split_seed": 1}) == 1
    assert resolve_split_seed(None, None) == 0
    assert resolve_split_seed(3, {"split_seed": 1}) == 3


def test_an_existing_manifest_is_not_overwritten(tmp_path):
    import pytest
    from prepare_clean_retraining_data import check_manifest_target
    locked = tmp_path / "dataset_manifest.json"
    locked.write_text("{}")
    with pytest.raises(FileExistsError):
        check_manifest_target(locked, validate_existing=False)
    check_manifest_target(locked, validate_existing=True)
    check_manifest_target(tmp_path / "new_manifest.json", validate_existing=False)


def test_a_bis_range_is_parsed_and_checked():
    import pytest
    from prepare_clean_retraining_data import parse_bis_range
    assert parse_bis_range(None) is None
    assert parse_bis_range([30, 100]) == (30.0, 100.0)
    with pytest.raises(ValueError):
        parse_bis_range([70, 30])
    with pytest.raises(ValueError):
        parse_bis_range([30, 120])
