"""Runner of the single-drug baseline (experiments/run_propofol_only.py)."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

import recorded_remifentanil as rr  # noqa: E402
import run_pilot  # noqa: E402
import run_propofol_only as rpo  # noqa: E402

CAMPAIGN_RUN = ROOT / "logs/revision_v3/campaign/primary/classical_matched/seed_0"


def _config(**changes):
    config = vars(run_pilot.build_parser().parse_args(["--out", "campaign/run"]))
    config.update(skip_offline=True, clean_temporal_protocol=True, skip_baselines=True,
                  residual_pid="pid.json", residual_range="4.0,0.05",
                  classical_actor_lr=0.0001, online_episodes=150, sequence_length=60,
                  arms="classical_matched", classical_action_prior="0.5,0.5")
    config.update(changes)
    return config


def test_campaign_arguments_round_trip_with_a_new_out_and_the_suffix(tmp_path):
    argv = rpo.campaign_argv(_config(), tmp_path / "unit")
    parsed = vars(run_pilot.build_parser().parse_args(argv))
    assert parsed == _config(out=str(tmp_path / "unit"), method_suffix="_ppfonly")


def test_campaign_arguments_refuse_what_the_command_line_cannot_express(tmp_path):
    with pytest.raises(ValueError, match="round-trip"):
        rpo.campaign_argv(_config(classical_action_prior=None), tmp_path / "unit")


@pytest.mark.skipif(not CAMPAIGN_RUN.exists(), reason="campaign logs not present")
def test_the_real_campaign_run_round_trips(tmp_path):
    config = json.loads((CAMPAIGN_RUN / "run_config.json").read_text(encoding="utf-8"))
    argv = rpo.campaign_argv(config, tmp_path / "unit")
    assert "--skip_offline" in argv and "--method_suffix" in argv


def _write_run(folder, split, timings):
    folder.mkdir(parents=True)
    (folder / "patient_split.json").write_text(json.dumps(split))
    (folder / "timings.json").write_text(json.dumps(timings))


SPLIT = {"train_caseids": [1, 2], "val_caseids": [3], "test_caseids": [4, 5],
         "evaluated_caseids": [4, 5], "online_caseids": [2, 1], "checkpoint_caseids": [3]}


def test_verification_accepts_a_run_with_the_campaign_patients_and_network(tmp_path):
    _write_run(tmp_path / "campaign", SPLIT,
               {"classical_matched": {"actor_params": 6991, "critic_params": 150018}})
    _write_run(tmp_path / "unit", SPLIT,
               {"classical_matched_ppfonly": {"actor_params": 6991, "critic_params": 150018}})
    rpo.verify_against_campaign(tmp_path / "unit", tmp_path / "campaign")
    verification = json.loads((tmp_path / "unit" / "verification.json").read_text())
    assert verification["identical"]
    assert verification["environment"] == "RecordedRemifentanilEnv"


@pytest.mark.parametrize("split, actor", [({**SPLIT, "online_caseids": [1, 2]}, 6991),
                                          (SPLIT, 7000)])
def test_verification_refuses_other_patients_or_another_network(tmp_path, split, actor):
    _write_run(tmp_path / "campaign", SPLIT,
               {"classical_matched": {"actor_params": 6991, "critic_params": 150018}})
    _write_run(tmp_path / "unit", split,
               {"classical_matched_ppfonly": {"actor_params": actor, "critic_params": 150018}})
    with pytest.raises(RuntimeError, match="campaign"):
        rpo.verify_against_campaign(tmp_path / "unit", tmp_path / "campaign")
    assert not (tmp_path / "unit" / "verification.json").exists()


def test_unverified_training_units_are_listed(tmp_path):
    units = rpo.build_units(tmp_path, seeds=range(2))
    for unit in units:
        unit.out.mkdir(parents=True)
    (units[1].out / "verification.json").write_text("{}")
    assert rpo.unverified(units) == ["classical_matched_ppfonly/seed1"]


def test_locked_schedules_are_checked_against_their_hash(tmp_path):
    schedules = {16: rr.RemifentanilSchedule(rates=(0.5,), hold=0.5)}
    rr.save_schedules(schedules, tmp_path / rpo.SCHEDULES, {"rule": "test"})
    rpo.write_hash(tmp_path)
    assert rpo.load_locked_schedules(tmp_path) == schedules
    path = tmp_path / rpo.SCHEDULES
    path.write_text(path.read_text().replace("0.5", "0.6"))
    with pytest.raises(RuntimeError, match="sha256"):
        rpo.load_locked_schedules(tmp_path)


def test_units_are_the_baseline_and_one_training_run_per_seed(tmp_path):
    units = rpo.build_units(tmp_path, seeds=range(3), campaign=tmp_path / "campaign")
    assert units[2].command[units[2].command.index("--campaign") + 1] == str(
        (tmp_path / "campaign").resolve())
    assert [u.label for u in units] == ["pid_tuned_ppfonly", "classical_matched_ppfonly/seed0",
                                        "classical_matched_ppfonly/seed1",
                                        "classical_matched_ppfonly/seed2"]
    assert units[2].methods == frozenset({"classical_matched_ppfonly"})
    assert units[2].out == tmp_path / "units" / "classical_matched_ppfonly_seed1"
    assert units[2].command[-2:] == ("--one", "train:1")
    assert all(u.n_patients == 600 for u in units)


def test_run_pilot_is_patched_for_the_recorded_remifentanil_and_restored(tmp_path, monkeypatch):
    # The loader returns its class-level cache first, so a one-row table keeps the test offline.
    table = pd.DataFrame({"subjectid": [5], "age": [63.0], "sex": ["F"], "height": [160.0],
                          "weight": [58.0], "bmi": [22.7]}, index=pd.Index([16], name="caseid"))
    monkeypatch.setattr(rpo._ClinicalWithCaseid.__bases__[0], "_clinical_cache", table)
    schedules = {16: rr.RemifentanilSchedule(rates=(0.5,), hold=0.5)}
    originals = (run_pilot.VitalDBLoader, run_pilot.patient_from_record, run_pilot.DualDrugEnv)
    try:
        rpo.install_into_run_pilot(schedules)
        clinical = run_pilot.VitalDBLoader(cache_dir=str(tmp_path)).load_clinical_info()
        patient = run_pilot.patient_from_record(clinical.loc[16].to_dict())
        assert patient.caseid == 16 and patient.remifentanil == schedules[16]
        assert run_pilot.DualDrugEnv is rr.RecordedRemifentanilEnv
    finally:
        rpo.restore_run_pilot()
    assert (run_pilot.VitalDBLoader, run_pilot.patient_from_record, run_pilot.DualDrugEnv) == originals
    assert rr.per_patient_eval.DualDrugEnv is rr._ORIGINAL_ENV
