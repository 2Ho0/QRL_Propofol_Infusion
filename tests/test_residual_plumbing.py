"""Residual mode and reward weights must reach every environment (v3, D14-D15).

Training (sanity, rollout and checkpoint environments) and the final
per-patient evaluation build their environments in different places. A
setting that reaches only one of them would train one controller and
evaluate another.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from agents.baseline_agents import PIDController  # noqa: E402
from agents.classical_agent import ClassicalDDPGAgent  # noqa: E402
from environment.dual_drug_env import DualDrugEnv  # noqa: E402
from evaluation.per_patient_eval import evaluate_cohort, evaluate_patient  # noqa: E402
from models.pharmacokinetics import PatientParameters  # noqa: E402
from train_hybrid import stage2_online_finetuning  # noqa: E402

PID = json.loads(Path("logs/revision_v2/pid_tuning/selected_pid.json").read_text())["parameters"]
V3 = {"residual_pid": PID, "residual_range": (4.0, 0.05),
      "reward_w3": 0.03, "reward_w_smooth": 0.5}


class _RecordingEnv(DualDrugEnv):
    kwargs = []

    def __init__(self, *args, **kwargs):
        _RecordingEnv.kwargs.append(kwargs)
        super().__init__(*args, **kwargs)


class _Middle:
    """Zero residual: always the middle of the action range."""
    sequence_length = 1

    def select_action(self, state, deterministic=True):
        return np.array([6.0, 1.0])


def _agent():
    with open("config/hyperparameters.yaml", encoding="utf-8") as stream:
        config = copy.deepcopy(yaml.safe_load(stream))
    config["training"]["batch_size"] = 32
    return ClassicalDDPGAgent(state_dim=13, action_dim=2, config=config,
                              encoder_type="none", seed=0)


def test_stage2_builds_every_environment_with_the_v3_settings(tmp_path):
    torch.manual_seed(0)
    patients = [PatientParameters(age=50, weight=71.0, height=170, gender="male")]
    dirs = {"stage2": tmp_path / "stage2", "base": tmp_path}
    dirs["stage2"].mkdir()
    _RecordingEnv.kwargs = []
    stage2_online_finetuning(
        agent=_agent(), n_episodes=1, warmup_episodes=0, dirs=dirs, eval_interval=1,
        seed=0, env_class=_RecordingEnv, device=torch.device("cpu"), n_eval_patients=1,
        observation_interval_seconds=60, control_interval_seconds=60,
        train_patients=patients, validation_patients=patients, env_kwargs=V3)
    assert len(_RecordingEnv.kwargs) >= 3        # sanity, training, checkpoint
    for kwargs in _RecordingEnv.kwargs:
        assert kwargs["residual_pid"] == PID
        assert kwargs["reward_w3"] == 0.03
        assert kwargs["reward_w_smooth"] == 0.5


def test_evaluate_patient_records_the_dose_the_residual_environment_applied():
    patient = PatientParameters(age=60, weight=58.0, height=160, gender="female")
    common = dict(seed=7, steps=30, observation_interval_seconds=10,
                  control_interval_seconds=60, variability=None)
    _, bis_pid, ppf_pid, rftn_pid = evaluate_patient(PIDController(**PID), patient, **common)
    _, bis_res, ppf_res, rftn_res = evaluate_patient(_Middle(), patient, env_kwargs=V3, **common)
    np.testing.assert_allclose(ppf_res, ppf_pid, atol=1e-5)
    np.testing.assert_allclose(rftn_res, rftn_pid, atol=1e-5)
    np.testing.assert_allclose(bis_res, bis_pid, atol=1e-5)
    assert not np.allclose(ppf_res, 6.0)


def test_evaluate_cohort_forwards_the_environment_settings():
    cohort = [(101, PatientParameters(age=45, weight=80.0, height=180, gender="male")),
              (102, PatientParameters(age=70, weight=55.0, height=158, gender="female"))]
    common = dict(seed=0, environment_seed=0, observation_interval_seconds=10,
                  control_interval_seconds=60, variability=None)
    pid_rows = evaluate_cohort(PIDController(**PID), "pid_tuned", cohort, **common)
    res_rows = evaluate_cohort(_Middle(), "zero_residual", cohort, env_kwargs=V3, **common)
    for pid_row, res_row in zip(pid_rows, res_rows):
        assert res_row.mdape == pytest.approx(pid_row.mdape, abs=1e-6)
        assert res_row.mean_ppf == pytest.approx(pid_row.mean_ppf, abs=1e-6)
        assert res_row.mean_remi == pytest.approx(pid_row.mean_remi, abs=1e-6)


class _AppliedRecordingEnv(DualDrugEnv):
    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed_used = kwargs.get("seed")
        self.applied_ppf = []
        _AppliedRecordingEnv.instances.append(self)

    def step(self, action):
        result = super().step(action)
        self.applied_ppf.append(result[4]["applied_action"][0])
        return result


def test_the_training_log_reports_the_applied_dose(tmp_path, capsys):
    """In residual mode the policy's action is a residual, not a dose (review, LOW)."""
    import re
    from environment.env_seeds import training_env_seed

    torch.manual_seed(0)
    patients = [PatientParameters(age=50, weight=71.0, height=170, gender="male")]
    dirs = {"stage2": tmp_path / "stage2", "base": tmp_path}
    dirs["stage2"].mkdir()
    _AppliedRecordingEnv.instances = []
    stage2_online_finetuning(
        agent=_agent(), n_episodes=1, warmup_episodes=0, dirs=dirs, eval_interval=5,
        seed=0, env_class=_AppliedRecordingEnv, device=torch.device("cpu"),
        n_eval_patients=1, observation_interval_seconds=60, control_interval_seconds=60,
        train_patients=patients, validation_patients=patients, env_kwargs=V3)
    printed = re.search(r"Main Phase.*?Avg propofol dose: ([0-9.]+)",
                        capsys.readouterr().out, re.S)
    training = [env for env in _AppliedRecordingEnv.instances
                if env.seed_used == training_env_seed(0, 0)]
    assert len(training) == 1
    assert float(printed.group(1)) == pytest.approx(np.mean(training[0].applied_ppf), abs=0.006)
