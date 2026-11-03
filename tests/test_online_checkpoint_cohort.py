"""The simulator-stage checkpoint cohort must evaluate every patient it lists.

v1 called ``env.reset(options={'patient': p})``, but DualDrugEnv ignores
``options``, so the "30-patient" checkpoint cohort re-ran the last training
patient 30 times. The checkpoint picked on it (best_mdape.pt) became the final
model. These tests run a tiny stage 2 and record which patients were simulated.
"""
import copy
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from agents.classical_agent import ClassicalDDPGAgent  # noqa: E402
from environment.dual_drug_env import DualDrugEnv  # noqa: E402
from models.pharmacokinetics import PatientParameters  # noqa: E402
from environment.env_seeds import checkpoint_env_seed, training_env_seed  # noqa: E402
from train_hybrid import stage2_online_finetuning  # noqa: E402


class _RecordingEnv(DualDrugEnv):
    built = []
    seeds = []

    def __init__(self, *args, patient=None, **kwargs):
        _RecordingEnv.built.append(None if patient is None else patient.weight)
        _RecordingEnv.seeds.append((None if patient is None else patient.weight,
                                    kwargs.get("seed")))
        super().__init__(*args, patient=patient, **kwargs)


def _agent():
    with open("config/hyperparameters.yaml", encoding="utf-8") as stream:
        config = copy.deepcopy(yaml.safe_load(stream))
    config["training"]["batch_size"] = 32
    return ClassicalDDPGAgent(state_dim=13, action_dim=2, config=config,
                              encoder_type="none", seed=0)


def test_checkpoint_cohort_simulates_each_validation_patient(tmp_path):
    torch.manual_seed(0)
    train = [PatientParameters(age=50, weight=71.0, height=170, gender="male")]
    validation = [PatientParameters(age=60, weight=w, height=165, gender="female")
                  for w in (52.0, 83.0)]
    dirs = {"stage2": tmp_path / "stage2", "base": tmp_path}
    dirs["stage2"].mkdir()
    _RecordingEnv.built = []
    _RecordingEnv.seeds = []
    stage2_online_finetuning(
        agent=_agent(), n_episodes=1, warmup_episodes=0, dirs=dirs,
        eval_interval=1, seed=0, env_class=_RecordingEnv,
        device=torch.device("cpu"), n_eval_patients=2,
        observation_interval_seconds=60, control_interval_seconds=60,
        train_patients=train, validation_patients=validation)
    assert 52.0 in _RecordingEnv.built
    assert 83.0 in _RecordingEnv.built


def test_simulator_episodes_and_checkpoints_use_disjoint_env_seeds(tmp_path):
    torch.manual_seed(0)
    train = [PatientParameters(age=50, weight=71.0, height=170, gender="male")]
    validation = [PatientParameters(age=60, weight=w, height=165, gender="female")
                  for w in (52.0, 83.0)]
    dirs = {"stage2": tmp_path / "stage2", "base": tmp_path}
    dirs["stage2"].mkdir()
    _RecordingEnv.built = []
    _RecordingEnv.seeds = []
    stage2_online_finetuning(
        agent=_agent(), n_episodes=1, warmup_episodes=0, dirs=dirs,
        eval_interval=1, seed=4, env_class=_RecordingEnv,
        device=torch.device("cpu"), n_eval_patients=2, validation_seed=1000,
        observation_interval_seconds=60, control_interval_seconds=60,
        train_patients=train, validation_patients=validation)
    seeds = dict(_RecordingEnv.seeds)
    assert seeds[71.0] == training_env_seed(4, 0)
    assert seeds[52.0] == checkpoint_env_seed(1000, 0)
    assert seeds[83.0] == checkpoint_env_seed(1000, 1)


def _run_stage2(agent, tmp_path):
    train = [PatientParameters(age=50, weight=71.0, height=170, gender="male")]
    dirs = {"stage2": tmp_path / "stage2", "base": tmp_path}
    dirs["stage2"].mkdir(exist_ok=True)
    stage2_online_finetuning(
        agent=agent, n_episodes=1, warmup_episodes=0, dirs=dirs,
        eval_interval=5, seed=0, env_class=DualDrugEnv,
        device=torch.device("cpu"), n_eval_patients=1,
        observation_interval_seconds=60, control_interval_seconds=60,
        train_patients=train, validation_patients=train)


def test_simulator_stage_trains_every_arm_without_dropout(tmp_path):
    # v1 left the mode to stage 1, so only the online-only arm (no stage 1)
    # trained with dropout on.
    agent = _agent()
    agent.actor.train()
    modes = []
    update = agent.update

    def spy():
        modes.append(agent.actor.training)
        return update()

    agent.update = spy
    _run_stage2(agent, tmp_path)
    assert modes and not any(modes)


def test_simulator_stage_rejects_a_discount_that_breaks_the_shaping(tmp_path):
    agent = _agent()
    agent.gamma = 0.95
    import pytest
    with pytest.raises(ValueError, match="gamma"):
        _run_stage2(agent, tmp_path)
