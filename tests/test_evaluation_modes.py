import sys
from pathlib import Path

import numpy as np
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environment.patient_simulator import PatientParameters
from evaluation.per_patient_eval import evaluate_patient


class _DropoutPolicy:
    def __init__(self):
        self.actor = nn.Dropout(0.5)
        self.encoder = nn.Dropout(0.5)
        self.actor.train()
        self.encoder.eval()
        self.observed_modes = []

    def reset(self):
        pass

    def select_action(self, state, deterministic=True):
        self.observed_modes.append((self.actor.training, self.encoder.training))
        return np.array([6.0, 0.1], dtype=np.float32)


def test_patient_evaluation_disables_dropout_and_restores_modes():
    policy = _DropoutPolicy()
    patient = PatientParameters(
        age=45.0, weight=70.0, height=170.0, gender="male")

    evaluate_patient(policy, patient, seed=3, steps=2, variability=None)

    assert policy.observed_modes == [(False, False), (False, False)]
    assert policy.actor.training is True
    assert policy.encoder.training is False
