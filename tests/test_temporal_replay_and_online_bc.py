import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.quantum_agent import QuantumDDPGAgent, SequenceReplayBuffer
from agents.classical_agent import ClassicalDDPGAgent
from data.vitaldb_loader import PatientSequenceDataset


def test_patient_sequence_dataset_is_chronological_and_boundary_safe():
    states = np.array([[1.0], [2.0], [10.0], [3.0]], dtype=np.float32)
    next_states = np.array([[2.0], [3.0], [11.0], [4.0]], dtype=np.float32)
    data = {
        "states": states,
        "actions": np.zeros((4, 2), dtype=np.float32),
        "rewards": np.zeros((4, 1), dtype=np.float32),
        "next_states": next_states,
        "dones": np.zeros((4, 1), dtype=np.float32),
        "caseids": np.array([1, 1, 2, 1]),
    }

    dataset = PatientSequenceDataset(data, sequence_length=3)

    np.testing.assert_array_equal(dataset.states[0, :, 0], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(dataset.states[1, :, 0], [1.0, 1.0, 2.0])
    np.testing.assert_array_equal(dataset.states[2, :, 0], [10.0, 10.0, 10.0])
    np.testing.assert_array_equal(dataset.states[3, :, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(dataset.next_states[3, :, 0], [2.0, 3.0, 4.0])


def test_patient_sequence_dataset_accepts_dense_precomputed_windows():
    states = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    next_states = states + 1
    data = {
        "states": states,
        "actions": np.zeros((2, 2), dtype=np.float32),
        "rewards": np.zeros(2, dtype=np.float32),
        "next_states": next_states,
        "dones": np.array([False, True]),
        "caseids": np.array([10, 10]),
    }
    dataset = PatientSequenceDataset(data, sequence_length=4)
    np.testing.assert_array_equal(dataset.states.numpy(), states)
    np.testing.assert_array_equal(dataset.next_states.numpy(), next_states)


def test_sequence_replay_emits_induction_steps_and_resets_episode():
    replay = SequenceReplayBuffer(capacity=10, sequence_length=3, seed=0)
    replay.push(np.array([1.0]), np.array([0.1]), 0.0, np.array([2.0]), False)
    replay.push(np.array([2.0]), np.array([0.2]), 0.0, np.array([3.0]), True)
    replay.push(np.array([9.0]), np.array([0.3]), 0.0, np.array([10.0]), False)

    assert len(replay) == 3
    np.testing.assert_array_equal(replay.buffer[0].states[:, 0], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(replay.buffer[1].states[:, 0], [1.0, 1.0, 2.0])
    np.testing.assert_array_equal(replay.buffer[1].next_states[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(replay.buffer[2].states[:, 0], [9.0, 9.0, 9.0])


def test_sequence_replay_accepts_observation_dense_control_windows():
    replay = SequenceReplayBuffer(capacity=10, sequence_length=4, seed=0)
    state_window = np.arange(8, dtype=np.float32).reshape(4, 2)
    next_window = state_window + 10
    replay.push(state_window, np.array([0.1, 0.2]), 1.0, next_window, True)
    assert len(replay) == 1
    np.testing.assert_array_equal(replay.buffer[0].states, state_window)
    np.testing.assert_array_equal(replay.buffer[0].next_states, next_window)


def test_online_td3_bc_reports_bc_loss_and_updates_encoder():
    config = {
        "quantum": {
            "n_qubits": 2,
            "n_layers": 1,
            "action_scale": 30.0,
            "action_prior": [0.5, 0.075],
            "residual_init_scale": 0.02,
        },
        "encoder": {
            "transformer": {
                "d_model": 8,
                "n_heads": 2,
                "num_layers": 1,
                "dropout": 0.0,
            }
        },
        "networks": {"critic": {"hidden_dims": [8, 8]}},
        "training": {
            "batch_size": 2,
            "policy_delay": 1,
            "warmup_steps": 0,
            "online_bc_weight": 1.0,
            "online_rl_scale": 0.05,
            "bc_action_weights": [1.0, 4.0],
        },
    }
    agent = QuantumDDPGAgent(
        state_dim=3,
        action_dim=2,
        config=config,
        encoder_type="transformer",
        sequence_length=2,
        seed=0,
    )
    expert_states = np.stack([
        np.zeros((2, 3), dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
    ])
    expert_actions = np.array([[0.2, 0.3], [0.4, 0.5]], dtype=np.float32)
    agent.configure_online_bc(
        expert_states, expert_actions, weight=1.0, action_weights=[1.0, 4.0])
    agent.replay_buffer.push(
        np.zeros(3), np.array([0.2, 0.3]), 1.0, np.ones(3), False)
    agent.replay_buffer.push(
        np.ones(3), np.array([0.4, 0.5]), 1.0, np.ones(3) * 2, False)

    encoder_before = [parameter.detach().clone() for parameter in agent.encoder.parameters()]
    metrics = agent.update()

    assert metrics["actor_bc_loss"] > 0.0
    assert metrics["actor_rl_coefficient"] > 0.0
    assert any(
        not torch.equal(before, after)
        for before, after in zip(encoder_before, agent.encoder.parameters())
    )


def test_classical_large_uses_same_temporal_td3_bc_pipeline():
    config = {
        "encoder": {
            "d_model": 8,
            "n_heads": 2,
            "num_layers": 1,
            "output_dim": 6,
            "dropout": 0.0,
        },
        "networks": {
            "actor": {"hidden_dims": [12, 8], "activation": "relu"},
            "critic": {"hidden_dims": [8, 8]},
        },
        "training": {
            "batch_size": 2,
            "policy_delay": 1,
            "warmup_steps": 0,
            "online_bc_weight": 1.0,
            "online_rl_scale": 0.05,
            "bc_action_weights": [1.0, 4.0],
        },
    }
    agent = ClassicalDDPGAgent(
        state_dim=3,
        action_dim=2,
        config=config,
        encoder_type="transformer",
        sequence_length=2,
        seed=0,
    )
    expert_states = np.stack([
        np.zeros((2, 3), dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
    ])
    expert_actions = np.array([[0.2, 0.3], [0.4, 0.5]], dtype=np.float32)
    agent.configure_online_bc(
        expert_states, expert_actions, weight=1.0, action_weights=[1.0, 4.0])
    agent.replay_buffer.push(
        np.zeros(3), np.array([0.2, 0.3]), 1.0, np.ones(3), False)
    agent.replay_buffer.push(
        np.ones(3), np.array([0.4, 0.5]), 1.0, np.ones(3) * 2, False)

    encoder_before = [parameter.detach().clone() for parameter in agent.encoder.parameters()]
    metrics = agent.update()

    assert metrics["actor_bc_loss"] > 0.0
    assert metrics["actor_rl_coefficient"] > 0.0
    assert any(
        not torch.equal(before, after)
        for before, after in zip(encoder_before, agent.encoder.parameters())
    )
