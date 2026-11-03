"""The variant arms (quantum critic, ansatz families, Fourier surrogate) build through run_pilot.build_agent."""
import copy
import sys
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from models.quantum_critic import HybridQuantumTwinCritic  # noqa: E402
from models.vqc import FourierSurrogatePolicy  # noqa: E402
from run_pilot import TRAINED_ARMS, build_agent  # noqa: E402


def n_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _config():
    with open("config/hyperparameters.yaml", encoding="utf-8") as stream:
        config = copy.deepcopy(yaml.safe_load(stream))
    config.setdefault("encoder", {})["sequence_length"] = 60
    config["experiment_overrides"] = {"classical_dropout": 0.0,
                                      "classical_action_prior": [0.5, 0.075],
                                      "encoder_type": "transformer"}
    return config


@pytest.mark.parametrize("arm, family", [
    ("ansatz_realamplitudes", "real_amplitudes"),
    ("ansatz_efficientsu2", "efficient_su2"),
    ("ansatz_twolocal", "two_local"),
])
def test_ansatz_arms_use_their_family(arm, family):
    agent = build_agent(arm, TRAINED_ARMS[arm], _config(), torch.device("cpu"), 0)
    assert agent.actor.vqc.ansatz == family
    assert agent.actor_target.vqc.ansatz == family


def test_fourier_arm_replaces_the_circuit():
    agent = build_agent("fourier_surrogate", TRAINED_ARMS["fourier_surrogate"],
                        _config(), torch.device("cpu"), 0)
    assert isinstance(agent.actor, FourierSurrogatePolicy)
    assert agent.actor(torch.randn(3, agent.encoded_dim)).shape == (3, 2)


def test_quantum_critic_arm_has_a_hybrid_critic_and_classical_actor():
    agent = build_agent("quantum_critic", TRAINED_ARMS["quantum_critic"],
                        _config(), torch.device("cpu"), 0)
    assert isinstance(agent.critic, HybridQuantumTwinCritic)
    assert isinstance(agent.critic_target, HybridQuantumTwinCritic)
    assert not any(p.requires_grad for p in agent.critic_target.parameters())
    optimised = {id(p) for group in agent.critic_optimizer.param_groups for p in group["params"]}
    assert optimised == {id(p) for p in agent.critic.parameters()}
    assert n_params(agent.actor) == n_params(build_agent(
        "classical_matched", TRAINED_ARMS["classical_matched"], _config(),
        torch.device("cpu"), 0).actor)


def test_actor_replacement_keeps_the_requested_ansatz():
    spec = {"kind": "quantum", "use_skip": False, "ansatz": "two_local"}
    agent = build_agent("custom", spec, _config(), torch.device("cpu"), 0)
    assert agent.actor.vqc.ansatz == "two_local"
    assert agent.actor_target.vqc.ansatz == "two_local"


@pytest.mark.parametrize("arm", ["quantum_critic", "fourier_surrogate", "ansatz_realamplitudes",
                                 "ansatz_efficientsu2", "ansatz_twolocal", "quantum_full"])
def test_arm_trains_and_reloads(arm, tmp_path):
    import numpy as np
    T = 6
    config = _config()
    config["encoder"]["sequence_length"] = T
    config["training"]["batch_size"] = 16
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    agent = build_agent(arm, TRAINED_ARMS[arm], config, torch.device("cpu"), 0)
    for _ in range(40):
        agent.replay_buffer.push(rng.normal(size=(T, 13)).astype(np.float32),
                                 rng.random(2).astype(np.float32), float(rng.normal()),
                                 rng.normal(size=(T, 13)).astype(np.float32), False)
    agent.configure_online_bc(states=rng.normal(size=(50, T, 13)).astype(np.float32),
                              actions=rng.random((50, 2)).astype(np.float32),
                              weight=1.0, rl_scale=0.05, action_weights=[1.0, 4.0])
    before = {k: v.clone() for k, v in agent.critic.state_dict().items()}
    metrics = [agent.update() for _ in range(2)]
    assert all(np.isfinite(v) for m in metrics for v in m.values())
    assert any(not torch.equal(before[k], v) for k, v in agent.critic.state_dict().items())
    state, sequence = np.zeros(13, np.float32), np.zeros((T, 13), np.float32)

    def act(policy):
        # Evaluation runs with dropout off (per_patient_eval sets eval mode).
        for module in (policy.actor, policy.encoder):
            if module is not None:
                module.eval()
        return policy.select_action(state, deterministic=True, state_sequence=sequence)

    action = act(agent)
    path = str(tmp_path / "checkpoint.pt")
    agent.save(path)
    clone = build_agent(arm, TRAINED_ARMS[arm], config, torch.device("cpu"), 1)
    clone.load(path)
    assert np.allclose(action, act(clone), atol=1e-6)
