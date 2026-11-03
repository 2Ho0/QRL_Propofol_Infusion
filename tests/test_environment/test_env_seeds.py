"""Environment seeds must keep runs independent (review MED 4).

The env seed fixes a simulated patient's random effects, its BIS noise and the
warm-up exploration doses. v1 used ``seed + episode``, so training seeds 0 and
1 shared 29 of 30 simulator patients, and the checkpoint cohort (1000 + i) and
low training seeds collided with test case ids (evaluation uses caseid).
"""
import numpy as np
import pytest

from agents.classical_agent import ClassicalDDPGAgent
from environment.env_seeds import checkpoint_env_seed, training_env_seed


def test_training_checkpoint_and_evaluation_seeds_are_disjoint():
    training = {training_env_seed(s, e) for s in range(10) for e in range(30)}
    checkpoint = {checkpoint_env_seed(1000, i) for i in range(30)}
    evaluation = set(range(0, 10_000))   # evaluation_seed 0 + VitalDB case id
    assert len(training) == 300 and len(checkpoint) == 30
    assert not training & checkpoint
    assert not training & evaluation
    assert not checkpoint & evaluation


def test_episode_index_is_bounded():
    with pytest.raises(ValueError):
        training_env_seed(0, 1000)
    with pytest.raises(ValueError):
        checkpoint_env_seed(1000, 1000)


def test_classical_exploration_noise_follows_the_agent_seed(tmp_path):
    import yaml
    with open("config/hyperparameters.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    def draws(seed):
        agent = ClassicalDDPGAgent(state_dim=13, action_dim=2, config=config,
                                   encoder_type="none", seed=seed)
        return np.array([agent.noise.sample() for _ in range(5)])

    assert np.allclose(draws(3), draws(3))
    assert not np.allclose(draws(3), draws(4))
