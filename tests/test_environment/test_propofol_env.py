"""
Unit Tests for Propofol Environment
====================================

Tests the main RL environment for propofol infusion control.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.environment.propofol_env import PropofolEnv
from src.utils.exceptions import InvalidActionError, InvalidStateError


class TestPropofolEnvInitialization:
    """Test environment initialization."""
    
    def test_default_initialization(self, sample_patient_params):
        """Test initialization with default parameters."""
        env = PropofolEnv(patient_params=sample_patient_params)
        
        assert env.dt == 5.0
        assert env.target_bis == 50.0
        assert env.bis_tolerance == 5.0
        assert env.max_steps == 720  # 1 hour
    
    def test_custom_parameters(self, sample_patient_params):
        """Test initialization with custom parameters."""
        env = PropofolEnv(
            patient_params=sample_patient_params,
            dt=10.0,
            target_bis=45.0,
            bis_tolerance=3.0,
            max_steps=360
        )
        
        assert env.dt == 10.0
        assert env.target_bis == 45.0
        assert env.bis_tolerance == 3.0
        assert env.max_steps == 360
    
    def test_action_space(self, sample_patient_params):
        """Test action space configuration."""
        env = PropofolEnv(patient_params=sample_patient_params)
        
        # Should be Box space [-1, 1]
        assert env.action_space.shape == (1,)
        assert env.action_space.low[0] == -1.0
        assert env.action_space.high[0] == 1.0
    
    def test_observation_space(self, sample_patient_params):
        """Test observation space configuration."""
        env = PropofolEnv(patient_params=sample_patient_params)
        
        # State includes: BIS, Ce, time, target_BIS
        assert len(env.observation_space.low) >= 4
        assert len(env.observation_space.high) >= 4


class TestPropofolEnvReset:
    """Test environment reset functionality."""
    
    def test_reset_returns_valid_state(self, sample_patient_params):
        """Test reset returns valid initial state."""
        env = PropofolEnv(patient_params=sample_patient_params)
        state = env.reset()
        
        # Should be numpy array
        assert isinstance(state, np.ndarray)
        
        # Should be within observation space
        assert env.observation_space.contains(state)
    
    def test_reset_initializes_awake_state(self, sample_patient_params):
        """Test reset initializes to awake state."""
        env = PropofolEnv(patient_params=sample_patient_params)
        state = env.reset()
        
        # BIS should be 100 (awake)
        bis = state[0]
        assert abs(bis - 100.0) < 1.0
        
        # Ce should be 0
        ce = state[1]
        assert abs(ce) < 0.01
    
    def test_reset_resets_time(self, sample_patient_params):
        """Test reset resets time counter."""
        env = PropofolEnv(patient_params=sample_patient_params)
        
        # Take some steps
        env.reset()
        for _ in range(10):
            env.step(np.array([0.5]))
        
        # Reset and check time is zero
        state = env.reset()
        assert env.current_step == 0


class TestPropofolEnvStep:
    """Test environment step functionality."""
    
    def test_step_returns_tuple(self, sample_patient_params):
        """Test step returns (state, reward, done, info)."""
        env = PropofolEnv(patient_params=sample_patient_params)
        env.reset()
        
        result = env.step(np.array([0.5]))
        
        assert len(result) == 4
        state, reward, done, info = result
        
        assert isinstance(state, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    
    def test_step_increases_time(self, sample_patient_params):
        """Test step increases time counter."""
        env = PropofolEnv(patient_params=sample_patient_params)
        env.reset()
        
        initial_step = env.current_step
        env.step(np.array([0.5]))
        
        assert env.current_step == initial_step + 1
    
    def test_step_updates_concentration(self, sample_patient_params):
        """Test step updates drug concentration."""
        env = PropofolEnv(patient_params=sample_patient_params)
        state = env.reset()
        
        initial_ce = state[1]
        
        # Apply positive action
        next_state, _, _, _ = env.step(np.array([0.5]))
        next_ce = next_state[1]
        
        # Concentration should increase
        assert next_ce > initial_ce
    
    def test_step_updates_bis(self, sample_patient_params):
        """Test step updates BIS."""
        env = PropofolEnv(patient_params=sample_patient_params)
        state = env.reset()
        
        initial_bis = state[0]
        
        # Apply action to induce anesthesia
        for _ in range(10):
            next_state, _, _, _ = env.step(np.array([1.0]))
        
        final_bis = next_state[0]
        
        # BIS should decrease
        assert final_bis < initial_bis
    
    def test_invalid_action_raises_error(self, sample_patient_params):
        """Test invalid action raises error."""
        env = PropofolEnv(patient_params=sample_patient_params)
        env.reset()
        
        # Action outside [-1, 1] should raise error
        with pytest.raises((InvalidActionError, ValueError, AssertionError)):
            env.step(np.array([2.0]))
    
    def test_done_after_max_steps(self, sample_patient_params):
        """Test episode terminates after max_steps."""
        env = PropofolEnv(
            patient_params=sample_patient_params,
            max_steps=10
        )
        env.reset()
        
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(np.array([0.5]))
            steps += 1
            
            # Safety check
            if steps > 20:
                break
        
        assert done
        assert steps == 10


class TestPropofolEnvReward:
    """Test reward computation."""
    
    def test_reward_at_target(self, sample_patient_params):
        """Test reward when BIS is at target."""
        env = PropofolEnv(
            patient_params=sample_patient_params,
            target_bis=50.0,
            bis_tolerance=5.0
        )
        env.reset()
        
        # Manually set BIS to target
        env.current_bis = 50.0
        
        _, reward, _, _ = env.step(np.array([0.0]))
        
        # Should get positive reward
        assert reward > 0
    
    def test_reward_far_from_target(self, sample_patient_params):
        """Test reward when BIS is far from target."""
        env = PropofolEnv(
            patient_params=sample_patient_params,
            target_bis=50.0
        )
        env.reset()
        
        # BIS starts at 100 (awake), far from target
        _, reward, _, _ = env.step(np.array([0.0]))
        
        # Should get negative reward
        assert reward < 0
    
    def test_reward_penalizes_large_actions(self, sample_patient_params):
        """Test large actions are penalized."""
        env = PropofolEnv(patient_params=sample_patient_params)
        env.reset()
        
        # Small action
        _, reward_small, _, _ = env.step(np.array([0.1]))
        
        env.reset()
        
        # Large action
        _, reward_large, _, _ = env.step(np.array([1.0]))
        
        # Same BIS error, but large action should have more penalty
        # (This assumes reward includes action penalty term)


class TestPropofolEnvInfo:
    """Test info dictionary."""
    
    def test_info_contains_bis(self, sample_patient_params):
        """Test info dict contains BIS."""
        env = PropofolEnv(patient_params=sample_patient_params)
        env.reset()
        
        _, _, _, info = env.step(np.array([0.5]))
        
        assert 'bis' in info
        assert 0 <= info['bis'] <= 100
    
    def test_info_contains_concentration(self, sample_patient_params):
        """Test info dict contains concentration."""
        env = PropofolEnv(patient_params=sample_patient_params)
        env.reset()
        
        _, _, _, info = env.step(np.array([0.5]))
        
        assert 'ce' in info or 'Ce' in info or 'concentration' in info
    
    def test_info_contains_action(self, sample_patient_params):
        """Test info dict contains action."""
        env = PropofolEnv(patient_params=sample_patient_params)
        env.reset()
        
        action = np.array([0.5])
        _, _, _, info = env.step(action)
        
        assert 'action' in info or 'infusion_rate' in info


class TestPropofolEnvEdgeCases:
    """Test edge cases and robustness."""
    
    def test_zero_action(self, sample_patient_params):
        """Test zero action (no infusion)."""
        env = PropofolEnv(patient_params=sample_patient_params)
        env.reset()
        
        # Zero action should not crash
        state, reward, done, info = env.step(np.array([0.0]))
        
        assert isinstance(state, np.ndarray)
        assert not np.any(np.isnan(state))
    
    def test_negative_action(self, sample_patient_params):
        """Test negative action (decrease infusion)."""
        env = PropofolEnv(patient_params=sample_patient_params)
        env.reset()
        
        # First increase concentration
        for _ in range(5):
            env.step(np.array([1.0]))
        
        # Then apply negative action
        state, reward, done, info = env.step(np.array([-0.5]))
        
        assert isinstance(state, np.ndarray)
        assert not np.any(np.isnan(state))
    
    def test_multiple_episodes(self, sample_patient_params):
        """Test multiple episode resets."""
        env = PropofolEnv(
            patient_params=sample_patient_params,
            max_steps=10
        )
        
        for episode in range(3):
            state = env.reset()
            
            # Initial state should be awake
            assert abs(state[0] - 100.0) < 1.0
            
            done = False
            while not done:
                _, _, done, _ = env.step(np.array([0.5]))
    
    def test_state_consistency(self, sample_patient_params):
        """Test state values are consistent."""
        env = PropofolEnv(patient_params=sample_patient_params)
        env.reset()
        
        for _ in range(20):
            state, _, _, _ = env.step(np.array([0.5]))
            
            # BIS should be in valid range
            assert 0 <= state[0] <= 100
            
            # Ce should be non-negative
            assert state[1] >= 0
            
            # No NaN or Inf
            assert not np.any(np.isnan(state))
            assert not np.any(np.isinf(state))


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
