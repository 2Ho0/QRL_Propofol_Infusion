"""
Propofol Environment - Gymnasium Compatible RL Environment
============================================================

This module implements a Gymnasium-compatible environment for propofol
infusion control using reinforcement learning. The environment wraps
the patient simulator and provides the standard RL interface.

Environment Specification:
--------------------------
- Observation Space: Box(2,) - [BIS_error_normalized, Ce_normalized]
- Action Space: Box(1,) - Continuous propofol infusion rate [0, 1] scaled to [0, max_dose]
- Reward: Negative squared BIS error with safety penalties
- Episode Length: Configurable (default 60 minutes / 720 steps at 5s intervals)

Following the CBIM paper formulation:
- Target BIS: 50 (moderate hypnosis)
- Safe Range: 40-60
- Infusion Rate: 0-200 μg/kg/min
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List
import yaml
from pathlib import Path

from .patient_simulator import PatientSimulator, PatientParameters, SchniderModel


class PropofolEnv(gym.Env):
    """
    Gymnasium Environment for Propofol Infusion Control.
    
    This environment simulates closed-loop anesthesia control where
    the agent must maintain BIS (Bispectral Index) within a target
    range by adjusting propofol infusion rates.
    
    State Space:
        The observation is a 4-dimensional vector:
        - obs[0]: Normalized BIS error = (BIS - target) / 50
        - obs[1]: Normalized effect-site concentration = Ce / EC50
        - obs[2]: Normalized rate of change of BIS (dBIS/dt approximation)
        - obs[3]: Normalized previous dose = prev_dose / max_dose
    
    Action Space:
        Continuous action in [0, 1] representing normalized infusion rate.
        Scaled to [0, max_dose] μg/kg/min for the simulator.
    
    Reward Function:
        Following CBIM paper formulation:
        r(t) = -α * (BIS - target)² - β * dose - γ * |dose_change| + safety_penalties
    
    Attributes:
        simulator: Patient PK/PD simulator
        config: Environment configuration
        max_steps: Maximum steps per episode
        current_step: Current step in episode
        prev_dose: Previous infusion rate (for dose change penalty)
        bis_history: Recent BIS values for dBIS/dt calculation
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        patient: Optional[PatientParameters] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the Propofol environment.
        
        Args:
            config: Configuration dictionary (overrides config_path)
            config_path: Path to YAML configuration file
            patient: Specific patient parameters (uses config defaults if None)
            render_mode: Rendering mode ('human' or 'rgb_array')
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        # Load configuration
        self.config = self._load_config(config, config_path)
        
        # Environment parameters from config
        env_config = self.config.get('environment', {})
        self.bis_target = env_config.get('bis_target', 50)
        self.bis_min = env_config.get('bis_min', 40)
        self.bis_max = env_config.get('bis_max', 60)
        self.dose_min = env_config.get('dose_min', 0.0)
        self.dose_max = env_config.get('dose_max', 200.0)
        self.dt = env_config.get('dt', 5.0)
        self.episode_duration = env_config.get('episode_duration', 3600)
        self.enable_bolus = env_config.get('enable_bolus', True)
        self.bolus_dose = env_config.get('bolus_dose', 2.0)
        
        # Reward configuration
        reward_config = self.config.get('reward', {})
        self.reward_weights = reward_config.get('weights', {
            'bis_error': 1.0,
            'dose_penalty': 0.01,
            'dose_change': 0.001,
            'stability_bonus': 0.1
        })
        self.safety_config = reward_config.get('safety', {
            'overdose_threshold': 150,
            'overdose_penalty': -10.0,
            'underdose_bis': 70,
            'underdose_penalty': -5.0,
            'critical_low_bis': 35,
            'critical_penalty': -20.0
        })
        
        # Calculate max steps
        self.max_steps = int(self.episode_duration / self.dt)
        
        # Initialize patient parameters
        if patient is not None:
            self.patient = patient
        else:
            patient_config = self.config.get('pkpd_model', {}).get('default_patient', {})
            self.patient = PatientParameters(
                age=patient_config.get('age', 40),
                weight=patient_config.get('weight', 70),
                height=patient_config.get('height', 170),
                gender=patient_config.get('gender', 'male')
            )
        
        # Create simulator
        self.seed_value = seed
        self.simulator = PatientSimulator(
            patient=self.patient,
            dt=self.dt,
            noise_std=2.0,
            seed=seed
        )
        
        # Define observation and action spaces
        # Observation: [bis_error_norm, ce_norm, dbis_norm, prev_dose_norm]
        self.observation_space = spaces.Box(
            low=np.array([-2.0, 0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 5.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action: Normalized dose [0, 1]
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # State tracking
        self.current_step = 0
        self.prev_dose = 0.0
        self.prev_bis = 97.0  # Awake BIS
        self.bis_history = []
        
        # Rendering
        self.render_mode = render_mode
        
        # Episode history for analysis
        self.episode_history = {
            'bis': [],
            'ce': [],
            'dose': [],
            'reward': [],
            'time': []
        }
    
    def _load_config(
        self, 
        config: Optional[Dict], 
        config_path: Optional[str]
    ) -> Dict:
        """Load configuration from dictionary or YAML file."""
        if config is not None:
            return config
        
        if config_path is not None:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Try to load default config
        default_path = Path(__file__).parent.parent.parent / 'config' / 'hyperparameters.yaml'
        if default_path.exists():
            with open(default_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Return minimal default config
        return {
            'environment': {
                'bis_target': 50,
                'bis_min': 40,
                'bis_max': 60,
                'dose_min': 0.0,
                'dose_max': 200.0,
                'dt': 5.0,
                'episode_duration': 3600,
                'enable_bolus': True,
                'bolus_dose': 2.0
            },
            'reward': {
                'weights': {
                    'bis_error': 1.0,
                    'dose_penalty': 0.01,
                    'dose_change': 0.001,
                    'stability_bonus': 0.1
                }
            }
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for this episode
            options: Additional options (e.g., patient parameters)
        
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Handle patient variation if specified in options
        if options is not None and 'patient' in options:
            self.patient = options['patient']
            self.simulator = PatientSimulator(
                patient=self.patient,
                dt=self.dt,
                noise_std=2.0,
                seed=seed or self.seed_value
            )
        else:
            self.simulator.reset()
        
        # Reset state tracking
        self.current_step = 0
        self.prev_dose = 0.0
        self.prev_bis = 97.0  # Awake BIS
        self.bis_history = [97.0] * 5  # Initialize with awake BIS
        
        # Clear episode history
        self.episode_history = {
            'bis': [],
            'ce': [],
            'dose': [],
            'reward': [],
            'time': []
        }
        
        # Apply induction bolus if enabled
        if self.enable_bolus:
            self.simulator.administer_bolus(self.bolus_dose)
            # Step forward a bit to let bolus take effect
            for _ in range(3):  # 15 seconds
                _, bis = self.simulator.step(0.0)
                self.bis_history.append(bis)
                self.prev_bis = bis
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            'bis': self.prev_bis,
            'ce': self.simulator.state[3],
            'patient': {
                'age': self.patient.age,
                'weight': self.patient.weight,
                'height': self.patient.height,
                'gender': self.patient.gender
            }
        }
        
        return observation, info
    
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Normalized infusion rate [0, 1]
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert normalized action to actual dose
        action = np.clip(action, 0, 1)
        dose = float(action[0]) * self.dose_max
        
        # Step the simulator
        state, bis = self.simulator.step(dose)
        
        # Update history
        self.bis_history.append(bis)
        if len(self.bis_history) > 10:
            self.bis_history.pop(0)
        
        # Calculate reward
        reward = self._calculate_reward(bis, dose)
        
        # Record episode history
        self.episode_history['bis'].append(bis)
        self.episode_history['ce'].append(state[3])
        self.episode_history['dose'].append(dose)
        self.episode_history['reward'].append(reward)
        self.episode_history['time'].append(self.simulator.time)
        
        # Update state
        self.prev_bis = bis
        self.prev_dose = dose
        self.current_step += 1
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Episode ends if BIS goes critically low (patient safety)
        if bis < 20:
            terminated = True
            reward += -100.0  # Large penalty for dangerous state
        
        # Episode truncated if max steps reached
        if self.current_step >= self.max_steps:
            truncated = True
        
        # Get observation
        observation = self._get_observation()
        
        info = {
            'bis': bis,
            'ce': state[3],
            'dose': dose,
            'time': self.simulator.time,
            'step': self.current_step,
            'in_target_range': self.bis_min <= bis <= self.bis_max
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector from current state.
        
        Returns:
            Observation array: [bis_error_norm, ce_norm, dbis_norm, prev_dose_norm]
        """
        # Current BIS and concentrations
        bis = self.prev_bis
        ce = self.simulator.state[3]
        ec50 = self.simulator.model.params.ec50
        
        # Normalized BIS error: (BIS - target) / 50
        bis_error_norm = (bis - self.bis_target) / 50.0
        
        # Normalized effect-site concentration
        ce_norm = ce / ec50
        
        # Approximate dBIS/dt from history
        if len(self.bis_history) >= 2:
            dbis = (self.bis_history[-1] - self.bis_history[-2]) / self.dt
            dbis_norm = np.clip(dbis / 10.0, -1.0, 1.0)  # Normalize
        else:
            dbis_norm = 0.0
        
        # Normalized previous dose
        prev_dose_norm = self.prev_dose / self.dose_max
        
        return np.array([
            bis_error_norm,
            ce_norm,
            dbis_norm,
            prev_dose_norm
        ], dtype=np.float32)
    
    def _calculate_reward(self, bis: float, dose: float) -> float:
        """
        Calculate reward based on CBIM paper formulation.
        
        Reward components:
        1. BIS error penalty: -α * ((BIS - target) / target)²
        2. Dose magnitude penalty: -β * dose
        3. Dose change penalty: -γ * |dose - prev_dose|
        4. Stability bonus: +δ if BIS in target range for sustained period
        5. Safety penalties for dangerous states
        
        Args:
            bis: Current BIS value
            dose: Current infusion rate
        
        Returns:
            Total reward value
        """
        w = self.reward_weights
        s = self.safety_config
        
        # 1. BIS error penalty (Performance Error based)
        # PE = (BIS - target) / target * 100
        pe = (bis - self.bis_target) / self.bis_target
        bis_error_penalty = -w['bis_error'] * (pe ** 2)
        
        # 2. Dose magnitude penalty (encourage minimal effective dose)
        dose_penalty = -w['dose_penalty'] * (dose / self.dose_max)
        
        # 3. Dose change penalty (encourage smooth control)
        dose_change = abs(dose - self.prev_dose) / self.dose_max
        change_penalty = -w['dose_change'] * dose_change
        
        # 4. Stability bonus (in target range)
        if self.bis_min <= bis <= self.bis_max:
            stability_bonus = w['stability_bonus']
        else:
            stability_bonus = 0.0
        
        # 5. Safety penalties
        safety_penalty = 0.0
        
        # Overdose (high infusion rate)
        if dose > s['overdose_threshold']:
            safety_penalty += s['overdose_penalty']
        
        # Patient awakening (BIS too high)
        if bis > s['underdose_bis']:
            safety_penalty += s['underdose_penalty']
        
        # Dangerous depth (BIS too low)
        if bis < s['critical_low_bis']:
            safety_penalty += s['critical_penalty']
        
        # Total reward
        reward = (
            bis_error_penalty +
            dose_penalty +
            change_penalty +
            stability_bonus +
            safety_penalty
        )
        
        return reward
    
    def get_episode_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics for the episode.
        
        Following CBIM paper metrics:
        - MDPE: Median Performance Error
        - MDAPE: Median Absolute Performance Error  
        - Wobble: Intra-individual variability
        - Time in Target: Percentage of time BIS in target range
        
        Returns:
            Dictionary of performance metrics
        """
        if len(self.episode_history['bis']) == 0:
            return {}
        
        bis_array = np.array(self.episode_history['bis'])
        
        # Performance Error (PE)
        pe = (bis_array - self.bis_target) / self.bis_target * 100
        
        # MDPE: Median Performance Error
        mdpe = np.median(pe)
        
        # MDAPE: Median Absolute Performance Error
        mdape = np.median(np.abs(pe))
        
        # Wobble: Median absolute deviation from MDPE
        wobble = np.median(np.abs(pe - mdpe))
        
        # Time in Target Range
        in_target = np.logical_and(
            bis_array >= self.bis_min,
            bis_array <= self.bis_max
        )
        time_in_target = np.mean(in_target) * 100
        
        # Total reward
        total_reward = sum(self.episode_history['reward'])
        
        # Mean dose
        mean_dose = np.mean(self.episode_history['dose'])
        
        return {
            'mdpe': mdpe,
            'mdape': mdape,
            'wobble': wobble,
            'time_in_target': time_in_target,
            'total_reward': total_reward,
            'mean_dose': mean_dose,
            'final_bis': bis_array[-1] if len(bis_array) > 0 else None
        }
    
    def render(self):
        """Render the environment (placeholder for visualization)."""
        if self.render_mode == "human":
            print(f"Step: {self.current_step}, BIS: {self.prev_bis:.1f}, "
                  f"Dose: {self.prev_dose:.1f} μg/kg/min")
    
    def close(self):
        """Clean up resources."""
        pass


def make_env(
    config_path: Optional[str] = None,
    patient: Optional[PatientParameters] = None,
    seed: Optional[int] = None
) -> PropofolEnv:
    """
    Factory function to create a PropofolEnv instance.
    
    Args:
        config_path: Path to configuration file
        patient: Patient parameters
        seed: Random seed
    
    Returns:
        Configured PropofolEnv instance
    """
    return PropofolEnv(
        config_path=config_path,
        patient=patient,
        seed=seed
    )


if __name__ == "__main__":
    # Test the environment
    print("Testing PropofolEnv...")
    
    env = PropofolEnv(seed=42)
    
    # Reset
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    
    # Run a few steps with random actions
    print("\nRunning episode with random actions...")
    total_reward = 0
    
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 10 == 0:
            print(f"Step {step}: BIS={info['bis']:.1f}, "
                  f"Dose={info['dose']:.1f}, Reward={reward:.3f}")
        
        if terminated or truncated:
            break
    
    # Get episode metrics
    metrics = env.get_episode_metrics()
    print(f"\nEpisode Metrics:")
    print(f"  MDPE: {metrics['mdpe']:.2f}%")
    print(f"  MDAPE: {metrics['mdape']:.2f}%")
    print(f"  Wobble: {metrics['wobble']:.2f}%")
    print(f"  Time in Target: {metrics['time_in_target']:.1f}%")
    print(f"  Total Reward: {metrics['total_reward']:.2f}")
    
    env.close()
    print("\nTest complete!")
