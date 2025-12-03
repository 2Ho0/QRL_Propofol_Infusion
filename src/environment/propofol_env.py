"""
Propofol Environment - Gymnasium Compatible RL Environment
============================================================

This module implements a Gymnasium-compatible environment for propofol
infusion control using reinforcement learning. The environment wraps
the patient simulator and provides the standard RL interface.

Following CBIM Paper Formulations:
----------------------------------
- Formulation (36): BIS slope = BIS(t) - BIS(t-1)
- Formulation (37): BIS error = BIS(t) - BIS_target(t)
- Formulation (38): PPF_acc(t) = Σ PPF(i) for i=t-W to t (cumulative propofol)
- Formulation (39): RFTN_acc(t) = Σ RFTN(i) for i=t-W to t (cumulative remifentanil)
- Formulation (40): R(s,a) = 1/(|g - BIS(t+1)| + α) (reward function)

Environment Specification:
--------------------------
- Observation Space: Extended state vector per Fig.4 of paper
  [BIS_error, BIS_slope, Ce_ppf, Ce_rftn, PPF_acc, RFTN_acc, prev_dose, ...]
- Action Space: Box(1,) - Continuous propofol infusion rate [0, 1] scaled to [0, max_dose]
- Reward: Formulation (40) based reward with safety penalties
- Episode Length: Configurable (default 60 minutes / 720 steps at 5s intervals)

Remifentanil External Input:
----------------------------
Remifentanil is treated as external input (not controlled by RL agent).
Random values are sampled based on typical surgical infusion ranges:
- Induction: 0.5-1.0 μg/kg/min
- Maintenance: 0.1-0.4 μg/kg/min
- Variable patterns simulating surgical stimulation changes
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List
import yaml
from pathlib import Path

from .patient_simulator import (
    PatientSimulator, 
    PatientParameters, 
    SchniderModel,
    BISModelType,
    DrugInteractionParams
)


class RemifentanilSchedule:
    """
    External remifentanil infusion schedule based on surgical data.
    
    Simulates realistic remifentanil administration patterns during surgery:
    - Induction phase: Higher initial rate
    - Maintenance phase: Lower baseline with variations
    - Stimulation events: Temporary increases during surgical stimulation
    
    Based on typical clinical ranges from CBIM paper and surgical data.
    """
    
    def __init__(
        self,
        base_rate: float = 0.2,           # ng/kg/min baseline
        induction_rate: float = 0.5,      # ng/kg/min during induction
        induction_duration: float = 300,  # seconds
        stimulation_prob: float = 0.02,   # Probability of stimulation event per step
        stimulation_increase: float = 0.3, # Additional rate during stimulation
        stimulation_duration: int = 12,    # Steps (60 seconds at dt=5)
        noise_std: float = 0.05,           # Random variation
        seed: Optional[int] = None
    ):
        """
        Initialize remifentanil schedule.
        
        Args:
            base_rate: Baseline maintenance rate (ng/kg/min)
            induction_rate: Rate during induction phase (ng/kg/min)
            induction_duration: Duration of induction phase (seconds)
            stimulation_prob: Probability of surgical stimulation event
            stimulation_increase: Additional rate during stimulation
            stimulation_duration: Duration of stimulation response (steps)
            noise_std: Standard deviation of random noise
            seed: Random seed
        """
        self.base_rate = base_rate
        self.induction_rate = induction_rate
        self.induction_duration = induction_duration
        self.stimulation_prob = stimulation_prob
        self.stimulation_increase = stimulation_increase
        self.stimulation_duration = stimulation_duration
        self.noise_std = noise_std
        
        self.rng = np.random.default_rng(seed)
        self.reset()
    
    def reset(self):
        """Reset the schedule state."""
        self.time = 0.0
        self.stimulation_countdown = 0
        self.current_rate = self.induction_rate
    
    def get_rate(self, time: float, dt: float = 5.0) -> float:
        """
        Get remifentanil rate for current time.
        
        Args:
            time: Current simulation time (seconds)
            dt: Time step (seconds)
        
        Returns:
            Remifentanil infusion rate (ng/kg/min)
        """
        self.time = time
        
        # Induction phase
        if time < self.induction_duration:
            # Gradual decrease from induction to maintenance
            progress = time / self.induction_duration
            base = self.induction_rate * (1 - progress) + self.base_rate * progress
        else:
            base = self.base_rate
        
        # Handle ongoing stimulation
        if self.stimulation_countdown > 0:
            self.stimulation_countdown -= 1
            base += self.stimulation_increase
        
        # Check for new stimulation event
        if self.rng.random() < self.stimulation_prob:
            self.stimulation_countdown = self.stimulation_duration
            base += self.stimulation_increase
        
        # Add noise
        rate = base + self.rng.normal(0, self.noise_std)
        
        return max(0.0, rate)


class PropofolEnv(gym.Env):
    """
    Gymnasium Environment for Propofol Infusion Control.
    
    This environment simulates closed-loop anesthesia control where
    the agent must maintain BIS (Bispectral Index) within a target
    range by adjusting propofol infusion rates.
    
    Extended State Space (following CBIM paper Fig.4):
        The observation includes time-series features:
        - obs[0]: Normalized BIS error = (BIS - target) / 50  [Formulation 37]
        - obs[1]: Normalized BIS slope = dBIS/dt / 10         [Formulation 36]
        - obs[2]: Normalized Ce_ppf = Ce_ppf / EC50
        - obs[3]: Normalized Ce_rftn = Ce_rftn / EC50
        - obs[4]: Normalized PPF_acc = cumulative PPF dose    [Formulation 38]
        - obs[5]: Normalized RFTN_acc = cumulative RFTN dose  [Formulation 39]
        - obs[6]: Normalized previous dose = prev_dose / max_dose
        - obs[7]: Normalized time = current_step / max_steps
    
    Action Space:
        Continuous action in [0, 1] representing normalized infusion rate.
        Scaled to [0, max_dose] μg/kg/min for the simulator.
    
    Reward Function - Formulation (40):
        R(s,a) = 1 / (|g - BIS(t+1)| + α)
        
        With additional safety penalties for dangerous states.
    
    Attributes:
        simulator: Patient PK/PD simulator (with drug interaction model)
        rftn_schedule: External remifentanil infusion schedule
        config: Environment configuration
        max_steps: Maximum steps per episode
        current_step: Current step in episode
        dose_history: History window for cumulative dose calculation
        bis_history: Recent BIS values for slope calculation
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        patient: Optional[PatientParameters] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        use_drug_interaction: bool = True,
        reward_type: str = "paper",  # "paper" for Formulation 40, "shaped" for original
        history_window: int = 12  # Window size for cumulative dose (60 seconds at dt=5)
    ):
        """
        Initialize the Propofol environment.
        
        Args:
            config: Configuration dictionary (overrides config_path)
            config_path: Path to YAML configuration file
            patient: Specific patient parameters (uses config defaults if None)
            render_mode: Rendering mode ('human' or 'rgb_array')
            seed: Random seed for reproducibility
            use_drug_interaction: If True, use drug interaction BIS model (Formulation 32)
            reward_type: "paper" for Formulation 40, "shaped" for original shaped reward
            history_window: Number of steps for cumulative dose calculation
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
        self.reward_type = reward_type
        self.reward_alpha = reward_config.get('alpha', 1.0)  # For Formulation 40
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
        
        # History window for Formulations (38)-(39)
        self.history_window = history_window
        
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
        
        # BIS model type
        self.use_drug_interaction = use_drug_interaction
        bis_model_type = (BISModelType.DRUG_INTERACTION if use_drug_interaction 
                         else BISModelType.HILL_SIGMOID)
        
        # Create simulator with drug interaction model
        self.seed_value = seed
        self.simulator = PatientSimulator(
            patient=self.patient,
            dt=self.dt,
            noise_std=2.0,
            seed=seed,
            bis_model_type=bis_model_type
        )
        
        # Remifentanil external schedule
        rftn_config = self.config.get('remifentanil', {})
        self.rftn_schedule = RemifentanilSchedule(
            base_rate=rftn_config.get('base_rate', 0.2),
            induction_rate=rftn_config.get('induction_rate', 0.5),
            induction_duration=rftn_config.get('induction_duration', 300),
            stimulation_prob=rftn_config.get('stimulation_prob', 0.02),
            seed=seed
        )
        
        # Define extended observation space (8 features)
        # [bis_error, bis_slope, ce_ppf, ce_rftn, ppf_acc, rftn_acc, prev_dose, time]
        self.observation_space = spaces.Box(
            low=np.array([-2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 1.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action: Normalized propofol dose [0, 1]
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # State tracking
        self.current_step = 0
        self.prev_dose = 0.0
        self.prev_bis = 97.0  # Awake BIS
        self.bis_history: List[float] = []
        self.dose_ppf_history: List[float] = []  # For Formulation (38)
        self.dose_rftn_history: List[float] = []  # For Formulation (39)
        
        # Rendering
        self.render_mode = render_mode
        
        # Episode history for analysis
        self.episode_history = {
            'bis': [],
            'ce_ppf': [],
            'ce_rftn': [],
            'dose_ppf': [],
            'dose_rftn': [],
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
        
        # BIS model type
        bis_model_type = (BISModelType.DRUG_INTERACTION if self.use_drug_interaction 
                         else BISModelType.HILL_SIGMOID)
        
        # Handle patient variation if specified in options
        if options is not None and 'patient' in options:
            self.patient = options['patient']
            self.simulator = PatientSimulator(
                patient=self.patient,
                dt=self.dt,
                noise_std=2.0,
                seed=seed or self.seed_value,
                bis_model_type=bis_model_type
            )
        else:
            self.simulator.reset()
        
        # Reset remifentanil schedule
        self.rftn_schedule.reset()
        
        # Reset state tracking
        self.current_step = 0
        self.prev_dose = 0.0
        self.prev_bis = 97.0  # Awake BIS
        self.bis_history = [97.0] * 5  # Initialize with awake BIS
        self.dose_ppf_history = [0.0] * self.history_window  # Formulation (38)
        self.dose_rftn_history = [0.0] * self.history_window  # Formulation (39)
        
        # Clear episode history
        self.episode_history = {
            'bis': [],
            'ce_ppf': [],
            'ce_rftn': [],
            'dose_ppf': [],
            'dose_rftn': [],
            'reward': [],
            'time': []
        }
        
        # Apply induction bolus if enabled
        if self.enable_bolus:
            self.simulator.administer_bolus(self.bolus_dose, drug="propofol")
            # Step forward a bit to let bolus take effect
            for _ in range(3):  # 15 seconds
                rftn_rate = self.rftn_schedule.get_rate(self.simulator.time, self.dt)
                _, bis = self.simulator.step(0.0, rftn_rate=rftn_rate)
                self.bis_history.append(bis)
                self.prev_bis = bis
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            'bis': self.prev_bis,
            'ce_ppf': self.simulator.state_ppf[3],
            'ce_rftn': self.simulator.state_rftn[3],
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
            action: Normalized propofol infusion rate [0, 1]
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert normalized action to actual propofol dose
        action = np.clip(action, 0, 1)
        dose_ppf = float(action[0]) * self.dose_max
        
        # Get remifentanil rate from external schedule
        dose_rftn = self.rftn_schedule.get_rate(self.simulator.time, self.dt)
        
        # Step the simulator with both drugs
        state, bis = self.simulator.step(dose_ppf, rftn_rate=dose_rftn)
        
        # Update BIS history for slope calculation (Formulation 36)
        self.bis_history.append(bis)
        if len(self.bis_history) > 10:
            self.bis_history.pop(0)
        
        # Update dose histories for cumulative calculation (Formulations 38-39)
        self.dose_ppf_history.append(dose_ppf)
        self.dose_rftn_history.append(dose_rftn)
        if len(self.dose_ppf_history) > self.history_window:
            self.dose_ppf_history.pop(0)
        if len(self.dose_rftn_history) > self.history_window:
            self.dose_rftn_history.pop(0)
        
        # Calculate reward
        reward = self._calculate_reward(bis, dose_ppf)
        
        # Record episode history
        self.episode_history['bis'].append(bis)
        self.episode_history['ce_ppf'].append(self.simulator.state_ppf[3])
        self.episode_history['ce_rftn'].append(self.simulator.state_rftn[3])
        self.episode_history['dose_ppf'].append(dose_ppf)
        self.episode_history['dose_rftn'].append(dose_rftn)
        self.episode_history['reward'].append(reward)
        self.episode_history['time'].append(self.simulator.time)
        
        # Update state
        self.prev_bis = bis
        self.prev_dose = dose_ppf
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
            'ce_ppf': self.simulator.state_ppf[3],
            'ce_rftn': self.simulator.state_rftn[3],
            'dose_ppf': dose_ppf,
            'dose_rftn': dose_rftn,
            'time': self.simulator.time,
            'step': self.current_step,
            'in_target_range': self.bis_min <= bis <= self.bis_max
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct extended observation vector following CBIM paper Fig.4.
        
        Returns:
            Observation array: [bis_error, bis_slope, ce_ppf, ce_rftn, 
                               ppf_acc, rftn_acc, prev_dose, time]
        """
        # Current BIS and concentrations
        bis = self.prev_bis
        ce_ppf = self.simulator.state_ppf[3]
        ce_rftn = self.simulator.state_rftn[3]
        ec50_ppf = self.simulator.ppf_model.params.ec50
        ec50_rftn = self.simulator.rftn_model.params.ec50
        
        # Formulation (37): BIS error = BIS - target
        bis_error_norm = (bis - self.bis_target) / 50.0
        
        # Formulation (36): BIS slope = dBIS/dt
        if len(self.bis_history) >= 2:
            bis_slope = (self.bis_history[-1] - self.bis_history[-2]) / self.dt
            bis_slope_norm = np.clip(bis_slope / 10.0, -1.0, 1.0)
        else:
            bis_slope_norm = 0.0
        
        # Normalized concentrations
        ce_ppf_norm = ce_ppf / ec50_ppf
        ce_rftn_norm = ce_rftn / ec50_rftn
        
        # Formulation (38): Cumulative propofol dose
        ppf_acc = sum(self.dose_ppf_history)
        ppf_acc_norm = ppf_acc / (self.dose_max * self.history_window)
        
        # Formulation (39): Cumulative remifentanil dose
        rftn_acc = sum(self.dose_rftn_history)
        rftn_acc_norm = rftn_acc / (1.0 * self.history_window)  # Normalize by typical max
        
        # Normalized previous dose
        prev_dose_norm = self.prev_dose / self.dose_max
        
        # Normalized time
        time_norm = self.current_step / self.max_steps
        
        return np.array([
            np.clip(bis_error_norm, -2.0, 2.0),
            bis_slope_norm,
            np.clip(ce_ppf_norm, 0.0, 5.0),
            np.clip(ce_rftn_norm, 0.0, 5.0),
            np.clip(ppf_acc_norm, 0.0, 5.0),
            np.clip(rftn_acc_norm, 0.0, 5.0),
            prev_dose_norm,
            time_norm
        ], dtype=np.float32)
    
    def _calculate_reward(self, bis: float, dose: float) -> float:
        """
        Calculate reward based on CBIM paper Formulation (40).
        
        Formulation (40): R(s,a) = 1 / (|g - BIS(t+1)| + α)
        
        Or shaped reward with multiple components:
        1. BIS error penalty: -α * ((BIS - target) / target)²
        2. Dose magnitude penalty: -β * dose
        3. Dose change penalty: -γ * |dose - prev_dose|
        4. Stability bonus: +δ if BIS in target range
        5. Safety penalties for dangerous states
        
        Args:
            bis: Current BIS value
            dose: Current propofol infusion rate
        
        Returns:
            Total reward value
        """
        s = self.safety_config
        
        if self.reward_type == "paper":
            # Formulation (40): R(s,a) = 1 / (|g - BIS| + α)
            bis_error = abs(self.bis_target - bis)
            reward = 1.0 / (bis_error + self.reward_alpha)
            
            # Scale to reasonable range [0, 1] -> [0, 0.1] or so
            reward = reward * 0.1
            
        else:
            # Shaped reward (original implementation)
            w = self.reward_weights
            
            # 1. BIS error penalty (Performance Error based)
            pe = (bis - self.bis_target) / self.bis_target
            bis_error_penalty = -w['bis_error'] * (pe ** 2)
            
            # 2. Dose magnitude penalty
            dose_penalty = -w['dose_penalty'] * (dose / self.dose_max)
            
            # 3. Dose change penalty
            dose_change = abs(dose - self.prev_dose) / self.dose_max
            change_penalty = -w['dose_change'] * dose_change
            
            # 4. Stability bonus
            if self.bis_min <= bis <= self.bis_max:
                stability_bonus = w['stability_bonus']
            else:
                stability_bonus = 0.0
            
            reward = bis_error_penalty + dose_penalty + change_penalty + stability_bonus
        
        # Safety penalties (applied to both reward types)
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
        
        return reward + safety_penalty
    
    def get_episode_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics for the episode.
        
        Following CBIM paper metrics - Formulations (50)-(52):
        - Formulation (50): PE = 100 * (BIS - target) / target
        - Formulation (51): MDPE = median(PE)
        - Formulation (52): MDAPE = median(|PE|)
        - Wobble: Intra-individual variability
        - Time in Target: Percentage of time BIS in target range
        
        Returns:
            Dictionary of performance metrics
        """
        if len(self.episode_history['bis']) == 0:
            return {}
        
        bis_array = np.array(self.episode_history['bis'])
        
        # Formulation (50): Performance Error (PE)
        pe = (bis_array - self.bis_target) / self.bis_target * 100
        
        # Formulation (51): MDPE - Median Performance Error
        mdpe = np.median(pe)
        
        # Formulation (52): MDAPE - Median Absolute Performance Error
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
        
        # Mean doses
        mean_dose_ppf = np.mean(self.episode_history['dose_ppf'])
        mean_dose_rftn = np.mean(self.episode_history['dose_rftn'])
        
        return {
            'mdpe': mdpe,
            'mdape': mdape,
            'wobble': wobble,
            'time_in_target': time_in_target,
            'total_reward': total_reward,
            'mean_dose_ppf': mean_dose_ppf,
            'mean_dose_rftn': mean_dose_rftn,
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
