"""
Dual Drug Environment: Propofol + Remifentanil
===============================================

Two-drug interaction model for anesthesia control based on:
- Schnider model for Propofol (3-compartment PK/PD)
- Minto model for Remifentanil (3-compartment PK/PD)  
- Synergistic drug interaction on BIS effect

Action Space: 2D [propofol_rate, remifentanil_rate]
State Space: 10D [BIS_error, Ce_PPF, Ce_RFTN, dBIS/dt, u_ppf, u_rftn, 
                  PPF_acc, RFTN_acc, BIS_slope, interaction_factor]

Based on CBIM Paper Formulations:
- Schnider (1-17): Propofol pharmacokinetics
- Minto (18-29): Remifentanil pharmacokinetics
- Drug Interaction (32): Combined BIS effect
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class PatientParameters:
    """Patient-specific parameters for PK/PD models."""
    # Demographics
    age: float  # years
    weight: float  # kg
    height: float  # cm
    gender: int  # 0: female, 1: male
    
    # Propofol (Schnider) parameters
    v1_ppf: float  # Central compartment volume
    v2_ppf: float  # Rapid peripheral compartment
    v3_ppf: float  # Slow peripheral compartment
    cl1_ppf: float  # Central clearance
    cl2_ppf: float  # Rapid clearance
    cl3_ppf: float  # Slow clearance
    ke0_ppf: float  # Effect-site rate constant
    
    # Remifentanil (Minto) parameters  
    v1_rftn: float  # Central compartment volume
    v2_rftn: float  # Rapid peripheral compartment
    v3_rftn: float  # Slow peripheral compartment
    cl1_rftn: float  # Central clearance
    cl2_rftn: float  # Rapid clearance
    cl3_rftn: float  # Slow clearance
    ke0_rftn: float  # Effect-site rate constant
    
    # Interaction parameters
    interaction_strength: float  # Synergy factor (0-1)


def create_patient_parameters(
    age: float = 40.0,
    weight: float = 70.0,
    height: float = 170.0,
    gender: int = 1,
    variability: float = 0.2
) -> PatientParameters:
    """
    Create patient-specific PK/PD parameters with inter-individual variability.
    
    Args:
        age: Patient age in years
        weight: Patient weight in kg
        height: Patient height in cm
        gender: 0 for female, 1 for male
        variability: Coefficient of variation for parameters
    
    Returns:
        PatientParameters with computed PK/PD parameters
    """
    # Lean body mass (LBM) for Schnider model
    if gender == 1:  # Male
        lbm = 1.10 * weight - 128 * (weight / height) ** 2
    else:  # Female
        lbm = 1.07 * weight - 148 * (weight / height) ** 2
    
    # Propofol (Schnider) parameters
    v1_ppf = 4.27  # L
    v2_ppf = 18.9 - 0.391 * (age - 53)  # L
    v3_ppf = 238  # L
    
    cl1_ppf = 1.89 + 0.0456 * (weight - 77) - 0.0681 * (lbm - 59) + 0.0264 * (height - 177)  # L/min
    cl2_ppf = 1.29 - 0.024 * (age - 53)  # L/min
    cl3_ppf = 0.836  # L/min
    
    ke0_ppf = 0.456  # 1/min (effect-site equilibration)
    
    # Remifentanil (Minto) parameters
    v1_rftn = 5.1 - 0.0201 * (age - 40) + 0.072 * (lbm - 55)  # L
    v2_rftn = 9.82 - 0.0811 * (age - 40) + 0.108 * (lbm - 55)  # L
    v3_rftn = 5.42  # L
    
    cl1_rftn = 2.6 - 0.0162 * (age - 40) + 0.0191 * (lbm - 55)  # L/min
    cl2_rftn = 2.05 - 0.0301 * (age - 40)  # L/min
    cl3_rftn = 0.076 - 0.00113 * (age - 40)  # L/min
    
    ke0_rftn = 0.595 - 0.007 * (age - 40)  # 1/min
    
    # Add inter-individual variability
    def add_variability(value, cv=variability):
        return value * np.random.lognormal(0, cv)
    
    # Interaction strength (synergistic effect)
    interaction_strength = np.random.uniform(0.3, 0.5)  # Moderate synergy
    
    return PatientParameters(
        age=age,
        weight=weight,
        height=height,
        gender=gender,
        v1_ppf=add_variability(v1_ppf),
        v2_ppf=add_variability(v2_ppf),
        v3_ppf=add_variability(v3_ppf),
        cl1_ppf=add_variability(cl1_ppf),
        cl2_ppf=add_variability(cl2_ppf),
        cl3_ppf=add_variability(cl3_ppf),
        ke0_ppf=add_variability(ke0_ppf, 0.1),
        v1_rftn=add_variability(v1_rftn),
        v2_rftn=add_variability(v2_rftn),
        v3_rftn=add_variability(v3_rftn),
        cl1_rftn=add_variability(cl1_rftn),
        cl2_rftn=add_variability(cl2_rftn),
        cl3_rftn=add_variability(cl3_rftn),
        ke0_rftn=add_variability(ke0_rftn, 0.1),
        interaction_strength=interaction_strength
    )


class DualDrugEnv(gym.Env):
    """
    Gymnasium environment for dual drug (Propofol + Remifentanil) anesthesia control.
    
    The agent controls both propofol and remifentanil infusion rates to maintain
    target BIS level, accounting for drug interaction effects.
    
    Action Space:
        Box(2): [propofol_rate, remifentanil_rate]
        - propofol_rate: 0-30 mg/kg/h
        - remifentanil_rate: 0-50 μg/kg/min
    
    Observation Space:
        Box(10): Extended state with dual drug information
    
    Reward:
        Based on BIS error, drug consumption, and safety
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        target_bis: float = 50.0,
        bis_noise_std: float = 2.0,
        max_steps: int = 200,
        dt: float = 1.0,
        patient_params: Optional[PatientParameters] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize dual drug environment.
        
        Args:
            target_bis: Target BIS value (40-60 for general anesthesia)
            bis_noise_std: Standard deviation of BIS measurement noise
            max_steps: Maximum episode length (minutes)
            dt: Time step (minutes)
            patient_params: Patient-specific parameters
            seed: Random seed
        """
        super().__init__()
        
        self.target_bis = target_bis
        self.bis_noise_std = bis_noise_std
        self.max_steps = max_steps
        self.dt = dt  # Time step in minutes
        
        if seed is not None:
            np.random.seed(seed)
        
        # Action space: [propofol_rate, remifentanil_rate]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([30.0, 50.0]),  # [mg/kg/h, μg/kg/min]
            dtype=np.float32
        )
        
        # Observation space: 10D extended state
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32
        )
        
        # Patient parameters
        if patient_params is None:
            self.patient = create_patient_parameters()
        else:
            self.patient = patient_params
        
        # PK/PD state variables
        self.reset()
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset time
        self.current_step = 0
        
        # Initial PK/PD state (all concentrations start at 0)
        # Propofol compartments: C1, C2, C3, Ce
        self.c1_ppf = 0.0
        self.c2_ppf = 0.0
        self.c3_ppf = 0.0
        self.ce_ppf = 0.0
        
        # Remifentanil compartments: C1, C2, C3, Ce
        self.c1_rftn = 0.0
        self.c2_rftn = 0.0
        self.c3_rftn = 0.0
        self.ce_rftn = 0.0
        
        # Initial BIS (awake state)
        self.bis = 95.0
        self.bis_history = [self.bis]
        
        # Previous actions
        self.prev_action_ppf = 0.0
        self.prev_action_rftn = 0.0
        
        # Drug accumulation tracking (1-minute window)
        self.ppf_accumulation = deque(maxlen=60)
        self.rftn_accumulation = deque(maxlen=60)
        
        # BIS slope (3-minute window)
        self.bis_window = deque(maxlen=180)
        self.bis_window.append(self.bis)
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one time step.
        
        Args:
            action: [propofol_rate, remifentanil_rate]
        
        Returns:
            observation: Next state
            reward: Reward for this step
            terminated: Whether episode ended naturally
            truncated: Whether episode was cut off
            info: Additional information
        """
        # Clip actions to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        ppf_rate, rftn_rate = action
        
        # Update PK/PD models
        self._update_pk_pd(ppf_rate, rftn_rate)
        
        # Compute BIS from drug interaction
        self._compute_bis()
        
        # Update history
        self.bis_history.append(self.bis)
        self.bis_window.append(self.bis)
        self.ppf_accumulation.append(ppf_rate)
        self.rftn_accumulation.append(rftn_rate)
        
        # Compute reward
        reward = self._compute_reward(ppf_rate, rftn_rate)
        
        # Update step counter
        self.current_step += 1
        self.prev_action_ppf = ppf_rate
        self.prev_action_rftn = rftn_rate
        
        # Check termination
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # Get next observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _update_pk_pd(self, ppf_rate: float, rftn_rate: float):
        """
        Update pharmacokinetic/pharmacodynamic models.
        
        Uses 3-compartment models for both drugs with effect-site equilibration.
        """
        # Convert rates: propofol mg/kg/h → mg/kg/min, remifentanil μg/kg/min
        ppf_input = ppf_rate / 60.0  # mg/kg/min
        rftn_input = rftn_rate  # μg/kg/min
        
        # Propofol (Schnider model) - Formulations (1)-(17)
        # dC1/dt = input/V1 + (Cl2/V1)(C2-C1) + (Cl3/V1)(C3-C1) - (Cl1/V1)C1
        dc1_ppf = (
            ppf_input / self.patient.v1_ppf +
            (self.patient.cl2_ppf / self.patient.v1_ppf) * (self.c2_ppf - self.c1_ppf) +
            (self.patient.cl3_ppf / self.patient.v1_ppf) * (self.c3_ppf - self.c1_ppf) -
            (self.patient.cl1_ppf / self.patient.v1_ppf) * self.c1_ppf
        )
        
        dc2_ppf = (self.patient.cl2_ppf / self.patient.v2_ppf) * (self.c1_ppf - self.c2_ppf)
        dc3_ppf = (self.patient.cl3_ppf / self.patient.v3_ppf) * (self.c1_ppf - self.c3_ppf)
        dce_ppf = self.patient.ke0_ppf * (self.c1_ppf - self.ce_ppf)
        
        # Update propofol concentrations
        self.c1_ppf += dc1_ppf * self.dt
        self.c2_ppf += dc2_ppf * self.dt
        self.c3_ppf += dc3_ppf * self.dt
        self.ce_ppf += dce_ppf * self.dt
        
        # Remifentanil (Minto model) - Formulations (18)-(29)
        # Similar structure but different parameters
        dc1_rftn = (
            rftn_input / self.patient.v1_rftn +
            (self.patient.cl2_rftn / self.patient.v1_rftn) * (self.c2_rftn - self.c1_rftn) +
            (self.patient.cl3_rftn / self.patient.v1_rftn) * (self.c3_rftn - self.c1_rftn) -
            (self.patient.cl1_rftn / self.patient.v1_rftn) * self.c1_rftn
        )
        
        dc2_rftn = (self.patient.cl2_rftn / self.patient.v2_rftn) * (self.c1_rftn - self.c2_rftn)
        dc3_rftn = (self.patient.cl3_rftn / self.patient.v3_rftn) * (self.c1_rftn - self.c3_rftn)
        dce_rftn = self.patient.ke0_rftn * (self.c1_rftn - self.ce_rftn)
        
        # Update remifentanil concentrations
        self.c1_rftn += dc1_rftn * self.dt
        self.c2_rftn += dc2_rftn * self.dt
        self.c3_rftn += dc3_rftn * self.dt
        self.ce_rftn += dce_rftn * self.dt
    
    def _compute_bis(self):
        """
        Compute BIS from drug interaction model.
        
        Based on formulation (32): Synergistic drug interaction
        BIS = f(Ce_propofol, Ce_remifentanil, interaction_strength)
        
        Models:
        1. Independent effect: Each drug reduces BIS separately
        2. Synergistic effect: Combined effect > sum of individual effects
        """
        # Individual effects (Hill equation)
        # E_max = 98, E_0 = 0, EC50_ppf = 4.47, EC50_rftn = 19.3
        e_max = 98.0
        ec50_ppf = 4.47  # μg/mL for propofol
        ec50_rftn = 19.3  # ng/mL for remifentanil (0.0193 μg/mL)
        gamma = 1.43  # Hill coefficient
        
        # Individual effects
        if self.ce_ppf > 0:
            effect_ppf = (self.ce_ppf / ec50_ppf) ** gamma
        else:
            effect_ppf = 0.0
        
        if self.ce_rftn > 0:
            effect_rftn = (self.ce_rftn / ec50_rftn) ** gamma
        else:
            effect_rftn = 0.0
        
        # Synergistic interaction model
        # U = effect_ppf + effect_rftn + α * effect_ppf * effect_rftn
        interaction = self.patient.interaction_strength
        combined_effect = effect_ppf + effect_rftn + interaction * effect_ppf * effect_rftn
        
        # BIS calculation
        bis_true = e_max / (1 + combined_effect)
        
        # Add measurement noise
        noise = np.random.normal(0, self.bis_noise_std)
        self.bis = np.clip(bis_true + noise, 0, 100)
    
    def _compute_reward(self, ppf_rate: float, rftn_rate: float) -> float:
        """
        Compute reward for current state and action.
        
        Reward components:
        1. BIS tracking: Penalize deviation from target
        2. Drug efficiency: Penalize excessive drug use
        3. Safety: Penalize dangerous states (BIS too low/high)
        4. Smoothness: Penalize rapid action changes
        """
        # BIS tracking reward (formulation 40)
        bis_error = abs(self.target_bis - self.bis)
        bis_reward = 1.0 / (bis_error + 1.0)
        
        # Drug efficiency penalty
        # Penalize high drug consumption
        drug_penalty = -0.001 * (ppf_rate + 0.1 * rftn_rate)  # Weight rftn less (more potent)
        
        # Safety penalties
        safety_penalty = 0.0
        if self.bis < 20:  # Too deep (dangerous)
            safety_penalty -= 10.0
        elif self.bis > 70:  # Risk of awareness
            safety_penalty -= 5.0
        
        # Action smoothness reward (penalize rapid changes)
        action_change_ppf = abs(ppf_rate - self.prev_action_ppf)
        action_change_rftn = abs(rftn_rate - self.prev_action_rftn)
        smoothness_penalty = -0.1 * (action_change_ppf + action_change_rftn)
        
        # Total reward
        reward = bis_reward + drug_penalty + safety_penalty + smoothness_penalty
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (state).
        
        State vector (10D):
        [0] BIS error = target_bis - current_bis
        [1] Ce_propofol (effect-site concentration)
        [2] Ce_remifentanil (effect-site concentration)
        [3] dBIS/dt (BIS derivative)
        [4] u_ppf (previous propofol action)
        [5] u_rftn (previous remifentanil action)
        [6] PPF accumulation (1-min sum)
        [7] RFTN accumulation (1-min sum)
        [8] BIS slope (3-min linear fit)
        [9] Interaction factor
        """
        # BIS error
        bis_error = self.target_bis - self.bis
        
        # dBIS/dt (simple finite difference)
        if len(self.bis_history) >= 2:
            dbis_dt = (self.bis_history[-1] - self.bis_history[-2]) / self.dt
        else:
            dbis_dt = 0.0
        
        # Drug accumulations
        ppf_acc = sum(self.ppf_accumulation) if self.ppf_accumulation else 0.0
        rftn_acc = sum(self.rftn_accumulation) if self.rftn_accumulation else 0.0
        
        # BIS slope (linear regression over 3-min window)
        if len(self.bis_window) >= 3:
            time = np.arange(len(self.bis_window))
            bis_values = np.array(list(self.bis_window))
            bis_slope = np.polyfit(time, bis_values, 1)[0]
        else:
            bis_slope = 0.0
        
        # Interaction factor (current synergy level)
        interaction_factor = self.patient.interaction_strength
        
        state = np.array([
            bis_error,
            self.ce_ppf,
            self.ce_rftn,
            dbis_dt,
            self.prev_action_ppf,
            self.prev_action_rftn,
            ppf_acc,
            rftn_acc,
            bis_slope,
            interaction_factor
        ], dtype=np.float32)
        
        return state
    
    def _get_info(self) -> dict:
        """Get additional information."""
        return {
            'bis': self.bis,
            'ce_propofol': self.ce_ppf,
            'ce_remifentanil': self.ce_rftn,
            'step': self.current_step,
            'patient_age': self.patient.age,
            'patient_weight': self.patient.weight
        }
    
    def render(self):
        """Render environment (text-based)."""
        if self.current_step % 10 == 0:  # Print every 10 steps
            print(f"Step {self.current_step:3d} | "
                  f"BIS: {self.bis:5.1f} | "
                  f"Ce_PPF: {self.ce_ppf:5.2f} | "
                  f"Ce_RFTN: {self.ce_rftn:5.2f} | "
                  f"PPF: {self.prev_action_ppf:5.1f} | "
                  f"RFTN: {self.prev_action_rftn:5.1f}")


# Import deque for tracking
from collections import deque


# Test function
if __name__ == "__main__":
    print("Testing Dual Drug Environment...")
    print("=" * 70)
    
    # Create environment
    env = DualDrugEnv(seed=42)
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Target BIS: {env.target_bis}")
    print(f"Patient: {env.patient.age}y, {env.patient.weight}kg, {env.patient.height}cm")
    print("=" * 70)
    
    # Test episode
    obs, info = env.reset()
    print(f"\nInitial state: {obs}")
    print(f"Initial BIS: {info['bis']:.1f}")
    
    # Run a few steps with random actions
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
    
    print("\n✓ Dual Drug Environment test complete!")
