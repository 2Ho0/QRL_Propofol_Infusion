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
from collections import deque

# Import PatientSimulator as the physics engine
from .patient_simulator import PatientSimulator, BISModelType, DrugInteractionParams
from models.pharmacokinetics import PatientParameters


@dataclass
class DualDrugPatientParams:
    """
    Patient-specific computed PK/PD parameters for dual drug environment.
    
    NOTE: This is different from models.pharmacokinetics.PatientParameters,
    which stores basic demographics. This class stores computed volumes,
    clearances, and rate constants for both drugs.
    """
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
) -> DualDrugPatientParams:
    """
    Create patient-specific PK/PD parameters with inter-individual variability.
    
    Args:
        age: Patient age in years
        weight: Patient weight in kg
        height: Patient height in cm
        gender: 0 for female, 1 for male
        variability: Coefficient of variation for parameters
    
    Returns:
        DualDrugPatientParams with computed PK/PD parameters
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
    
    return DualDrugPatientParams(
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
        Box(13): Extended state with dual drug information + patient demographics
        - Includes: BIS, drug concentrations, derivatives, accumulations, age, sex, BMI
    
    Reward:
        Based on BIS error, drug consumption, and safety
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        target_bis: float = 50.0,
        bis_noise_std: float = 2.0,
        max_steps: int = 360,  # 360 minutes = 6 hours (typical surgery duration)
        dt: float = 1.0,
        patient_params: Optional[DualDrugPatientParams] = None,
        patient: Optional[PatientParameters] = None,
        seed: Optional[int] = None,
        reward_type: str = 'potential',
        initial_bis: Optional[float] = None
    ):
        """
        Initialize dual drug environment.
        
        Args:
            target_bis: Target BIS value (40-60 for general anesthesia)
            bis_noise_std: Standard deviation of BIS measurement noise
            max_steps: Maximum episode length (minutes)
            dt: Time step (minutes)
            patient_params: Legacy patient-specific parameters (deprecated)
            patient: PatientParameters object (preferred)
            seed: Random seed
            reward_type: Reward function type ('simple', 'paper', 'hybrid', 'potential')
            initial_bis: Initial BIS value (None=start from 98, or specify 40-60 for VitalDB-like start)
        
        Note:
            This environment now uses PatientSimulator as the internal physics engine
            for accurate PK/PD simulation using ODE integration.
        """
        super().__init__()
        
        self.target_bis = target_bis
        self.bis_noise_std = bis_noise_std
        self.initial_bis = initial_bis  # Store for reset
        self.max_steps = max_steps
        self.dt = dt  # Time step in minutes
        self.reward_type = reward_type
        self.gamma = 0.99  # Discount factor for potential-based shaping
        
        if seed is not None:
            np.random.seed(seed)
        
        # Action space: [propofol_rate, remifentanil_rate]
        # Ranges match VitalDB data:
        # - Propofol: 0-30 mg/kg/h (typical: 4-12, max observed ~20)
        # - Remifentanil: 0-1.0 μg/kg/min (typical: 0.05-0.3, max observed ~0.9)
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([30.0, 1.0]),  # [mg/kg/h, μg/kg/min]
            dtype=np.float32
        )
        
        # Observation space: 13D extended state (with patient demographics)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32
        )
        
        # Create patient for PatientSimulator
        if patient is not None:
            self.patient_obj = patient
        elif patient_params is not None:
            # Convert legacy DualDrugPatientParams to PatientParameters
            self.patient_obj = PatientParameters(
                age=patient_params.age,
                weight=patient_params.weight,
                height=patient_params.height,
                gender='male' if patient_params.gender == 1 else 'female'
            )
            # Keep legacy params for backward compatibility
            self.patient = patient_params
        else:
            # Create default patient
            self.patient_obj = PatientParameters()
        
        # Initialize PatientSimulator as the physics engine
        self.simulator = PatientSimulator(
            patient=self.patient_obj,
            dt=dt * 60,  # Convert minutes to seconds for simulator
            noise_std=bis_noise_std,
            seed=seed,
            bis_model_type=BISModelType.DRUG_INTERACTION,
            drug_interaction_params=DrugInteractionParams(
                bis0=98.0,
                c50_ppf=4.47,
                c50_rftn=19.3,
                gamma=1.43,
                noise_std=bis_noise_std
            )
        )
        
        # Initialize state tracking
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
        
        # Reset PatientSimulator (physics engine)
        simulator_state = self.simulator.reset()
        
        # Set initial BIS (either from parameter or default from simulator)
        if self.initial_bis is not None:
            # Initialize with target BIS by setting appropriate drug concentrations
            # Use inverse PD model to find required concentrations
            target_ce_ppf, target_ce_remi = self._compute_target_concentrations(self.initial_bis)
            self.simulator.state_ppf[3] = target_ce_ppf  # Effect-site
            self.simulator.state_ppf[0] = target_ce_ppf  # Central compartment
            self.simulator.state_rftn[3] = target_ce_remi
            self.simulator.state_rftn[0] = target_ce_remi
            self.bis = self.initial_bis
        else:
            # Get initial BIS from simulator (default: 98)
            self.bis = self.simulator.get_bis()
        
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
        
        # Episode history for metrics tracking
        self.episode_history = {
            'bis': [],
            'ce_propofol': [],
            'ce_remifentanil': [],
            'action_propofol': [],
            'action_remifentanil': [],
            'reward': []
        }
        
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
        
        # Update physics using PatientSimulator
        # CRITICAL UNIT CONVERSION:
        # Environment units: mg/kg/h (propofol), μg/kg/min (remifentanil)
        # Simulator expects: μg/kg/min (propofol), ng/kg/min (remifentanil)
        
        # Propofol: mg/kg/h → μg/kg/min
        # 1 mg/kg/h = (1/60) mg/kg/min = (1000/60) μg/kg/min
        ppf_rate_ug_per_min = (ppf_rate / 60.0) * 1000.0  # μg/kg/min
        
        # Remifentanil: μg/kg/min → ng/kg/min
        # 1 μg/kg/min = 1000 ng/kg/min
        rftn_rate_ng_per_min = rftn_rate * 1000.0  # ng/kg/min
        
        # PatientSimulator.step expects rates in units per minute and dt in seconds
        simulator_state, bis_value = self.simulator.step(
            infusion_rate=ppf_rate_ug_per_min,  # μg/kg/min
            dt=self.dt * 60,  # Convert minutes to seconds
            rftn_rate=rftn_rate_ng_per_min  # ng/kg/min
        )
        
        # Get BIS from simulator
        self.bis = bis_value
        
        # Update history
        self.bis_history.append(self.bis)
        self.bis_window.append(self.bis)
        self.ppf_accumulation.append(ppf_rate)
        self.rftn_accumulation.append(rftn_rate)
        
        # Store previous BIS for potential-based shaping
        prev_bis = self.bis_history[-2] if len(self.bis_history) >= 2 else self.bis
        
        # Compute reward
        reward = self._compute_reward(ppf_rate, rftn_rate, prev_bis)
        
        # Track episode history for metrics
        self.episode_history['bis'].append(self.bis)
        self.episode_history['ce_propofol'].append(self.simulator.state_ppf[3])
        self.episode_history['ce_remifentanil'].append(self.simulator.state_rftn[3])
        self.episode_history['action_propofol'].append(ppf_rate)
        self.episode_history['action_remifentanil'].append(rftn_rate)
        self.episode_history['reward'].append(reward)
        
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
    
    # Note: _update_pk_pd and _compute_bis are now handled by PatientSimulator
    # This eliminates code duplication and uses accurate ODE integration
    
    def _potential_function(self, bis: float) -> float:
        """
        Potential function Φ(s) for reward shaping.
        
        This function represents the "value" or "promise" of being in a state
        with a given BIS value. Higher potential means closer to the goal.
        
        The potential function is designed to:
        - Be highest when BIS is at target (50)
        - Decrease smoothly as BIS deviates from target
        - Be continuous and differentiable for stable gradients
        
        Args:
            bis: Current BIS value
            
        Returns:
            Potential value (higher is better)
        """
        e_bis = abs(bis - self.target_bis)
        
        # Zone-based potential (mirrors reward structure)
        if e_bis <= 5:
            # Perfect zone: highest potential
            return 10.0
        elif e_bis <= 10:
            # Acceptable zone: linear decay from 10.0 to 5.0
            return 10.0 - (e_bis - 5) * 1.0
        elif e_bis <= 20:
            # Outside target: linear decay from 5.0 to 2.5
            return 5.0 - (e_bis - 10) * 0.25
        else:
            # Far from target: slow decay, minimum 0
            return max(0.0, 2.5 - (e_bis - 20) * 0.05)
    
    def _compute_reward(self, ppf_rate: float, rftn_rate: float, prev_bis: float) -> float:
        """
        Compute reward using potential-based reward shaping.
        
        Formula: r_total = r_base + F(s, s')
        where F(s, s') = γΦ(s') - Φ(s)
        
        This method guarantees:
        1. Optimal policy is preserved (Ng et al. 1999)
        2. Faster learning through shaped rewards
        3. Smooth gradients throughout state space
        
        Args:
            ppf_rate: Propofol infusion rate (mg/kg/h)
            rftn_rate: Remifentanil infusion rate (μg/kg/min)
            prev_bis: Previous BIS value (for potential shaping)
            
        Returns:
            Total reward (base + potential shaping)
        """
        if self.reward_type == 'potential':
            return self._reward_potential_based(ppf_rate, rftn_rate, prev_bis)
        elif self.reward_type == 'paper':
            return self._reward_paper(ppf_rate, rftn_rate)
        elif self.reward_type == 'hybrid':
            return self._reward_hybrid(ppf_rate, rftn_rate)
        else:  # 'simple' or default
            return self._reward_simple(ppf_rate, rftn_rate)
    
    def _reward_potential_based(self, ppf_rate: float, rftn_rate: float, prev_bis: float) -> float:
        """
        Potential-based reward shaping (Ng et al. 1999).
        
        Combines:
        1. Base reward (paper formulation)
        2. Potential-based shaping: F(s,s') = γΦ(s') - Φ(s)
        
        This preserves optimal policy while accelerating learning.
        """
        # === Base Reward (Paper Formulation) ===
        e_bis = abs(self.bis - self.target_bis)
        
        # Weights (from paper)
        w1, w2, w3 = 1.0, 0.5, 0.1
        
        # R_track: BIS tracking reward
        if e_bis <= 5:
            r_track = 1.0
        elif e_bis <= 10:
            r_track = 0.5 * (1 - (e_bis - 5) / 5)
        else:
            r_track = -0.5 * min((e_bis - 10) / 40, 1.0)
        
        # R_safe: Safety reward
        if self.bis < 30:
            r_safe = -2.0 * ((30 - self.bis) / 30)
        elif self.bis < 40:
            r_safe = -0.5 * ((40 - self.bis) / 10)
        elif self.bis <= 60:
            r_safe = 0.0
        elif self.bis <= 70:
            r_safe = -0.2 * ((self.bis - 60) / 10)
        else:
            r_safe = -1.0 * min((self.bis - 70) / 30, 1.0)
        
        # R_eff: Drug efficiency
        normalized_ppf = ppf_rate / 12.0  # Normalize by typical max
        normalized_remi = rftn_rate / 0.3
        r_eff = -(normalized_ppf + normalized_remi)
        
        # Base reward
        r_base = w1 * r_track + w2 * r_safe + w3 * r_eff
        
        # === Potential-based Shaping ===
        # F(s, s') = γΦ(s') - Φ(s)
        phi_prev = self._potential_function(prev_bis)
        phi_current = self._potential_function(self.bis)
        shaping = self.gamma * phi_current - phi_prev
        
        # Total reward
        r_total = r_base + shaping
        
        return r_total
    
    def _reward_paper(self, ppf_rate: float, rftn_rate: float) -> float:
        """
        Paper formulation (Equation 1-3) without potential shaping.
        
        r_t = w1 * R_track + w2 * R_safe + w3 * R_eff
        """
        e_bis = abs(self.bis - self.target_bis)
        
        # Weights
        w1, w2, w3 = 1.0, 0.5, 0.1
        
        # R_track
        if e_bis <= 5:
            r_track = 1.0
        elif e_bis <= 10:
            r_track = 0.5 * (1 - (e_bis - 5) / 5)
        else:
            r_track = -0.5 * min((e_bis - 10) / 40, 1.0)
        
        # R_safe
        if self.bis < 30:
            r_safe = -2.0 * ((30 - self.bis) / 30)
        elif self.bis < 40:
            r_safe = -0.5 * ((40 - self.bis) / 10)
        elif self.bis <= 60:
            r_safe = 0.0
        elif self.bis <= 70:
            r_safe = -0.2 * ((self.bis - 60) / 10)
        else:
            r_safe = -1.0 * min((self.bis - 70) / 30, 1.0)
        
        # R_eff
        normalized_ppf = ppf_rate / 12.0
        normalized_remi = rftn_rate / 0.3
        r_eff = -(normalized_ppf + normalized_remi)
        
        return w1 * r_track + w2 * r_safe + w3 * r_eff
    
    def _reward_hybrid(self, ppf_rate: float, rftn_rate: float) -> float:
        """
        Hybrid: Dense (paper) + Sparse (bonus/penalty).
        
        Combines continuous paper formulation with discrete bonuses/penalties.
        """
        # Dense component (paper)
        r_dense = self._reward_paper(ppf_rate, rftn_rate)
        
        # Sparse component
        r_sparse = 0.0
        
        # Perfect zone bonus
        if 45 <= self.bis <= 55:
            r_sparse += 10.0
        elif 40 <= self.bis <= 60:
            r_sparse += 3.0
        
        # Danger penalties
        if self.bis < 20:
            r_sparse -= 20.0
        elif self.bis < 30:
            r_sparse -= 5.0
        elif self.bis > 80:
            r_sparse -= 20.0
        elif self.bis > 70:
            r_sparse -= 5.0
        
        # Combine (0.7 dense + 0.3 sparse)
        return 0.7 * r_dense + 0.3 * r_sparse
    
    def _reward_simple(self, ppf_rate: float, rftn_rate: float) -> float:
        """
        Simple reward (original implementation).
        
        Enhanced reward components:
        1. BIS tracking: Strong penalty for deviation from target (amplified)
        2. Target range bonus: Large bonus for staying in optimal range
        3. Drug efficiency: Penalize excessive drug use
        4. Safety: Heavy penalties for dangerous states
        5. Smoothness: Penalize rapid action changes
        """
        bis_error = abs(self.target_bis - self.bis)
        
        # 1. BIS tracking reward (amplified by 10x for stronger signal)
        bis_reward = -bis_error * 10.0  # Higher penalty for deviation
        
        # 2. Target range bonuses (encourage staying in optimal BIS range)
        if 45 <= self.bis <= 55:  # Optimal range
            bis_reward += 50.0  # Large bonus
        elif 40 <= self.bis <= 60:  # Acceptable range
            bis_reward += 20.0  # Moderate bonus
        
        # 3. Drug efficiency penalty (adjusted for new action space)
        # Penalize excessive drug use, with remifentanil weighted more (more potent)
        drug_penalty = -0.01 * (ppf_rate + rftn_rate * 10.0)
        
        # 4. Safety penalties (stronger)
        safety_penalty = 0.0
        if self.bis < 30:  # Dangerously deep
            safety_penalty = -100.0
        elif self.bis < 20:  # Critically deep
            safety_penalty = -200.0
        elif self.bis > 70:  # Risk of awareness
            safety_penalty = -100.0
        elif bis_error > 30:  # Far from target
            safety_penalty = -10.0
        
        # 5. Action smoothness reward (penalize rapid changes)
        action_change_ppf = abs(ppf_rate - self.prev_action_ppf)
        action_change_rftn = abs(rftn_rate - self.prev_action_rftn)
        smoothness_penalty = -0.1 * (action_change_ppf + action_change_rftn * 2.0)  # Weight rftn changes more
        
        # Total reward
        reward = bis_reward + drug_penalty + safety_penalty + smoothness_penalty
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (state).
        
        State vector (13D):
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
        [10] Age normalized (age / 100)
        [11] Sex (0=Female, 1=Male)
        [12] BMI normalized (bmi / 40)
        """
        # Get effect-site concentrations from simulator
        ce_ppf = self.simulator.state_ppf[3]  # Effect-site concentration (index 3)
        ce_rftn = self.simulator.state_rftn[3]  # Effect-site concentration
        
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
        
        # Interaction factor from drug interaction params
        interaction_factor = self.simulator.drug_interaction_params.gamma
        
        # Patient demographics (normalized)
        age_norm = self.patient_obj.age / 100.0  # 0-100 years → 0-1
        sex_numeric = 1.0 if self.patient_obj.gender == 'male' else 0.0
        # Compute BMI
        height_m = self.patient_obj.height / 100.0
        bmi = self.patient_obj.weight / (height_m ** 2)
        bmi_norm = bmi / 40.0  # 15-40 BMI → 0.375-1.0
        
        state = np.array([
            bis_error,
            ce_ppf,
            ce_rftn,
            dbis_dt,
            self.prev_action_ppf,
            self.prev_action_rftn,
            ppf_acc,
            rftn_acc,
            bis_slope,
            interaction_factor,
            age_norm,
            sex_numeric,
            bmi_norm
        ], dtype=np.float32)
        
        return state
    
    def _get_info(self) -> dict:
        """Get additional information."""
        return {
            'bis': self.bis,
            'ce_propofol': self.simulator.state_ppf[3],
            'ce_remifentanil': self.simulator.state_rftn[3],
            'c1_propofol': self.simulator.state_ppf[0],
            'c1_remifentanil': self.simulator.state_rftn[0],
            'step': self.current_step,
            'patient_age': self.patient_obj.age,
            'patient_weight': self.patient_obj.weight,
            'simulator': 'PatientSimulator (ODE-based)'
        }
    
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
        pe = (bis_array - self.target_bis) / self.target_bis * 100
        
        # Formulation (51): MDPE - Median Performance Error
        mdpe = np.median(pe)
        
        # Formulation (52): MDAPE - Median Absolute Performance Error
        mdape = np.median(np.abs(pe))
        
        # Wobble: Median absolute deviation from MDPE
        wobble = np.median(np.abs(pe - mdpe))
        
        # Time in Target Range (BIS 45-55)
        bis_min = 45.0
        bis_max = 55.0
        in_target = np.logical_and(
            bis_array >= bis_min,
            bis_array <= bis_max
        )
        time_in_target = np.mean(in_target) * 100
        
        # Induction Time: Time to reach target range from awake state
        induction_time = None
        for i, bis_val in enumerate(bis_array):
            if bis_min <= bis_val <= bis_max:
                induction_time = i * self.dt  # Minutes
                break
        
        # Total reward
        total_reward = sum(self.episode_history['reward'])
        
        # Mean drug doses
        mean_dose_ppf = np.mean(self.episode_history['action_propofol'])
        mean_dose_rftn = np.mean(self.episode_history['action_remifentanil'])
        
        # Performance Error statistics
        pe_mean = np.mean(pe)
        pe_std = np.std(pe)
        
        return {
            'mdpe': mdpe,                      # Formulation (51)
            'mdape': mdape,                    # Formulation (52)
            'wobble': wobble,                  # Intra-individual variability
            'time_in_target': time_in_target,  # % time in BIS 45-55
            'induction_time': induction_time,  # Minutes to reach target
            'pe_mean': pe_mean,                # Mean performance error
            'pe_std': pe_std,                  # PE variability
            'total_reward': total_reward,
            'mean_dose_ppf': mean_dose_ppf,    # mg/kg/h
            'mean_dose_rftn': mean_dose_rftn,  # μg/kg/min
            'final_bis': bis_array[-1] if len(bis_array) > 0 else None
        }
    
    def _compute_target_concentrations(self, target_bis: float) -> Tuple[float, float]:
        """
        Compute drug concentrations needed to achieve target BIS.
        
        Uses inverse PD model to find propofol and remifentanil concentrations
        that result in the desired BIS value.
        
        Args:
            target_bis: Desired BIS value (e.g., 50)
        
        Returns:
            (ce_ppf, ce_rftn): Effect-site concentrations for propofol and remifentanil
        """
        # Use typical clinical ratios from VitalDB data
        # Propofol: ~4-6 mg/kg/h -> Ce ~3-5 μg/ml
        # Remifentanil: ~0.1-0.15 μg/kg/min -> Ce ~3-5 ng/ml
        
        # From drug interaction model: BIS = 98 * (1 + Ce_ppf/4.47 + Ce_remi/19.3)^(-1.43)
        # Rearrange: (98/BIS)^(1/1.43) = 1 + Ce_ppf/4.47 + Ce_remi/19.3
        
        if target_bis >= 98:
            return 0.0, 0.0
        
        # Target ratio from equation
        target_ratio = (98.0 / target_bis) ** (1.0 / 1.43) - 1.0
        
        # Use typical clinical ratio: propofol contributes ~60%, remifentanil ~40%
        ppf_contribution = 0.6 * target_ratio
        remi_contribution = 0.4 * target_ratio
        
        # Convert to concentrations
        ce_ppf = ppf_contribution * 4.47  # μg/ml
        ce_rftn = remi_contribution * 19.3  # ng/ml
        
        return ce_ppf, ce_rftn
    
    def render(self):
        """Render environment (text-based)."""
        if self.current_step % 10 == 0:  # Print every 10 steps
            ce_ppf = self.simulator.state_ppf[3]
            ce_rftn = self.simulator.state_rftn[3]
            print(f"Step {self.current_step:3d} | "
                  f"BIS: {self.bis:5.1f} | "
                  f"Ce_PPF: {ce_ppf:5.2f} | "
                  f"Ce_RFTN: {ce_rftn:5.2f} | "
                  f"PPF: {self.prev_action_ppf:5.1f} | "
                  f"RFTN: {self.prev_action_rftn:5.1f}")


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
