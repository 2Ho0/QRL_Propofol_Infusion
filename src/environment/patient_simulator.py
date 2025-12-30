"""
Patient Simulator - PK/PD Model for Propofol and Remifentanil
==============================================================

This module implements the pharmacokinetic/pharmacodynamic (PK/PD) model
for propofol (Schnider) and remifentanil (Minto) based on the CBIM paper.

The model consists of:
1. Three-compartment PK model (Central, Shallow Peripheral, Deep Peripheral)
2. Effect-site compartment with ke0 equilibration
3. Drug interaction model for BIS prediction

Mathematical Formulation (CBIM Paper):
--------------------------------------
State-Space Representation - Formulations (1)-(3):
    ẋ(t) = Ax(t) + Bu(t)
    
    where x = [x₁, x₂, x₃]ᵀ (compartment drug amounts in μg)
    
    Formulation (1): dx₁/dt = -(k₁₀+k₁₂+k₁₃)x₁ + k₂₁x₂ + k₃₁x₃ + u(t)
    Formulation (2): dx₂/dt = k₁₂x₁ - k₂₁x₂
    Formulation (3): dx₃/dt = k₁₃x₁ - k₃₁x₃
    
    A = [-(k₁₀+k₁₂+k₁₃)  k₂₁         k₃₁      ]
        [k₁₂             -k₂₁        0        ]
        [k₁₃             0           -k₃₁     ]
    
    B = [1, 0, 0]ᵀ

Effect-Site Model - Formulations (30)-(31):
    Formulation (30): ẋₑ(t) = -ke₀·xₑ(t) + k₁ₑ·x̂₁(t)
    Formulation (31): Ċₑ(t) = ke₀·(Cₚ(t) - Cₑ(t))

Drug Interaction BIS Model - Formulation (32):
    BIS(t) = 98.0 · (1 + Ĉₑ^PPF(t)/4.47 + Ĉₑ^RFTN(t)/19.3)^(-1.43) + ε

    Simplified (Propofol only) Hill/Sigmoid Emax:
    BIS = E0 - Emax * (Ce^γ / (Ce^γ + EC50^γ))

References:
-----------
- Schnider TW, et al. "The influence of method of administration and 
  covariates on the pharmacokinetics of propofol in adult volunteers."
  Anesthesiology. 1998.
- Minto CF, et al. "Pharmacokinetics and pharmacodynamics of remifentanil."
  Anesthesiology. 1997.
- CBIM (Closed-loop BIS-guided Infusion Model) paper
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Union
from scipy.integrate import odeint
from enum import Enum

# Import from centralized pharmacokinetics module
from models.pharmacokinetics import (
    BasePKModel,
    PatientParameters,
    MintoModel as MintoModelBase,
    MintoParameters,
    SchniderModel,
    SchniderParameters
)


class BISModelType(Enum):
    """Type of BIS prediction model."""
    HILL_SIGMOID = "hill_sigmoid"           # Propofol only Hill model
    DRUG_INTERACTION = "drug_interaction"   # Combined propofol-remifentanil model (Formulation 32)


@dataclass
class DrugInteractionParams:
    """
    Parameters for Drug Interaction BIS Model - Formulation (32).
    
    BIS(t) = BIS0 · (1 + e^(Ce_PPF/C50_PPF) + e^(Ce_RFTN/C50_RFTN))^(-gamma) + ε
    
    Default values from CBIM paper Table A.5.
    """
    bis0: float = 98.0           # Baseline BIS
    c50_ppf: float = 4.47        # Propofol C50 (μg/ml)
    c50_rftn: float = 19.3       # Remifentanil C50 (ng/ml)
    gamma: float = 1.43          # Interaction exponent
    noise_std: float = 2.0       # Measurement noise std


# PatientParameters is now imported from models.pharmacokinetics.base
# No need to redefine it here


# SchniderModel and SchniderParameters have been moved to:
# src/models/pharmacokinetics/schnider_model.py
# Import them using: from models.pharmacokinetics import SchniderModel, SchniderParameters


# MintoModelParameters and MintoModel classes have been moved to:
# src/models/pharmacokinetics/minto_model.py
# Import them using: from models.pharmacokinetics import MintoModel, MintoParameters


class PatientSimulator:
    """
    Patient Simulator for Propofol and Remifentanil Anesthesia.
    
    This simulator implements the complete PK/PD model for propofol
    and remifentanil anesthesia control, computing the dynamic response of 
    drug concentrations and BIS to drug infusions.
    
    Following CBIM paper formulations:
    - Formulations (1)-(3): State-space representation ẋ(t) = Ax(t) + Bu(t)
    - Formulations (4)-(17): Schnider model for Propofol
    - Formulations (18)-(29): Minto model for Remifentanil
    - Formulation (31): Effect-site dynamics Ċₑ = ke₀·(C₁ - Cₑ)
    - Formulation (32): Drug interaction BIS model
    
    State Variables (Propofol):
        - C1_ppf: Central compartment concentration (μg/ml)
        - C2_ppf: Shallow peripheral concentration (μg/ml)
        - C3_ppf: Deep peripheral concentration (μg/ml)
        - Ce_ppf: Effect-site concentration (μg/ml)
    
    State Variables (Remifentanil):
        - C1_rftn: Central compartment concentration (ng/ml)
        - C2_rftn: Shallow peripheral concentration (ng/ml)
        - C3_rftn: Deep peripheral concentration (ng/ml)
        - Ce_rftn: Effect-site concentration (ng/ml)
    
    Attributes:
        ppf_model: Schnider PK/PD model for propofol
        rftn_model: Minto PK/PD model for remifentanil
        state_ppf: Propofol concentrations [C1, C2, C3, Ce]
        state_rftn: Remifentanil concentrations [C1, C2, C3, Ce]
        time: Current simulation time (seconds)
        dt: Default time step (seconds)
        bis_model: Type of BIS model (Hill or Drug Interaction)
        history: Record of states, actions, and BIS values
    """
    
    def __init__(
        self,
        patient: Optional[PatientParameters] = None,
        model: Optional[SchniderModel] = None,
        dt: float = 5.0,
        noise_std: float = 2.0,
        seed: Optional[int] = None,
        bis_model_type: BISModelType = BISModelType.DRUG_INTERACTION,
        drug_interaction_params: Optional[DrugInteractionParams] = None
    ):
        """
        Initialize the patient simulator.
        
        Args:
            patient: Patient demographic parameters
            model: Pre-configured Schnider model (creates new if None)
            dt: Default time step in seconds
            noise_std: Standard deviation of BIS measurement noise
            seed: Random seed for reproducibility
            bis_model_type: Type of BIS model (HILL_SIGMOID or DRUG_INTERACTION)
            drug_interaction_params: Parameters for drug interaction model
        """
        # Propofol model (Schnider)
        if model is not None:
            self.ppf_model = model
            self.patient = model.patient
        else:
            self.patient = patient or PatientParameters()
            self.ppf_model = SchniderModel(patient=self.patient)
        
        # Remifentanil model (Minto) - using centralized model
        self.rftn_model = MintoModelBase(patient=self.patient)
        
        # Alias for backward compatibility
        self.model = self.ppf_model
        
        self.dt = dt
        self.noise_std = noise_std
        self.bis_model_type = bis_model_type
        self.drug_interaction_params = drug_interaction_params or DrugInteractionParams()
        
        # Random state for reproducibility
        self.rng = np.random.default_rng(seed)
        
        # Initialize state
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        Reset the simulator to initial conditions.
        
        Returns:
            Initial state vector [C1_ppf, C2_ppf, C3_ppf, Ce_ppf]
        """
        # All concentrations start at zero (no drug in system)
        self.state_ppf = np.zeros(4)   # [C1, C2, C3, Ce] for propofol
        self.state_rftn = np.zeros(4)  # [C1, C2, C3, Ce] for remifentanil
        
        # Alias for backward compatibility
        self.state = self.state_ppf
        
        self.time = 0.0
        
        # Clear history
        self.history = {
            'time': [],
            'state': [],           # Propofol state (backward compat)
            'state_ppf': [],
            'state_rftn': [],
            'bis': [],
            'dose': [],            # Propofol dose (backward compat)
            'dose_ppf': [],
            'dose_rftn': [],
            'ce': [],              # Propofol Ce (backward compat)
            'ce_ppf': [],
            'ce_rftn': []
        }
        
        return self.state_ppf.copy()
    
    def _pk_pd_ode_combined(
        self, 
        y: np.ndarray, 
        t: float, 
        ppf_rate: float, 
        rftn_rate: float
    ) -> np.ndarray:
        """
        Compute the derivatives for combined PK/PD ODE system.
        
        Implements Formulations (1)-(3) in state-space form:
        ẋ(t) = Ax(t) + Bu(t)
        
        Args:
            y: Combined state vector [C1_ppf, C2_ppf, C3_ppf, Ce_ppf, 
                                      C1_rftn, C2_rftn, C3_rftn, Ce_rftn]
            t: Current time (not used directly, required by odeint)
            ppf_rate: Propofol infusion rate (μg/kg/min)
            rftn_rate: Remifentanil infusion rate (ng/kg/min)
        
        Returns:
            Derivatives for all compartments
        """
        # Unpack states
        C1_ppf, C2_ppf, C3_ppf, Ce_ppf = y[:4]
        C1_rftn, C2_rftn, C3_rftn, Ce_rftn = y[4:]
        
        # Propofol PK/PD
        p = self.ppf_model.params
        weight = self.patient.weight
        
        # Formulation (1): dx₁/dt = -(k₁₀+k₁₂+k₁₃)x₁ + k₂₁x₂ + k₃₁x₃ + u(t)
        ppf_infusion_ug_per_min = ppf_rate * weight
        dC1_ppf = (ppf_infusion_ug_per_min / (p.V1 * 1000)
                   - (p.k10 + p.k12 + p.k13) * C1_ppf
                   + p.k21 * C2_ppf * p.V2 / p.V1
                   + p.k31 * C3_ppf * p.V3 / p.V1)
        
        # Formulation (2): dx₂/dt = k₁₂x₁ - k₂₁x₂
        dC2_ppf = p.k12 * C1_ppf * p.V1 / p.V2 - p.k21 * C2_ppf
        
        # Formulation (3): dx₃/dt = k₁₃x₁ - k₃₁x₃
        dC3_ppf = p.k13 * C1_ppf * p.V1 / p.V3 - p.k31 * C3_ppf
        
        # Formulation (31): Ċₑ(t) = ke₀·(C₁(t) - Cₑ(t))
        dCe_ppf = p.ke0 * (C1_ppf - Ce_ppf)
        
        # Remifentanil PK/PD
        r = self.rftn_model.params
        
        # Same formulations (1)-(3) for remifentanil
        rftn_infusion_ng_per_min = rftn_rate * weight
        dC1_rftn = (rftn_infusion_ng_per_min / (r.V1 * 1000)
                    - (r.k10 + r.k12 + r.k13) * C1_rftn
                    + r.k21 * C2_rftn * r.V2 / r.V1
                    + r.k31 * C3_rftn * r.V3 / r.V1)
        
        dC2_rftn = r.k12 * C1_rftn * r.V1 / r.V2 - r.k21 * C2_rftn
        dC3_rftn = r.k13 * C1_rftn * r.V1 / r.V3 - r.k31 * C3_rftn
        
        # Formulation (31) for remifentanil
        dCe_rftn = r.ke0 * (C1_rftn - Ce_rftn)
        
        return np.array([dC1_ppf, dC2_ppf, dC3_ppf, dCe_ppf,
                         dC1_rftn, dC2_rftn, dC3_rftn, dCe_rftn])
    
    def _pk_pd_ode(self, y: np.ndarray, t: float, infusion_rate: float) -> np.ndarray:
        """
        Compute the derivatives for the PK/PD ODE system (propofol only).
        
        Backward compatible interface that only updates propofol.
        
        Args:
            y: State vector [C1, C2, C3, Ce]
            t: Current time (not used directly, required by odeint)
            infusion_rate: Propofol infusion rate (μg/kg/min)
        
        Returns:
            Derivatives [dC1/dt, dC2/dt, dC3/dt, dCe/dt]
        """
        C1, C2, C3, Ce = y
        p = self.ppf_model.params
        
        weight = self.patient.weight
        infusion_ug_per_min = infusion_rate * weight
        
        # Formulation (1)
        dC1 = (infusion_ug_per_min / (p.V1 * 1000)
               - (p.k10 + p.k12 + p.k13) * C1
               + p.k21 * C2 * p.V2 / p.V1
               + p.k31 * C3 * p.V3 / p.V1)
        
        # Formulation (2)
        dC2 = p.k12 * C1 * p.V1 / p.V2 - p.k21 * C2
        
        # Formulation (3)
        dC3 = p.k13 * C1 * p.V1 / p.V3 - p.k31 * C3
        
        # Formulation (31)
        dCe = p.ke0 * (C1 - Ce)
        
        return np.array([dC1, dC2, dC3, dCe])
    
    def step(
        self, 
        infusion_rate: float, 
        dt: Optional[float] = None,
        rftn_rate: float = 0.0
    ) -> Tuple[np.ndarray, float]:
        """
        Advance the simulation by one time step.
        
        Args:
            infusion_rate: Propofol infusion rate (μg/kg/min)
            dt: Time step in seconds (uses default if None)
            rftn_rate: Remifentanil infusion rate (ng/kg/min), default 0
        
        Returns:
            Tuple of (new_state, measured_bis)
        """
        if dt is None:
            dt = self.dt
        
        # Convert dt from seconds to minutes for the model
        dt_min = dt / 60.0
        
        # Combined state vector
        combined_state = np.concatenate([self.state_ppf, self.state_rftn])
        
        # Integrate the ODE system
        t_span = np.array([0, dt_min])
        solution = odeint(
            self._pk_pd_ode_combined,
            combined_state,
            t_span,
            args=(infusion_rate, rftn_rate)
        )
        
        # Update states
        self.state_ppf = solution[-1, :4]
        self.state_rftn = solution[-1, 4:]
        self.state = self.state_ppf  # Backward compatibility
        self.time += dt
        
        # Compute BIS with measurement noise
        bis_true = self._compute_bis(self.state_ppf[3], self.state_rftn[3])
        bis_measured = bis_true + self.rng.normal(0, self.noise_std)
        bis_measured = np.clip(bis_measured, 0, 100)
        
        # Record history
        self.history['time'].append(self.time)
        self.history['state'].append(self.state_ppf.copy())
        self.history['state_ppf'].append(self.state_ppf.copy())
        self.history['state_rftn'].append(self.state_rftn.copy())
        self.history['bis'].append(bis_measured)
        self.history['dose'].append(infusion_rate)
        self.history['dose_ppf'].append(infusion_rate)
        self.history['dose_rftn'].append(rftn_rate)
        self.history['ce'].append(self.state_ppf[3])
        self.history['ce_ppf'].append(self.state_ppf[3])
        self.history['ce_rftn'].append(self.state_rftn[3])
        
        return self.state_ppf.copy(), bis_measured
    
    def _compute_bis(
        self, 
        ce_ppf: float, 
        ce_rftn: float = 0.0
    ) -> float:
        """
        Compute BIS from effect-site concentrations.
        
        Formulation (32) - Drug Interaction Model:
        BIS(t) = BIS0 · (1 + Ĉₑ^PPF(t)/C50_PPF + Ĉₑ^RFTN(t)/C50_RFTN)^(-γ) + ε
        
        Or Hill/Sigmoid Emax (propofol only):
        BIS = E0 - Emax * (Ce^γ / (Ce^γ + EC50^γ))
        
        Args:
            ce_ppf: Propofol effect-site concentration (μg/ml)
            ce_rftn: Remifentanil effect-site concentration (ng/ml)
        
        Returns:
            BIS value (0-100)
        """
        if self.bis_model_type == BISModelType.DRUG_INTERACTION:
            # Formulation (32): Drug Interaction Model
            # BIS(t) = 98.0 · (1 + Ce_PPF/4.47 + Ce_RFTN/19.3)^(-1.43) + ε
            params = self.drug_interaction_params
            
            if ce_ppf <= 0 and ce_rftn <= 0:
                return params.bis0
            
            # Linear combination (not exponential)
            ratio_ppf = max(0, ce_ppf) / params.c50_ppf
            ratio_rftn = max(0, ce_rftn) / params.c50_rftn
            
            bis = params.bis0 * np.power(1 + ratio_ppf + ratio_rftn, -params.gamma)  # Formulation (32)
            
        else:
            # Hill/Sigmoid Emax (propofol only)
            p = self.ppf_model.params
            
            if ce_ppf <= 0:
                return p.e0
            
            ce_gamma = ce_ppf ** p.gamma
            ec50_gamma = p.ec50 ** p.gamma
            
            effect = p.emax * (ce_gamma / (ce_gamma + ec50_gamma))
            bis = p.e0 - effect
        
        return np.clip(bis, 0, 100)
    
    def get_bis(self, add_noise: bool = True) -> float:
        """
        Get current BIS value.
        
        Args:
            add_noise: Whether to add measurement noise
        
        Returns:
            Current BIS value
        """
        bis_true = self._compute_bis(self.state_ppf[3], self.state_rftn[3])
        
        if add_noise:
            bis = bis_true + self.rng.normal(0, self.noise_std)
            return np.clip(bis, 0, 100)
        
        return bis_true
    
    def get_concentrations(self) -> Dict[str, float]:
        """
        Get current drug concentrations.
        
        Returns:
            Dictionary with propofol and remifentanil concentrations
        """
        return {
            # Propofol (μg/ml)
            'C1_ppf': self.state_ppf[0],
            'C2_ppf': self.state_ppf[1],
            'C3_ppf': self.state_ppf[2],
            'Ce_ppf': self.state_ppf[3],
            # Remifentanil (ng/ml)
            'C1_rftn': self.state_rftn[0],
            'C2_rftn': self.state_rftn[1],
            'C3_rftn': self.state_rftn[2],
            'Ce_rftn': self.state_rftn[3],
            # Backward compatibility aliases
            'C1': self.state_ppf[0],
            'C2': self.state_ppf[1],
            'C3': self.state_ppf[2],
            'Ce': self.state_ppf[3]
        }
    
    def get_observation(self) -> np.ndarray:
        """
        Get the full observation vector for RL.
        
        The observation includes:
        - Normalized BIS error: (BIS - target) / 50
        - Normalized propofol effect-site concentration: Ce_ppf / EC50
        - Normalized remifentanil effect-site concentration: Ce_rftn / EC50
        
        Returns:
            Observation vector for the RL agent
        """
        bis = self.get_bis(add_noise=True)
        ce_ppf = self.state_ppf[3]
        ce_rftn = self.state_rftn[3]
        
        # Normalize features
        bis_error = (bis - 50) / 50.0  # Normalized error from target
        ce_ppf_normalized = ce_ppf / self.ppf_model.params.ec50
        ce_rftn_normalized = ce_rftn / self.rftn_model.params.ec50
        
        return np.array([bis_error, ce_ppf_normalized, ce_rftn_normalized], dtype=np.float32)
    
    def administer_bolus(self, dose_mg_per_kg: float, drug: str = "propofol"):
        """
        Administer a bolus dose of drug.
        
        This simulates rapid injection by instantly increasing
        the central compartment concentration.
        
        Args:
            dose_mg_per_kg: Bolus dose in mg/kg (propofol) or μg/kg (remifentanil)
            drug: Which drug ("propofol" or "remifentanil")
        """
        weight = self.patient.weight
        
        if drug.lower() == "propofol":
            v1_ml = self.ppf_model.params.V1 * 1000  # Convert L to ml
            # Convert mg/kg to μg and calculate concentration increase
            dose_ug = dose_mg_per_kg * weight * 1000  # mg to μg
            concentration_increase = dose_ug / v1_ml
            self.state_ppf[0] += concentration_increase
            self.state[0] += concentration_increase  # Backward compat
        else:  # remifentanil
            v1_ml = self.rftn_model.params.V1 * 1000  # Convert L to ml
            # dose_mg_per_kg is actually μg/kg for remifentanil
            dose_ng = dose_mg_per_kg * weight * 1000  # μg to ng
            concentration_increase = dose_ng / v1_ml
            self.state_rftn[0] += concentration_increase
    
    def get_history_arrays(self) -> Dict[str, np.ndarray]:
        """
        Get simulation history as numpy arrays.
        
        Returns:
            Dictionary of numpy arrays for time, states, BIS, doses
        """
        return {
            'time': np.array(self.history['time']),
            'state': np.array(self.history['state']),
            'state_ppf': np.array(self.history['state_ppf']) if self.history['state_ppf'] else np.array([]),
            'state_rftn': np.array(self.history['state_rftn']) if self.history['state_rftn'] else np.array([]),
            'bis': np.array(self.history['bis']),
            'dose': np.array(self.history['dose']),
            'dose_ppf': np.array(self.history['dose_ppf']) if self.history['dose_ppf'] else np.array([]),
            'dose_rftn': np.array(self.history['dose_rftn']) if self.history['dose_rftn'] else np.array([]),
            'ce': np.array(self.history['ce']),
            'ce_ppf': np.array(self.history['ce_ppf']) if self.history['ce_ppf'] else np.array([]),
            'ce_rftn': np.array(self.history['ce_rftn']) if self.history['ce_rftn'] else np.array([])
        }
    
    def get_state_space_matrices(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get state-space matrices for both drug models.
        
        Returns formulations (1)-(3) in matrix form ẋ = Ax + Bu for both drugs.
        
        Returns:
            Dictionary with 'propofol' and 'remifentanil' keys, each containing
            (A_matrix, B_vector) tuples
        """
        return {
            'propofol': self.ppf_model.get_state_space_matrices(),
            'remifentanil': self.rftn_model.get_state_space_matrices()
        }
    
    def get_extended_state_space_matrices(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get extended state-space matrices including effect-site compartments.
        
        Returns:
            Dictionary with 'propofol' and 'remifentanil' keys
        """
        return {
            'propofol': self.ppf_model.get_extended_state_space_matrices(),
            'remifentanil': self.rftn_model.get_extended_state_space_matrices()
        }
    
    # =============================================================================
    # VitalDB Validation Methods
    # =============================================================================
    
    @classmethod
    def from_vitaldb_demographics(
        cls,
        age: float,
        weight: float,
        height: float,
        sex: str,
        dt: float = 5.0,
        noise_std: float = 2.0,
        seed: Optional[int] = None,
        bis_model_type: BISModelType = BISModelType.DRUG_INTERACTION,
    ) -> 'PatientSimulator':
        """
        Create PatientSimulator from VitalDB patient demographics.
        
        Args:
            age: Patient age (years)
            weight: Patient weight (kg)
            height: Patient height (cm)
            sex: Patient sex ('male' or 'female')
            dt: Time step (seconds)
            noise_std: BIS measurement noise standard deviation
            seed: Random seed
            bis_model_type: BIS model type
        
        Returns:
            PatientSimulator initialized with VitalDB demographics
        """
        # Convert sex nomenclature if needed
        gender = sex.lower()
        if gender not in ['male', 'female']:
            gender = 'male'  # Default
        
        patient = PatientParameters(
            age=age,
            weight=weight,
            height=height,
            gender=gender,
        )
        
        return cls(
            patient=patient,
            dt=dt,
            noise_std=noise_std,
            seed=seed,
            bis_model_type=bis_model_type,
        )
    
    def validate_against_vitaldb(
        self,
        times: np.ndarray,
        bis_observed: np.ndarray,
        ce_ppf_observed: np.ndarray,
        ppf_doses: np.ndarray,
        ce_rftn_observed: Optional[np.ndarray] = None,
        rftn_doses: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Validate PK/PD model predictions against VitalDB ground truth.
        
        Compares simulator BIS predictions to actual BIS measurements from VitalDB,
        given the same drug infusion trajectories.
        
        Args:
            times: Time points (seconds)
            bis_observed: Observed BIS values from VitalDB
            ce_ppf_observed: Observed propofol effect-site concentrations (μg/mL)
            ppf_doses: Propofol infusion rates (μg/kg/min)
            ce_rftn_observed: Observed remifentanil effect-site concentrations (ng/mL)
            rftn_doses: Remifentanil infusion rates (μg/kg/min)
        
        Returns:
            Dictionary with validation metrics:
                - bis_rmse: Root mean squared error for BIS
                - bis_mae: Mean absolute error for BIS
                - bis_r2: R-squared coefficient for BIS
                - ce_ppf_rmse: RMSE for propofol effect-site concentration
                - ce_ppf_mae: MAE for propofol effect-site concentration
                - ce_ppf_r2: R-squared for propofol effect-site concentration
        """
        # Reset simulator
        self.reset()
        
        # Initialize arrays for predictions
        bis_predicted = []
        ce_ppf_predicted = []
        ce_rftn_predicted = []
        
        # Simulate trajectory
        for i in range(len(times)):
            # Get current doses
            ppf_dose = ppf_doses[i] if i < len(ppf_doses) else 0.0
            rftn_dose = rftn_doses[i] if rftn_doses is not None and i < len(rftn_doses) else 0.0
            
            # Step simulator
            state_ppf, state_rftn, bis = self.step_dual_drug(ppf_dose, rftn_dose)
            
            # Record predictions (without noise for comparison)
            bis_predicted.append(self._compute_bis(state_ppf[3], state_rftn[3]))
            ce_ppf_predicted.append(state_ppf[3])
            ce_rftn_predicted.append(state_rftn[3])
        
        # Convert to arrays
        bis_predicted = np.array(bis_predicted)
        ce_ppf_predicted = np.array(ce_ppf_predicted)
        ce_rftn_predicted = np.array(ce_rftn_predicted)
        
        # Filter out NaN/invalid values
        valid_mask = ~np.isnan(bis_observed)
        bis_obs_valid = bis_observed[valid_mask]
        bis_pred_valid = bis_predicted[valid_mask]
        
        # Compute BIS validation metrics
        bis_rmse = np.sqrt(np.mean((bis_pred_valid - bis_obs_valid) ** 2))
        bis_mae = np.mean(np.abs(bis_pred_valid - bis_obs_valid))
        
        # R-squared
        ss_res = np.sum((bis_obs_valid - bis_pred_valid) ** 2)
        ss_tot = np.sum((bis_obs_valid - np.mean(bis_obs_valid)) ** 2)
        bis_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Compute Ce propofol validation metrics
        valid_ce_ppf_mask = ~np.isnan(ce_ppf_observed)
        ce_ppf_obs_valid = ce_ppf_observed[valid_ce_ppf_mask]
        ce_ppf_pred_valid = ce_ppf_predicted[valid_ce_ppf_mask]
        
        ce_ppf_rmse = np.sqrt(np.mean((ce_ppf_pred_valid - ce_ppf_obs_valid) ** 2))
        ce_ppf_mae = np.mean(np.abs(ce_ppf_pred_valid - ce_ppf_obs_valid))
        
        ss_res_ce = np.sum((ce_ppf_obs_valid - ce_ppf_pred_valid) ** 2)
        ss_tot_ce = np.sum((ce_ppf_obs_valid - np.mean(ce_ppf_obs_valid)) ** 2)
        ce_ppf_r2 = 1 - (ss_res_ce / ss_tot_ce) if ss_tot_ce > 0 else 0.0
        
        metrics = {
            'bis_rmse': bis_rmse,
            'bis_mae': bis_mae,
            'bis_r2': bis_r2,
            'ce_ppf_rmse': ce_ppf_rmse,
            'ce_ppf_mae': ce_ppf_mae,
            'ce_ppf_r2': ce_ppf_r2,
        }
        
        # Add remifentanil metrics if available
        if ce_rftn_observed is not None:
            valid_ce_rftn_mask = ~np.isnan(ce_rftn_observed)
            ce_rftn_obs_valid = ce_rftn_observed[valid_ce_rftn_mask]
            ce_rftn_pred_valid = ce_rftn_predicted[valid_ce_rftn_mask]
            
            ce_rftn_rmse = np.sqrt(np.mean((ce_rftn_pred_valid - ce_rftn_obs_valid) ** 2))
            ce_rftn_mae = np.mean(np.abs(ce_rftn_pred_valid - ce_rftn_obs_valid))
            
            ss_res_rftn = np.sum((ce_rftn_obs_valid - ce_rftn_pred_valid) ** 2)
            ss_tot_rftn = np.sum((ce_rftn_obs_valid - np.mean(ce_rftn_obs_valid)) ** 2)
            ce_rftn_r2 = 1 - (ss_res_rftn / ss_tot_rftn) if ss_tot_rftn > 0 else 0.0
            
            metrics['ce_rftn_rmse'] = ce_rftn_rmse
            metrics['ce_rftn_mae'] = ce_rftn_mae
            metrics['ce_rftn_r2'] = ce_rftn_r2
        
        return metrics


def create_patient_population(
    n_patients: int,
    age_range: Tuple[float, float] = (20, 80),
    weight_range: Tuple[float, float] = (50, 100),
    height_range: Tuple[float, float] = (150, 190),
    seed: Optional[int] = None
) -> List[PatientParameters]:
    """
    Create a population of patients with varied demographics.
    
    Args:
        n_patients: Number of patients to generate
        age_range: (min, max) age in years
        weight_range: (min, max) weight in kg
        height_range: (min, max) height in cm
        seed: Random seed for reproducibility
    
    Returns:
        List of PatientParameters objects
    """
    rng = np.random.default_rng(seed)
    
    patients = []
    for _ in range(n_patients):
        age = rng.uniform(*age_range)
        weight = rng.uniform(*weight_range)
        height = rng.uniform(*height_range)
        gender = rng.choice(['male', 'female'])
        
        patients.append(PatientParameters(
            age=age,
            weight=weight,
            height=height,
            gender=gender
        ))
    
    return patients


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Patient Simulator...")
    
    # Create a patient
    patient = PatientParameters(age=45, weight=70, height=175, gender="male")
    print(f"Patient: {patient}")
    print(f"LBM: {patient.lbm:.2f} kg")
    
    # Create simulator
    simulator = PatientSimulator(patient=patient, dt=5.0, seed=42)
    
    # Simulate a simple scenario
    # 1. Bolus induction
    simulator.administer_bolus(2.0)  # 2 mg/kg bolus
    
    # 2. Run simulation with constant infusion
    print("\nSimulation (constant 100 μg/kg/min infusion):")
    print(f"{'Time (s)':<10} {'BIS':<10} {'Ce (μg/ml)':<15}")
    print("-" * 35)
    
    for i in range(24):  # 2 minutes simulation
        state, bis = simulator.step(100.0)  # 100 μg/kg/min
        if i % 4 == 0:  # Print every 20 seconds
            print(f"{simulator.time:<10.0f} {bis:<10.1f} {state[3]:<15.3f}")
    
    print("\nSimulation complete!")
