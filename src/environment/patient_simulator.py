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


@dataclass
class PatientParameters:
    """
    Patient demographic parameters for PK/PD model individualization.
    
    Attributes:
        age: Patient age in years
        weight: Patient weight in kg
        height: Patient height in cm
        gender: Patient gender ('male' or 'female')
        lbm: Lean body mass in kg (calculated if not provided)
    """
    age: float = 40.0
    weight: float = 70.0
    height: float = 170.0
    gender: str = "male"
    lbm: Optional[float] = None
    
    def __post_init__(self):
        """Calculate lean body mass if not provided."""
        if self.lbm is None:
            self.lbm = self._calculate_lbm()
    
    def _calculate_lbm(self) -> float:
        """
        Calculate Lean Body Mass using James formula.
        
        For males: LBM = 1.1 * weight - 128 * (weight/height)^2
        For females: LBM = 1.07 * weight - 148 * (weight/height)^2
        """
        height_m = self.height / 100.0  # Convert to meters
        weight_height_ratio = self.weight / (self.height ** 2) * 10000  # (weight/height^2) in proper units
        
        if self.gender.lower() == "male":
            lbm = 1.1 * self.weight - 128 * (self.weight / self.height) ** 2 # Formulation (4), it uses **2 in the original paper
        else:
            lbm = 1.07 * self.weight - 148 * (self.weight / self.height) ** 2 # Formulation (5), it uses **2 in the original paper
        
        # Ensure LBM is reasonable (at least 30% of body weight)
        return max(lbm, 0.3 * self.weight)


@dataclass
class SchniderModelParameters:
    """
    Schnider model PK/PD parameters.
    
    These parameters define the three-compartment pharmacokinetic model
    and the effect-site pharmacodynamic model for propofol.
    """

    # Table A.4 notation
    h_1: float = 4.27      # V1 (L)
    h_2: float = 18.9      # V2 base
    h_3: float = 0.391     # V2 age coefficient
    h_4: float = 53        # Age reference
    h_5: float = 238.0     # V3 (L)
    h_6: float = 1.89      # Cl1 base
    h_7: float = 0.0456    # Weight coefficient
    h_8: float = 77        # Weight reference (kg)
    h_9: float = 0.0681    # LBM coefficient
    h_10: float = 59       # LBM reference (kg)
    h_11: float = 0.0264   # Height coefficient
    h_12: float = 177      # Height reference (cm)
    h_13: float = 1.29     # Cl2 base
    h_14: float = 0.024    # Cl2 age coefficient
    h_15: float = 53       # Cl2 age reference
    h_16: float = 0.836    # Cl3 (L/min)
    h_17: float = 0.456    # ke0 (min^-1)

    # Compartment volumes (L)
    v1: float = 4.27      # Central compartment
    v2: float = 18.9      # Shallow peripheral compartment  
    v3: float = 238.0     # Deep peripheral compartment
    
    # Rate constants (min^-1)
    k10: float = 0.443    # Elimination from central
    k12: float = 0.302    # Central to shallow peripheral
    k13: float = 0.196    # Central to deep peripheral
    k21: float = 0.092    # Shallow peripheral to central
    k31: float = 0.0048   # Deep peripheral to central
    ke0: float = 0.456    # Effect-site equilibration
    
    # Hill model parameters for BIS
    e0: float = 97.0      # Baseline BIS (awake)
    emax: float = 97.0    # Maximum effect
    ec50: float = 3.4     # Effect-site concentration at 50% effect (μg/ml)
    gamma: float = 3.0    # Hill coefficient (steepness)


class SchniderModel:
    """
    Schnider PK/PD Model for Propofol.
    
    Implements the three-compartment pharmacokinetic model with effect-site
    compartment and Hill sigmoid pharmacodynamic model for BIS prediction.
    
    The model can be individualized based on patient demographics following
    the Schnider covariate model equations.
    
    Attributes:
        patient: Patient demographic parameters
        params: Schnider model parameters
        use_covariates: Whether to adjust parameters based on patient covariates
    """
    
    def __init__(
        self,
        patient: Optional[PatientParameters] = None,
        params: Optional[SchniderModelParameters] = None,
        use_covariates: bool = True
    ):
        """
        Initialize the Schnider model.
        
        Args:
            patient: Patient demographic parameters (uses defaults if None)
            params: Model parameters (uses population defaults if None)
            use_covariates: If True, adjust parameters based on patient demographics
        """
        self.patient = patient or PatientParameters()
        self.params = params or SchniderModelParameters()
        self.use_covariates = use_covariates
        
        # Apply covariate adjustments if requested
        if use_covariates:
            self._apply_schnider_covariates()
    
    def _apply_schnider_covariates(self):
        """
        Apply Schnider covariate model to adjust PK parameters.
        
        The Schnider model adjusts volumes and clearances based on:
        - Age
        - Weight  
        - Lean Body Mass (LBM)
        - Height
        
        References: Schnider et al., Anesthesiology 1998
        """
        age = self.patient.age
        weight = self.patient.weight
        lbm = self.patient.lbm
        height = self.patient.height
        
        # Schnider covariate equations for volumes
        # V1 is fixed at 4.27 L
        self.params.v1 = self.params.h_1 # Formulation (6)
        
        # V2 depends on age
        self.params.v2 = self.params.h_2 - self.params.h_3 * (age - self.params.h_4) # Formulation (7)
        
        # V3 is fixed
        self.params.v3 = self.params.h_5 # Formulation (8)
        
        # Clearances and rate constants
        # Cl1 (elimination clearance) depends on weight
        cl1 = self.params.h_6 + self.params.h_7 * (weight - self.params.h_8) - self.params.h_9 * (lbm - self.params.h_10) + self.params.h_11 * (height - self.params.h_12) # Formulation (9)
        cl2 = self.params.h_13 - self.params.h_14 * (age - self.params.h_15) # Formulation (10)
        cl3 = self.params.h_16 # Formulation (11)
        
        # Convert clearances to rate constants
        self.params.k10 = cl1 / self.params.v1 # Formulation (12)
        self.params.k12 = cl2 / self.params.v1 # Formulation (13)
        self.params.k13 = cl3 / self.params.v1 # Formulation (14)
        self.params.k21 = cl2 / self.params.v2 # Formulation (15)
        self.params.k31 = cl3 / self.params.v3 # Formulation (16)
        
        # ke0 is typically fixed but can vary with age
        self.params.ke0 = self.params.h_17 # Formulation (17)
    
    def get_rate_constants(self) -> Dict[str, float]:
        """Return all rate constants as a dictionary."""
        return {
            'k10': self.params.k10,
            'k12': self.params.k12,
            'k13': self.params.k13,
            'k21': self.params.k21,
            'k31': self.params.k31,
            'ke0': self.params.ke0
        }
    
    def get_volumes(self) -> Dict[str, float]:
        """Return compartment volumes as a dictionary."""
        return {
            'v1': self.params.v1,
            'v2': self.params.v2,
            'v3': self.params.v3
        }
    
    def get_state_space_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the state-space matrices A and B for the 3-compartment model.
        
        Formulations (1)-(3) in matrix form: ẋ(t) = Ax(t) + Bu(t)
        
        A = [-(k₁₀+k₁₂+k₁₃)  k₂₁         k₃₁      ]
            [k₁₂             -k₂₁        0        ]
            [k₁₃             0           -k₃₁     ]
        
        B = [1/V₁, 0, 0]ᵀ  (converts infusion rate to concentration rate)
        
        Returns:
            Tuple of (A_matrix, B_vector) numpy arrays
        """
        p = self.params

        # Volume ratios
        r21 = p.v2 / p.v1
        r31 = p.v3 / p.v1
        r12 = p.v1 / p.v2
        r13 = p.v1 / p.v3
        
        # System matrix A - Formulations (1)-(3) coefficients
        A = np.array([
            [-(p.k10 + p.k12 + p.k13), p.k21 * r21, p.k31 * r31],
        [p.k12 * r12, -p.k21, 0.0],
        [p.k13 * r13, 0.0, -p.k31]
        ])
        
        # Input vector B (1/V1 to convert infusion to concentration)
        # Infusion rate is μg/kg/min, divide by V1*1000 to get μg/ml/min
        B = np.array([1.0 / (p.v1 * 1000), 0.0, 0.0])
        
        return A, B
    
    def get_extended_state_space_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get extended state-space matrices including effect-site compartment.
        
        Extended state: x = [C₁, C₂, C₃, Cₑ]ᵀ
        
        Formulation (31): Ċₑ(t) = ke₀·(C₁(t) - Cₑ(t))
        
        Returns:
            Tuple of (A_extended, B_extended) numpy arrays
        """
        p = self.params

        # Volume ratios
        r21 = p.v2 / p.v1
        r31 = p.v3 / p.v1
        r12 = p.v1 / p.v2
        r13 = p.v1 / p.v3
        
        # Extended system matrix (4x4)
        A_ext = np.array([
            [-(p.k10 + p.k12 + p.k13), p.k21 * r21, p.k31 * r31, 0.0],
        [p.k12 * r12, -p.k21, 0.0, 0.0],
        [p.k13 * r13, 0.0, -p.k31, 0.0],
        [p.ke0, 0.0, 0.0, -p.ke0]  # Formulation (31)
        ])
        
        # Extended input vector
        B_ext = np.array([1.0 / (p.v1 * 1000), 0.0, 0.0, 0.0])
        
        return A_ext, B_ext


@dataclass
class MintoModelParameters:
    """
    Minto model PK/PD parameters for Remifentanil.
    
    These parameters define the three-compartment pharmacokinetic model
    and the effect-site pharmacodynamic model for remifentanil.
    """

    # Table A.4 notation
    f_1: float = 5.1       # V1 base (L)
    f_2: float = 0.0201    # V1 age coefficient
    f_3: float = 0.072     # V1 LBM coefficient
    f_4: float = 9.82      # V2 base (L)
    f_5: float = 0.0811    # V2 age coefficient
    f_6: float = 0.108     # V2 LBM coefficient
    f_7: float = 5.42      # V3 (L)
    f_8: float = 2.6       # Cl1 base (L/min)
    f_9: float = 0.0162    # Cl1 age coefficient
    f_10: float = 0.0191   # Cl1 LBM coefficient
    f_11: float = 2.05     # Cl2 base (L/min)
    f_12: float = 0.030   # Cl2 age coefficient
    f_13: float = 0.076   # Cl2 LBM coefficient
    f_14: float = 0.0013    # Cl3 base (L/min)
    f_15: float = 0.595   # Cl3 weight coefficient
    f_16: float = 0.007   # Cl3 age coefficient
    f_17: float = 40       # Age reference
    f_18: float = 55       # LBM reference (kg)
    
    # Compartment volumes (L)
    v1: float = 5.1       # Central compartment
    v2: float = 9.82      # Shallow peripheral compartment
    v3: float = 5.42      # Deep peripheral compartment
    
    # Rate constants (min^-1)
    k10: float = 0.51     # Elimination from central
    k12: float = 0.40     # Central to shallow peripheral
    k13: float = 0.015    # Central to deep peripheral
    k21: float = 0.21     # Shallow peripheral to central
    k31: float = 0.014    # Deep peripheral to central
    ke0: float = 0.595    # Effect-site equilibration
    
    # Hill model parameters for effect
    e0: float = 97.0      # Baseline effect
    emax: float = 97.0    # Maximum effect
    ec50: float = 12.0    # Effect-site concentration at 50% effect (ng/ml)
    gamma: float = 2.0    # Hill coefficient (steepness)


class MintoModel:
    """
    Minto PK/PD Model for Remifentanil.
    
    Implements the three-compartment pharmacokinetic model with effect-site
    compartment and Hill sigmoid pharmacodynamic model for effect prediction.
    
    The model can be individualized based on patient demographics following
    the Minto covariate model equations.
    
    Attributes:
        patient: Patient demographic parameters
        params: Minto model parameters
        use_covariates: Whether to adjust parameters based on patient covariates
    """
    
    def __init__(
        self,
        patient: Optional[PatientParameters] = None,
        params: Optional[MintoModelParameters] = None,
        use_covariates: bool = True
    ):
        """
        Initialize the Minto model.
        
        Args:
            patient: Patient demographic parameters (uses defaults if None)
            params: Model parameters (uses population defaults if None)
            use_covariates: If True, adjust parameters based on patient demographics
        """
        self.patient = patient or PatientParameters()
        self.params = params or MintoModelParameters()
        self.use_covariates = use_covariates
        
        # Apply covariate adjustments if requested
        if use_covariates:
            self._apply_minto_covariates()
    
    def _apply_minto_covariates(self):
        """
        Apply Minto covariate model to adjust PK parameters.
        
        The Minto model adjusts volumes and clearances based on:
        - Age
        - Weight
        - Lean Body Mass (LBM)
        
        References: Minto et al., Anesthesiology 1997
        """
        age = self.patient.age
        weight = self.patient.weight
        lbm = self.patient.lbm
        
        # Minto covariate equations for volumes
        # V1 depends on age and LBM
        self.params.v1 = self.params.f_1 - self.params.f_2 * (age - self.params.f_17) + self.params.f_3 * (lbm - self.params.f_18)  # Formulation (18)
        
        # V2 depends on age and LBM
        self.params.v2 = self.params.f_4 - self.params.f_5 * (age - self.params.f_17) + self.params.f_6 * (lbm - self.params.f_18)  # Formulation (19)
        
        # V3 is fixed
        self.params.v3 = self.params.f_7  # Formulation (20)
        
        # Clearances
        cl1 = self.params.f_8 - self.params.f_9 * (age - self.params.f_17) + self.params.f_10 * (lbm - self.params.f_18)  # Formulation (21)
        cl2 = self.params.f_11 - self.params.f_12 * (age - self.params.f_17)  # Formulation (22)
        cl3 = self.params.f_13 - self.params.f_14 * (age - self.params.f_17)  # Formulation (23)
        
        # Convert clearances to rate constants
        self.params.k10 = cl1 / self.params.v1  # Formulation (24)
        self.params.k12 = cl2 / self.params.v1  # Formulation (25)
        self.params.k13 = cl3 / self.params.v1  # Formulation (26)
        self.params.k21 = cl2 / self.params.v2  # Formulation (27)
        self.params.k31 = cl3 / self.params.v3  # Formulation (28)
        
        # ke0 is fixed
        self.params.ke0 = self.params.f_15 - self.params.f_16 * (age - self.params.f_17)  # Formulation (29)
    
    def get_rate_constants(self) -> Dict[str, float]:
        """Return all rate constants as a dictionary."""
        return {
            'k10': self.params.k10,
            'k12': self.params.k12,
            'k13': self.params.k13,
            'k21': self.params.k21,
            'k31': self.params.k31,
            'ke0': self.params.ke0
        }
    
    def get_volumes(self) -> Dict[str, float]:
        """Return compartment volumes as a dictionary."""
        return {
            'v1': self.params.v1,
            'v2': self.params.v2,
            'v3': self.params.v3
        }
    
    def get_state_space_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the state-space matrices A and B for Remifentanil 3-compartment model.
        
        Formulations (1)-(3) adapted for remifentanil: ẋ(t) = Ax(t) + Bu(t)
        
        Returns:
            Tuple of (A_matrix, B_vector) numpy arrays
        """
        p = self.params

        # Volume ratios
        r21 = p.v2 / p.v1
        r31 = p.v3 / p.v1
        r12 = p.v1 / p.v2
        r13 = p.v1 / p.v3
        
        # System matrix A
        A = np.array([
            [-(p.k10 + p.k12 + p.k13), p.k21 * r21, p.k31 * r31],
        [p.k12 * r12, -p.k21, 0.0],
        [p.k13 * r13, 0.0, -p.k31]
        ])
        
        # Input vector B (1/V1 to convert infusion to concentration)
        # Remifentanil: ng/kg/min → ng/ml
        B = np.array([1.0 / (p.v1 * 1000), 0.0, 0.0])
        
        return A, B
    
    def get_extended_state_space_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get extended state-space matrices including effect-site compartment.
        
        Extended state: x = [C₁, C₂, C₃, Cₑ]ᵀ
        
        Returns:
            Tuple of (A_extended, B_extended) numpy arrays
        """
        p = self.params

        # Volume ratios
        r21 = p.v2 / p.v1
        r31 = p.v3 / p.v1
        r12 = p.v1 / p.v2
        r13 = p.v1 / p.v3
        
        # Extended system matrix (4x4)
        A_ext = np.array([
            [-(p.k10 + p.k12 + p.k13), p.k21 * r21, p.k31 * r31, 0.0],
        [p.k12 * r12, -p.k21, 0.0, 0.0],
        [p.k13 * r13, 0.0, -p.k31, 0.0],
        [p.ke0, 0.0, 0.0, -p.ke0]
        ])
        
        # Extended input vector
        B_ext = np.array([1.0 / (p.v1 * 1000), 0.0, 0.0, 0.0])
        
        return A_ext, B_ext


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
        
        # Remifentanil model (Minto)
        self.rftn_model = MintoModel(patient=self.patient)
        
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
        dC1_ppf = (ppf_infusion_ug_per_min / (p.v1 * 1000)
                   - (p.k10 + p.k12 + p.k13) * C1_ppf
                   + p.k21 * C2_ppf * p.v2 / p.v1
                   + p.k31 * C3_ppf * p.v3 / p.v1)
        
        # Formulation (2): dx₂/dt = k₁₂x₁ - k₂₁x₂
        dC2_ppf = p.k12 * C1_ppf * p.v1 / p.v2 - p.k21 * C2_ppf
        
        # Formulation (3): dx₃/dt = k₁₃x₁ - k₃₁x₃
        dC3_ppf = p.k13 * C1_ppf * p.v1 / p.v3 - p.k31 * C3_ppf
        
        # Formulation (31): Ċₑ(t) = ke₀·(C₁(t) - Cₑ(t))
        dCe_ppf = p.ke0 * (C1_ppf - Ce_ppf)
        
        # Remifentanil PK/PD
        r = self.rftn_model.params
        
        # Same formulations (1)-(3) for remifentanil
        rftn_infusion_ng_per_min = rftn_rate * weight
        dC1_rftn = (rftn_infusion_ng_per_min / (r.v1 * 1000)
                    - (r.k10 + r.k12 + r.k13) * C1_rftn
                    + r.k21 * C2_rftn * r.v2 / r.v1
                    + r.k31 * C3_rftn * r.v3 / r.v1)
        
        dC2_rftn = r.k12 * C1_rftn * r.v1 / r.v2 - r.k21 * C2_rftn
        dC3_rftn = r.k13 * C1_rftn * r.v1 / r.v3 - r.k31 * C3_rftn
        
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
        dC1 = (infusion_ug_per_min / (p.v1 * 1000)
               - (p.k10 + p.k12 + p.k13) * C1
               + p.k21 * C2 * p.v2 / p.v1
               + p.k31 * C3 * p.v3 / p.v1)
        
        # Formulation (2)
        dC2 = p.k12 * C1 * p.v1 / p.v2 - p.k21 * C2
        
        # Formulation (3)
        dC3 = p.k13 * C1 * p.v1 / p.v3 - p.k31 * C3
        
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
            v1_ml = self.ppf_model.params.v1 * 1000  # Convert L to ml
            # Convert mg/kg to μg and calculate concentration increase
            dose_ug = dose_mg_per_kg * weight * 1000  # mg to μg
            concentration_increase = dose_ug / v1_ml
            self.state_ppf[0] += concentration_increase
            self.state[0] += concentration_increase  # Backward compat
        else:  # remifentanil
            v1_ml = self.rftn_model.params.v1 * 1000  # Convert L to ml
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
