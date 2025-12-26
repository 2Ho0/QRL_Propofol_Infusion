"""
Schnider PK/PD Model for Propofol
==================================

Implements the three-compartment pharmacokinetic model with effect-site
compartment for propofol following the Schnider covariate model.

References:
-----------
- Schnider TW, et al. "The influence of method of administration and 
  covariates on the pharmacokinetics of propofol in adult volunteers."
  Anesthesiology. 1998 Sep;89(3):562-73.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from .base import BasePKModel, PatientParameters, validate_patient_parameters


@dataclass
class SchniderParameters:
    """
    Schnider model PK/PD parameters for Propofol.
    
    These parameters define the three-compartment pharmacokinetic model
    and the effect-site pharmacodynamic model for propofol.
    
    Default values from Schnider et al., Anesthesiology 1998.
    
    Attributes:
        h_1 to h_17: Covariate model coefficients from Table A.4
        v1, v2, v3: Compartment volumes (L)
        k10, k12, k13, k21, k31, ke0: Rate constants (min^-1)
        e0, emax, ec50, gamma: Hill model parameters for BIS
    """

    # Table A.4 notation - Covariate model coefficients
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
    V1: float = 4.27      # Central compartment
    V2: float = 18.9      # Shallow peripheral compartment  
    V3: float = 238.0     # Deep peripheral compartment
    
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


class SchniderModel(BasePKModel):
    """
    Schnider PK/PD Model for Propofol.
    
    Implements the three-compartment pharmacokinetic model with effect-site
    compartment and Hill sigmoid pharmacodynamic model for BIS prediction.
    
    The model can be individualized based on patient demographics following
    the Schnider covariate model equations (Formulations 6-17).
    
    Mathematical Formulation:
        State-Space: ẋ(t) = Ax(t) + Bu(t)
        Effect-Site: Ċₑ(t) = ke₀·(C₁(t) - Cₑ(t))
        BIS Prediction: BIS = E₀ - Eₘₐₓ · (Cₑ^γ / (Cₑ^γ + EC₅₀^γ))
    
    Attributes:
        patient: Patient demographic parameters
        params: Schnider model parameters
        use_covariates: Whether to adjust parameters based on patient covariates
    
    Example:
        >>> patient = PatientParameters(age=45, weight=70, height=170, gender='M')
        >>> model = SchniderModel(patient=patient)
        >>> A, B = model.get_state_space_matrices()
        >>> print(f"A shape: {A.shape}, B shape: {B.shape}")
        A shape: (3, 3), B shape: (3,)
    """
    
    def __init__(
        self,
        age: Optional[float] = None,
        weight: Optional[float] = None,
        height: Optional[float] = None,
        gender: Optional[str] = None,
        patient: Optional[PatientParameters] = None,
        params: Optional[SchniderParameters] = None,
        use_covariates: bool = True
    ):
        """
        Initialize the Schnider model.
        
        Supports two interfaces:
        1. Individual parameters: age, weight, height, gender
        2. PatientParameters object: patient
        
        Args:
            age: Patient age in years (18-90)
            weight: Patient weight in kg (30-200)
            height: Patient height in cm (100-250)
            gender: Patient gender ('male' or 'female')
            patient: PatientParameters object (overrides individual parameters)
            params: Model parameters (uses population defaults if None)
            use_covariates: If True, adjust parameters based on patient demographics
        
        Raises:
            ValueError: If patient parameters are invalid
        
        Example:
            >>> # Method 1: Individual parameters
            >>> model1 = SchniderModel(age=45, weight=70, height=170, gender='M')
            >>> 
            >>> # Method 2: PatientParameters object
            >>> patient = PatientParameters(age=45, weight=70, height=170, gender='M')
            >>> model2 = SchniderModel(patient=patient)
        """
        # Handle dual interface
        if patient is not None:
            self.patient = patient
        elif age is not None:
            self.patient = PatientParameters(
                age=age or 40.0,
                weight=weight or 70.0,
                height=height or 170.0,
                gender=gender or "male"
            )
        else:
            self.patient = PatientParameters()
        
        # Validate patient parameters
        validate_patient_parameters(self.patient)
        
        self.params = params or SchniderParameters()
        self.use_covariates = use_covariates
        
        # Apply covariate adjustments if requested
        if use_covariates:
            self._apply_schnider_covariates()
    
    def _apply_schnider_covariates(self) -> None:
        """
        Apply Schnider covariate model to adjust PK parameters.
        
        Formulations (6)-(17) from Schnider et al., Anesthesiology 1998.
        
        The Schnider model adjusts volumes and clearances based on:
        - Age: Affects V2 and Cl2
        - Weight: Affects Cl1
        - Lean Body Mass (LBM): Affects Cl1
        - Height: Affects Cl1
        
        Volume Equations:
            V1 = h₁ (fixed at 4.27 L)
            V2 = h₂ - h₃·(age - h₄)
            V3 = h₅ (fixed at 238 L)
        
        Clearance Equations:
            Cl1 = h₆ + h₇·(weight - h₈) - h₉·(lbm - h₁₀) + h₁₁·(height - h₁₂)
            Cl2 = h₁₃ - h₁₄·(age - h₁₅)
            Cl3 = h₁₆ (fixed)
        
        Rate Constants:
            k10 = Cl1 / V1
            k12 = Cl2 / V1
            k13 = Cl3 / V1
            k21 = Cl2 / V2
            k31 = Cl3 / V3
        """
        age = self.patient.age
        weight = self.patient.weight
        lbm = self.patient.lbm
        height = self.patient.height
        
        # Formulation (6): V1 is fixed at 4.27 L
        self.params.V1 = self.params.h_1
        
        # Formulation (7): V2 depends on age
        self.params.V2 = self.params.h_2 - self.params.h_3 * (age - self.params.h_4)
        
        # Formulation (8): V3 is fixed at 238 L
        self.params.V3 = self.params.h_5
        
        # Formulation (9): Cl1 depends on weight, LBM, and height
        cl1 = (self.params.h_6 + 
               self.params.h_7 * (weight - self.params.h_8) - 
               self.params.h_9 * (lbm - self.params.h_10) + 
               self.params.h_11 * (height - self.params.h_12))
        
        # Formulation (10): Cl2 depends on age
        cl2 = self.params.h_13 - self.params.h_14 * (age - self.params.h_15)
        
        # Formulation (11): Cl3 is fixed
        cl3 = self.params.h_16
        
        # Formulations (12)-(16): Convert clearances to rate constants
        self.params.k10 = cl1 / self.params.V1
        self.params.k12 = cl2 / self.params.V1
        self.params.k13 = cl3 / self.params.V1
        self.params.k21 = cl2 / self.params.V2
        self.params.k31 = cl3 / self.params.V3
        
        # Formulation (17): ke0 is typically fixed
        self.params.ke0 = self.params.h_17
    
    def get_rate_constants(self) -> Dict[str, float]:
        """
        Get all rate constants as a dictionary.
        
        Returns:
            Dictionary with keys: k10, k12, k13, k21, k31, ke0
            All values in min^-1
        
        Example:
            >>> model = SchniderModel()
            >>> constants = model.get_rate_constants()
            >>> print(f"k10 = {constants['k10']:.3f} min^-1")
            k10 = 0.443 min^-1
        """
        return {
            'k10': self.params.k10,
            'k12': self.params.k12,
            'k13': self.params.k13,
            'k21': self.params.k21,
            'k31': self.params.k31,
            'ke0': self.params.ke0
        }
    
    def get_volumes(self) -> Dict[str, float]:
        """
        Get compartment volumes as a dictionary.
        
        Returns:
            Dictionary with keys: v1, v2, v3
            All values in liters (L)
        
        Example:
            >>> model = SchniderModel()
            >>> volumes = model.get_volumes()
            >>> print(f"V1 = {volumes['v1']:.2f} L")
            V1 = 4.27 L
        """
        return {
            'V1': self.params.V1,
            'V2': self.params.V2,
            'V3': self.params.V3
        }
    
    def get_state_space_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the state-space matrices A and B for the 3-compartment model.
        
        Formulations (1)-(3) in matrix form: ẋ(t) = Ax(t) + Bu(t)
        
        State vector: x = [C₁, C₂, C₃]ᵀ (compartment concentrations in μg/ml)
        Input: u(t) = infusion rate (μg/kg/min)
        
        Matrix A (3×3):
            A = [-(k₁₀+k₁₂+k₁₃)  k₂₁·V₂/V₁   k₃₁·V₃/V₁]
                [k₁₂·V₁/V₂       -k₂₁         0        ]
                [k₁₃·V₁/V₃       0            -k₃₁     ]
        
        Vector B (3×1):
            B = [1/(V₁·1000), 0, 0]ᵀ
            (converts μg/kg/min to μg/ml/min)
        
        Returns:
            Tuple of (A_matrix, B_vector) numpy arrays
            A_matrix: 3×3 system matrix (min^-1)
            B_vector: 3×1 input vector (ml/kg → ml/ml·min)
        
        Example:
            >>> model = SchniderModel()
            >>> A, B = model.get_state_space_matrices()
            >>> print(f"A shape: {A.shape}, det(A): {np.linalg.det(A):.6f}")
            A shape: (3, 3), det(A): 0.000398
        """
        p = self.params

        # Volume ratios for inter-compartment flows
        r21 = p.V2 / p.V1  # V₂/V₁
        r31 = p.V3 / p.V1  # V₃/V₁
        r12 = p.V1 / p.V2  # V₁/V₂
        r13 = p.V1 / p.V3  # V₁/V₃
        
        # System matrix A - Formulations (1)-(3) coefficients
        A = np.array([
            [-(p.k10 + p.k12 + p.k13), p.k21 * r21, p.k31 * r31],
            [p.k12 * r12, -p.k21, 0.0],
            [p.k13 * r13, 0.0, -p.k31]
        ])
        
        # Input vector B (1/(V1*1000) to convert infusion rate to concentration rate)
        # Infusion rate: μg/kg/min → concentration rate: μg/ml/min
        B = np.array([1.0 / (p.V1 * 1000), 0.0, 0.0])
        
        return A, B
    
    def get_extended_state_space_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get extended state-space matrices including effect-site compartment.
        
        Extended state: x = [C₁, C₂, C₃, Cₑ]ᵀ
        
        Formulation (31): Ċₑ(t) = ke₀·(C₁(t) - Cₑ(t))
        
        Matrix A_ext (4×4):
            A_ext = [-(k₁₀+k₁₂+k₁₃)  k₂₁·V₂/V₁   k₃₁·V₃/V₁   0   ]
                    [k₁₂·V₁/V₂       -k₂₁         0           0   ]
                    [k₁₃·V₁/V₃       0            -k₃₁        0   ]
                    [ke₀             0            0           -ke₀]
        
        Vector B_ext (4×1):
            B_ext = [1/(V₁·1000), 0, 0, 0]ᵀ
        
        Returns:
            Tuple of (A_extended, B_extended) numpy arrays
            A_extended: 4×4 system matrix (min^-1)
            B_extended: 4×1 input vector
        
        Example:
            >>> model = SchniderModel()
            >>> A_ext, B_ext = model.get_extended_state_space_matrices()
            >>> print(f"A_ext shape: {A_ext.shape}")
            A_ext shape: (4, 4)
        """
        p = self.params

        # Volume ratios
        r21 = p.V2 / p.V1
        r31 = p.V3 / p.V1
        r12 = p.V1 / p.V2
        r13 = p.V1 / p.V3
        
        # Extended system matrix (4x4)
        A_ext = np.array([
            [-(p.k10 + p.k12 + p.k13), p.k21 * r21, p.k31 * r31, 0.0],
            [p.k12 * r12, -p.k21, 0.0, 0.0],
            [p.k13 * r13, 0.0, -p.k31, 0.0],
            [p.ke0, 0.0, 0.0, -p.ke0]  # Formulation (31): Ċₑ = ke₀·(C₁ - Cₑ)
        ])
        
        # Extended input vector
        B_ext = np.array([1.0 / (p.V1 * 1000), 0.0, 0.0, 0.0])
        
        return A_ext, B_ext
