"""
Patient Simulator - PK/PD Model for Propofol
=============================================

This module implements the pharmacokinetic/pharmacodynamic (PK/PD) model
for propofol based on the Schnider model.

The model consists of:
1. Three-compartment PK model (Central, Shallow Peripheral, Deep Peripheral)
2. Effect-site compartment with ke0 equilibration
3. Hill/Sigmoid Emax model for BIS prediction

Mathematical Formulation:
-------------------------
PK Model (Three-Compartment Mammillary Model):
    dC1/dt = u(t)/V1 - (k10 + k12 + k13)*C1 + (V2/V1)*k21*C2 + (V3/V1)*k31*C3
    dC2/dt = (V1/V2)*k12*C1 - k21*C2
    dC3/dt = (V1/V3)*k13*C1 - k31*C3

Effect-Site Model:
    dCe/dt = ke0 * (C1 - Ce)

BIS Prediction (Hill/Sigmoid Emax):
    BIS = E0 - Emax * (Ce^gamma / (Ce^gamma + EC50^gamma))

References:
-----------
- Schnider TW, et al. "The influence of method of administration and 
  covariates on the pharmacokinetics of propofol in adult volunteers."
  Anesthesiology. 1998.
- CBIM (Closed-loop BIS-guided Infusion Model) paper
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from scipy.integrate import odeint


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
            lbm = 1.1 * self.weight - 128 * (self.weight / self.height) ** 2
        else:
            lbm = 1.07 * self.weight - 148 * (self.weight / self.height) ** 2
        
        # Ensure LBM is reasonable (at least 30% of body weight)
        return max(lbm, 0.3 * self.weight)


@dataclass
class SchniderModelParameters:
    """
    Schnider model PK/PD parameters.
    
    These parameters define the three-compartment pharmacokinetic model
    and the effect-site pharmacodynamic model for propofol.
    """
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
        self.params.v1 = 4.27
        
        # V2 depends on age
        self.params.v2 = 18.9 - 0.391 * (age - 53)
        
        # V3 is fixed
        self.params.v3 = 238.0
        
        # Clearances and rate constants
        # Cl1 (elimination clearance) depends on weight
        cl1 = 1.89 + 0.0456 * (weight - 77) - 0.0681 * (lbm - 59) + 0.0264 * (height - 177)
        cl2 = 1.29 - 0.024 * (age - 53)
        cl3 = 0.836
        
        # Convert clearances to rate constants
        self.params.k10 = cl1 / self.params.v1
        self.params.k12 = cl2 / self.params.v1
        self.params.k21 = cl2 / self.params.v2
        self.params.k13 = cl3 / self.params.v1
        self.params.k31 = cl3 / self.params.v3
        
        # ke0 is typically fixed but can vary with age
        self.params.ke0 = 0.456
    
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


class PatientSimulator:
    """
    Patient Simulator for Propofol Anesthesia.
    
    This simulator implements the complete PK/PD model for propofol
    anesthesia control, computing the dynamic response of drug
    concentrations and BIS to propofol infusion.
    
    State Variables:
        - C1: Central compartment concentration (μg/ml)
        - C2: Shallow peripheral concentration (μg/ml)
        - C3: Deep peripheral concentration (μg/ml)
        - Ce: Effect-site concentration (μg/ml)
    
    The simulator maintains internal state and can be stepped forward
    in time with varying infusion rates.
    
    Attributes:
        model: Schnider PK/PD model
        state: Current concentrations [C1, C2, C3, Ce]
        time: Current simulation time (seconds)
        dt: Default time step (seconds)
        history: Record of states, actions, and BIS values
    """
    
    def __init__(
        self,
        patient: Optional[PatientParameters] = None,
        model: Optional[SchniderModel] = None,
        dt: float = 5.0,
        noise_std: float = 2.0,
        seed: Optional[int] = None
    ):
        """
        Initialize the patient simulator.
        
        Args:
            patient: Patient demographic parameters
            model: Pre-configured Schnider model (creates new if None)
            dt: Default time step in seconds
            noise_std: Standard deviation of BIS measurement noise
            seed: Random seed for reproducibility
        """
        if model is not None:
            self.model = model
        else:
            self.model = SchniderModel(patient=patient)
        
        self.dt = dt
        self.noise_std = noise_std
        
        # Random state for reproducibility
        self.rng = np.random.default_rng(seed)
        
        # Initialize state
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        Reset the simulator to initial conditions.
        
        Returns:
            Initial state vector [C1, C2, C3, Ce]
        """
        # All concentrations start at zero (no drug in system)
        self.state = np.zeros(4)  # [C1, C2, C3, Ce]
        self.time = 0.0
        
        # Clear history
        self.history = {
            'time': [],
            'state': [],
            'bis': [],
            'dose': [],
            'ce': []
        }
        
        return self.state.copy()
    
    def _pk_pd_ode(self, y: np.ndarray, t: float, infusion_rate: float) -> np.ndarray:
        """
        Compute the derivatives for the PK/PD ODE system.
        
        Args:
            y: State vector [C1, C2, C3, Ce]
            t: Current time (not used directly, required by odeint)
            infusion_rate: Propofol infusion rate (μg/kg/min)
        
        Returns:
            Derivatives [dC1/dt, dC2/dt, dC3/dt, dCe/dt]
        """
        C1, C2, C3, Ce = y
        p = self.model.params
        
        # Convert infusion rate from μg/kg/min to μg/min
        # Then to concentration rate by dividing by V1 (in L) * 1000 (convert L to ml)
        # Note: V1 is in L, concentrations are in μg/ml
        weight = self.model.patient.weight
        infusion_ug_per_min = infusion_rate * weight  # μg/min
        
        # Rate of change of central compartment concentration
        # dC1/dt = infusion/V1 - (k10 + k12 + k13)*C1 + k21*V2/V1*C2 + k31*V3/V1*C3
        dC1 = (infusion_ug_per_min / (p.v1 * 1000)  # Convert V1 from L to ml
               - (p.k10 + p.k12 + p.k13) * C1
               + p.k21 * C2 * p.v2 / p.v1
               + p.k31 * C3 * p.v3 / p.v1)
        
        # Rate of change of shallow peripheral
        dC2 = p.k12 * C1 * p.v1 / p.v2 - p.k21 * C2
        
        # Rate of change of deep peripheral
        dC3 = p.k13 * C1 * p.v1 / p.v3 - p.k31 * C3
        
        # Rate of change of effect-site
        dCe = p.ke0 * (C1 - Ce)
        
        return np.array([dC1, dC2, dC3, dCe])
    
    def step(self, infusion_rate: float, dt: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """
        Advance the simulation by one time step.
        
        Args:
            infusion_rate: Propofol infusion rate (μg/kg/min)
            dt: Time step in seconds (uses default if None)
        
        Returns:
            Tuple of (new_state, measured_bis)
        """
        if dt is None:
            dt = self.dt
        
        # Convert dt from seconds to minutes for the model
        dt_min = dt / 60.0
        
        # Integrate the ODE system
        t_span = np.array([0, dt_min])
        solution = odeint(
            self._pk_pd_ode,
            self.state,
            t_span,
            args=(infusion_rate,)
        )
        
        # Update state
        self.state = solution[-1]
        self.time += dt
        
        # Compute BIS with measurement noise
        bis_true = self._compute_bis(self.state[3])
        bis_measured = bis_true + self.rng.normal(0, self.noise_std)
        bis_measured = np.clip(bis_measured, 0, 100)
        
        # Record history
        self.history['time'].append(self.time)
        self.history['state'].append(self.state.copy())
        self.history['bis'].append(bis_measured)
        self.history['dose'].append(infusion_rate)
        self.history['ce'].append(self.state[3])
        
        return self.state.copy(), bis_measured
    
    def _compute_bis(self, ce: float) -> float:
        """
        Compute BIS from effect-site concentration using Hill model.
        
        BIS = E0 - Emax * (Ce^gamma / (Ce^gamma + EC50^gamma))
        
        Args:
            ce: Effect-site concentration (μg/ml)
        
        Returns:
            BIS value (0-100)
        """
        p = self.model.params
        
        if ce <= 0:
            return p.e0
        
        # Hill/Sigmoid Emax model
        ce_gamma = ce ** p.gamma
        ec50_gamma = p.ec50 ** p.gamma
        
        effect = p.emax * (ce_gamma / (ce_gamma + ec50_gamma))
        bis = p.e0 - effect
        
        # Clip to valid BIS range
        return np.clip(bis, 0, 100)
    
    def get_bis(self, add_noise: bool = True) -> float:
        """
        Get current BIS value.
        
        Args:
            add_noise: Whether to add measurement noise
        
        Returns:
            Current BIS value
        """
        bis_true = self._compute_bis(self.state[3])
        
        if add_noise:
            bis = bis_true + self.rng.normal(0, self.noise_std)
            return np.clip(bis, 0, 100)
        
        return bis_true
    
    def get_concentrations(self) -> Dict[str, float]:
        """
        Get current drug concentrations.
        
        Returns:
            Dictionary with C1, C2, C3, Ce concentrations
        """
        return {
            'C1': self.state[0],
            'C2': self.state[1],
            'C3': self.state[2],
            'Ce': self.state[3]
        }
    
    def get_observation(self) -> np.ndarray:
        """
        Get the full observation vector for RL.
        
        The observation includes:
        - Normalized BIS error: (BIS - target) / 50
        - Normalized effect-site concentration: Ce / EC50
        - Additional features can be added here
        
        Returns:
            Observation vector for the RL agent
        """
        bis = self.get_bis(add_noise=True)
        ce = self.state[3]
        
        # Normalize features
        bis_error = (bis - 50) / 50.0  # Normalized error from target
        ce_normalized = ce / self.model.params.ec50  # Normalized by EC50
        
        return np.array([bis_error, ce_normalized], dtype=np.float32)
    
    def administer_bolus(self, dose_mg_per_kg: float):
        """
        Administer a bolus dose of propofol.
        
        This simulates rapid injection by instantly increasing
        the central compartment concentration.
        
        Args:
            dose_mg_per_kg: Bolus dose in mg/kg
        """
        weight = self.model.patient.weight
        v1_ml = self.model.params.v1 * 1000  # Convert L to ml
        
        # Convert mg/kg to μg and calculate concentration increase
        dose_ug = dose_mg_per_kg * weight * 1000  # mg to μg
        concentration_increase = dose_ug / v1_ml
        
        self.state[0] += concentration_increase
    
    def get_history_arrays(self) -> Dict[str, np.ndarray]:
        """
        Get simulation history as numpy arrays.
        
        Returns:
            Dictionary of numpy arrays for time, states, BIS, doses
        """
        return {
            'time': np.array(self.history['time']),
            'state': np.array(self.history['state']),
            'bis': np.array(self.history['bis']),
            'dose': np.array(self.history['dose']),
            'ce': np.array(self.history['ce'])
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
