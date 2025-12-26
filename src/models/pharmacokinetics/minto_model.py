"""
Minto Pharmacokinetic Model for Remifentanil
=============================================

This module implements the Minto 3-compartment pharmacokinetic model
for remifentanil, as described in the literature.

Reference:
----------
Minto CF, Schnider TW, Egan TD, et al.
"Influence of age and gender on the pharmacokinetics and pharmacodynamics
of remifentanil. I. Model development."
Anesthesiology. 1997;86(1):10-23.

CBIM Paper Formulations (18-29):
---------------------------------
The Minto model describes remifentanil kinetics using a 3-compartment model:

(18) dA1/dt = -k10·A1 - k12·A1 + k21·A2 - k13·A1 + k31·A3 + u(t)
(19) dA2/dt = k12·A1 - k21·A2
(20) dA3/dt = k13·A1 - k31·A3

Where:
- A1, A2, A3: Drug amounts in central, rapid peripheral, slow peripheral compartments
- k10, k12, k21, k13, k31: Rate constants
- u(t): Infusion rate (μg/kg/min)

Rate constants are derived from volumes and clearances:
(21) V1 = 5.1 - 0.0201*(age-40) + 0.072*(LBM-55)
(22) V2 = 9.82 - 0.0811*(age-40) + 0.108*(LBM-55)
(23) V3 = 5.42
(24) Cl1 = 2.6 - 0.0162*(age-40) + 0.0191*(LBM-55)
(25) Cl2 = 2.05 - 0.0301*(age-40)
(26) Cl3 = 0.076 - 0.00113*(age-40)
(27) k10 = Cl1/V1
(28) k12 = Cl2/V1, k21 = Cl2/V2
(29) k13 = Cl3/V1, k31 = Cl3/V3

Lean Body Mass (LBM):
- Male: LBM = 1.10 × Weight - 128 × (Weight/Height)²
- Female: LBM = 1.07 × Weight - 148 × (Weight/Height)²

Effect-Site:
- ke0 = 0.595 min⁻¹ (t1/2ke0 = 1.16 min)
- Ce = A1/V1 (simplified, or use effect-site compartment)

Pharmacodynamics:
- EC50 = 2.0 ng/mL (effect-site concentration for 50% effect)
- Gamma = 1.2 (Hill coefficient)
"""

import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class MintoParameters:
    """Minto model parameters for a patient."""
    # Patient characteristics
    age: float  # years
    weight: float  # kg
    height: float  # cm
    gender: str  # 'M' or 'F'
    
    # Derived
    lbm: float  # Lean body mass (kg)
    
    # Volumes (L)
    V1: float  # Central compartment
    V2: float  # Rapid peripheral
    V3: float  # Slow peripheral
    
    # Clearances (L/min)
    Cl1: float  # Elimination clearance
    Cl2: float  # Central <-> Rapid peripheral
    Cl3: float  # Central <-> Slow peripheral
    
    # Rate constants (min⁻¹)
    k10: float  # Central -> Elimination
    k12: float  # Central -> Rapid peripheral
    k21: float  # Rapid peripheral -> Central
    k13: float  # Central -> Slow peripheral
    k31: float  # Slow peripheral -> Central
    ke0: float  # Effect-site equilibration
    
    # Pharmacodynamics
    EC50: float  # ng/mL
    gamma: float  # Hill coefficient


class MintoModel:
    """
    Minto 3-compartment pharmacokinetic model for remifentanil.
    
    This model accurately represents remifentanil pharmacokinetics
    based on patient covariates (age, weight, height, gender).
    
    State Variables:
    ----------------
    - A1: Amount in central compartment (μg)
    - A2: Amount in rapid peripheral compartment (μg)
    - A3: Amount in slow peripheral compartment (μg)
    - Ae: Amount in effect site (μg)
    
    Concentrations:
    ---------------
    - Cp: Plasma concentration = A1/V1 (ng/mL)
    - Ce: Effect-site concentration = Ae/V1 (ng/mL)
    
    Example:
    --------
    >>> model = MintoModel(age=45, weight=70, height=170, gender='M')
    >>> model.set_infusion_rate(0.3)  # μg/kg/min
    >>> for _ in range(60):  # Simulate 60 minutes
    ...     model.step(dt=1.0)  # 1-minute steps
    ...     print(f"Ce: {model.Ce:.2f} ng/mL")
    """
    
    def __init__(
        self,
        age: float,
        weight: float,
        height: float,
        gender: str,
        dt: float = 0.0833  # 5 seconds in minutes
    ):
        """
        Initialize Minto model.
        
        Args:
            age: Patient age (years), typically 18-90
            weight: Patient weight (kg), typically 40-150
            height: Patient height (cm), typically 140-200
            gender: Patient gender ('M' or 'F')
            dt: Time step for simulation (minutes)
        """
        self.age = age
        self.weight = weight
        self.height = height
        self.gender = gender.upper()
        self.dt = dt
        
        # Compute parameters
        self.params = self._compute_parameters()
        
        # Initialize state
        self.reset()
    
    def _compute_parameters(self) -> MintoParameters:
        """
        Compute Minto model parameters from patient covariates.
        
        Implements Formulations (21)-(29).
        
        Returns:
            MintoParameters object
        """
        age = self.age
        weight = self.weight
        height_m = self.height / 100.0  # Convert cm to m
        
        # Compute Lean Body Mass (LBM)
        if self.gender == 'M':
            # Male: LBM = 1.10 × Weight - 128 × (Weight/Height²)
            lbm = 1.10 * weight - 128 * (weight / (height_m ** 2))
        else:
            # Female: LBM = 1.07 × Weight - 148 × (Weight/Height²)
            lbm = 1.07 * weight - 148 * (weight / (height_m ** 2))
        
        # Ensure LBM is reasonable
        lbm = max(20, min(lbm, weight))
        
        # Formulation (21): V1 (L)
        V1 = 5.1 - 0.0201 * (age - 40) + 0.072 * (lbm - 55)
        V1 = max(2.0, V1)  # Minimum 2L
        
        # Formulation (22): V2 (L)
        V2 = 9.82 - 0.0811 * (age - 40) + 0.108 * (lbm - 55)
        V2 = max(3.0, V2)  # Minimum 3L
        
        # Formulation (23): V3 (L)
        V3 = 5.42  # Constant
        
        # Formulation (24): Cl1 (L/min)
        Cl1 = 2.6 - 0.0162 * (age - 40) + 0.0191 * (lbm - 55)
        Cl1 = max(0.5, Cl1)  # Minimum 0.5 L/min
        
        # Formulation (25): Cl2 (L/min)
        Cl2 = 2.05 - 0.0301 * (age - 40)
        Cl2 = max(0.5, Cl2)
        
        # Formulation (26): Cl3 (L/min)
        Cl3 = 0.076 - 0.00113 * (age - 40)
        Cl3 = max(0.01, Cl3)
        
        # Formulation (27): k10 = Cl1/V1
        k10 = Cl1 / V1
        
        # Formulation (28): k12, k21
        k12 = Cl2 / V1
        k21 = Cl2 / V2
        
        # Formulation (29): k13, k31
        k13 = Cl3 / V1
        k31 = Cl3 / V3
        
        # Effect-site equilibration rate constant
        ke0 = 0.595  # min⁻¹ (t1/2 = 1.16 min)
        
        # Pharmacodynamics
        EC50 = 2.0  # ng/mL
        gamma = 1.2  # Hill coefficient
        
        return MintoParameters(
            age=age,
            weight=weight,
            height=self.height,
            gender=self.gender,
            lbm=lbm,
            V1=V1, V2=V2, V3=V3,
            Cl1=Cl1, Cl2=Cl2, Cl3=Cl3,
            k10=k10, k12=k12, k21=k21, k13=k13, k31=k31,
            ke0=ke0,
            EC50=EC50,
            gamma=gamma
        )
    
    def reset(self):
        """Reset state to initial conditions (no drug)."""
        self.A1 = 0.0  # Central compartment (μg)
        self.A2 = 0.0  # Rapid peripheral (μg)
        self.A3 = 0.0  # Slow peripheral (μg)
        self.Ae = 0.0  # Effect site (μg)
        
        self.infusion_rate = 0.0  # μg/kg/min
        self.time = 0.0  # minutes
    
    def set_infusion_rate(self, rate: float):
        """
        Set remifentanil infusion rate.
        
        Args:
            rate: Infusion rate (μg/kg/min)
        """
        self.infusion_rate = max(0.0, rate)
    
    def step(self, dt: Optional[float] = None) -> Tuple[float, float]:
        """
        Advance simulation by one time step.
        
        Implements Formulations (18)-(20) using Euler integration.
        
        Args:
            dt: Time step (minutes). If None, uses self.dt
        
        Returns:
            Tuple of (Cp, Ce) - plasma and effect-site concentrations (ng/mL)
        """
        if dt is None:
            dt = self.dt
        
        p = self.params
        
        # Infusion rate in μg/min
        u = self.infusion_rate * self.weight
        
        # Formulation (18): dA1/dt
        dA1_dt = -p.k10 * self.A1 - p.k12 * self.A1 + p.k21 * self.A2 \
                 - p.k13 * self.A1 + p.k31 * self.A3 + u
        
        # Formulation (19): dA2/dt
        dA2_dt = p.k12 * self.A1 - p.k21 * self.A2
        
        # Formulation (20): dA3/dt
        dA3_dt = p.k13 * self.A1 - p.k31 * self.A3
        
        # Effect-site dynamics
        dAe_dt = p.ke0 * (self.A1 - self.Ae)
        
        # Update state (Euler integration)
        self.A1 += dA1_dt * dt
        self.A2 += dA2_dt * dt
        self.A3 += dA3_dt * dt
        self.Ae += dAe_dt * dt
        
        # Ensure non-negative
        self.A1 = max(0.0, self.A1)
        self.A2 = max(0.0, self.A2)
        self.A3 = max(0.0, self.A3)
        self.Ae = max(0.0, self.Ae)
        
        # Update time
        self.time += dt
        
        return self.Cp, self.Ce
    
    @property
    def Cp(self) -> float:
        """Plasma concentration (ng/mL)."""
        return (self.A1 / self.params.V1) if self.params.V1 > 0 else 0.0
    
    @property
    def Ce(self) -> float:
        """Effect-site concentration (ng/mL)."""
        return (self.Ae / self.params.V1) if self.params.V1 > 0 else 0.0
    
    def get_effect(self) -> float:
        """
        Compute pharmacodynamic effect (0-1).
        
        Uses Hill equation:
        Effect = Ce^γ / (EC50^γ + Ce^γ)
        
        Returns:
            Effect magnitude (0 = no effect, 1 = maximal effect)
        """
        Ce = self.Ce
        EC50 = self.params.EC50
        gamma = self.params.gamma
        
        if Ce <= 0:
            return 0.0
        
        numerator = Ce ** gamma
        denominator = (EC50 ** gamma) + (Ce ** gamma)
        
        return numerator / denominator
    
    def get_state(self) -> np.ndarray:
        """
        Get current state vector.
        
        Returns:
            State array [A1, A2, A3, Ae, Cp, Ce]
        """
        return np.array([
            self.A1,
            self.A2,
            self.A3,
            self.Ae,
            self.Cp,
            self.Ce
        ])
    
    def set_state(self, state: np.ndarray):
        """
        Set state from vector.
        
        Args:
            state: State array [A1, A2, A3, Ae]
        """
        self.A1 = max(0.0, state[0])
        self.A2 = max(0.0, state[1])
        self.A3 = max(0.0, state[2])
        self.Ae = max(0.0, state[3])
    
    def get_info(self) -> Dict:
        """Get model information and parameters."""
        p = self.params
        return {
            'model': 'Minto',
            'patient': {
                'age': self.age,
                'weight': self.weight,
                'height': self.height,
                'gender': self.gender,
                'lbm': p.lbm
            },
            'volumes': {
                'V1': p.V1,
                'V2': p.V2,
                'V3': p.V3
            },
            'clearances': {
                'Cl1': p.Cl1,
                'Cl2': p.Cl2,
                'Cl3': p.Cl3
            },
            'rate_constants': {
                'k10': p.k10,
                'k12': p.k12,
                'k21': p.k21,
                'k13': p.k13,
                'k31': p.k31,
                'ke0': p.ke0
            },
            'pharmacodynamics': {
                'EC50': p.EC50,
                'gamma': p.gamma
            },
            'state': {
                'A1': self.A1,
                'A2': self.A2,
                'A3': self.A3,
                'Ae': self.Ae,
                'Cp': self.Cp,
                'Ce': self.Ce,
                'effect': self.get_effect(),
                'time': self.time
            }
        }


def test_minto_model():
    """Test the Minto model with typical patient."""
    print("Testing Minto Model...")
    print("=" * 70)
    
    # Create model for typical patient
    model = MintoModel(
        age=45,
        weight=70,
        height=170,
        gender='M'
    )
    
    # Print parameters
    info = model.get_info()
    print("\nPatient:")
    for key, value in info['patient'].items():
        print(f"  {key}: {value}")
    
    print("\nPK Parameters:")
    print(f"  V1: {info['volumes']['V1']:.2f} L")
    print(f"  V2: {info['volumes']['V2']:.2f} L")
    print(f"  V3: {info['volumes']['V3']:.2f} L")
    print(f"  Cl1: {info['clearances']['Cl1']:.3f} L/min")
    print(f"  k10: {info['rate_constants']['k10']:.4f} min⁻¹")
    print(f"  ke0: {info['rate_constants']['ke0']:.4f} min⁻¹")
    
    # Simulate induction
    print("\nSimulating remifentanil induction (0.5 μg/kg/min)...")
    print(f"{'Time (min)':>12} {'Cp (ng/mL)':>12} {'Ce (ng/mL)':>12} {'Effect':>10}")
    print("-" * 50)
    
    model.set_infusion_rate(0.5)  # μg/kg/min
    
    for minute in range(15):  # 15 minutes
        for _ in range(int(60 / (model.dt * 60))):  # Steps per minute
            model.step()
        
        if minute % 3 == 0:
            print(f"{model.time:12.1f} {model.Cp:12.2f} {model.Ce:12.2f} {model.get_effect():10.3f}")
    
    # Maintenance
    print("\nSwitching to maintenance (0.2 μg/kg/min)...")
    model.set_infusion_rate(0.2)
    
    for minute in range(15, 30):
        for _ in range(int(60 / (model.dt * 60))):
            model.step()
        
        if minute % 3 == 0:
            print(f"{model.time:12.1f} {model.Cp:12.2f} {model.Ce:12.2f} {model.get_effect():10.3f}")
    
    # Stop infusion
    print("\nStopping infusion (recovery)...")
    model.set_infusion_rate(0.0)
    
    for minute in range(30, 45):
        for _ in range(int(60 / (model.dt * 60))):
            model.step()
        
        if minute % 3 == 0:
            print(f"{model.time:12.1f} {model.Cp:12.2f} {model.Ce:12.2f} {model.get_effect():10.3f}")
    
    print("\n✓ Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_minto_model()
