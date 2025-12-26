"""
Drug Interaction Models for Anesthesia
=======================================

This module implements pharmacodynamic interaction models for
multi-drug anesthesia, particularly propofol-remifentanil interaction.

Models Implemented:
-------------------
1. Greco Response Surface Model (synergistic interaction)
2. Minto Interaction Model (BIS prediction with dual drugs)
3. Additive Model (simple addition)
4. Multiplicative Model (product of effects)

CBIM Paper Formulation (32):
-----------------------------
BIS = f(Ce_propofol, Ce_remifentanil, interaction_params)

The Greco model captures synergistic interaction between propofol and
remifentanil, where combined effect exceeds sum of individual effects.

Reference:
----------
Minto CF, Schnider TW, Short TG, Gregg KM, Gentilini A, Shafer SL.
"Response surface model for anesthetic drug interactions."
Anesthesiology. 2000;92(6):1603-1616.

Greco WR, Bravo G, Parsons JC.
"The search for synergy: a critical review from a response surface perspective."
Pharmacol Rev. 1995;47(2):331-385.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum


class InteractionType(Enum):
    """Type of drug interaction."""
    ADDITIVE = "additive"  # α = 0
    SYNERGISTIC = "synergistic"  # α > 0
    ANTAGONISTIC = "antagonistic"  # α < 0


@dataclass
class DrugParameters:
    """Pharmacodynamic parameters for a single drug."""
    C50: float  # EC50 - concentration for 50% effect
    gamma: float  # Hill coefficient (slope)
    E0: float = 100.0  # Baseline effect (awake BIS = 100)
    Emax: float = 100.0  # Maximum effect


class GrecoInteractionModel:
    """
    Greco Response Surface Model for Drug Interaction.
    
    This model describes the combined effect of two drugs on BIS
    using a response surface approach that accounts for synergy.
    
    Mathematical Model:
    -------------------
    BIS = E0 - Emax × U / (1 + U)
    
    where:
    U = U_ppf + U_rftn + α × U_ppf × U_rftn
    
    U_ppf = (C_ppf / C50_ppf)^γ₁
    U_rftn = (C_rftn / C50_rftn)^γ₂
    
    Parameters:
    -----------
    - C_ppf, C_rftn: Effect-site concentrations (μg/mL, ng/mL)
    - C50_ppf, C50_rftn: EC50 values
    - γ₁, γ₂: Hill coefficients
    - α: Interaction parameter
      α = 0: Additive
      α > 0: Synergistic (combined effect > sum)
      α < 0: Antagonistic (combined effect < sum)
    
    Typical Values:
    ---------------
    Propofol:
    - C50 = 3.4 μg/mL
    - γ = 1.47
    
    Remifentanil:
    - C50 = 2.0 ng/mL
    - γ = 1.2
    
    Interaction:
    - α = 1.2 (synergistic for BIS depression)
    
    Example:
    --------
    >>> model = GrecoInteractionModel()
    >>> bis = model.compute_bis(ce_propofol=3.0, ce_remifentanil=1.5)
    >>> print(f"BIS: {bis:.1f}")
    """
    
    def __init__(
        self,
        propofol_params: Optional[DrugParameters] = None,
        remifentanil_params: Optional[DrugParameters] = None,
        alpha: float = 1.2,
        E0: float = 100.0,
        Emax: float = 100.0
    ):
        """
        Initialize Greco interaction model.
        
        Args:
            propofol_params: PD parameters for propofol
            remifentanil_params: PD parameters for remifentanil
            alpha: Interaction parameter (0=additive, >0=synergistic, <0=antagonistic)
            E0: Baseline BIS (awake = 100)
            Emax: Maximum BIS depression (100)
        """
        # Default propofol parameters
        if propofol_params is None:
            propofol_params = DrugParameters(
                C50=3.4,  # μg/mL
                gamma=1.47,  # Hill coefficient
                E0=100.0,
                Emax=100.0
            )
        
        # Default remifentanil parameters
        if remifentanil_params is None:
            remifentanil_params = DrugParameters(
                C50=2.0,  # ng/mL
                gamma=1.2,  # Hill coefficient
                E0=100.0,
                Emax=100.0
            )
        
        self.ppf_params = propofol_params
        self.rftn_params = remifentanil_params
        self.alpha = alpha
        self.E0 = E0
        self.Emax = Emax
        
        # Determine interaction type
        if abs(alpha) < 1e-6:
            self.interaction_type = InteractionType.ADDITIVE
        elif alpha > 0:
            self.interaction_type = InteractionType.SYNERGISTIC
        else:
            self.interaction_type = InteractionType.ANTAGONISTIC
    
    def compute_u(self, concentration: float, params: DrugParameters) -> float:
        """
        Compute normalized potency U for a drug.
        
        U = (C / C50)^γ
        
        Args:
            concentration: Effect-site concentration
            params: Drug parameters
        
        Returns:
            Normalized potency U
        """
        if concentration <= 0:
            return 0.0
        
        u = (concentration / params.C50) ** params.gamma
        return u
    
    def compute_bis(
        self,
        ce_propofol: float,
        ce_remifentanil: float
    ) -> float:
        """
        Compute BIS from dual drug concentrations using Greco model.
        
        Implements Formulation (32).
        
        Args:
            ce_propofol: Propofol effect-site concentration (μg/mL)
            ce_remifentanil: Remifentanil effect-site concentration (ng/mL)
        
        Returns:
            BIS value (0-100)
        """
        # Compute individual potencies
        u_ppf = self.compute_u(ce_propofol, self.ppf_params)
        u_rftn = self.compute_u(ce_remifentanil, self.rftn_params)
        
        # Compute combined potency with interaction
        # U_total = U_ppf + U_rftn + α × U_ppf × U_rftn
        U_total = u_ppf + u_rftn + self.alpha * u_ppf * u_rftn
        
        # Compute BIS
        # BIS = E0 - Emax × U / (1 + U)
        if U_total < 0:
            U_total = 0.0
        
        bis = self.E0 - self.Emax * (U_total / (1.0 + U_total))
        
        # Clamp to valid range
        bis = np.clip(bis, 0.0, 100.0)
        
        return bis
    
    def compute_effect(
        self,
        ce_propofol: float,
        ce_remifentanil: float
    ) -> float:
        """
        Compute effect (0-1) instead of BIS.
        
        Effect = 1 - (BIS / 100)
        
        Args:
            ce_propofol: Propofol Ce (μg/mL)
            ce_remifentanil: Remifentanil Ce (ng/mL)
        
        Returns:
            Effect magnitude (0 = awake, 1 = maximal sedation)
        """
        bis = self.compute_bis(ce_propofol, ce_remifentanil)
        effect = 1.0 - (bis / 100.0)
        return np.clip(effect, 0.0, 1.0)
    
    def compute_isobol(
        self,
        target_bis: float = 50.0,
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute isobol (line of equal effect) for given target BIS.
        
        An isobol shows all combinations of propofol and remifentanil
        concentrations that produce the same BIS.
        
        Args:
            target_bis: Target BIS value
            n_points: Number of points to compute
        
        Returns:
            Tuple of (ce_propofol_array, ce_remifentanil_array)
        """
        ce_ppf_array = []
        ce_rftn_array = []
        
        # Range of propofol concentrations
        ce_ppf_range = np.linspace(0.0, 10.0, n_points)  # μg/mL
        
        for ce_ppf in ce_ppf_range:
            # Binary search for matching remifentanil concentration
            ce_rftn_low = 0.0
            ce_rftn_high = 20.0  # ng/mL
            
            for _ in range(20):  # Binary search iterations
                ce_rftn_mid = (ce_rftn_low + ce_rftn_high) / 2.0
                bis = self.compute_bis(ce_ppf, ce_rftn_mid)
                
                if abs(bis - target_bis) < 0.1:
                    ce_ppf_array.append(ce_ppf)
                    ce_rftn_array.append(ce_rftn_mid)
                    break
                elif bis > target_bis:
                    ce_rftn_low = ce_rftn_mid
                else:
                    ce_rftn_high = ce_rftn_mid
        
        return np.array(ce_ppf_array), np.array(ce_rftn_array)
    
    def get_info(self) -> Dict:
        """Get model information."""
        return {
            'model': 'Greco Response Surface',
            'interaction_type': self.interaction_type.value,
            'alpha': self.alpha,
            'E0': self.E0,
            'Emax': self.Emax,
            'propofol': {
                'C50': self.ppf_params.C50,
                'gamma': self.ppf_params.gamma
            },
            'remifentanil': {
                'C50': self.rftn_params.C50,
                'gamma': self.rftn_params.gamma
            }
        }


class MintoInteractionModel:
    """
    Minto Interaction Model (simplified for BIS).
    
    This is a simplified version based on the Minto paper,
    using a hierarchical sigmoid model.
    
    BIS = E0 × (1 - Effect_ppf) × (1 - Effect_rftn)
    
    where:
    Effect_drug = Emax × C^γ / (C50^γ + C^γ)
    """
    
    def __init__(
        self,
        propofol_C50: float = 3.4,
        propofol_gamma: float = 1.47,
        remifentanil_C50: float = 2.0,
        remifentanil_gamma: float = 1.2,
        E0: float = 100.0
    ):
        """
        Initialize Minto interaction model.
        
        Args:
            propofol_C50: Propofol EC50 (μg/mL)
            propofol_gamma: Propofol Hill coefficient
            remifentanil_C50: Remifentanil EC50 (ng/mL)
            remifentanil_gamma: Remifentanil Hill coefficient
            E0: Baseline BIS
        """
        self.ppf_C50 = propofol_C50
        self.ppf_gamma = propofol_gamma
        self.rftn_C50 = remifentanil_C50
        self.rftn_gamma = remifentanil_gamma
        self.E0 = E0
    
    def compute_effect(self, concentration: float, C50: float, gamma: float) -> float:
        """Compute fractional effect for single drug."""
        if concentration <= 0:
            return 0.0
        
        numerator = concentration ** gamma
        denominator = (C50 ** gamma) + (concentration ** gamma)
        
        return numerator / denominator
    
    def compute_bis(
        self,
        ce_propofol: float,
        ce_remifentanil: float
    ) -> float:
        """
        Compute BIS from dual drug concentrations.
        
        Args:
            ce_propofol: Propofol Ce (μg/mL)
            ce_remifentanil: Remifentanil Ce (ng/mL)
        
        Returns:
            BIS value (0-100)
        """
        # Individual effects
        effect_ppf = self.compute_effect(ce_propofol, self.ppf_C50, self.ppf_gamma)
        effect_rftn = self.compute_effect(ce_remifentanil, self.rftn_C50, self.rftn_gamma)
        
        # Combined effect (multiplicative)
        bis = self.E0 * (1.0 - effect_ppf) * (1.0 - effect_rftn)
        
        return np.clip(bis, 0.0, 100.0)


class AdditiveInteractionModel:
    """
    Simple additive interaction model.
    
    Effect_total = Effect_propofol + Effect_remifentanil
    BIS = E0 - Emax × Effect_total
    
    This assumes no interaction (α = 0 in Greco model).
    """
    
    def __init__(
        self,
        propofol_C50: float = 3.4,
        propofol_gamma: float = 1.47,
        remifentanil_C50: float = 2.0,
        remifentanil_gamma: float = 1.2,
        E0: float = 100.0,
        Emax: float = 100.0
    ):
        """Initialize additive model."""
        self.ppf_C50 = propofol_C50
        self.ppf_gamma = propofol_gamma
        self.rftn_C50 = remifentanil_C50
        self.rftn_gamma = remifentanil_gamma
        self.E0 = E0
        self.Emax = Emax
    
    def compute_effect(self, concentration: float, C50: float, gamma: float) -> float:
        """Compute fractional effect for single drug."""
        if concentration <= 0:
            return 0.0
        
        numerator = concentration ** gamma
        denominator = (C50 ** gamma) + (concentration ** gamma)
        
        return numerator / denominator
    
    def compute_bis(
        self,
        ce_propofol: float,
        ce_remifentanil: float
    ) -> float:
        """Compute BIS with additive interaction."""
        effect_ppf = self.compute_effect(ce_propofol, self.ppf_C50, self.ppf_gamma)
        effect_rftn = self.compute_effect(ce_remifentanil, self.rftn_C50, self.rftn_gamma)
        
        # Additive effect (clamped to [0, 1])
        effect_total = min(1.0, effect_ppf + effect_rftn)
        
        bis = self.E0 - self.Emax * effect_total
        
        return np.clip(bis, 0.0, 100.0)


def test_interaction_models():
    """Test and compare interaction models."""
    print("Testing Drug Interaction Models")
    print("=" * 70)
    
    # Create models
    greco = GrecoInteractionModel(alpha=1.2)  # Synergistic
    minto = MintoInteractionModel()
    additive = AdditiveInteractionModel()
    
    # Test concentrations
    ce_ppf = 3.0  # μg/mL (near EC50)
    ce_rftn = 2.0  # ng/mL (at EC50)
    
    print(f"\nTest Case:")
    print(f"  Propofol Ce: {ce_ppf:.1f} μg/mL")
    print(f"  Remifentanil Ce: {ce_rftn:.1f} ng/mL")
    
    print(f"\nBIS Predictions:")
    print(f"  Greco (synergistic, α=1.2): {greco.compute_bis(ce_ppf, ce_rftn):.1f}")
    print(f"  Minto (multiplicative):     {minto.compute_bis(ce_ppf, ce_rftn):.1f}")
    print(f"  Additive (α=0):             {additive.compute_bis(ce_ppf, ce_rftn):.1f}")
    
    # Test range of concentrations
    print(f"\n{'Propofol':>12} {'Remifent':>12} {'Greco BIS':>12} {'Minto BIS':>12} {'Additive':>12}")
    print("-" * 65)
    
    for ppf in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
        for rftn in [0.0, 1.0, 2.0]:
            bis_greco = greco.compute_bis(ppf, rftn)
            bis_minto = minto.compute_bis(ppf, rftn)
            bis_additive = additive.compute_bis(ppf, rftn)
            
            if rftn == 0.0 or ppf in [0.0, 3.0]:
                print(f"{ppf:12.1f} {rftn:12.1f} {bis_greco:12.1f} {bis_minto:12.1f} {bis_additive:12.1f}")
    
    # Compute isobol for BIS = 50
    print("\nComputing isobol for BIS = 50...")
    ce_ppf_iso, ce_rftn_iso = greco.compute_isobol(target_bis=50.0, n_points=50)
    print(f"  Found {len(ce_ppf_iso)} points on isobol")
    print(f"  Propofol alone (for BIS=50): ~{ce_ppf_iso[0]:.2f} μg/mL")
    print(f"  Remifentanil alone (for BIS=50): ~{ce_rftn_iso[-1]:.2f} ng/mL")
    
    # Model info
    print(f"\nGreco Model Info:")
    info = greco.get_info()
    print(f"  Interaction: {info['interaction_type']} (α={info['alpha']})")
    print(f"  Propofol C50: {info['propofol']['C50']} μg/mL")
    print(f"  Remifentanil C50: {info['remifentanil']['C50']} ng/mL")
    
    print("\n✓ Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_interaction_models()
