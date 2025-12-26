"""
Common Base Classes and Utilities for PK/PD Models
===================================================

This module provides base classes and common utilities for pharmacokinetic
and pharmacodynamic models.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class PatientParameters:
    """
    Patient demographic parameters for PK/PD model individualization.
    
    Attributes:
        age: Patient age in years (typically 18-90)
        weight: Patient weight in kg (typically 40-150)
        height: Patient height in cm (typically 140-200)
        gender: Patient gender ('male'/'M' or 'female'/'F')
        lbm: Lean body mass in kg (calculated if not provided)
    """
    age: float = 40.0
    weight: float = 70.0
    height: float = 170.0
    gender: str = "male"
    lbm: Optional[float] = None
    
    def __post_init__(self):
        """Calculate lean body mass if not provided and normalize gender."""
        # Normalize gender to lowercase
        self.gender = self.gender.lower()
        if self.gender not in ['male', 'female', 'm', 'f']:
            self.gender = 'male'  # Default to male
        
        # Normalize gender abbreviations
        if self.gender in ['m', 'male']:
            self.gender = 'male'
        elif self.gender in ['f', 'female']:
            self.gender = 'female'
        
        # Calculate LBM if not provided
        if self.lbm is None:
            self.lbm = self._calculate_lbm()
    
    def _calculate_lbm(self) -> float:
        """
        Calculate Lean Body Mass using James formula.
        
        For males: LBM = 1.1 × weight - 128 × (weight/height)²
        For females: LBM = 1.07 × weight - 148 × (weight/height)²
        
        Returns:
            Lean body mass in kg
        """
        # Convert height to meters for calculation
        height_m = self.height / 100.0
        
        if self.gender in ['male', 'm']:
            # Male formula (Formulation 4)
            lbm = 1.1 * self.weight - 128 * (self.weight / (height_m ** 2))
        else:
            # Female formula (Formulation 5)
            lbm = 1.07 * self.weight - 148 * (self.weight / (height_m ** 2))
        
        # Ensure LBM is reasonable (at least 30% of body weight, max = weight)
        lbm = max(lbm, 0.3 * self.weight)
        lbm = min(lbm, self.weight)
        
        return lbm
    
    @property
    def gender_code(self) -> str:
        """Return gender as single letter code ('M' or 'F')."""
        return 'M' if self.gender in ['male', 'm'] else 'F'


class BasePKModel(ABC):
    """
    Abstract base class for pharmacokinetic models.
    
    All PK models should implement this interface for consistency
    across Schnider, Minto, and future models.
    """
    
    @abstractmethod
    def get_state_space_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get state-space representation matrices A and B.
        
        For 3-compartment model: ẋ = Ax + Bu
        where x = [C1, C2, C3] (concentrations)
        
        Returns:
            Tuple of (A, B) matrices:
                - A: 3x3 state transition matrix
                - B: 3x1 input matrix
        """
        pass
    
    @abstractmethod
    def get_extended_state_space_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get extended state-space matrices including effect-site.
        
        For 4-compartment model: ẋ = Ax + Bu
        where x = [C1, C2, C3, Ce] (including effect-site)
        
        Returns:
            Tuple of (A_ext, B_ext) matrices:
                - A_ext: 4x4 extended state transition matrix
                - B_ext: 4x1 extended input matrix
        """
        pass
    
    @abstractmethod
    def get_rate_constants(self) -> Dict[str, float]:
        """
        Get all rate constants as a dictionary.
        
        Returns:
            Dictionary containing k10, k12, k21, k13, k31, ke0
        """
        pass
    
    @abstractmethod
    def get_volumes(self) -> Dict[str, float]:
        """
        Get compartment volumes as a dictionary.
        
        Returns:
            Dictionary containing V1, V2, V3
        """
        pass


def validate_patient_parameters(
    patient_or_age: Union[PatientParameters, float],
    weight: Optional[float] = None,
    height: Optional[float] = None,
    gender: Optional[str] = None
) -> None:
    """
    Validate patient parameters are within reasonable physiological ranges.
    
    Supports two calling conventions:
    1. validate_patient_parameters(patient_obj)
    2. validate_patient_parameters(age, weight, height, gender)
    
    Args:
        patient_or_age: Either a PatientParameters object or age (float)
        weight: Patient weight (kg) - required if first arg is age
        height: Patient height (cm) - required if first arg is age
        gender: Patient gender - required if first arg is age
    
    Raises:
        ValueError: If parameters are out of valid ranges
    """
    # Handle both interfaces
    if isinstance(patient_or_age, PatientParameters):
        patient = patient_or_age
        age = patient.age
        weight = patient.weight
        height = patient.height
        gender = patient.gender
    else:
        age = patient_or_age
        if weight is None or height is None or gender is None:
            raise ValueError("When passing age directly, must also provide weight, height, and gender")
    
    if not (0 <= age <= 120):
        raise ValueError(f"Age must be between 0 and 120 years, got {age}")
    
    if not (30 <= weight <= 200):
        raise ValueError(f"Weight must be between 30 and 200 kg, got {weight}")
    
    if not (100 <= height <= 250):
        raise ValueError(f"Height must be between 100 and 250 cm, got {height}")
    
    if gender.lower() not in ['male', 'female', 'm', 'f']:
        raise ValueError(f"Gender must be 'male'/'M' or 'female'/'F', got {gender}")



def calculate_lean_body_mass(
    weight: float,
    height: float,
    gender: str
) -> float:
    """
    Calculate Lean Body Mass using James formula.
    
    Args:
        weight: Patient weight (kg)
        height: Patient height (cm)
        gender: Patient gender ('male'/'M' or 'female'/'F')
    
    Returns:
        Lean body mass in kg
    """
    height_m = height / 100.0
    
    if gender.lower() in ['male', 'm']:
        lbm = 1.1 * weight - 128 * (weight / (height_m ** 2))
    else:
        lbm = 1.07 * weight - 148 * (weight / (height_m ** 2))
    
    # Ensure reasonable bounds
    lbm = max(lbm, 0.3 * weight)
    lbm = min(lbm, weight)
    
    return lbm
