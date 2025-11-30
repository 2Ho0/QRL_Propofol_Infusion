"""
Environment module for Propofol Infusion Control
=================================================

Contains:
    - PatientSimulator: PK/PD model for propofol pharmacokinetics/pharmacodynamics
    - PropofolEnv: Gymnasium-compatible environment for RL training
"""

from .patient_simulator import PatientSimulator, SchniderModel
from .propofol_env import PropofolEnv

__all__ = ["PatientSimulator", "SchniderModel", "PropofolEnv"]
