"""
Environment module for Propofol Infusion Control
=================================================

Contains:
    - PatientSimulator: Integrated PK/PD simulator for propofol+remifentanil
    - PropofolEnv: Gymnasium-compatible environment for RL training
    - DualDrugEnv: Dual drug (propofol+remifentanil) environment

Note: PK/PD models (SchniderModel, MintoModel) are now in models.pharmacokinetics
"""

from .patient_simulator import PatientSimulator
from .propofol_env import PropofolEnv
from .dual_drug_env import DualDrugEnv, create_patient_parameters

# Re-export from pharmacokinetics module for backward compatibility
from models.pharmacokinetics import (
    PatientParameters,
    SchniderModel,
    SchniderParameters,
    MintoModel,
    MintoParameters
)

__all__ = [
    "PatientSimulator",
    "PatientParameters",
    "SchniderParameters",  # Updated name from SchniderModelParameters
    "SchniderModel",
    "MintoParameters",  # Updated name from MintoModelParameters
    "MintoModel",
    "PropofolEnv",
    "DualDrugEnv",
    "create_patient_parameters",
]
