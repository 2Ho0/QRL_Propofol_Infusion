"""
Custom Exceptions for Quantum RL Propofol Infusion
===================================================

Defines custom exception classes for better error handling and debugging
throughout the project.

Exception Hierarchy:
--------------------
QRLPropofolError (base)
├── ModelError
│   ├── PKPDModelError
│   ├── QuantumCircuitError
│   └── EncoderError
├── EnvironmentError
│   ├── InvalidStateError
│   ├── InvalidActionError
│   └── SimulationError
├── DataError
│   ├── VitalDBError
│   ├── DataQualityError
│   └── DataLoadError
├── TrainingError
│   ├── ConvergenceError
│   ├── CheckpointError
│   └── ConfigurationError
└── AgentError
    ├── PolicyError
    └── ValueEstimationError

Usage:
------
    from src.utils.exceptions import PKPDModelError, InvalidStateError
    
    if age < 0 or age > 120:
        raise PKPDModelError(f"Invalid age: {age}. Must be between 0-120 years.")
    
    if not (0 <= bis <= 100):
        raise InvalidStateError(f"BIS out of valid range: {bis}")
"""


class QRLPropofolError(Exception):
    """Base exception for all QRL Propofol project errors."""
    pass


# ============================================================================
# Model Errors
# ============================================================================

class ModelError(QRLPropofolError):
    """Base exception for model-related errors."""
    pass


class PKPDModelError(ModelError):
    """
    Exception raised for pharmacokinetic/pharmacodynamic model errors.
    
    Examples:
        - Invalid patient parameters
        - Numerical instability in PK/PD simulation
        - Concentration out of valid range
    """
    pass


class QuantumCircuitError(ModelError):
    """
    Exception raised for quantum circuit errors.
    
    Examples:
        - Invalid number of qubits
        - Circuit compilation failure
        - Measurement error
    """
    pass


class EncoderError(ModelError):
    """
    Exception raised for encoder network errors.
    
    Examples:
        - Invalid sequence length
        - LSTM/Transformer initialization failure
        - Encoding dimension mismatch
    """
    pass


# ============================================================================
# Environment Errors
# ============================================================================

class EnvironmentError(QRLPropofolError):
    """Base exception for environment-related errors."""
    pass


class InvalidStateError(EnvironmentError):
    """
    Exception raised when environment state is invalid.
    
    Examples:
        - BIS out of valid range (0-100)
        - Negative concentrations
        - NaN or Inf values in state
    """
    pass


class InvalidActionError(EnvironmentError):
    """
    Exception raised when action is invalid.
    
    Examples:
        - Action out of bounds
        - NaN or Inf action values
        - Wrong action dimension
    """
    pass


class SimulationError(EnvironmentError):
    """
    Exception raised when simulation fails.
    
    Examples:
        - PK/PD integration divergence
        - Patient model instability
        - Time step too large
    """
    pass


# ============================================================================
# Data Errors
# ============================================================================

class DataError(QRLPropofolError):
    """Base exception for data-related errors."""
    pass


class VitalDBError(DataError):
    """
    Exception raised for VitalDB data access errors.
    
    Examples:
        - Case not found
        - Download failure
        - Corrupted data
    """
    pass


class DataQualityError(DataError):
    """
    Exception raised when data quality is insufficient.
    
    Examples:
        - Too much missing data
        - Signal quality too low
        - Duration out of valid range
    """
    pass


class DataLoadError(DataError):
    """
    Exception raised when data loading fails.
    
    Examples:
        - File not found
        - Incorrect format
        - Deserialization error
    """
    pass


# ============================================================================
# Training Errors
# ============================================================================

class TrainingError(QRLPropofolError):
    """Base exception for training-related errors."""
    pass


class ConvergenceError(TrainingError):
    """
    Exception raised when training fails to converge.
    
    Examples:
        - Loss becomes NaN or Inf
        - No improvement for too many episodes
        - Gradient explosion
    """
    pass


class CheckpointError(TrainingError):
    """
    Exception raised for checkpoint save/load errors.
    
    Examples:
        - Cannot save checkpoint
        - Checkpoint file corrupted
        - Model architecture mismatch
    """
    pass


class ConfigurationError(TrainingError):
    """
    Exception raised for configuration errors.
    
    Examples:
        - Missing required configuration
        - Invalid hyperparameter values
        - Conflicting settings
    """
    pass


# ============================================================================
# Agent Errors
# ============================================================================

class AgentError(QRLPropofolError):
    """Base exception for agent-related errors."""
    pass


class PolicyError(AgentError):
    """
    Exception raised for policy network errors.
    
    Examples:
        - Policy output out of bounds
        - NaN policy values
        - Policy network forward pass failure
    """
    pass


class ValueEstimationError(AgentError):
    """
    Exception raised for value estimation errors.
    
    Examples:
        - Q-value becomes Inf or NaN
        - Value network failure
        - TD error explosion
    """
    pass


# ============================================================================
# Helper Functions
# ============================================================================

def validate_patient_parameters(
    age: float,
    weight: float,
    height: float,
    gender: str
) -> None:
    """
    Validate patient parameters for PK/PD models.
    
    Args:
        age: Patient age (years)
        weight: Patient weight (kg)
        height: Patient height (cm)
        gender: Patient gender ('M' or 'F')
    
    Raises:
        PKPDModelError: If any parameter is invalid
    """
    if not (0 < age <= 120):
        raise PKPDModelError(
            f"Invalid age: {age}. Must be between 0 and 120 years."
        )
    
    if not (30 <= weight <= 200):
        raise PKPDModelError(
            f"Invalid weight: {weight}. Must be between 30 and 200 kg."
        )
    
    if not (100 <= height <= 250):
        raise PKPDModelError(
            f"Invalid height: {height}. Must be between 100 and 250 cm."
        )
    
    if gender not in ['M', 'F', 'male', 'female']:
        raise PKPDModelError(
            f"Invalid gender: {gender}. Must be 'M', 'F', 'male', or 'female'."
        )


def validate_bis(bis: float) -> None:
    """
    Validate BIS value.
    
    Args:
        bis: Bispectral index value
    
    Raises:
        InvalidStateError: If BIS is invalid
    """
    import numpy as np
    
    if np.isnan(bis) or np.isinf(bis):
        raise InvalidStateError(f"BIS is NaN or Inf: {bis}")
    
    if not (0 <= bis <= 100):
        raise InvalidStateError(
            f"BIS out of valid range [0, 100]: {bis}"
        )


def validate_concentration(
    concentration: float,
    name: str = "concentration",
    min_value: float = 0.0,
    max_value: float = 100.0
) -> None:
    """
    Validate drug concentration.
    
    Args:
        concentration: Concentration value
        name: Name of the concentration (for error message)
        min_value: Minimum valid value
        max_value: Maximum valid value
    
    Raises:
        InvalidStateError: If concentration is invalid
    """
    import numpy as np
    
    if np.isnan(concentration) or np.isinf(concentration):
        raise InvalidStateError(f"{name} is NaN or Inf: {concentration}")
    
    if not (min_value <= concentration <= max_value):
        raise InvalidStateError(
            f"{name} out of valid range [{min_value}, {max_value}]: {concentration}"
        )


def validate_action(action: float, min_action: float, max_action: float) -> None:
    """
    Validate agent action.
    
    Args:
        action: Action value
        min_action: Minimum valid action
        max_action: Maximum valid action
    
    Raises:
        InvalidActionError: If action is invalid
    """
    import numpy as np
    
    if np.isnan(action) or np.isinf(action):
        raise InvalidActionError(f"Action is NaN or Inf: {action}")
    
    if not (min_action <= action <= max_action):
        raise InvalidActionError(
            f"Action out of bounds [{min_action}, {max_action}]: {action}"
        )


def check_convergence(
    loss: float,
    loss_history: list,
    patience: int = 50,
    min_delta: float = 1e-6
) -> None:
    """
    Check if training has converged or diverged.
    
    Args:
        loss: Current loss value
        loss_history: History of recent loss values
        patience: Number of episodes without improvement
        min_delta: Minimum change to be considered improvement
    
    Raises:
        ConvergenceError: If loss is NaN/Inf or no improvement for too long
    """
    import numpy as np
    
    # Check for NaN or Inf
    if np.isnan(loss) or np.isinf(loss):
        raise ConvergenceError(f"Loss became NaN or Inf: {loss}")
    
    # Check for no improvement
    if len(loss_history) >= patience:
        recent_best = min(loss_history[-patience:])
        if loss > recent_best - min_delta:
            raise ConvergenceError(
                f"No improvement for {patience} episodes. "
                f"Current loss: {loss:.6f}, Best recent: {recent_best:.6f}"
            )


# Example usage and testing
if __name__ == "__main__":
    print("Testing custom exceptions...")
    
    # Test patient validation
    try:
        validate_patient_parameters(age=150, weight=70, height=170, gender='M')
    except PKPDModelError as e:
        print(f"✓ Caught expected error: {e}")
    
    # Test BIS validation
    try:
        validate_bis(150.0)
    except InvalidStateError as e:
        print(f"✓ Caught expected error: {e}")
    
    # Test action validation
    try:
        validate_action(action=300.0, min_action=0.0, max_action=200.0)
    except InvalidActionError as e:
        print(f"✓ Caught expected error: {e}")
    
    # Test convergence check
    try:
        import numpy as np
        check_convergence(loss=np.nan, loss_history=[1.0, 0.9, 0.8])
    except ConvergenceError as e:
        print(f"✓ Caught expected error: {e}")
    
    print("\nAll exception tests passed!")
