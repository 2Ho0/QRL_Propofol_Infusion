"""
Test Configuration and Fixtures
================================

Provides pytest configuration and shared fixtures for testing.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def random_seed():
    """Set random seeds for reproducibility."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed


@pytest.fixture
def sample_patient_params():
    """Sample patient parameters for testing."""
    return {
        'age': 45,
        'weight': 70,
        'height': 170,
        'gender': 'M'
    }


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'environment': {
            'bis_target': 50,
            'dt': 5.0,
            'max_steps': 720
        },
        'quantum': {
            'n_qubits': 2,
            'n_layers': 3
        },
        'ddpg': {
            'learning_rate_actor': 0.0003,
            'learning_rate_critic': 0.001,
            'gamma': 0.99
        }
    }


@pytest.fixture
def temp_log_dir(tmp_path):
    """Temporary directory for logs."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir
