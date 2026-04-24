"""
Shared pytest fixtures and configuration for all test modules.
"""

import pytest
import numpy as np
from pathlib import Path


# ============================================================================
# Fixtures - Sample Data
# ============================================================================

@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def data_dir(project_root):
    """Return the data directory path."""
    data_path = project_root / "data"
    data_path.mkdir(exist_ok=True)
    return data_path


@pytest.fixture(scope="session")
def sample_eeg_data():
    """Generate sample EEG data for testing."""
    fs = 250  # sampling frequency (Hz)
    duration = 30  # seconds
    n_channels = 4
    n_samples = fs * duration
    
    # Generate synthetic EEG (realistic noise pattern)
    data = np.random.randn(n_channels, n_samples) * 50  # in microvolts
    return data, fs


@pytest.fixture(scope="session")
def sample_emg_data():
    """Generate sample EMG data for testing."""
    fs = 250  # sampling frequency (Hz)
    duration = 30  # seconds
    n_channels = 2
    n_samples = fs * duration
    
    # Generate synthetic EMG (higher amplitude)
    data = np.random.randn(n_channels, n_samples) * 200  # in microvolts
    return data, fs


@pytest.fixture
def sample_features_dict():
    """Sample feature dictionary for testing."""
    return {
        'patient_id': 'TEST001',
        'eeg_alpha_power': 2.5,
        'eeg_beta_power': 1.2,
        'emg_mean_amplitude': 150.0,
        'emg_rms': 180.0,
        'label': 'RBD',
    }


# ============================================================================
# Fixtures - Temporary Paths
# ============================================================================

@pytest.fixture
def tmp_config_yaml(tmp_path):
    """Create a temporary YAML config file for testing."""
    config = tmp_path / "test_config.yaml"
    config.write_text("""
raw_root: /tmp/raw
out_root: /tmp/processed
eeg:
  l_freq: 0.5
  h_freq: 80.0
""")
    return config


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
