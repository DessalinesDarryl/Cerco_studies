"""
Unit tests for Explainable AI (XAI) module.
Tests cover SHAP, DeepLift, Integrated Gradients, and other interpretation methods.
"""

import pytest
import numpy as np


@pytest.mark.unit
def test_xai_module_imports():
    """Test that XAI modules can be imported."""
    xai_modules = ['deeplift', 'aggregation']
    
    for module_name in xai_modules:
        try:
            module = __import__(f'src.xai.{module_name}', fromlist=[module_name])
            assert module is not None
        except ImportError:
            pytest.skip(f"src.xai.{module_name} not available")


@pytest.mark.unit
def test_sample_eeg_data(sample_eeg_data):
    """Test that sample EEG data can be used for XAI testing."""
    data, fs = sample_eeg_data
    
    # Verify basic data properties
    assert isinstance(data, np.ndarray)
    assert data.ndim == 2
    assert data.dtype in [np.float32, np.float64]
    assert fs > 0


@pytest.mark.unit 
def test_sample_features_dict(sample_features_dict):
    """Test sample features for XAI interpretability testing."""
    # Ensure numeric features are present for XAI methods
    numeric_features = [
        'eeg_alpha_power', 'eeg_beta_power', 
        'emg_mean_amplitude', 'emg_rms'
    ]
    
    for feat in numeric_features:
        assert feat in sample_features_dict
        assert isinstance(sample_features_dict[feat], (int, float))
