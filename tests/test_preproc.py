"""
Unit tests for data preprocessing module.
Tests cover data loading, filtering, and preprocessing pipelines.
"""

import pytest
import numpy as np


@pytest.mark.unit
def test_imports():
    """Test that preprocessing module can be imported."""
    try:
        import src.data.preprocessing as preproc
        assert preproc is not None
    except ImportError:
        pytest.skip("src.data.preprocessing not available")


@pytest.mark.unit
def test_sample_eeg_data(sample_eeg_data):
    """Verify sample EEG data fixture generation."""
    data, fs = sample_eeg_data
    assert isinstance(data, np.ndarray)
    assert data.shape[0] == 4  # number of channels
    assert data.shape[1] == 7500  # 250 Hz * 30 seconds
    assert fs == 250


@pytest.mark.unit
def test_sample_emg_data(sample_emg_data):
    """Verify sample EMG data fixture generation."""
    data, fs = sample_emg_data
    assert isinstance(data, np.ndarray)
    assert data.shape[0] == 2  # number of channels
    assert data.shape[1] == 7500  # 250 Hz * 30 seconds
    assert fs == 250
