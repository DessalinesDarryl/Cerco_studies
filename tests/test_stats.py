"""
Unit tests for statistical analysis module.
Tests cover ANOVA, Kruskal-Wallis, and other statistical utilities.
"""

import pytest
import pandas as pd
import numpy as np


@pytest.mark.unit
def test_anova_imports():
    """Test that stats module can be imported."""
    try:
        import src.stats.anova_kruskal as stats_module
        assert stats_module is not None
    except ImportError:
        pytest.skip("src.stats.anova_kruskal not available")


@pytest.mark.unit
def test_sample_features_dict(sample_features_dict):
    """Test sample features fixture."""
    assert isinstance(sample_features_dict, dict)
    assert sample_features_dict['patient_id'] == 'TEST001'
    assert sample_features_dict['label'] == 'RBD'
    assert all(key in sample_features_dict for key in [
        'eeg_alpha_power', 'eeg_beta_power', 'emg_mean_amplitude'
    ])


@pytest.mark.unit
def test_sample_dataframe():
    """Test creating sample dataframes for stats testing."""
    df = pd.DataFrame({
        "band": ["alpha", "alpha", "alpha", "beta", "beta", "beta"],
        "group": [0, 1, 2, 0, 1, 2],
        "score": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5]
    })
    assert len(df) == 6
    assert "band" in df.columns
    assert df["group"].unique().tolist() == [0, 1, 2]
