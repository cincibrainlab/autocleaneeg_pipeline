"""Tests for preprocessing standalone functions.

This module tests all preprocessing functions including filtering, resampling,
referencing, and basic channel operations.
"""

import numpy as np
import pytest

# Import test utilities
from tests.fixtures.synthetic_data import create_synthetic_raw


class TestFiltering:
    """Test filtering function."""

    def test_filter_data_import(self):
        """Test that filter_data can be imported."""
        from autoclean import filter_data
        from autoclean.functions.preprocessing import filter_data as filter_data_direct

        # Both imports should work and be the same function
        assert filter_data is filter_data_direct

    def test_filter_data_basic_functionality(self):
        """Test basic filtering functionality."""
        from autoclean import filter_data

        # Create test data
        raw = create_synthetic_raw(
            n_channels=16, sfreq=250, duration=5, montage="standard_1020"
        )

        # Test highpass filtering
        filtered_raw = filter_data(raw, l_freq=1.0)

        assert filtered_raw is not raw  # Should be a copy
        assert filtered_raw.info["sfreq"] == raw.info["sfreq"]  # Same sampling rate
        assert len(filtered_raw.ch_names) == len(raw.ch_names)  # Same channels

    def test_filter_data_parameter_validation(self):
        """Test parameter validation."""
        from autoclean import filter_data

        raw = create_synthetic_raw(n_channels=4, sfreq=250, duration=2)

        # Test invalid data type
        with pytest.raises(TypeError):
            filter_data("not_mne_data")

        # Test invalid frequencies
        with pytest.raises(ValueError):
            filter_data(raw, l_freq=-1.0)

        with pytest.raises(ValueError):
            filter_data(raw, h_freq=-1.0)

        with pytest.raises(ValueError):
            filter_data(raw, l_freq=40.0, h_freq=30.0)  # l_freq >= h_freq

    def test_filter_data_no_filtering(self):
        """Test that no filtering returns a copy."""
        from autoclean import filter_data

        raw = create_synthetic_raw(n_channels=4, sfreq=250, duration=2)

        # No filtering parameters
        result = filter_data(raw)

        assert result is not raw  # Should be a copy
        assert np.array_equal(result.get_data(), raw.get_data())  # Same data


class TestResampling:
    """Test resampling function."""

    def test_placeholder(self):
        """Placeholder test - will be implemented with resample_data function."""
        # This will be replaced with actual tests when resample_data is implemented
        assert True


class TestReferencing:
    """Test referencing function."""

    def test_placeholder(self):
        """Placeholder test - will be implemented with rereference_data function."""
        # This will be replaced with actual tests when rereference_data is implemented
        assert True


class TestBasicOperations:
    """Test basic operations (drop, crop, trim)."""

    def test_placeholder(self):
        """Placeholder test - will be implemented with basic ops functions."""
        # This will be replaced with actual tests when basic ops are implemented
        assert True
