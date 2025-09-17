"""Tests for preprocessing standalone functions.

This module tests all preprocessing functions including filtering, resampling,
referencing, and basic channel operations.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import pywt

# Import test utilities
from tests.fixtures.synthetic_data import create_synthetic_raw

# Load wavelet module directly to avoid package side effects during testing
_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "autoclean"
    / "functions"
    / "preprocessing"
    / "wavelet_thresholding.py"
)
_SPEC = importlib.util.spec_from_file_location("wavelet_thresholding", _MODULE_PATH)
wavelet_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(wavelet_module)

wavelet_threshold = wavelet_module.wavelet_threshold
_resolve_decomposition_level = wavelet_module._resolve_decomposition_level


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


class TestWaveletThreshold:
    """Test wavelet thresholding function."""

    def test_wavelet_threshold_basic(self):
        """Wavelet thresholding should reduce artifact amplitude."""

        raw = create_synthetic_raw(n_channels=1, sfreq=250, duration=1)
        raw_artifact = raw.copy()
        raw_artifact._data[0, 100] += 1.0  # inject transient artifact

        cleaned = wavelet_threshold(raw_artifact)

        assert abs(cleaned.get_data()[0, 100]) < abs(raw_artifact.get_data()[0, 100])

    def test_wavelet_threshold_clamps_level_for_short_segments(self):
        """Short recordings should clamp the decomposition level safely."""

        raw = create_synthetic_raw(n_channels=2, sfreq=250, duration=0.2)
        cleaned = wavelet_threshold(raw, level=10)

        assert cleaned.get_data().shape == raw.get_data().shape

        max_level = _resolve_decomposition_level(raw.n_times, "sym4", 10)
        assert max_level <= 10
        if max_level == 0:
            assert np.allclose(cleaned.get_data(), raw.get_data())

    def test_wavelet_threshold_supports_hard_mode(self):
        """Hard thresholding should preserve more of a large transient."""

        raw = create_synthetic_raw(n_channels=1, sfreq=250, duration=1)
        raw_artifact = raw.copy()
        spike_index = 125
        raw_artifact._data[0, spike_index] += 2.5

        cleaned_soft = wavelet_threshold(raw_artifact, threshold_mode="soft")
        cleaned_hard = wavelet_threshold(raw_artifact, threshold_mode="hard")

        assert not np.allclose(cleaned_soft.get_data(), cleaned_hard.get_data())
        soft_value = np.abs(cleaned_soft.get_data()[0, spike_index])
        hard_value = np.abs(cleaned_hard.get_data()[0, spike_index])
        assert hard_value >= soft_value

    def test_wavelet_threshold_invalid_mode_raises(self):
        """Unsupported threshold modes should raise a helpful error."""

        raw = create_synthetic_raw(n_channels=1, sfreq=250, duration=1)

        with pytest.raises(ValueError):
            wavelet_threshold(raw, threshold_mode="invalid")

    def test_wavelet_threshold_erp_mode_matches_single_filter_when_clean(self):
        """ERP mode should reduce to a single filter when no artifact is present."""

        raw = create_synthetic_raw(n_channels=1, sfreq=250, duration=10)
        raw._data[:] = 0.0

        erp_cleaned = wavelet_threshold(
            raw, is_erp=True, bandpass=(1.0, 30.0)
        ).get_data()
        expected = raw.copy().filter(l_freq=1.0, h_freq=30.0, verbose=False).get_data()

        assert np.allclose(erp_cleaned, expected, atol=1e-12)

    def test_wavelet_threshold_erp_mode_bandpass_validation(self):
        """ERP mode should validate the supplied band-pass tuple."""

        raw = create_synthetic_raw(n_channels=1, sfreq=250, duration=1)

        with pytest.raises(ValueError):
            wavelet_threshold(raw, is_erp=True, bandpass=(30.0, 1.0))

        with pytest.raises(ValueError):
            wavelet_threshold(raw, is_erp=True, bandpass=None)

    def test_resolve_decomposition_level_matches_pywt(self):
        """Helper should agree with PyWavelets max level calculation."""

        wavelet = "sym4"
        wavelet_obj = pywt.Wavelet(wavelet)

        data_len = 100
        requested = 8
        expected_max = pywt.dwt_max_level(data_len, wavelet_obj.dec_len)
        resolved = _resolve_decomposition_level(data_len, wavelet, requested)

        assert resolved == min(requested, expected_max)
        assert resolved <= requested

        very_short = 5
        assert _resolve_decomposition_level(very_short, wavelet, 5) == 0
