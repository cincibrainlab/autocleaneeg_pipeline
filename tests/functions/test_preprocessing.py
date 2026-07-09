"""Tests for preprocessing standalone functions.

This module tests all preprocessing functions including filtering, resampling,
referencing, and basic channel operations.
"""

import importlib.util
from pathlib import Path
from typing import List

import mne
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
_compute_psd_metrics = wavelet_module._compute_psd_metrics


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

    def test_resample_data_import(self):
        """Test that resample_data can be imported."""
        from autoclean import resample_data
        from autoclean.functions.preprocessing import resample_data as resample_direct

        # Both imports should work and be the same function
        assert resample_data is resample_direct

    def test_resample_data_basic_functionality(self):
        """Test basic resampling functionality."""
        from autoclean import resample_data

        raw = create_synthetic_raw(n_channels=4, sfreq=250, duration=2)
        resampled = resample_data(raw, sfreq=125)

        assert resampled.info["sfreq"] == 125
        assert len(resampled.ch_names) == len(raw.ch_names)

    def test_resample_data_with_epochsarray(self):
        """Test resampling with EpochsArray (BaseEpochs subclass)."""
        from autoclean import resample_data

        # Create EpochsArray
        n_channels, n_samples, n_epochs = 4, 256, 10
        data = np.random.randn(n_epochs, n_channels, n_samples)
        info = mne.create_info(n_channels, 256, "eeg")
        epochs = mne.EpochsArray(data, info)

        # Should accept BaseEpochs subclass
        resampled = resample_data(epochs, sfreq=128)

        assert isinstance(resampled, mne.BaseEpochs)
        assert resampled.info["sfreq"] == 128


class TestReferencing:
    """Test referencing function."""

    def test_placeholder(self):
        """Placeholder test - will be implemented with rereference_data function."""
        # This will be replaced with actual tests when rereference_data is implemented
        assert True

    def test_rereference_data_import(self):
        """Test that rereference_data can be imported."""
        from autoclean import rereference_data
        from autoclean.functions.preprocessing import (
            rereference_data as rereference_direct,
        )

        assert rereference_data is rereference_direct

    def test_rereference_data_basic_functionality(self):
        """Test basic rereferencing functionality."""
        from autoclean import rereference_data

        raw = create_synthetic_raw(n_channels=4, sfreq=250, duration=2)
        rereferenced = rereference_data(raw, ref_channels="average")

        assert rereferenced is not raw
        assert len(rereferenced.ch_names) == len(raw.ch_names)

    def test_rereference_data_with_epochsarray(self):
        """Test rereferencing with EpochsArray (BaseEpochs subclass)."""
        from autoclean import rereference_data

        # Create EpochsArray
        n_channels, n_samples, n_epochs = 4, 256, 10
        data = np.random.randn(n_epochs, n_channels, n_samples)
        info = mne.create_info(n_channels, 256, "eeg")
        epochs = mne.EpochsArray(data, info)

        # Should accept BaseEpochs subclass
        rereferenced = rereference_data(epochs, ref_channels="average")

        assert isinstance(rereferenced, mne.BaseEpochs)


class TestBasicOperations:
    """Test basic operations (drop, crop, trim)."""

    def test_placeholder(self):
        """Placeholder test - will be implemented with basic ops functions."""
        # This will be replaced with actual tests when basic ops are implemented
        assert True

    def test_drop_channels_import(self):
        """Test that drop_channels can be imported."""
        from autoclean.functions.preprocessing.basic_ops import drop_channels

        assert callable(drop_channels)

    def test_drop_channels_basic_functionality(self):
        """Test basic channel dropping functionality."""
        from autoclean.functions.preprocessing.basic_ops import drop_channels

        raw = create_synthetic_raw(
            n_channels=4, sfreq=250, duration=2, montage="standard_1020"
        )
        result = drop_channels(raw, raw.ch_names[0])  # Use actual channel name

        assert result.info["nchan"] == 3
        assert raw.ch_names[0] not in result.ch_names

    def test_drop_channels_with_epochsarray(self):
        """Test drop_channels with EpochsArray (BaseEpochs subclass)."""
        from autoclean.functions.preprocessing.basic_ops import drop_channels

        # Create EpochsArray with proper channel names
        n_channels, n_samples, n_epochs = 4, 256, 10
        data = np.random.randn(n_epochs, n_channels, n_samples)
        ch_names = ["Fp1", "Fp2", "Fz", "Cz"]
        info = mne.create_info(ch_names, 256, "eeg")
        epochs = mne.EpochsArray(data, info)

        # Should accept BaseEpochs subclass
        result = drop_channels(epochs, "Fp1")

        assert isinstance(result, mne.BaseEpochs)
        assert result.info["nchan"] == 3

    def test_crop_data_with_epochsarray(self):
        """Test crop_data with EpochsArray (BaseEpochs subclass)."""
        from autoclean.functions.preprocessing.basic_ops import crop_data

        # Create EpochsArray
        n_channels, n_samples, n_epochs = 4, 256, 10
        data = np.random.randn(n_epochs, n_channels, n_samples)
        info = mne.create_info(n_channels, 256, "eeg")
        epochs = mne.EpochsArray(data, info)

        # Should accept BaseEpochs subclass
        result = crop_data(epochs, tmin=0, tmax=0.5)

        assert isinstance(result, mne.BaseEpochs)

    def test_trim_edges_with_epochsarray(self):
        """Test trim_edges with EpochsArray (BaseEpochs subclass)."""
        from autoclean.functions.preprocessing.basic_ops import trim_edges

        # Create EpochsArray
        n_channels, n_samples, n_epochs = 4, 256, 10
        data = np.random.randn(n_epochs, n_channels, n_samples)
        info = mne.create_info(n_channels, 256, "eeg")
        epochs = mne.EpochsArray(data, info)

        # Should accept BaseEpochs subclass
        result = trim_edges(epochs, duration=0.01)

        assert isinstance(result, mne.BaseEpochs)

    def test_assign_channel_types_with_epochsarray(self):
        """Test assign_channel_types with EpochsArray (BaseEpochs subclass)."""
        from autoclean.functions.preprocessing.basic_ops import assign_channel_types

        # Create EpochsArray with proper channel names
        n_channels, n_samples, n_epochs = 4, 256, 10
        data = np.random.randn(n_epochs, n_channels, n_samples)
        ch_names = ["Fp1", "Fp2", "Fz", "Cz"]
        info = mne.create_info(ch_names, 256, "eeg")
        epochs = mne.EpochsArray(data, info)

        # Should accept BaseEpochs subclass
        types = {"Fp1": "eeg", "Fp2": "eeg"}
        result = assign_channel_types(epochs, types)

        assert isinstance(result, mne.BaseEpochs)


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

    def test_wavelet_threshold_threshold_scale(self):
        """Scaling the threshold should modulate artifact attenuation."""

        raw = create_synthetic_raw(n_channels=1, sfreq=250, duration=1)
        artifact = raw.copy()
        idx = 150
        artifact._data[0, idx] += 3.0

        cleaned_low = wavelet_threshold(artifact, threshold_scale=0.5)
        cleaned_high = wavelet_threshold(artifact, threshold_scale=2.0)

        low_value = np.abs(cleaned_low.get_data()[0, idx])
        high_value = np.abs(cleaned_high.get_data()[0, idx])
        assert high_value <= low_value

    def test_wavelet_threshold_invalid_mode_raises(self):
        """Unsupported threshold modes should raise a helpful error."""

        raw = create_synthetic_raw(n_channels=1, sfreq=250, duration=1)

        with pytest.raises(ValueError):
            wavelet_threshold(raw, threshold_mode="invalid")

    def test_wavelet_threshold_auto_level(self):
        """Auto level should match the maximum safe decomposition level."""

        raw = create_synthetic_raw(n_channels=1, sfreq=250, duration=1)
        auto_cleaned = wavelet_threshold(raw, level="auto")

        max_level = _resolve_decomposition_level(raw.n_times, "sym4", raw.n_times)
        explicit_cleaned = wavelet_threshold(raw, level=max_level)

        assert np.allclose(auto_cleaned.get_data(), explicit_cleaned.get_data())

    def test_wavelet_threshold_picks_subset(self):
        """Channel picks should confine denoising to selected channels."""

        raw = create_synthetic_raw(
            montage="standard_1020", n_channels=4, sfreq=250, duration=1
        )
        artifact = raw.copy()
        artifact._data[0, 80] += 4.0

        cleaned = wavelet_threshold(artifact, picks=[artifact.ch_names[0]])

        assert not np.allclose(cleaned.get_data()[0], artifact.get_data()[0])
        assert np.allclose(cleaned.get_data()[1], artifact.get_data()[1])

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

    def test_wavelet_psd_metrics_honour_ceiling(self, monkeypatch):
        """Wavelet PSD metrics should respect the configured frequency ceiling."""

        captured_fmax: List[float] = []

        def _fake_psd(data, sfreq, **kwargs):
            captured_fmax.append(kwargs.get("fmax", -1))
            freqs = np.linspace(1.0, kwargs.get("fmax", 10.0), 8)
            if freqs[0] >= freqs[-1]:
                freqs = np.linspace(1.0, 10.0, 8)
            return np.ones((data.shape[0], freqs.size)), freqs

        monkeypatch.setattr(wavelet_module, "psd_array_welch", _fake_psd)

        baseline = np.random.randn(2, 500)
        cleaned = baseline.copy()

        _compute_psd_metrics(
            baseline,
            cleaned,
            sfreq=200.0,
            ch_names=["Fz", "Cz"],
            psd_fmax=25.0,
        )

        assert captured_fmax
        assert all(fmax <= 25.0 for fmax in captured_fmax)

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
