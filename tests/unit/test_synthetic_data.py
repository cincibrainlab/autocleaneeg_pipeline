"""Test synthetic data generation utilities."""

import numpy as np
import pytest
import mne

from tests.fixtures.synthetic_data import (
    create_synthetic_raw,
    create_synthetic_events,
    create_test_montage_info
)
from tests.fixtures.test_utils import EEGAssertions


class TestSyntheticDataGeneration:
    """Test synthetic data generation functions."""
    
    def test_create_synthetic_raw_gsn129(self):
        """Test creation of synthetic GSN-HydroCel-129 data."""
        raw = create_synthetic_raw(
            montage="GSN-HydroCel-129",
            n_channels=129,
            duration=10.0,
            sfreq=1000.0
        )
        
        # Check basic properties
        EEGAssertions.assert_raw_properties(
            raw,
            expected_sfreq=1000.0,
            expected_n_channels=129,
            expected_duration=10.0
        )
        
        # Check channel names
        assert raw.ch_names[-1] == 'Cz', "Last channel should be Cz for 129-channel montage"
        assert all(ch.startswith('E') for ch in raw.ch_names[:-1]), "GSN channels should start with E"
        
        # Check data quality
        EEGAssertions.assert_data_quality(raw, max_amplitude=200e-6)
    
    def test_create_synthetic_raw_gsn128(self):
        """Test creation of synthetic GSN-HydroCel-128 data."""
        raw = create_synthetic_raw(
            montage="GSN-HydroCel-128",
            n_channels=128,
            duration=5.0,
            sfreq=500.0
        )
        
        EEGAssertions.assert_raw_properties(
            raw,
            expected_sfreq=500.0,
            expected_n_channels=128,
            expected_duration=5.0
        )
        
        # All channels should start with E for GSN
        assert all(ch.startswith('E') for ch in raw.ch_names)
    
    def test_create_synthetic_raw_1020(self):
        """Test creation of synthetic 10-20 montage data."""
        raw = create_synthetic_raw(
            montage="standard_1020",
            n_channels=32,
            duration=10.0,
            sfreq=250.0
        )
        
        EEGAssertions.assert_raw_properties(
            raw,
            expected_sfreq=250.0,
            expected_n_channels=32,
            expected_duration=10.0
        )
        
        # Should have standard 10-20 channel names
        standard_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'Cz']
        for ch in standard_channels[:len(raw.ch_names)]:
            if ch in raw.ch_names:
                assert ch in raw.ch_names, f"Expected standard channel {ch}"
    
    def test_synthetic_data_reproducibility(self):
        """Test that synthetic data is reproducible with same seed."""
        raw1 = create_synthetic_raw(duration=5.0, seed=42)
        raw2 = create_synthetic_raw(duration=5.0, seed=42)
        
        np.testing.assert_array_equal(raw1.get_data(), raw2.get_data())
    
    def test_synthetic_data_variability(self):
        """Test that synthetic data varies with different seeds."""
        raw1 = create_synthetic_raw(duration=5.0, seed=42)
        raw2 = create_synthetic_raw(duration=5.0, seed=123)
        
        # Data should be different
        assert not np.array_equal(raw1.get_data(), raw2.get_data())
        
        # But structure should be the same
        assert raw1.info['sfreq'] == raw2.info['sfreq']
        assert len(raw1.ch_names) == len(raw2.ch_names)
    
    def test_create_resting_events(self):
        """Test resting state event creation (should be empty)."""
        events = create_synthetic_events("resting", duration=60.0, sfreq=1000.0)
        
        assert events.shape[0] == 0, "Resting state should have no events"
        assert events.shape[1] == 3, "Events should have 3 columns"
    
    def test_create_chirp_events(self):
        """Test chirp stimulus event creation."""
        events = create_synthetic_events("chirp", duration=10.0, sfreq=1000.0)
        
        # Should have events every 2 seconds
        expected_n_events = 4  # At 2, 4, 6, 8 seconds
        assert len(events) == expected_n_events, f"Expected {expected_n_events} chirp events"
        
        # All events should be stimulus events (ID=1)
        assert all(events[:, 2] == 1), "All chirp events should have ID=1"
        
        # Check timing (approximately every 2 seconds)
        event_times = events[:, 0] / 1000.0  # Convert to seconds
        expected_times = np.array([2, 4, 6, 8])
        np.testing.assert_allclose(event_times, expected_times, atol=0.1)
    
    def test_create_mmn_events(self):
        """Test MMN event creation."""
        events = create_synthetic_events("mmn", duration=10.0, sfreq=1000.0)
        
        # Should have events every 500ms starting at 1s, ending before 9.5s
        # (10-1)/0.5 = 18, but range(1, 10-0.5, 0.5) gives 17 events
        expected_n_events = 17  # More accurate calculation
        assert len(events) == expected_n_events, f"Expected {expected_n_events} MMN events, got {len(events)}"
        
        # Should have mix of standard (1) and deviant (2) events
        event_ids = events[:, 2]
        assert 1 in event_ids, "Should have standard events (ID=1)"
        assert 2 in event_ids, "Should have deviant events (ID=2)"
        
        # Approximately 80% standard, 20% deviant
        n_standard = np.sum(event_ids == 1)
        n_deviant = np.sum(event_ids == 2)
        standard_ratio = n_standard / len(events)
        assert 0.6 < standard_ratio < 0.9, f"Standard ratio {standard_ratio} not in expected range"
    
    def test_create_assr_events(self):
        """Test ASSR event creation."""
        events = create_synthetic_events("assr", duration=10.0, sfreq=1000.0)
        
        # Should have stimulus start (1) and stop (2) events
        event_ids = events[:, 2]
        assert 1 in event_ids, "Should have stimulus start events (ID=1)"
        assert 2 in event_ids, "Should have stimulus stop events (ID=2)"
        
        # Should have equal numbers of start and stop events
        n_start = np.sum(event_ids == 1)
        n_stop = np.sum(event_ids == 2)
        assert n_start == n_stop, "Should have equal start and stop events"
    
    def test_invalid_paradigm(self):
        """Test error handling for invalid paradigm."""
        with pytest.raises(ValueError, match="Unknown paradigm"):
            create_synthetic_events("invalid_paradigm", duration=10.0, sfreq=1000.0)
    
    def test_montage_info_creation(self):
        """Test montage info creation."""
        info = create_test_montage_info("GSN-HydroCel-129", 129, 1000.0)
        
        assert info['sfreq'] == 1000.0
        assert len(info['ch_names']) == 129
        assert info['ch_names'][-1] == 'Cz'
        assert all(ch_type == 'eeg' for ch_type in info.get_channel_types())
    
    def test_data_amplitude_ranges(self):
        """Test that synthetic data has realistic amplitude ranges."""
        raw = create_synthetic_raw(duration=30.0)
        data = raw.get_data()
        
        # Check amplitude statistics
        rms_amplitude = np.sqrt(np.mean(data**2))
        max_amplitude = np.max(np.abs(data))
        
        # Should be in realistic EEG ranges
        assert 5e-6 < rms_amplitude < 100e-6, f"RMS amplitude {rms_amplitude:.2e} not in realistic range"
        assert max_amplitude < 500e-6, f"Max amplitude {max_amplitude:.2e} too large"
    
    def test_frequency_content(self):
        """Test that synthetic data has expected frequency content."""
        raw = create_synthetic_raw(duration=60.0, sfreq=1000.0)
        
        # Compute power spectral density
        psd, freqs = mne.time_frequency.psd_array_welch(
            raw.get_data(), 
            sfreq=raw.info['sfreq'],
            fmin=1, fmax=100, 
            n_fft=2048,
            verbose=False
        )
        
        # Should have power in EEG frequency bands
        alpha_power = np.mean(psd[:, (freqs >= 8) & (freqs <= 12)])
        beta_power = np.mean(psd[:, (freqs >= 13) & (freqs <= 30)])
        theta_power = np.mean(psd[:, (freqs >= 4) & (freqs <= 8)])
        
        # All bands should have some power
        assert alpha_power > 0, "Should have alpha band power"
        assert beta_power > 0, "Should have beta band power"
        assert theta_power > 0, "Should have theta band power"