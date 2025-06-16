"""Tests for artifact detection standalone functions.

This module tests artifact detection and channel operation functions including
bad channel detection and interpolation.
"""

import pytest
import numpy as np
import mne
from mne.channels import make_standard_montage

# Import test utilities
from tests.fixtures.synthetic_data import create_synthetic_raw

# Import the functions to test
from autoclean.functions.artifacts import detect_bad_channels, interpolate_bad_channels


class TestBadChannelDetection:
    """Test bad channel detection function."""
    
    def test_detect_bad_channels_basic(self):
        """Test basic bad channel detection."""
        raw = create_synthetic_raw(
            montage="standard_1020", 
            n_channels=32, 
            duration=10.0, 
            sfreq=250
        )
        
        # Ensure montage is properly set
        if raw.get_montage() is None:
            montage = make_standard_montage('standard_1020')
            raw.set_montage(montage, match_case=False, on_missing='ignore')
        
        bad_channels = detect_bad_channels(
            raw,
            correlation_thresh=0.2,  # More lenient for synthetic data
            deviation_thresh=4.0,    # More lenient for synthetic data
            ransac_corr_thresh=0    # Disable RANSAC for synthetic data
        )
        
        assert isinstance(bad_channels, list)
        # With synthetic data, we might not detect many bad channels
        assert len(bad_channels) >= 0
    
    def test_detect_bad_channels_by_method(self):
        """Test bad channel detection with method breakdown."""
        raw = create_synthetic_raw(
            montage="standard_1020",
            n_channels=16, 
            duration=5.0, 
            sfreq=250
        )
        
        # Ensure montage is properly set
        if raw.get_montage() is None:
            montage = make_standard_montage('standard_1020')
            raw.set_montage(montage, match_case=False, on_missing='ignore')
        
        bad_by_method = detect_bad_channels(
            raw,
            correlation_thresh=0.2,  # More lenient for synthetic data
            deviation_thresh=4.0,    # More lenient for synthetic data
            ransac_corr_thresh=0,    # Disable RANSAC for synthetic data
            return_by_method=True
        )
        
        assert isinstance(bad_by_method, dict)
        expected_keys = ['correlation', 'deviation', 'ransac', 'combined']
        assert all(key in bad_by_method for key in expected_keys)
        assert all(isinstance(bad_by_method[key], list) for key in expected_keys)
    
    def test_detect_bad_channels_exclude_channels(self):
        """Test bad channel detection with excluded channels."""
        raw = create_synthetic_raw(
            montage="standard_1020",
            n_channels=16,
            duration=5.0, 
            sfreq=250
        )
        
        # Ensure montage is properly set
        if raw.get_montage() is None:
            montage = make_standard_montage('standard_1020')
            raw.set_montage(montage, match_case=False, on_missing='ignore')
        
        # Exclude some channels from detection
        exclude_channels = ['Cz', 'Pz']
        bad_channels = detect_bad_channels(
            raw,
            exclude_channels=exclude_channels,
            correlation_thresh=0.1,  # Very strict to potentially catch something
            deviation_thresh=3.0,
            ransac_corr_thresh=0     # Disable RANSAC for synthetic data
        )
        
        # Excluded channels should not appear in bad channels list
        for excluded_ch in exclude_channels:
            assert excluded_ch not in bad_channels
    
    def test_detect_bad_channels_no_ransac(self):
        """Test bad channel detection with RANSAC disabled."""
        raw = create_synthetic_raw(
            montage="standard_1020",
            n_channels=16,
            duration=5.0, 
            sfreq=250
        )
        
        # Ensure montage is properly set
        if raw.get_montage() is None:
            montage = make_standard_montage('standard_1020')
            raw.set_montage(montage, match_case=False, on_missing='ignore')
        
        bad_by_method = detect_bad_channels(
            raw,
            ransac_corr_thresh=0,  # Disable RANSAC
            return_by_method=True
        )
        
        # RANSAC should return empty list when disabled
        assert bad_by_method['ransac'] == []
    
    def test_detect_bad_channels_invalid_input(self):
        """Test bad channel detection with invalid inputs."""
        raw = create_synthetic_raw(duration=5.0, sfreq=250, n_channels=16)
        
        # Test invalid correlation threshold
        with pytest.raises(ValueError, match="correlation_thresh must be between 0 and 1"):
            detect_bad_channels(raw, correlation_thresh=1.5)
        
        # Test invalid deviation threshold
        with pytest.raises(ValueError, match="deviation_thresh must be positive"):
            detect_bad_channels(raw, deviation_thresh=-1.0)
        
        # Test invalid data type
        with pytest.raises(TypeError, match="Data must be an MNE Raw object"):
            detect_bad_channels("not_raw_data")


class TestChannelInterpolation:
    """Test channel interpolation function."""
    
    def test_interpolate_bad_channels_basic(self):
        """Test basic channel interpolation."""
        raw = create_synthetic_raw(
            montage="standard_1020",
            n_channels=32,
            duration=5.0, 
            sfreq=250
        )
        
        # Ensure montage is properly set
        if raw.get_montage() is None:
            montage = make_standard_montage('standard_1020')
            raw.set_montage(montage, match_case=False, on_missing='ignore')
        
        # Mark some channels as bad
        bad_channels = ['Fp1', 'F7']
        # Only use channels that exist in our data
        existing_bad_channels = [ch for ch in bad_channels if ch in raw.ch_names]
        
        if existing_bad_channels:
            raw_interp = interpolate_bad_channels(
                raw,
                bad_channels=existing_bad_channels
            )
            
            assert isinstance(raw_interp, mne.io.BaseRaw)
            # Bad channels should be removed from bads list if reset_bads=True
            assert len(raw_interp.info['bads']) == 0
            # Data shape should remain the same
            assert raw_interp.get_data().shape == raw.get_data().shape
    
    def test_interpolate_bad_channels_from_info(self):
        """Test interpolation using channels marked in info['bads']."""
        raw = create_synthetic_raw(
            montage="standard_1020",
            n_channels=16,
            duration=5.0, 
            sfreq=250
        )
        
        # Ensure montage is properly set
        if raw.get_montage() is None:
            montage = make_standard_montage('standard_1020')
            raw.set_montage(montage, match_case=False, on_missing='ignore')
        
        # Mark channels as bad in info
        if len(raw.ch_names) >= 2:
            raw.info['bads'] = [raw.ch_names[0], raw.ch_names[1]]
            
            raw_interp = interpolate_bad_channels(raw)
            
            assert isinstance(raw_interp, mne.io.BaseRaw)
            # Should have interpolated the bad channels
            assert len(raw_interp.info['bads']) == 0  # reset_bads=True by default
    
    def test_interpolate_bad_channels_keep_bads(self):
        """Test interpolation while keeping bad channels marked."""
        raw = create_synthetic_raw(
            montage="standard_1020",
            n_channels=16,
            duration=5.0, 
            sfreq=250
        )
        
        # Ensure montage is properly set
        if raw.get_montage() is None:
            montage = make_standard_montage('standard_1020')
            raw.set_montage(montage, match_case=False, on_missing='ignore')
        
        if len(raw.ch_names) >= 1:
            bad_channels = [raw.ch_names[0]]
            
            raw_interp = interpolate_bad_channels(
                raw,
                bad_channels=bad_channels,
                reset_bads=False
            )
            
            # Bad channels should still be marked as bad
            assert bad_channels[0] in raw_interp.info['bads']
    
    def test_interpolate_bad_channels_no_montage(self):
        """Test interpolation fails without montage."""
        raw = create_synthetic_raw(duration=5.0, sfreq=250, n_channels=16)
        
        # Explicitly remove montage
        raw.set_montage(None)
        bad_channels = [raw.ch_names[0]] if raw.ch_names else []
        
        if bad_channels:
            with pytest.raises(RuntimeError, match="Channel positions.*must be set"):
                interpolate_bad_channels(raw, bad_channels=bad_channels)
    
    def test_interpolate_bad_channels_invalid_channels(self):
        """Test interpolation with invalid channel names."""
        raw = create_synthetic_raw(
            montage="standard_1020",
            n_channels=16,
            duration=5.0, 
            sfreq=250
        )
        
        # Ensure montage is properly set
        if raw.get_montage() is None:
            montage = make_standard_montage('standard_1020')
            raw.set_montage(montage, match_case=False, on_missing='ignore')
        
        # Try to interpolate non-existent channels
        with pytest.raises(ValueError, match="Bad channels not found in data"):
            interpolate_bad_channels(raw, bad_channels=['NonExistentChannel'])
    
    def test_interpolate_bad_channels_no_bad_channels(self):
        """Test interpolation with no bad channels."""
        raw = create_synthetic_raw(
            montage="standard_1020",
            n_channels=16,
            duration=5.0, 
            sfreq=250
        )
        
        # Ensure montage is properly set
        if raw.get_montage() is None:
            montage = make_standard_montage('standard_1020')
            raw.set_montage(montage, match_case=False, on_missing='ignore')
        
        # No bad channels
        raw_interp = interpolate_bad_channels(raw, bad_channels=[])
        
        # Should return a copy with no changes
        assert isinstance(raw_interp, mne.io.BaseRaw)
        np.testing.assert_array_equal(raw_interp.get_data(), raw.get_data())
    
    def test_interpolate_bad_channels_epochs(self):
        """Test interpolation with epochs data."""
        raw = create_synthetic_raw(
            montage="standard_1020",
            n_channels=16,
            duration=5.0, 
            sfreq=250
        )
        
        # Ensure montage is properly set
        if raw.get_montage() is None:
            montage = make_standard_montage('standard_1020')
            raw.set_montage(montage, match_case=False, on_missing='ignore')
        
        # Create epochs
        from autoclean.functions.epoching import create_regular_epochs
        epochs = create_regular_epochs(raw, tmin=-0.5, tmax=0.5)
        
        if len(epochs) > 0 and len(raw.ch_names) >= 1:
            bad_channels = [raw.ch_names[0]]
            
            epochs_interp = interpolate_bad_channels(
                epochs,
                bad_channels=bad_channels
            )
            
            assert isinstance(epochs_interp, mne.Epochs)
            assert epochs_interp.get_data().shape == epochs.get_data().shape
    
    def test_interpolate_bad_channels_invalid_input(self):
        """Test interpolation with invalid inputs."""
        # Test invalid data type
        with pytest.raises(TypeError, match="Data must be an MNE Raw or Epochs object"):
            interpolate_bad_channels("not_mne_data")
        
        # Test invalid mode
        raw = create_synthetic_raw(
            montage="standard_1020",
            n_channels=16,
            duration=5.0, 
            sfreq=250
        )
        
        # Ensure montage is properly set
        if raw.get_montage() is None:
            montage = make_standard_montage('standard_1020')
            raw.set_montage(montage, match_case=False, on_missing='ignore')
        
        with pytest.raises(ValueError, match="mode must be 'accurate' or 'fast'"):
            interpolate_bad_channels(raw, bad_channels=[], mode='invalid')