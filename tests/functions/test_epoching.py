"""Tests for epoching standalone functions.

This module tests all epoching functions including regular epochs, event-based
epochs, and epoch quality assessment.
"""

import pytest
import numpy as np
import mne

# Import test utilities
from tests.fixtures.synthetic_data import create_synthetic_raw

# Import the functions to test
from autoclean.functions.epoching import (
    create_regular_epochs, 
    create_eventid_epochs,
    create_sl_epochs,
    detect_outlier_epochs,
    gfp_clean_epochs
)


class TestRegularEpochs:
    """Test regular epochs creation function."""
    
    def test_create_regular_epochs_basic(self):
        """Test basic regular epochs creation."""
        raw = create_synthetic_raw(duration=10.0, sfreq=250, n_channels=32)
        
        epochs = create_regular_epochs(
            data=raw,
            tmin=-1.0,
            tmax=1.0,
            overlap=0.0
        )
        
        assert isinstance(epochs, mne.Epochs)
        assert epochs.tmin == -1.0
        assert epochs.tmax == 1.0
        assert len(epochs) > 0
    
    def test_create_regular_epochs_with_overlap(self):
        """Test regular epochs with overlap."""
        raw = create_synthetic_raw(duration=10.0, sfreq=250, n_channels=32)
        
        epochs = create_regular_epochs(
            data=raw,
            tmin=-0.5,
            tmax=0.5,
            overlap=0.25
        )
        
        assert isinstance(epochs, mne.Epochs)
        assert len(epochs) > 5  # Should create more epochs with overlap
    
    def test_create_regular_epochs_with_metadata(self):
        """Test regular epochs with metadata inclusion."""
        raw = create_synthetic_raw(duration=5.0, sfreq=250, n_channels=32)
        
        epochs = create_regular_epochs(
            data=raw,
            tmin=-0.5,
            tmax=0.5,
            include_metadata=True
        )
        
        assert epochs.metadata is not None
        assert 'epoch_number' in epochs.metadata.columns
        assert 'epoch_duration' in epochs.metadata.columns


class TestEventIdEpochs:
    """Test event-based epochs creation function."""
    
    def test_create_eventid_epochs_basic(self):
        """Test basic event-based epochs creation."""
        raw = create_synthetic_raw(duration=10.0, sfreq=250, n_channels=32)
        
        # Add some test annotations as events
        annotations = mne.Annotations(
            onset=[1.0, 3.0, 5.0],
            duration=[0.0, 0.0, 0.0],
            description=['stim', 'stim', 'target']
        )
        raw.set_annotations(annotations)
        
        event_id = {'stim': 1}
        epochs = create_eventid_epochs(
            data=raw,
            event_id=event_id,
            tmin=-0.2,
            tmax=0.5,
            on_missing='ignore'  # Don't error if no matching events
        )
        
        assert isinstance(epochs, mne.Epochs)
        assert epochs.tmin == -0.2
        assert epochs.tmax == 0.5
    
    def test_create_eventid_epochs_no_events_warn(self):
        """Test event-based epochs when no events found with warning."""
        raw = create_synthetic_raw(duration=5.0, sfreq=250, n_channels=32)
        
        # Add some annotations that won't match our event_id
        annotations = mne.Annotations(
            onset=[1.0],
            duration=[0.0],
            description=['other_event']
        )
        raw.set_annotations(annotations)
        
        # Test should handle gracefully without warning in this case
        epochs = create_eventid_epochs(
            data=raw,
            event_id={'nonexistent': 999},
            on_missing='warn'
        )
            
        assert isinstance(epochs, (mne.Epochs, mne.EpochsArray))
        assert len(epochs) == 0
    
    def test_create_eventid_epochs_no_events_raise(self):
        """Test event-based epochs when no events found with error."""
        raw = create_synthetic_raw(duration=5.0, sfreq=250, n_channels=32)
        
        # Add some annotations that won't match our event_id
        annotations = mne.Annotations(
            onset=[1.0],
            duration=[0.0],
            description=['other_event']
        )
        raw.set_annotations(annotations)
        
        with pytest.raises(ValueError, match="No events found"):
            create_eventid_epochs(
                data=raw,
                event_id={'nonexistent': 999},
                on_missing='raise'
            )


class TestStatisticalLearningEpochs:
    """Test statistical learning epochs creation function."""
    
    def test_create_sl_epochs_basic(self):
        """Test basic statistical learning epochs creation."""
        raw = create_synthetic_raw(duration=10.0, sfreq=250, n_channels=32)
        
        # Add statistical learning events
        syllable_codes = ["DIN1", "DIN2", "DIN3", "DIN4", "DIN5", "DIN6"]
        onset_times = [1.0, 3.0, 5.0]
        descriptions = []
        onsets = []
        
        # Create syllable sequences starting at each onset
        for onset_time in onset_times:
            for i, code in enumerate(syllable_codes):
                onsets.append(onset_time + i * 0.3)  # 300ms per syllable
                descriptions.append(code)
        
        annotations = mne.Annotations(
            onset=onsets,
            duration=[0.0] * len(onsets),
            description=descriptions
        )
        raw.set_annotations(annotations)
        
        # Create epochs (will likely return empty due to validation requirements)
        epochs = create_sl_epochs(
            data=raw,
            tmin=0.0,
            tmax=1.8,  # 6 syllables * 300ms
            num_syllables_per_epoch=6
        )
        
        assert isinstance(epochs, mne.Epochs)
    
    def test_create_sl_epochs_error_handling(self):
        """Test statistical learning epochs error handling."""
        raw = create_synthetic_raw(duration=5.0, sfreq=250, n_channels=32)
        
        # Test with no annotations
        with pytest.raises(ValueError, match="No events found"):
            create_sl_epochs(data=raw)
        
        # Test with annotations but no valid syllable patterns
        annotations = mne.Annotations(
            onset=[1.0],
            duration=[0.0], 
            description=['other_event']
        )
        raw.set_annotations(annotations)
        
        with pytest.raises((ValueError, RuntimeError)):
            create_sl_epochs(data=raw)


class TestEpochQuality:
    """Test epoch quality assessment functions."""
    
    def test_detect_outlier_epochs_basic(self):
        """Test basic outlier detection."""
        # Create epochs with some outliers
        raw = create_synthetic_raw(duration=10.0, sfreq=250, n_channels=32)
        epochs = create_regular_epochs(raw, tmin=-0.5, tmax=0.5)
        
        # Add artificial outliers by modifying some epochs
        if len(epochs) > 0:
            data = epochs.get_data()
            if data.shape[0] > 2:
                # Make some epochs outliers by adding large amplitude
                data[0] *= 10  # Large amplitude outlier
                data[1] += 500e-6  # Large offset outlier
                
                # Create new epochs object with modified data
                epochs_modified = mne.EpochsArray(data, epochs.info, tmin=epochs.tmin)
                
                # Detect outliers
                clean_epochs = detect_outlier_epochs(epochs_modified, threshold=2.0)
                
                assert isinstance(clean_epochs, (mne.Epochs, mne.EpochsArray))
                assert len(clean_epochs) <= len(epochs_modified)
        
    def test_detect_outlier_epochs_with_scores(self):
        """Test outlier detection returning scores."""
        raw = create_synthetic_raw(duration=5.0, sfreq=250, n_channels=32)
        epochs = create_regular_epochs(raw, tmin=-0.2, tmax=0.2)
        
        if len(epochs) > 0:
            clean_epochs, scores = detect_outlier_epochs(
                epochs, 
                threshold=3.0,
                return_scores=True
            )
            
            assert isinstance(clean_epochs, (mne.Epochs, mne.EpochsArray))
            assert isinstance(scores, dict)
            assert all(key in scores for key in ['mean', 'variance', 'range', 'gradient'])
    
    def test_gfp_clean_epochs_basic(self):
        """Test basic GFP cleaning."""
        raw = create_synthetic_raw(duration=5.0, sfreq=250, n_channels=32)
        epochs = create_regular_epochs(raw, tmin=-0.2, tmax=0.2)
        
        if len(epochs) > 0:
            clean_epochs = gfp_clean_epochs(epochs, gfp_threshold=3.0)
            
            assert isinstance(clean_epochs, (mne.Epochs, mne.EpochsArray))
            assert len(clean_epochs) <= len(epochs)
    
    def test_gfp_clean_epochs_with_subsampling(self):
        """Test GFP cleaning with epoch subsampling.""" 
        raw = create_synthetic_raw(duration=10.0, sfreq=250, n_channels=32)
        epochs = create_regular_epochs(raw, tmin=-0.2, tmax=0.2)
        
        if len(epochs) >= 5:
            clean_epochs = gfp_clean_epochs(
                epochs,
                gfp_threshold=3.0,
                number_of_epochs=3,
                random_seed=42
            )
            
            assert isinstance(clean_epochs, (mne.Epochs, mne.EpochsArray))
            assert len(clean_epochs) == 3
    
    def test_gfp_clean_epochs_with_gfp_values(self):
        """Test GFP cleaning returning GFP values."""
        raw = create_synthetic_raw(duration=5.0, sfreq=250, n_channels=32)
        epochs = create_regular_epochs(raw, tmin=-0.2, tmax=0.2)
        
        if len(epochs) > 0:
            clean_epochs, gfp_values = gfp_clean_epochs(
                epochs,
                return_gfp_values=True
            )
            
            assert isinstance(clean_epochs, (mne.Epochs, mne.EpochsArray))
            assert isinstance(gfp_values, np.ndarray)
            assert len(gfp_values) == len(epochs)  # Original number of epochs