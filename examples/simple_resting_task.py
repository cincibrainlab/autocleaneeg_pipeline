"""Simple resting state EEG processing task.

This is the most basic example of a Python task file for AutoClean EEG.
It demonstrates the minimal configuration needed for resting state EEG processing.
"""

from typing import Any, Dict
from autoclean.core.task import Task

# Simple configuration for basic resting state processing
config = {
    'resample_step': {
        'enabled': True,
        'value': 250  # Downsample to 250 Hz
    },
    'filtering': {
        'enabled': True,
        'value': {
            'l_freq': 1,    # High-pass filter at 1 Hz
            'h_freq': 40,   # Low-pass filter at 40 Hz
            'notch_freqs': [60]  # Remove 60 Hz line noise
        }
    },
    'montage': {
        'enabled': True,
        'value': "GSN-HydroCel-129"  # EGI 129-channel cap
    },
    'reference_step': {
        'enabled': True,
        'value': "average"  # Average reference
    },
    'ICA': {
        'enabled': True,
        'value': {
            'method': 'picard',  # Fast ICA algorithm
            'n_components': None  # Auto-determine number of components
        }
    },
    'ICLabel': {
        'enabled': True,
        'value': {
            'ic_flags_to_reject': ['muscle', 'eye', 'heart'],
            'ic_rejection_threshold': 0.7
        }
    },
    'epoch_settings': {
        'enabled': True,
        'value': {
            'tmin': -2,  # 2 seconds before event
            'tmax': 2    # 2 seconds after event
        }
    }
}


class SimpleRestingTask(Task):
    """Simple resting state EEG processing task.
    
    This task performs basic preprocessing including:
    - Resampling to 250 Hz
    - Filtering (1-40 Hz bandpass, 60 Hz notch)
    - Average referencing
    - ICA artifact removal
    - Epoching into 4-second segments
    
    Only saves the final cleaned epochs to disk.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the simple resting task.
        
        Args:
            config: Configuration dictionary from the pipeline.
        """
        # Use the embedded configuration
        self.settings = globals()['config']
        super().__init__(config)

    def run(self) -> None:
        """Execute the simple resting state processing pipeline."""
        # Import raw EEG data
        self.import_raw()
        
        # Basic preprocessing (resampling, filtering, referencing)
        self.run_basic_steps()
        
        # Channel cleaning
        self.clean_bad_channels(cleaning_method="interpolate")
        
        # Re-referencing
        self.rereference_data()
        
        # Artifact detection
        self.annotate_noisy_epochs()
        self.detect_dense_oscillatory_artifacts()
        
        # ICA cleaning
        self.run_ica()
        self.run_ICLabel()
        
        # Create epochs and export only the final result
        self.create_regular_epochs(export=True)