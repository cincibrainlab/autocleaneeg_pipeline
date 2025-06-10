"""ASSR (Auditory Steady-State Response) EEG processing task.

This task demonstrates processing for ASSR paradigms with event-related epoching
and specialized filtering for auditory responses.
"""

from typing import Any, Dict
from autoclean.core.task import Task

# ASSR-specific configuration
config = {
    'resample_step': {
        'enabled': True,
        'value': 500  # Higher sampling rate for ASSR
    },
    'filtering': {
        'enabled': True,
        'value': {
            'l_freq': 0.5,   # Lower high-pass for ASSR
            'h_freq': 100,   # Higher low-pass for ASSR
            'notch_freqs': [60, 120],  # Remove line noise harmonics
            'notch_widths': 2
        }
    },
    'montage': {
        'enabled': True,
        'value': "GSN-HydroCel-129"
    },
    'reference_step': {
        'enabled': True,
        'value': "average"
    },
    'ICA': {
        'enabled': True,
        'value': {
            'method': 'picard',
            'n_components': 25,  # Fixed number for ASSR
            'fit_params': {
                'ortho': False,
                'extended': True
            }
        }
    },
    'ICLabel': {
        'enabled': True,
        'value': {
            'ic_flags_to_reject': ['muscle', 'eye', 'heart', 'line_noise'],
            'ic_rejection_threshold': 0.8  # Stricter threshold for ASSR
        }
    },
    'eventid_epoch_settings': {
        'enabled': True,
        'value': {
            'tmin': -0.5,  # 500ms pre-stimulus
            'tmax': 2.0,   # 2s post-stimulus
            'event_id': {'stimulus': 1},
            'baseline': (-0.2, 0),  # Baseline correction
            'reject': {'eeg': 100e-6}  # Strict artifact rejection
        }
    }
}


class ASSRDefault(Task):
    """ASSR (Auditory Steady-State Response) processing task.
    
    This task is optimized for processing ASSR data with:
    - Higher sampling rate preservation (500 Hz)
    - Wider frequency range (0.5-100 Hz)
    - Event-locked epoching
    - Strict artifact rejection
    - Multiple export points for analysis
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the ASSR processing task.
        
        Args:
            config: Configuration dictionary from the pipeline.
        """
        self.settings = globals()['config']
        super().__init__(config)

    def run(self) -> None:
        """Execute the ASSR processing pipeline."""
        # Import raw EEG data
        self.import_raw()
        
        # Basic preprocessing with export for quality control
        self.run_basic_steps(export=True)
        
        # Store original for comparison
        self.original_raw = self.raw.copy()
        
        # Create BIDS-compliant paths
        self.create_bids_path()
        
        # Channel operations
        self.clean_bad_channels(cleaning_method="interpolate", reset_bads=True)
        self.rereference_data()
        
        # Artifact detection and annotation
        self.annotate_noisy_epochs()
        self.annotate_uncorrelated_epochs()
        self.detect_dense_oscillatory_artifacts()
        
        # ICA processing with export
        self.run_ica(export=True)
        self.run_ICLabel()
        
        # Export cleaned continuous data
        from autoclean.io.export import save_raw_to_set
        save_raw_to_set(
            raw=self.raw,
            autoclean_dict=self.config,
            stage="post_clean_raw",
            flagged=self.flagged,
        )
        
        # Event-locked epoching
        self.create_eventid_epochs()
        
        # Prepare and clean epochs
        self.prepare_epochs_for_ica()
        self.gfp_clean_epochs(export=True)
        
        # Generate visualization reports
        if hasattr(self, 'original_raw') and self.original_raw is not None:
            self.plot_raw_vs_cleaned_overlay(self.original_raw, self.raw)
            self.step_psd_topo_figure(self.original_raw, self.raw)