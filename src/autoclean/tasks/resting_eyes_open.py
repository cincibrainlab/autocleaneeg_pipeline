"""New-style Python task file for resting state EEG preprocessing.

This demonstrates the new approach where configuration and processing logic
are combined in a single Python file.
"""

from typing import Any, Dict
from autoclean.core.task import Task

config = {'resample_step': {
            'enabled': True,
            'value': 250  # New sampling rate in Hz
        },
        'filtering': {
            'enabled': True,
            'value': {
                'l_freq': 1,
                'h_freq': 100,
                'notch_freqs': [60, 120],
                'notch_widths': 5
            }
        },
        'drop_outerlayer': {
            'enabled': False,
            'value': []
        },
        'eog_step': {
            'enabled': False,
            'value': []
        },
        'trim_step': {
            'enabled': True,
            'value': 4  # Number of seconds to trim from start/end
        },
        'crop_step': {
            'enabled': False,
            'value': {
                'start': 0,
                'end': 60
            }
        },
        'reference_step': {
            'enabled': True,
            'value': "average"
        },
        'montage': {
            'enabled': True,
            'value': "GSN-HydroCel-129"
        },
        'ICA': {
            'enabled': True,
            'value': {
                'method': 'picard',
                'n_components': None,
                'fit_params': {
                    'ortho': False,
                    'extended': True
                }
            }
        },
        'ICLabel': {
            'enabled': True,
            'value': {
                'ic_flags_to_reject': [
                    'muscle', 'heart', 'eog', 'ch_noise', 
                    'line_noise', 'eye', 'channel noise', 'line noise'
                ],
                'ic_rejection_threshold': 0.3
            }
        },
        'epoch_settings': {
            'enabled': True,
            'value': {
                'tmin': -1,
                'tmax': 1
            },
            'event_id': None,
            'remove_baseline': {
                'enabled': False,
                'window': [None, 0]
            },
            'threshold_rejection': {
                'enabled': False,
                'volt_threshold': {
                    'eeg': 125e-6
                }
            }
        }
    }

class RestingEyesOpen(Task):
    """Resting state EEG preprocessing task using the new Python task file approach.
    
    This class combines both configuration and processing logic in a single file,
    eliminating the need for separate YAML configuration files.
    """

    def run(self) -> None:
        """Execute the complete resting state EEG processing pipeline.
        
        This method demonstrates the new export=True functionality,
        allowing selective exporting at each processing step.
        """
        # Import raw EEG data
        self.import_raw()
        
        # Basic preprocessing with optional export
        self.filter_data()


        self.run_basic_steps(export=True)  # Export after basic steps
        
        # Create BIDS-compliant paths and filenames
        self.create_bids_path()
        
        # Channel cleaning
        self.clean_bad_channels(
            cleaning_method="interpolate", 
            reset_bads=True
        )
        
        # Re-referencing
        self.rereference_data()
        
        # Artifact detection
        self.annotate_noisy_epochs()
        self.annotate_uncorrelated_epochs()
        self.detect_dense_oscillatory_artifacts()
        
        # ICA processing with optional export
        self.run_ica(export=True)  # Export after ICA
        self.run_ICLabel()
        
        # Manual save for compatibility (can be removed once all mixins updated)
        from autoclean.io.export import save_raw_to_set
        save_raw_to_set(
            raw=self.raw,
            autoclean_dict=self.config,
            stage="post_clean_raw",
            flagged=self.flagged,
        )
        
        # Epoching with export
        self.create_regular_epochs(export=True)  # Export epochs
        
        # Prepare epochs for ICA
        self.prepare_epochs_for_ica()
        
        # Clean epochs using GFP with export
        self.gfp_clean_epochs(export=True)  # Export cleaned epochs
        
        # Generate visualization reports
        self.generate_reports()

    def generate_reports(self) -> None:
        """Generate quality control visualizations and reports."""
        if self.raw is None or self.original_raw is None:
            return
            
        # Plot raw vs cleaned overlay using mixin method
        self.plot_raw_vs_cleaned_overlay(self.original_raw, self.raw)
        
        # Plot PSD topography using mixin method
        self.step_psd_topo_figure(self.original_raw, self.raw)
        
        # Additional report generation can be added here


    def __init__(self, config: Dict[str, Any]):
        """Initialize the resting state task with embedded settings.
        
        Args:
            config: Configuration dictionary from the pipeline.
        """
        # Use the embedded settings from the module-level config variable
        self.settings = globals()['config']
        
        # Initialize the base class
        super().__init__(config)