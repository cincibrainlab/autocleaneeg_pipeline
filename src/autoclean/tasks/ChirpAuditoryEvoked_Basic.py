from autoclean.core.task import Task
from autoclean.task_config_schema import CONFIG_VERSION

# =============================================================================
#  Auditory Chirp EEG paradigm with stimulus triggers EEG PREPROCESSING CONFIGURATION
# =============================================================================
# This configuration controls how your Auditory Chirp EEG paradigm with stimulus triggers EEG data will be 
# automatically cleaned and processed. Each section handles a different aspect
# of the preprocessing pipeline.
#
# 🟢 enabled: True  = Apply this processing step
# 🔴 enabled: False = Skip this processing step
#
# =============================================================================

config = {
    'version': CONFIG_VERSION,
    # Optional: Specify a dataset name for organized output directories
    # Examples:
    #   With dataset_name: "Experiment1_07-03-2025"
    #   Without dataset_name: "CustomTask"
    # "dataset_name": "Experiment1",  # Uncomment and modify for your dataset
    # Optional: Specify default input file or directory for this task
    # This will be used when no input is provided via CLI or API
    # Examples:
    #   "input_path": "/path/to/my/data.raw",           # Single file
    #   "input_path": "/path/to/data/directory/",       # Directory
    #"input_path": "/path/to/my/data/",  # Uncomment and modify for your data
    # Optional: keep flagged files in standard output directories
    # "move_flagged_files": False,
    'resample_step': {
        'enabled': True,
        'value': 250
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
    "drop_outerlayer": {
        "enabled": False,
        "value": [],  # Channel indices to drop (e.g., [1, 32, 125, 126, 127, 128])
    },
    "eog_step": {
        "enabled": False,
        "value": [],  # EOG channel indices (e.g., [1, 32, 8, 14, 17, 21, 25, 125, 126, 127, 128])
    },
    "trim_step": {"enabled": True, "value": 4},  # Trim seconds from start/end
    "crop_step": {
        "enabled": True,
        "value": {"start": 0, "end": 60},  # Start time (seconds)  # End time (seconds)
    },
    "reference_step": {
        "enabled": True,
        "value": "average",  # Reference type: 'average', specific channels, or None
    },
    "montage": {
        "enabled": True,
        "value": "GSN-HydroCel-129",  # EEG montage (e.g., 'standard_1020', 'GSN-HydroCel-129')
    },
    "ICA": {
        "enabled": True,
        "value": {
            "method": "infomax",
            "n_components": None,
            "fit_params": {"extended": True},
            "temp_highpass_for_ica": None,
        },
    },
    "component_rejection": {
        "enabled": True,
        "method": "iclabel",  # Classification method: 'iclabel' or 'icvision' or 'hybrid'
        "value": {
            "ic_flags_to_reject": ["muscle", "heart", "eog", "ch_noise", "line_noise"],
            "ic_rejection_threshold": 0.3,
        },
        "psd_fmax": 40.0,  # NEW: Limit PSD plots to 40 Hz
    },
    'epoch_settings': {
        'enabled': True,
        'value': {
            'tmin': -0.5,
            'tmax': 2.75
        },
        'event_id': {
            'DIN64': 1
        },
        'remove_baseline': {
            'enabled': False,
            'window': [None, 0]
        },
        'threshold_rejection': {
            'enabled': False,
            'volt_threshold': {
                'eeg': 0.000125
            }
        }
    }
}

class ChirpDefault(Task):
    """Task implementation for Chirp default EEG preprocessing."""

    def run(self) -> None:
        # Import raw EEG data
        self.import_raw()

        #Basic preprocessing steps
        self.resample_data()

        self.filter_data()

        self.drop_outer_layer()

        self.assign_eog_channels()

        self.trim_edges()

        self.crop_duration()

        self.original_raw = self.raw.copy()
        
        # Channel cleaning
        self.clean_bad_channels()
        
        # Re-referencing
        self.rereference_data()
        
        # Artifact detection
        self.annotate_noisy_epochs()
        self.annotate_uncorrelated_epochs()
        self.detect_dense_oscillatory_artifacts()
        
        # ICA processing with optional export
        self.run_ica()  # Export after ICA
        self.classify_ica_components()
        
        # Epoching with export
        self.create_eventid_epochs() # Using event IDs
        
        # Detect outlier epochs
        self.detect_outlier_epochs()
        
        # Clean epochs using GFP with export
        self.gfp_clean_epochs() 

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

