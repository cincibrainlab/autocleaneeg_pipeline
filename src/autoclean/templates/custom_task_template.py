from autoclean.core.task import Task
from autoclean.task_config_schema import CONFIG_VERSION

# =============================================================================
#                     CUSTOM EEG PREPROCESSING TASK TEMPLATE
# =============================================================================
# This is a template for creating custom EEG preprocessing tasks.
# Customize the configuration below to match your specific EEG paradigm.
#
# Instructions:
# 1. Rename this file to match your task (e.g., my_experiment.py)
# 2. Update the class name below (e.g., MyExperiment)
# 3. Modify the config dictionary to match your data requirements
# 4. Customize the run() method to define your processing pipeline
#
# 🟢 enabled: True  = Apply this processing step
# 🔴 enabled: False = Skip this processing step
#
# 💡 TIP: Use the AutoClean configuration wizard to generate settings
#         automatically, or copy settings from existing tasks!
# =============================================================================

config = {
    "version": CONFIG_VERSION,
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
    # "input_path": "/path/to/my/data/",  # Uncomment and modify for your data
    # Optional: keep flagged files in standard output directories
    # "move_flagged_files": False,
    "resample_step": {"enabled": True, "value": 250},  # Resample to 250 Hz
    "filtering": {
        "enabled": True,
        "value": {
            "l_freq": 1,  # High-pass filter (Hz)
            "h_freq": 100,  # Low-pass filter (Hz)
            "notch_freqs": [60, 120],  # Notch filter frequencies
            "notch_widths": 5,  # Notch filter width
        },
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
        "enabled": False,
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
            "method": "fastica",  # ICA method
            "n_components": None,  # Number of components (None = auto)
            "fit_params": {},  # Additional ICA parameters
        },
    },
    "component_rejection": {
        "enabled": True,
        "method": "iclabel",  # Classification method: 'iclabel' or 'icvision'
        "value": {
            "ic_flags_to_reject": ["muscle", "heart", "eog", "ch_noise", "line_noise"],
            "ic_rejection_threshold": 0.3  # Threshold for automatic rejection
        },
    },
    "epoch_settings": {
        "enabled": True,
        "value": {
            "tmin": -1,  # Epoch start (seconds relative to event)
            "tmax": 1,  # Epoch end (seconds relative to event)
        },
        "event_id": None,  # Event IDs for epoching (None = auto-detect)
        "remove_baseline": {
            "enabled": False,
            "window": [None, 0],  # Baseline correction window
        },
        "threshold_rejection": {
            "enabled": False,
            "volt_threshold": {
                "eeg": 0.000125  # Voltage threshold for epoch rejection (V)
            },
        },
    },
}


class CustomTask(Task):
    """
    Custom EEG preprocessing task template.

    This template provides a starting point for creating custom EEG preprocessing
    pipelines. Modify the class name, configuration, and processing steps to
    match your specific experimental paradigm.
    """

    def run(self) -> None:
        """
        Define your custom EEG preprocessing pipeline.

        This method orchestrates the entire preprocessing workflow.
        Customize by adding, removing, or reordering processing steps.
        """
        # Import raw EEG data
        self.import_raw()

        # Basic preprocessing steps
        self.resample_data()
        self.filter_data()
        self.drop_outer_layer()
        self.assign_eog_channels()
        self.trim_edges()
        self.crop_duration()

        # Store original data for comparison
        self.original_raw = self.raw.copy()

        # Create BIDS-compliant paths and filenames
        self.create_bids_path()

        # Channel cleaning
        self.clean_bad_channels()

        # Re-referencing
        self.rereference_data()

        # Artifact detection
        self.annotate_noisy_epochs()
        self.annotate_uncorrelated_epochs()
        self.detect_dense_oscillatory_artifacts()

        # ICA processing with optional export
        self.run_ica()
        self.classify_ica_components()  # Uses method from component_rejection config

        # Epoching with export
        self.create_regular_epochs()  # Using auto-detected or configured event IDs

        # Detect outlier epochs
        self.detect_outlier_epochs()

        # Clean epochs using Global Field Power
        self.gfp_clean_epochs()

        # Generate visualization reports
        self.generate_reports()

    def generate_reports(self) -> None:
        """
        Generate quality control visualizations and reports.

        Customize this method to add task-specific visualizations
        and quality metrics for your EEG paradigm.
        """
        if self.raw is None or self.original_raw is None:
            return

        # Plot raw vs cleaned overlay
        self.plot_raw_vs_cleaned_overlay(self.original_raw, self.raw)

        # Plot power spectral density topography
        self.step_psd_topo_figure(self.original_raw, self.raw)

        # Add custom reports here:
        # - Event-related potential plots
        # - Time-frequency analyses
        # - Custom quality metrics
        # - Task-specific visualizations
