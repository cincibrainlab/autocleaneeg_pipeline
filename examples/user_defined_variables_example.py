"""
Example demonstrating user-defined variables in task configuration.

This example shows how to add custom variables to your EEG processing task
and access them throughout your pipeline using the existing config system.
"""

from autoclean.core.task import Task

# =============================================================================
#                           USER-DEFINED VARIABLES
# =============================================================================
# Add your custom variables directly to the config dictionary.
# These will be automatically available via self.settings in your task.

config = {
    # ========= STANDARD EEG PROCESSING SETTINGS =========
    "resample_step": {"enabled": True, "value": 250},
    "filtering": {
        "enabled": True,
        "value": {
            "l_freq": 1,
            "h_freq": 100,
            "notch_freqs": [60, 120],
            "notch_widths": 5,
        },
    },
    "montage": {
        "enabled": True,
        "value": "GSN-HydroCel-129",
    },
    "ICA": {
        "enabled": True,
        "value": {
            "method": "fastica",
            "n_components": None,
        },
    },
    
    # ========= CUSTOM USER-DEFINED VARIABLES =========
    # These are YOUR custom variables for your specific experiment
    "stimulus_duration": 500,           # Stimulus duration in milliseconds
    "trial_count": 120,                # Expected number of trials
    "response_window": 1000,           # Response window in milliseconds
    "baseline_period": [-200, 0],      # Baseline period in milliseconds
    "custom_threshold": 75.0,          # Custom rejection threshold
    "experiment_conditions": [         # List of experimental conditions
        "condition_A",
        "condition_B", 
        "control"
    ],
    "participant_group": "adults",     # Participant group identifier
    "analysis_parameters": {           # Nested custom parameters
        "window_size": 100,
        "overlap": 50,
        "method": "custom_algorithm"
    }
}


class UserDefinedVariablesExample(Task):
    """
    Example task demonstrating user-defined variables.
    
    This task shows how to:
    1. Define custom variables in the config dictionary
    2. Access them via self.settings in your task methods
    3. Use them throughout your processing pipeline
    """

    def run(self) -> None:
        """Run the processing pipeline using custom user-defined variables."""
        
        # Import raw EEG data
        self.import_raw()
        
        # Access your custom variables with defaults for safety
        stimulus_duration = self.settings.get("stimulus_duration", 250)
        trial_count = self.settings.get("trial_count", 100)
        response_window = self.settings.get("response_window", 800)
        baseline_period = self.settings.get("baseline_period", [-100, 0])
        threshold = self.settings.get("custom_threshold", 50.0)
        conditions = self.settings.get("experiment_conditions", ["default"])
        group = self.settings.get("participant_group", "unknown")
        analysis_params = self.settings.get("analysis_parameters", {})
        
        # Use your variables in processing
        print(f"Processing {group} participant data:")
        print(f"  - Stimulus duration: {stimulus_duration}ms")
        print(f"  - Expected trials: {trial_count}")
        print(f"  - Response window: {response_window}ms")
        print(f"  - Baseline period: {baseline_period}ms")
        print(f"  - Custom threshold: {threshold}")
        print(f"  - Conditions: {conditions}")
        print(f"  - Analysis params: {analysis_params}")
        
        # Run standard preprocessing
        self.resample_data()
        self.filter_data()
        
        # Store original for comparison
        self.original_raw = self.raw.copy()
        self.create_bids_path()
        
        # Use custom variables in your processing logic
        self._apply_custom_processing(
            duration=stimulus_duration,
            trials=trial_count,
            threshold=threshold
        )
        
        # Continue with standard pipeline
        self.clean_bad_channels()
        self.rereference_data()
        self.run_ica()
        
        # Create epochs using your custom parameters
        self._create_custom_epochs(
            baseline=baseline_period,
            response_window=response_window,
            conditions=conditions
        )
        
        # Generate reports
        self.generate_reports()
    
    def _apply_custom_processing(self, duration: int, trials: int, threshold: float) -> None:
        """Apply custom processing using user-defined parameters."""
        print(f"Applying custom processing:")
        print(f"  - Using duration: {duration}ms")
        print(f"  - Expecting {trials} trials")
        print(f"  - Threshold: {threshold}")
        
        # Your custom processing logic here
        # Example: Custom artifact detection based on your threshold
        if hasattr(self, 'raw') and self.raw is not None:
            # Use your custom threshold for artifact detection
            self.annotate_noisy_epochs()
            print(f"Applied custom threshold of {threshold} for artifact detection")
    
    def _create_custom_epochs(self, baseline: list, response_window: int, conditions: list) -> None:
        """Create epochs using custom user parameters."""
        print(f"Creating epochs with custom parameters:")
        print(f"  - Baseline: {baseline}ms")
        print(f"  - Response window: {response_window}ms")  
        print(f"  - Conditions: {conditions}")
        
        # Use your custom parameters for epoching
        # This would typically involve setting up epoch parameters
        # based on your experimental design
        
        # Standard epoching (customize based on your needs)
        self.create_regular_epochs()
        
    def generate_reports(self) -> None:
        """Generate reports including custom variable information."""
        if self.raw is None or self.original_raw is None:
            return
            
        # Standard reports
        self.plot_raw_vs_cleaned_overlay(self.original_raw, self.raw)
        self.step_psd_topo_figure(self.original_raw, self.raw)
        
        # Custom reporting using your variables
        print("=== CUSTOM ANALYSIS REPORT ===")
        print(f"Participant group: {self.settings.get('participant_group', 'unknown')}")
        print(f"Conditions processed: {self.settings.get('experiment_conditions', [])}")
        print(f"Analysis method: {self.settings.get('analysis_parameters', {}).get('method', 'standard')}")