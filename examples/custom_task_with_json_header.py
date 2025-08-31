"""
{
  "stimulus_duration": 500,
  "trial_count": 120,
  "response_window": 1000,
  "baseline_period": [-200, 0],
  "custom_threshold": 75.0,
  "experiment_name": "CustomExperiment"
}
"""

# Example task file demonstrating user-defined variables in JSON header
from typing import Any, Dict
from autoclean.core.task import Task


class CustomTaskWithJsonHeader(Task):
    """Example task demonstrating user-defined variables from JSON header.
    
    This task shows how to define custom variables in a JSON header
    at the top of the file and access them via task_context.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def run(self) -> None:
        """Run the processing pipeline using user-defined variables."""
        # Import raw EEG data
        self.import_raw()
        
        # Access user-defined variables from JSON header
        stimulus_duration = self.task_context.get("stimulus_duration", 250)
        trial_count = self.task_context.get("trial_count", 100)
        response_window = self.task_context.get("response_window", 800)
        baseline_period = self.task_context.get("baseline_period", [-100, 0])
        custom_threshold = self.task_context.get("custom_threshold", 50.0)
        experiment_name = self.task_context.get("experiment_name", "DefaultExperiment")
        
        print(f"Processing {experiment_name} with {trial_count} trials")
        print(f"Stimulus duration: {stimulus_duration}ms")
        print(f"Response window: {response_window}ms") 
        print(f"Baseline period: {baseline_period}")
        print(f"Custom threshold: {custom_threshold}")
        
        # Use variables in processing pipeline
        # Example: Configure epochs based on user-defined parameters
        if hasattr(self, 'raw') and self.raw is not None:
            # Could use baseline_period for epoching
            # Could use response_window for event detection
            # Could use custom_threshold for artifact rejection
            pass
        
        # Continue with standard preprocessing
        self.run_basic_steps()
        
        # Create epochs with user-defined parameters
        self.create_regular_epochs()


# Optional: Embedded configuration (alternative to JSON header)
config = {
    "eeg_system": {"montage": "GSN-HydroCel-129", "reference": "average"},
    "signal_processing": {"filter": {"highpass": 0.1, "lowpass": 50.0}},
    "output": {"save_stages": ["raw", "epochs"]}
}