"""Mixin classes for data processing operations in autoclean tasks.

This module contains mixin classes that provide common functionality for
data processing operations that can be used across different task types.
These mixins are designed to be included in task classes to provide
standardized implementations of common operations.

Examples:
    ```python
    from autoclean.mixins.data_processing import SignalProcessingMixin
    from autoclean.core.task import Task

    class MyTask(Task, SignalProcessingMixin):
        def run(self):
            # ... task initialization ...
            
            # Use the mixin method
            self.raw = self.resample_data(self.raw)
            
            # ... continue processing ...
    ```
"""

from typing import Union, Dict, Any
from datetime import datetime

import mne

from autoclean.utils.logging import message
from autoclean.utils.database import manage_database, get_run_record
from autoclean.step_functions.io import save_raw_to_set


class SignalProcessingMixin:
    """Mixin class providing signal processing functionality for EEG data.
    
    This mixin provides methods for common signal processing operations such as
    resampling, filtering, and other preprocessing steps for MNE Raw or Epochs objects.
    It respects the configuration toggles in the task configuration.
    
    Attributes:
        config: Task configuration dictionary (provided by the parent class)
    """
    
    def resample_data(self, data: Union[mne.io.Raw, mne.Epochs, None] = None, target_sfreq: float = None, 
                    stage_name: str = "resampled", use_epochs: bool = False) -> Union[mne.io.Raw, mne.Epochs]:
        """Resample raw or epoched data based on configuration settings.
        
        This method can work with self.raw, self.epochs, or a provided data object.
        It checks the resample_step toggle in the configuration if no target_sfreq is provided.
        
        Args:
            data: Optional MNE Raw or Epochs object to resample. If None, uses self.raw or self.epochs
            target_sfreq: Optional target sampling frequency. If None, reads from config
            stage_name: Name for saving the resampled data (default: "resampled")
            use_epochs: If True and data is None, uses self.epochs instead of self.raw
            
        Returns:
            The resampled data object (same type as input)
            
        Raises:
            TypeError: If data is not a Raw or Epochs object
            RuntimeError: If resampling fails
            AttributeError: If self.raw or self.epochs doesn't exist when needed
        """
        # Determine which data to use
        if data is None:
            if use_epochs:
                if not hasattr(self, 'epochs') or self.epochs is None:
                    raise AttributeError("No epochs data available for resampling")
                data = self.epochs
            else:
                if not hasattr(self, 'raw') or self.raw is None:
                    raise AttributeError("No raw data available for resampling")
                data = self.raw
        
        # Type checking
        if not isinstance(data, (mne.io.BaseRaw, mne.BaseEpochs)):
            raise TypeError("Data must be an MNE Raw or Epochs object")
            
        # Access configuration if needed
        if not hasattr(self, 'config'):
            raise AttributeError("SignalProcessingMixin requires a 'config' attribute")
            
        # If target_sfreq is not provided, get it from config
        if target_sfreq is None:
            task = self.config.get("task")
            
            # Check if resampling is enabled in the configuration
            resample_enabled = self.config.get("tasks", {}).get(task, {}).get("settings", {}).get(
                "resample_step", {}).get("enabled", False)
                
            if not resample_enabled:
                message("info", "Resampling step is disabled in configuration")
                return data
                
            # Get target sampling frequency from config
            target_sfreq = self.config.get("tasks", {}).get(task, {}).get("settings", {}).get(
                "resample_step", {}).get("value")
                
            if target_sfreq is None:
                message("warning", "Target sampling frequency not specified, skipping resampling")
                return data
            
        # Check if we need to resample (avoid unnecessary resampling)
        current_sfreq = data.info["sfreq"]
        if abs(current_sfreq - target_sfreq) < 0.01:  # Small threshold to account for floating point errors
            message("info", f"Data already at target frequency ({target_sfreq} Hz), skipping resampling")
            return data
            
        message("header", f"Resampling data from {current_sfreq} Hz to {target_sfreq} Hz...")
        
        try:
            # Resample based on data type
            if isinstance(data, mne.io.Raw):
                resampled_data = data.copy().resample(target_sfreq)
                # Save resampled raw data if it's a Raw object
                save_raw_to_set(resampled_data, self.config, f"post_{stage_name}")
            else:  # Epochs
                resampled_data = data.copy().resample(target_sfreq)
                
            message("info", f"Data successfully resampled to {target_sfreq} Hz")
            
            # Update metadata
            metadata = {
                "resample_data": {
                    "creationDateTime": datetime.now().isoformat(),
                    "original_sfreq": current_sfreq,
                    "target_sfreq": target_sfreq,
                    "data_type": "raw" if isinstance(data, mne.io.Raw) else "epochs"
                }
            }
            
            # Update the database if we have a run_id
            if hasattr(self, 'config') and self.config.get("run_id"):
                run_id = self.config.get("run_id")
                manage_database(
                    operation="update", update_record={"run_id": run_id, "metadata": metadata}
                )
            
            # Update self.raw or self.epochs if we're using those
            if data is None and hasattr(self, 'raw') and self.raw is not None and not use_epochs:
                self.raw = resampled_data
            elif data is None and hasattr(self, 'epochs') and self.epochs is not None and use_epochs:
                self.epochs = resampled_data
            breakpoint()
            return resampled_data
            
        except Exception as e:
            message("error", f"Error during resampling: {str(e)}")
            raise RuntimeError(f"Failed to resample data: {str(e)}") from e
