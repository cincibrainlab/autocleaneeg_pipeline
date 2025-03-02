"""Base signal processing mixin for autoclean tasks."""

from typing import Any, Dict, Optional, Tuple, Union
import mne

class SignalProcessingMixin:
    """Base mixin class providing signal processing functionality for EEG data.
    
    This mixin serves as a composition of all signal processing mixins and provides
    utility methods shared across the different signal processing operations.
    
    Attributes:
        config: Task configuration dictionary (provided by the parent class)
    """
    
    def _check_step_enabled(self, step_name: str) -> Tuple[bool, Optional[Any]]:
        """Check if a processing step is enabled in the configuration.
        
        Args:
            step_name: Name of the step to check in the configuration
            
        Returns:
            Tuple of (is_enabled, value) where is_enabled is a boolean indicating
            if the step is enabled, and value is the configuration value for the step
            if it exists, or None otherwise
        """
        if not hasattr(self, 'config'):
            return True, None
            
        task = self.config.get("task")
        if not task:
            return True, None
            
        settings = self.config.get("tasks", {}).get(task, {}).get("settings", {})
        step_settings = settings.get(step_name, {})
        
        is_enabled = step_settings.get("enabled", False)
        value = step_settings.get("value")
        
        return is_enabled, value
        
    def _report_step_status(self) -> None:
        """Report the enabled/disabled status of all processing steps in the configuration."""
        if not hasattr(self, 'config'):
            return
            
        task = self.config.get("task")
        if not task:
            return
            
        settings = self.config.get("tasks", {}).get(task, {}).get("settings", {})
        
        from autoclean.utils.logging import message
        
        message("header", f"Processing step status for task '{task}':")
        
        for step_name, step_settings in settings.items():
            if isinstance(step_settings, dict) and "enabled" in step_settings:
                is_enabled = step_settings.get("enabled", False)
                status = "✓" if is_enabled else "✗"
                message("info", f"{status} {step_name}")
                
    def _get_data_object(self, data: Union[mne.io.BaseRaw, mne.BaseEpochs, None],
                        use_epochs: bool = False) -> Union[mne.io.BaseRaw, mne.BaseEpochs]:
        """Get the appropriate data object based on the parameters.
        
        Args:
            data: Optional data object. If None, uses self.raw or self.epochs
            use_epochs: If True and data is None, uses self.epochs instead of self.raw
            
        Returns:
            The appropriate data object
            
        Raises:
            AttributeError: If self.raw or self.epochs doesn't exist when needed
        """
        if data is not None:
            return data
            
        if use_epochs:
            if not hasattr(self, 'epochs') or self.epochs is None:
                raise AttributeError("No epochs data available")
            return self.epochs
        else:
            if not hasattr(self, 'raw') or self.raw is None:
                raise AttributeError("No raw data available")
            return self.raw
            
    def _update_instance_data(self, data: Union[mne.io.BaseRaw, mne.BaseEpochs, None],
                             result_data: Union[mne.io.BaseRaw, mne.BaseEpochs],
                             use_epochs: bool = False) -> None:
        """Update the instance data attribute with the result data.
        
        Args:
            data: Original data object that was processed
            result_data: Result data object after processing
            use_epochs: If True, updates self.epochs instead of self.raw
        """
        if data is None:
            if use_epochs and hasattr(self, 'epochs'):
                self.epochs = result_data
            elif not use_epochs and hasattr(self, 'raw'):
                self.raw = result_data
        elif data is getattr(self, 'raw', None):
            self.raw = result_data
        elif data is getattr(self, 'epochs', None):
            self.epochs = result_data
    
    def _update_metadata(self, operation: str, metadata_dict: Dict[str, Any]) -> None:
        """Update the database with metadata about an operation.
        
        Args:
            operation: Name of the operation
            metadata_dict: Dictionary of metadata to store
        """
        if not hasattr(self, 'config') or not self.config.get("run_id"):
            return
            
        from datetime import datetime
        from autoclean.utils.database import manage_database
        
        # Add creation timestamp if not present
        if "creationDateTime" not in metadata_dict:
            metadata_dict["creationDateTime"] = datetime.now().isoformat()
            
        metadata = {operation: metadata_dict}
        
        run_id = self.config.get("run_id")
        manage_database(
            operation="update", update_record={"run_id": run_id, "metadata": metadata}
        )
        
    def _save_raw_result(self, result_data: mne.io.BaseRaw, stage_name: str) -> None:
        """Save the raw result data to a file.
        
        Args:
            result_data: Raw data to save
            stage_name: Name of the processing stage
        """
        if not hasattr(self, 'config'):
            return
            
        from autoclean.step_functions.io import save_raw_to_set
        
        if isinstance(result_data, mne.io.BaseRaw):
            save_raw_to_set(result_data, self.config, f"post_{stage_name}")
