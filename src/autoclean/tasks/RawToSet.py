# Standard library imports
from pathlib import Path
from typing import Any, Dict, Optional

# Third-party imports
import mne

# Local imports
from autoclean.core.task import Task
from autoclean.step_functions.io import save_raw_to_set, import_eeg


class RawToSet(Task):

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize a new task instance.

        Args:
            config: Configuration dictionary containing all settings.
                   See class docstring for configuration example.

        Note:
            The parent class handles basic initialization and validation.
            Task-specific setup should be added here if needed.
            
        Raises:
            ValueError: If required configuration fields are missing
            TypeError: If configuration fields have incorrect types
        """
        # Initialize instance variables
        self.raw: Optional[mne.io.Raw] = None
        self.pipeline: Optional[Any] = None
        self.epochs: Optional[mne.Epochs] = None
        self.original_raw: Optional[mne.io.Raw] = None

        # Call parent initialization with validated config
        super().__init__(config)

    def run(self) -> None:
        # Import raw data using standard function
        self.raw = import_eeg(self.config)

        # Save imported data if configured
        save_raw_to_set(self.raw, self.config, "post_import")

        exit 

    def _validate_task_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate task-specific configuration settings.

        This method checks that all required settings for the task
        are present and valid. The validation includes:
        - Required fields exist
        - Field types are correct
        - Stage files configuration is properly structured
        
        Args:
            config: Configuration dictionary that has passed common validation.
                   Contains all standard fields plus task-specific settings.

        Returns:
            Dict[str, Any]: The validated configuration dictionary.
                           You can add derived settings or defaults here.

        Raises:
            ValueError: If any required settings are missing or invalid
            TypeError: If settings are of wrong type

        Example:
            ```python
            def _validate_task_config(self, config):
                required_fields = {
                    'eeg_system': str,
                    'settings': dict,
                }
                for field, field_type in required_fields.items():
                    if field not in config:
                        raise ValueError(f"Missing required field: {field}")
                    if not isinstance(config[field], field_type):
                        raise TypeError(f"Field {field} must be {field_type}")
                return config
            ```
        """
        # Define required fields and their expected types
        required_fields = {
            "task": str,
            "eeg_system": str,
            "tasks": dict,
        }

        # Validate required fields exist and have correct types
        for field, field_type in required_fields.items():
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
            if not isinstance(config[field], field_type):
                raise TypeError(f"Field {field} must be {field_type}")

        # Validate stage_files structure
        # These are the processing stages where data can be saved
        required_stages = [
            "post_import",
        ]

        # Check each required stage has proper configuration
        for stage in required_stages:
            if stage not in config["stage_files"]:
                raise ValueError(f"Missing stage in stage_files: {stage}")
            stage_config = config["stage_files"][stage]
            if not isinstance(stage_config, dict):
                raise ValueError(f"Stage {stage} configuration must be a dictionary")
            if "enabled" not in stage_config:
                raise ValueError(f"Stage {stage} must have 'enabled' field")
            if "suffix" not in stage_config:
                raise ValueError(f"Stage {stage} must have 'suffix' field")

        return config
        