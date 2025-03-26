# Standard library imports
from abc import ABC, abstractmethod
from pathlib import Path
from struct import Struct
from typing import Any, Dict, Optional

# Third-party imports
import mne  # Core EEG processing library for data containers and processing

# Local imports
from autoclean.mixins.signal_processing.main import SignalProcessingMixin
from autoclean.mixins.reporting.main import ReportingMixin


class Task(ABC, SignalProcessingMixin, ReportingMixin):
    """Base class for all EEG processing tasks.

    This class defines the interface that all specific EEG tasks must implement.
    It provides the basic structure for:
    1. Loading and validating configuration
    2. Importing raw EEG data
    3. Running preprocessing steps
    4. Applying task-specific processing
    5. Saving results

    It should be inherited from to create new tasks in the autoclean.tasks module. 

    Notes
    -----
    Abstract base class that enforces a consistent interface across all EEG processing
    tasks through abstract methods and strict type checking. Manages state through
    MNE objects (Raw and Epochs) while maintaining processing history in a dictionary.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize a new task instance.

        Parameters
        ----------
        config : Dict[str, Any]
            A dictionary containing all configuration settings for the task.
            Must include:
            
            - run_id (str): Unique identifier for this processing run
            - unprocessed_file (Path): Path to the raw EEG data file
            - task (str): Name of the task (e.g., "rest_eyesopen")
            - tasks (dict): Task-specific settings
            - stage_files (dict): Configuration for saving intermediate results

        Raises
        ------
        ValueError
            If the configuration is missing required fields or contains invalid values.

        Examples
        --------
        >>> config = {
        ...     'run_id': '12345',
        ...     'unprocessed_file': Path('data/sub-01_task-rest_eeg.raw'),
        ...     'task': 'rest_eyesopen',
        ...     'tasks': {'rest_eyesopen': {...}},
        ...     'stage_files': {'post_import': {'enabled': True}}
        ... }
        >>> task = MyTask(config)
        """
        # Configuration must be validated first as other initializations depend on it
        self.config = self.validate_config(config)

        # Initialize MNE data containers to None
        # These will be populated during the processing pipeline
        self.raw: Optional[mne.io.Raw] = None  # Holds continuous EEG data
        self.epochs: Optional[mne.Epochs] = None  # Holds epoched data segments
        self.flagged = False
        self.flagged_reasons = []

    def import_raw(self) -> None:
        """Import the raw EEG data from file.
        
        Notes
        -----
        Imports data using the configured import function and flags files with
        duration less than 60 seconds. Saves the imported data as a post-import
        stage file.
        
        Raises
        ------
        RuntimeError
            If the import fails or the file cannot be read.
        """
        from autoclean.step_functions.io import import_eeg, save_raw_to_set
        self.raw = import_eeg(self.config)
        if self.raw.duration < 60:
            self.flagged = True
            self.flagged_reasons = [f"WARNING: Initial duration ({float(self.raw.duration):.1f}s) less than 1 minute"]
        save_raw_to_set(raw = self.raw, autoclean_dict = self.config, stage = "post_import", flagged = self.flagged)

    @abstractmethod
    def run(self) -> None:
        """Run the standard EEG preprocessing pipeline.

        Raises
        ------
        RuntimeError
            If no data has been imported (self.raw is None).
        ValueError
            If preprocessing parameters are invalid.
        RuntimeError
            If any preprocessing step fails.

        Notes
        -----
        Defines interface for MNE-based preprocessing operations including filtering,
        resampling, and artifact detection. Maintains processing state through
        self.raw modifications.
        
        The specific parameters for each preprocessing step should be
        defined in the task configuration and validated before use.
        """
        pass

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the complete task configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            The configuration dictionary to validate.
            See __init__ docstring for required fields.

        Returns
        -------
        Dict[str, Any]
            The validated configuration dictionary.
            May contain additional fields added during validation.

        Raises
        ------
        ValueError
            If any required fields are missing or invalid.
        TypeError
            If any fields are of the wrong type.

        Notes
        -----
        Implements two-stage validation pattern with base validation followed by
        task-specific checks. Uses type annotations and runtime checks to ensure
        configuration integrity before processing begins.

        Examples
        --------
        >>> config = {...}  # Your configuration dictionary
        >>> validated_config = task.validate_config(config)
        >>> print(f"Validation successful: {validated_config['task']}")
        """
        # Schema definition for base configuration requirements
        # All tasks must provide these fields with exact types
        required_fields = {
            "run_id": str,  # Unique identifier for tracking
            "unprocessed_file": Path,  # Input file path
            "task": str,  # Task identifier
            "tasks": dict,  # Task-specific settings
            "stage_files": dict,  # Intermediate file config
        }

        # Two-stage validation: first check existence, then type
        for field, field_type in required_fields.items():
            # Stage 1: Check field existence
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

            # Stage 2: Validate field type using isinstance for safety
            if not isinstance(config[field], field_type):
                raise TypeError(
                    f"Field '{field}' must be of type {field_type.__name__}, "
                    f"got {type(config[field]).__name__} instead"
                )

        # After base validation succeeds, delegate to task-specific validation
        # Using template method pattern for extensibility
        return self._validate_task_config(config)

    @abstractmethod
    def _validate_task_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate task-specific configuration settings.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary that has passed common validation.
            Contains all fields from validate_config plus task-specific settings.

        Returns
        -------
        Dict[str, Any]
            The validated configuration dictionary, potentially modified
            to include derived or default values.

        Raises
        ------
        ValueError
            If task-specific configuration is invalid.
        TypeError
            If task-specific fields are of wrong type.

        Notes
        -----
        This is an abstract method that must be implemented by all task classes.
        The implementation should validate all task-specific settings and parameters.

        Implementation Requirements:
        
        - Must validate all task-specific parameters
        - Should add any derived/computed values to config
        - Must maintain immutability of input config
        - Must implement type checking for all fields
        """
        pass


    def get_flagged_status(self) -> tuple[bool, list[str]]:
        """Get the flagged status of the task.

        Returns
        -------
        tuple of (bool, list of str)
            A tuple containing a boolean flag and a list of reasons for flagging.
        """
        return self.flagged, self.flagged_reasons
    
    def get_raw(self) -> Optional[mne.io.Raw]:
        """Get the raw data of the task.

        Returns
        -------
        mne.io.Raw
            The raw data of the task.
            
        Raises
        ------
        ValueError
            If raw data is not available.
        """
        if self.raw is None:
            raise ValueError("Raw data is not available.")
        return self.raw
    
    def get_epochs(self) -> Optional[mne.Epochs]:
        """Get the epochs of the task.

        Returns
        -------
        mne.Epochs
            The epochs of the task.
            
        Raises
        ------
        ValueError
            If epochs are not available.
        """
        if self.epochs is None:
            raise ValueError("Epochs are not available.")
        return self.epochs
    
    
