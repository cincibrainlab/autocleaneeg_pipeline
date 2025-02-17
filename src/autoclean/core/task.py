# src/autoclean/core/task.py
"""Base class for all EEG processing tasks.

This class defines the interface that all specific EEG processing tasks must implement.
It provides a standardized structure for processing EEG data through several stages:

1. Configuration validation - Ensures all required settings are present and valid
2. Data import - Loads raw EEG data from files
3. Preprocessing - Applies standard EEG preprocessing steps
4. Task-specific processing - Implements specialized analysis for each paradigm
5. Results saving - Stores processed data and analysis results

Each task (like resting state or ASSR) inherits from this class and implements
these stages according to their specific requirements.

Example:
    To implement a new task type:

    >>> class MyNewTask(Task):
    ...     def __init__(self, config):
    ...         super().__init__(config)
    ...
    ...     def _validate_task_config(self, config):
    ...         # Add task-specific validation
    ...         return config
    ...
    ...     def import_data(self, file_path):
    ...         # Implement data import logic
    ...         pass
    ...
    ...     def preprocess(self):
    ...         # Implement preprocessing logic
    ...         pass
    ...
    ...     def process(self):
    ...         # Implement task-specific processing logic
    ...         pass
"""

# Standard library imports
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

# Third-party imports
import mne  # Core EEG processing library for data containers and processing


class Task(ABC):
    """Base class for all EEG processing tasks.

    This class defines the interface that all specific EEG tasks must implement.
    It provides the basic structure for:
    1. Loading and validating configuration
    2. Importing raw EEG data
    3. Running preprocessing steps
    4. Applying task-specific processing
    5. Saving results

    Abstract base class that enforces a consistent interface across all EEG processing
    tasks through abstract methods and strict type checking. Manages state through
    MNE objects (Raw and Epochs) while maintaining processing history in a dictionary.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize a new task instance.

        Establishes task state by validating configuration dictionary and initializing
        MNE data containers. Sets up processing history tracking and ensures type safety
        through strict validation.

        Args:
            config: A dictionary containing all configuration settings for the task.
                   Must include:
                   - run_id (str): Unique identifier for this processing run
                   - unprocessed_file (Path): Path to the raw EEG data file
                   - task (str): Name of the task (e.g., "rest_eyesopen")
                   - tasks (dict): Task-specific settings
                   - stage_files (dict): Configuration for saving intermediate results

        Raises:
            ValueError: If the configuration is missing required fields or contains invalid values.

        Example:
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

        # Dictionary to track processing history, metrics, and state changes
        self.pipeline_results: Dict[str, Any] = {}

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the complete task configuration.

        Implements two-stage validation pattern with base validation followed by
        task-specific checks. Uses type annotations and runtime checks to ensure
        configuration integrity before processing begins.

        Args:
            config: The configuration dictionary to validate.
                   See __init__ docstring for required fields.

        Returns:
            Dict[str, Any]: The validated configuration dictionary.
                           May contain additional fields added during validation.

        Raises:
            ValueError: If any required fields are missing or invalid.
            TypeError: If any fields are of the wrong type.

        Example:
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

        Template method that enforces task-specific validation while allowing
        flexible configuration requirements. Child classes must implement this
        to define their unique validation rules.

        Args:
            config: Configuration dictionary that has passed common validation.
                   Contains all fields from validate_config plus task-specific settings.

        Returns:
            Dict[str, Any]: The validated configuration dictionary, potentially modified
                           to include derived or default values.

        Raises:
            ValueError: If task-specific configuration is invalid.
            TypeError: If task-specific fields are of wrong type.

        Note:
            This is an abstract method that must be implemented by all task classes.
            The implementation should validate all task-specific settings and parameters.

        Implementation Requirements:
        - Must validate all task-specific parameters
        - Should add any derived/computed values to config
        - Must maintain immutability of input config
        - Must implement type checking for all fields
        """
        pass

    @abstractmethod
    def import_data(self, file_path: Path) -> None:
        """Import raw EEG data for processing.

        Defines interface for data import operations, ensuring consistent handling
        of MNE Raw objects across tasks. Implementations must handle file I/O
        and initial data validation.

        Args:
            file_path: Path to the EEG data file. The file format should match
                      what's expected by the specific task implementation.

        Raises:
            FileNotFoundError: If the specified file doesn't exist.
            ValueError: If the file format is invalid or data is corrupted.
            RuntimeError: If there are problems reading the file.

        Note:
            This method typically sets self.raw with the imported MNE Raw object.
            The exact format and preprocessing steps depend on the task type.
        """
        pass

    @abstractmethod
    def preprocess(self) -> None:
        """Run the standard EEG preprocessing pipeline.

        Defines interface for MNE-based preprocessing operations including filtering,
        resampling, and artifact detection. Maintains processing state through
        self.raw modifications.

        Raises:
            RuntimeError: If no data has been imported (self.raw is None)
            ValueError: If preprocessing parameters are invalid
            RuntimeError: If any preprocessing step fails

        Note:
            The specific parameters for each preprocessing step should be
            defined in the task configuration and validated before use.
        """
        pass

    @abstractmethod
    def process(self) -> None:
        """Run task-specific processing steps.

        Defines interface for specialized analysis operations, working with both
        continuous (Raw) and epoched data. Implementations should update
        pipeline_results with processing outcomes.

        Raises:
            RuntimeError: If preprocessing hasn't been completed
            ValueError: If processing parameters are invalid
            RuntimeError: If any processing step fails

        Note:
            The exact processing steps and parameters depend on the task type
            and should be defined in the task configuration.
        """
        pass
