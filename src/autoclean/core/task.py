"""Base class for all EEG processing tasks."""

# Standard library imports
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

# Third-party imports
import mne  # Core EEG processing library for data containers and processing

from autoclean.io.export import save_epochs_to_set, save_raw_to_set
from autoclean.io.import_ import import_eeg

# Local imports
try:
    from autoclean.mixins import DISCOVERED_MIXINS
    if not DISCOVERED_MIXINS:
        print("🚨 CRITICAL ERROR: DISCOVERED_MIXINS is empty!")
        print("Task class will be missing all mixin functionality!")
        print("Check autoclean.mixins package for import errors.")
        # Create a minimal fallback
        class _EmptyMixinFallback:
            def __getattr__(self, name):
                raise AttributeError(
                    f"Method '{name}' not available - mixin discovery failed. "
                    f"Check autoclean.mixins package for import errors."
                )
        DISCOVERED_MIXINS = (_EmptyMixinFallback,)
except ImportError as e:
    print("🚨 CRITICAL ERROR: Could not import DISCOVERED_MIXINS!")
    print(f"Import error: {e}")
    print("Task class will be missing all mixin functionality!")
    # Create a minimal fallback
    class _ImportErrorMixinFallback:
        def __getattr__(self, name):
            raise AttributeError(
                f"Method '{name}' not available - mixin import failed: {e}"
            )
    DISCOVERED_MIXINS = (_ImportErrorMixinFallback,)


class Task(ABC, *DISCOVERED_MIXINS):
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

            The base class automatically detects a module-level 'config' variable
            and uses it for self.settings in Python-based tasks.

        Examples
        --------
        >>> # Python task file approach - no __init__ needed!
        >>> config = {'resample': {'enabled': True, 'value': 250}}
        >>> class MyTask(Task):
        ...     def run(self):
        ...         self.import_raw()
        ...         # Processing steps here
        """
        # Auto-detect module-level config for Python tasks
        if not hasattr(self, "settings"):
            # Get the module where this class was defined
            import inspect

            module = inspect.getmodule(self.__class__)
            if module and hasattr(module, "config"):
                self.settings = module.config
            else:
                self.settings = None

        # Extract EEG system from task settings before validation
        config['eeg_system'] = self._extract_eeg_system()
        
        # Configuration must be validated first as other initializations depend on it
        self.config = self.validate_config(config)

        # Initialize MNE data containers to None
        # These will be populated during the processing pipeline
        self.raw: Optional[mne.io.Raw] = None  # Holds continuous EEG data
        self.original_raw: Optional[mne.io.Raw] = None
        self.epochs: Optional[mne.Epochs] = None  # Holds epoched data segments
        self.flagged = False
        self.flagged_reasons = []
        self.fast_ica: Optional[mne.ICA] = None
        self.final_ica: Optional[mne.ICA] = None
        self.ica_flags = None

    def _extract_eeg_system(self) -> str:
        """Extract EEG system/montage from task settings.
        
        Returns
        -------
        str
            The montage name from task config, or "auto" as fallback
        """
        if (self.settings and 
            'montage' in self.settings and 
            self.settings['montage'].get('enabled', False)):
            return self.settings['montage']['value']
        return "auto"

    def import_raw(self) -> None:
        """Import the raw EEG data from file.

        Notes
        -----
        Imports data using the configured import function and flags files with
        duration less than 60 seconds. Saves the imported data as a post-import
        stage file.

        """

        self.raw = import_eeg(self.config)
        if self.raw.duration < 60:
            self.flagged = True
            self.flagged_reasons = [
                f"WARNING: Initial duration ({float(self.raw.duration):.1f}s) less than 1 minute"
            ]
        save_raw_to_set(
            raw=self.raw,
            autoclean_dict=self.config,
            stage="post_import",
            flagged=self.flagged,
        )

    def import_epochs(self) -> None:
        """Import the epochs from file.

        Notes
        -----
        Imports data using the configured import function and saves the imported
        data as a post-import stage file.

        """

        self.epochs = import_eeg(self.config)

        save_epochs_to_set(
            epochs=self.epochs,
            autoclean_dict=self.config,
            stage="post_import",
            flagged=self.flagged,
        )

    @abstractmethod
    def run(self) -> None:
        """Run the standard EEG preprocessing pipeline.

        Notes
        -----
        Defines interface for MNE-based preprocessing operations including filtering,
        resampling, and artifact detection. Maintains processing state through
        self.raw modifications.

        The specific parameters for each preprocessing step should be
        defined in the task configuration and validated before use.
        """

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

        # No longer validate required_stages - stages are created dynamically when export=True is used

        return config

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

        """
        if self.epochs is None:
            raise ValueError("Epochs are not available.")
        return self.epochs
