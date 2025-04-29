"""Base signal processing mixin for autoclean tasks.

This module provides the foundation for all signal processing mixins in the AutoClean
pipeline. It defines the base class that all specialized signal processing mixins
inherit from, providing common utility methods and a consistent interface for
working with EEG data.

The SignalProcessingMixin class is designed to be used as a mixin with Task classes,
providing them with signal processing capabilities while maintaining a clean separation
of concerns. This modular approach allows for flexible composition of processing
functionality across different task types.
"""

from typing import Any, Dict, Optional, Tuple, Union

import mne

from autoclean.utils.logging import message


class SignalProcessingMixin:
    """Base mixin class providing signal processing functionality for EEG data.

    This mixin serves as the foundation for all signal processing operations in the
    AutoClean pipeline. It provides utility methods for configuration management,
    data access, and metadata tracking that are shared across all signal processing
    operations.

    The SignalProcessingMixin is designed to be used with Task classes through multiple
    inheritance, allowing tasks to gain signal processing capabilities while maintaining
    a clean separation of concerns. Specialized signal processing mixins inherit from
    this base class and extend it with specific functionality.

    Attributes:
        config (Dict[str, Any]): Task configuration dictionary (provided by the parent class)
        raw (mne.io.Raw): MNE Raw object containing the EEG data (if available)
        epochs (mne.Epochs): MNE Epochs object containing epoched data (if available)
        metadata (Dict[str, Any]): Dictionary tracking processing metadata

    Note:
        This class expects to be mixed in with a class that provides access to
        configuration settings via the `config` attribute and data objects via
        the `raw` and/or `epochs` attributes.
    """

    # FLAG CRITERIA
    EPOCH_RETENTION_THRESHOLD = 0.5  # Flag if less than 50% of epochs are kept
    REFERENCE_ARTIFACT_THRESHOLD = (
        5  # Flag if more than 5 reference artifacts are detected
    )
    BAD_CHANNEL_THRESHOLD = 0.15  # Flag if more than 15% of channels are bad

    def _check_step_enabled(self, step_name: str) -> Tuple[bool, Optional[Any]]:
        """Check if a processing step is enabled in the configuration.

        This method examines the task configuration to determine if a specific
        processing step is enabled and retrieves its configuration value if available.
        It is used by signal processing methods to respect user configuration
        preferences and skip disabled steps.

        Args:
            step_name: Name of the step to check in the configuration

        Returns:
            Tuple of (is_enabled, value) where is_enabled is a boolean indicating
            if the step is enabled, and value is the configuration value for the step
            if it exists, or None otherwise

        Example:
            ```python
            # Check if resampling is enabled
            is_enabled, config_value = self._check_step_enabled("resample_step")
            if not is_enabled:
                return data  # Skip processing if disabled

            # Use configuration value if available
            target_sfreq = config_value.get("sfreq", 250)  # Default to 250 Hz
            ```
        """
        if not hasattr(self, "config"):
            return True, None

        task = self.config.get("task")
        if not task:
            return True, None

        settings = self.config.get("tasks", {}).get(task, {}).get("settings", {})
        step_settings = settings.get(step_name, {})

        is_enabled = step_settings.get("enabled", False)

        # Create a copy of step_settings without the 'enabled' key
        settings_copy = step_settings.copy()
        if "enabled" in settings_copy:
            settings_copy.pop("enabled")

        return is_enabled, settings_copy

    def _report_step_status(self) -> None:
        """Report the enabled/disabled status of all processing steps in the configuration.

        This method prints a formatted report of all processing steps defined in the
        task configuration, indicating which steps are enabled (✓) and which are
        disabled (✗). It provides a clear overview of the processing pipeline
        configuration at runtime.

        The report is organized by task and includes the step name, enabled status,
        and configuration value when available.

        Example output:
        ```
        Processing Steps Status for Task: resting_eyes_open
        ✓ resample_step: sfreq=250
        ✗ drop_outerlayer: disabled
        ✓ reference_step: type=average
        ```

        Returns:
            None
        """
        if not hasattr(self, "config"):
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

    def _get_data_object(
        self, data: Union[mne.io.Raw, mne.Epochs, None], use_epochs: bool = False
    ) -> Union[mne.io.Raw, mne.Epochs]:
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
            if not hasattr(self, "epochs") or self.epochs is None:
                raise AttributeError("No epochs data available")
            return self.epochs
        else:
            if not hasattr(self, "raw") or self.raw is None:
                raise AttributeError("No raw data available")
            return self.raw

    def _update_instance_data(
        self,
        data: Union[mne.io.Raw, mne.Epochs, None],
        result_data: Union[mne.io.Raw, mne.Epochs],
        use_epochs: bool = False,
    ) -> None:
        """Update the instance data attribute with the result data.

        Args:
            data: Original data object that was processed
            result_data: Result data object after processing
            use_epochs: If True, updates self.epochs instead of self.raw
        """
        if data is None:
            if use_epochs and hasattr(self, "epochs"):
                self.epochs = result_data
            elif not use_epochs and hasattr(self, "raw"):
                self.raw = result_data
        elif data is getattr(self, "raw", None):
            self.raw = result_data
        elif data is getattr(self, "epochs", None):
            self.epochs = result_data

    def _update_metadata(self, operation: str, metadata_dict: Dict[str, Any]) -> None:
        """Update the database with metadata about an operation.

        Args:
            operation: Name of the operation
            metadata_dict: Dictionary of metadata to store
        """
        if not hasattr(self, "config") or not self.config.get("run_id"):
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

    def _update_flagged_status(self, flagged: bool, reason: str) -> None:
        """Update the flagged status and reasons.

        Args:
            flagged: Boolean indicating if the data is flagged
            reason: Reason for flagging the data
        """
        if not hasattr(self, "flagged"):
            self.flagged = flagged
            self.flagged_reasons = [reason]
        else:
            self.flagged = flagged
            self.flagged_reasons.append(reason)

        message("warning", reason)

    def _save_raw_result(self, result_data: mne.io.Raw, stage_name: str) -> None:
        """Save the raw result data to a file.

        Args:
            result_data: Raw data to save
            stage_name: Name of the processing stage
        """
        if not hasattr(self, "config"):
            return

        from autoclean.io.export import save_raw_to_set

        if isinstance(result_data, mne.io.Raw):
            save_raw_to_set(
                raw=result_data,
                autoclean_dict=self.config,
                stage=stage_name,
                flagged=self.flagged,
            )

    def _save_epochs_result(self, result_data: mne.Epochs, stage_name: str) -> None:
        """Save the epochs result data to a file.

        Args:
            result_data: Epochs data to save
            stage_name: Name of the processing stage
        """
        if not hasattr(self, "config"):
            return

        from autoclean.io.export import save_epochs_to_set

        if isinstance(result_data, mne.Epochs):
            save_epochs_to_set(
                epochs=result_data,
                autoclean_dict=self.config,
                stage=stage_name,
                flagged=self.flagged,
            )
