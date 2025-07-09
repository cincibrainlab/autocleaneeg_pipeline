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

import inspect
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import mne
from matplotlib import pyplot as plt

from autoclean.io import save_epochs_to_set, save_raw_to_set
from autoclean.utils.database import manage_database_conditionally
from autoclean.utils.logging import message


class BaseMixin:
    """Base mixin class providing signal processing functionality for EEG data.

    This mixin serves as the foundation for all mixins in the
    AutoClean pipeline. It provides utility methods for configuration management,
    data access, and metadata tracking that are shared across all mixins.

    The BaseMixin is designed to be used with Task classes through multiple
    inheritance, allowing tasks to gain signal processing capabilities while maintaining
    a clean separation of concerns. Specialized mixins inherit from
    this base class and extend it with specific functionality.

    Note:
        This class expects to be mixed in with a class that provides access to
        configuration settings via the `config` attribute and data objects via
        the `raw` and/or `epochs` attributes.
    """

    # FLAG CRITERIA
    EPOCH_RETENTION_THRESHOLD = 0.5  # Flag if less than 50% of epochs are kept
    # Flag if more than 5 reference artifacts are detected
    REFERENCE_ARTIFACT_THRESHOLD = 5
    BAD_CHANNEL_THRESHOLD = 0.15  # Flag if more than 15% of channels are bad

    def _check_step_enabled(self, step_name: str) -> Tuple[bool, Optional[Any]]:
        """Check if a processing step is enabled in the configuration.

        This method examines the task configuration to determine if a specific
        processing step is enabled and retrieves its configuration value if available.
        It first checks self.settings (Python task files), then falls back to
        YAML configuration for backward compatibility.

        Args:
            step_name: Name of the step to check in the configuration

        Returns:
            Tuple of (is_enabled, value) where is_enabled is a boolean indicating
            if the step is enabled, and value is the configuration value for the step
            if it exists, or None otherwise

        Example:
            ```python
            # Check if resampling is enabled
            is_enabled, config_value = self._check_step_enabled("resample")
            if not is_enabled:
                return data  # Skip processing if disabled

            # Use configuration value if available
            target_sfreq = config_value.get("value", 250)  # Default to 250 Hz
            ```
        """
        # Priority 1: Check self.settings (Python task files)
        if hasattr(self, "settings") and self.settings is not None:
            step_settings = self.settings.get(step_name, {})
            if step_settings:  # If step exists in settings
                is_enabled = step_settings.get("enabled", False)

                # Create a copy of step_settings without the 'enabled' key
                settings_copy = step_settings.copy()
                if "enabled" in settings_copy:
                    settings_copy.pop("enabled")

                return is_enabled, settings_copy

        # Priority 2: Fall back to YAML config (backward compatibility)
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
        disabled (✗). It checks both self.settings (Python tasks) and YAML config.

        Example output:
        ```
        Processing Steps Status for Task: MyRestingTask
        ✓ resample: value=250
        ✗ drop_outerlayer: disabled
        ✓ reference_step: value=average
        ```

        Returns:
            None
        """
        task_name = "Unknown"
        if hasattr(self, "config") and self.config.get("task"):
            task_name = self.config.get("task")
        elif hasattr(self, "__class__"):
            task_name = self.__class__.__name__

        message("header", f"Processing step status for task '{task_name}':")

        # Priority 1: Report from self.settings (Python task files)
        if hasattr(self, "settings") and self.settings is not None:
            for step_name, step_settings in self.settings.items():
                if isinstance(step_settings, dict) and "enabled" in step_settings:
                    is_enabled = step_settings.get("enabled", False)
                    status = "✓" if is_enabled else "✗"
                    message("info", f"{status} {step_name}")
            return

        # Priority 2: Fall back to YAML config
        if not hasattr(self, "config"):
            message("info", "No configuration available")
            return

        task = self.config.get("task")
        if not task:
            message("info", "No task specified in configuration")
            return

        settings = self.config.get("tasks", {}).get(task, {}).get("settings", {})

        if not settings:
            message("info", "No step settings found in configuration")
            return

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

        # Add creation timestamp if not present
        if "creationDateTime" not in metadata_dict:
            metadata_dict["creationDateTime"] = datetime.now().isoformat()

        metadata = {operation: metadata_dict}

        run_id = self.config.get("run_id")
        manage_database_conditionally(
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

        if isinstance(result_data, mne.io.base.BaseRaw):
            save_raw_to_set(
                raw=result_data,
                autoclean_dict=self.config,
                stage=stage_name,
                flagged=self.flagged,
            )

    def _auto_export_if_enabled(
        self,
        data: Union[mne.io.Raw, mne.Epochs],
        stage_name: str,
        export_enabled: bool = False,
    ) -> None:
        """Automatically export data if export is enabled.

        Args:
            data: The data to export (Raw or Epochs)
            stage_name: Name of the processing stage for export
            export_enabled: Whether export is enabled for this call
        """
        if not export_enabled:
            return

        if not hasattr(self, "config"):
            message("warning", f"Cannot export {stage_name}: no config available")
            return

        # Ensure stage exists in config, create if needed
        self._ensure_stage_exists(stage_name)

        try:
            if isinstance(data, mne.io.base.BaseRaw):
                self._save_raw_result(data, stage_name)
            elif isinstance(data, mne.Epochs):
                self._save_epochs_result(data, stage_name)
            else:
                message(
                    "warning",
                    f"Cannot export {stage_name}: unsupported data type {type(data)}",
                )
        except Exception as e:
            message("error", f"Failed to export {stage_name}: {str(e)}")

    def _ensure_stage_exists(self, stage_name: str) -> None:
        """No longer needed - stages are handled by export functions.

        Args:
            stage_name: Name of the stage to ensure exists
        """
        # No action needed - export functions handle stage creation automatically
        pass

    def _generate_stage_name(self, method_name: str) -> str:
        """Generate appropriate stage name from method name.

        Args:
            method_name: Name of the method being called

        Returns:
            Generated stage name (e.g., "post_basic_steps")
        """
        # Map common method names to stage names
        method_to_stage = {
            "run_basic_steps": "post_basic_steps",
            "run_ica": "post_ica",
            "create_regular_epochs": "post_epochs",
            "create_epochs": "post_epochs",
            "filter_data": "post_filter",
            "resample_data": "post_resample",
            "rereference_data": "post_reference",
            "clean_bad_channels": "post_channel_cleaning",
            "gfp_clean_epochs": "post_gfp_cleaning",
            "detect_outlier_epochs": "post_epochs_prep",
        }

        return method_to_stage.get(method_name, f"post_{method_name}")

    def _get_current_method_name(self) -> str:
        """Get the name of the calling method for automatic stage naming.

        Returns:
            Name of the method that called this function
        """
        frame = inspect.currentframe()
        try:
            # Go up the call stack to find the calling method
            # Skip: _get_current_method_name -> _auto_export_if_enabled -> actual_method
            caller_frame = frame.f_back.f_back.f_back
            if caller_frame:
                return caller_frame.f_code.co_name
            return "unknown_method"
        finally:
            del frame

    def _save_epochs_result(self, result_data: mne.Epochs, stage_name: str) -> None:
        """Save the epochs result data to a file.

        Args:
            result_data: Epochs data to save
            stage_name: Name of the processing stage
        """
        if not hasattr(self, "config"):
            return

        if isinstance(
            result_data, mne.Epochs
        ):  # pylint: disable=isinstance-second-argument-not-valid-type
            save_epochs_to_set(
                epochs=result_data,
                autoclean_dict=self.config,
                stage=stage_name,
                flagged=self.flagged,
            )

    def _get_derivatives_path(self) -> Path:
        """Get the derivatives path for saving reports and visualizations.

        Args:
            bids_path: Optional BIDSPath object. If None, attempts to get from pipeline or config

        Returns:
            Path object pointing to the derivatives directory

        Raises:
            ValueError: If derivatives path cannot be determined
        """
        # TODO: Add backup try with bids_paths
        if self.config:
            if "derivatives_dir" in self.config:
                return Path(self.config["derivatives_dir"])

        raise ValueError("Could not determine derivatives path")

    def _save_figure(self, fig: plt.Figure, filename: str, dpi: int = 300) -> str:
        """Save a matplotlib figure to the derivatives directory.

        Args:
            fig: Matplotlib figure to save
            filename: Base filename (without path or extension)
            dpi: Resolution for saving the figure

        Returns:
            Full path to the saved figure
        """
        try:
            derivatives_path = self._get_derivatives_path()
            figure_path = derivatives_path / f"{filename}.png"

            # Save figure
            fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

            message("info", f"Figure saved to {figure_path}")
            return str(figure_path)

        except Exception as e:  # pylint: disable=broad-exception-caught
            message("error", f"Error saving figure: {str(e)}")
            return ""
