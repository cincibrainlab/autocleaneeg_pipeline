"""AutoReject epochs cleaning mixin for autoclean tasks.

This module provides functionality for cleaning epochs using AutoReject, a machine
learning-based method for automatic artifact rejection in EEG data. AutoReject
automatically identifies and removes bad epochs and interpolates bad channels
within epochs.

The AutoRejectEpochsMixin class implements methods for applying the AutoReject
algorithm to epoched EEG data, with configurable parameters for controlling the
behavior of the algorithm. It also provides functionality for tracking rejection
statistics and saving intermediate results.

AutoReject is particularly useful for automated preprocessing pipelines, as it
reduces the need for manual inspection and rejection of artifacts, while maintaining
the quality of the data for subsequent analysis.
"""

from pathlib import Path
from typing import List, Optional, Union

import mne
from autoreject import AutoReject

from autoclean.functions.advanced.autoreject_reporting import generate_autoreject_report
from autoclean.utils.logging import message


class AutoRejectEpochsMixin:
    """Mixin class providing functionality to clean epochs using AutoReject.

    This mixin provides methods for cleaning epochs using AutoReject, a machine
    learning-based method for automatic artifact rejection in EEG data. AutoReject
    automatically identifies and removes bad epochs and interpolates bad channels
    within epochs.

    The cleaning process involves training a model to identify bad channels and epochs
    based on statistical properties of the data, and then applying this model to clean
    the data. The mixin provides configurable parameters for controlling the behavior
    of the AutoReject algorithm, such as the number of channels to interpolate and the
    consensus threshold.

    The mixin respects configuration settings from the autoclean_config.yaml file,
    allowing users to customize the AutoReject parameters and enable/disable the step.
    """

    def apply_autoreject(
        self,
        epochs: Union[mne.Epochs, None] = None,
        n_interpolate: Optional[List[int]] = None,
        consensus: Optional[List[float]] = None,
        n_jobs: int = 1,
        stage_name: str = "apply_autoreject",
    ) -> mne.Epochs:
        """Apply AutoReject to clean epochs by removing artifacts and interpolating bad channels.

        This method applies the AutoReject algorithm to clean epochs by identifying and
        removing bad epochs and interpolating bad channels within epochs. AutoReject is a
        machine learning-based method that automatically determines optimal thresholds for
        artifact rejection, reducing the need for manual inspection.

        The method uses a cross-validation approach to determine the optimal parameters
        for artifact rejection, including the number of channels to interpolate and the
        consensus threshold. These parameters can be customized through the method arguments
        or the configuration file.

        The method requires the autoreject package to be installed. If it's not installed,
        an ImportError will be raised with instructions for installation.

        Args:
            epochs: Optional MNE Epochs object. If None, uses self.epochs
            n_interpolate: List of number of channels to interpolate. If None, uses default values
            consensus: List of consensus percentages. If None, uses default values
            n_jobs: Number of parallel jobs to run (default: 1)
            stage_name: Name for saving and metadata tracking

        Returns:
            mne.Epochs: The cleaned epochs object with bad epochs removed
            and bad channels interpolated

        Raises:
            AttributeError: If self.epochs doesn't exist when needed
            TypeError: If epochs is not an Epochs object
            RuntimeError: If AutoReject fails

        Example:
            ```python
            # Apply AutoReject with default parameters
            clean_epochs = task.apply_autoreject()

            # Apply AutoReject with custom parameters
            clean_epochs = task.apply_autoreject(
                n_interpolate=[1, 4, 8],
                consensus=[0.1, 0.25, 0.5, 0.75, 0.9],
                n_jobs=4
            )
            ```
        """
        # Check if this step is enabled in the configuration
        is_enabled, config_value = self._check_step_enabled("apply_autoreject")

        if not is_enabled:
            message("info", "AutoReject step is disabled in configuration")
            return None

        # Get parameters from config if available
        if config_value and isinstance(config_value, dict):
            n_interpolate = config_value.get("n_interpolate", n_interpolate)
            consensus = config_value.get("consensus", consensus)
            n_jobs = config_value.get("n_jobs", n_jobs)

        # Determine which data to use
        if epochs is None:
            epochs = self.epochs

        # Type checking
        if not isinstance(
            epochs, mne.Epochs
        ):  # pylint: disable=isinstance-second-argument-not-valid-type
            raise TypeError("Data must be an MNE Epochs object for AutoReject")

        try:
            message("header", "Applying AutoReject for artifact rejection")

            # Store copy of original epochs for reporting
            epochs_before = epochs.copy()

            # Create AutoReject object with parameters if provided
            if n_interpolate is not None and consensus is not None:
                ar = AutoReject(
                    n_interpolate=n_interpolate, consensus=consensus, n_jobs=n_jobs
                )
            else:
                ar = AutoReject(n_jobs=n_jobs)

            # Fit and transform epochs
            epochs_clean, rejection_log = ar.fit_transform(epochs, return_log=True)

            # Calculate statistics
            rejected_epochs = len(epochs) - len(epochs_clean)
            rejection_percent = (
                round((rejected_epochs / len(epochs)) * 100, 2)
                if len(epochs) > 0
                else 0
            )

            message(
                "info",
                f"Artifacts rejected: {rejected_epochs} epochs removed by "
                f"AutoReject ({rejection_percent}%)",
            )

            # Get block info for reproducibility
            block_info = self._get_block_info("autoreject")

            # Update metadata
            metadata = {
                "block_name": "autoreject",
                "initial_epochs": len(epochs),
                "final_epochs": len(epochs_clean),
                "rejected_epochs": rejected_epochs,
                "rejection_percent": rejection_percent,
                "epoch_duration": epochs.times[-1] - epochs.times[0],
                "samples_per_epoch": epochs.times.shape[0],
                "total_duration_sec": (epochs.times[-1] - epochs.times[0])
                * len(epochs_clean),
                "total_samples": epochs.times.shape[0] * len(epochs_clean),
                "channel_count": len(epochs.ch_names),
                "n_interpolate": n_interpolate,
                "consensus": consensus,
                "n_jobs": n_jobs,
            }

            # Add block version/commit info for reproducibility
            if block_info:
                metadata.update(block_info)

            self._update_metadata("step_apply_autoreject", metadata)

            # Store epochs
            if hasattr(self, "config") and self.config.get("run_id"):
                self.epochs = epochs_clean

            # Save epochs
            self._save_epochs_result(epochs_clean, stage_name)

            # Generate comprehensive PDF report
            report_path: Optional[Path] = None
            report_relative: Optional[Path] = None
            try:
                # Determine report location
                base_name = getattr(self, "base_name", "autoreject_report")
                filename = f"{base_name}_autoreject.pdf"

                if hasattr(self, "_resolve_report_path"):
                    report_path = self._resolve_report_path("autoreject", filename)
                elif hasattr(self, "directories") and "derivatives" in self.directories:
                    derivatives_dir = Path(self.directories["derivatives"])
                    if not derivatives_dir.exists():
                        raise ValueError(
                            "No derivatives directory available for autoreject report"
                        )
                    report_path = derivatives_dir / filename

                if report_path:
                    # Generate report with rejection log
                    ar_params = {
                        "n_interpolate": n_interpolate,
                        "consensus": consensus,
                        "n_jobs": n_jobs,
                    }
                    _report_result = generate_autoreject_report(
                        epochs_before=epochs_before,
                        epochs_after=epochs_clean,
                        output_pdf=report_path,
                        rejection_log=(
                            rejection_log.labels
                            if hasattr(rejection_log, "labels")
                            else None
                        ),
                        ar_params=ar_params,
                    )

                    message("success", f"AutoReject report saved to {report_path}")

                    # Try to get relative path for metadata
                    if hasattr(self, "_report_relative_path"):
                        try:
                            report_relative = self._report_relative_path(report_path)
                        except Exception:
                            report_relative = None

            except Exception as exc:
                message("warning", f"AutoReject report generation skipped: {exc}")
                report_path = None
                report_relative = None

            # Update metadata with report path
            if report_path:
                metadata["report_path"] = str(report_relative or report_path)

            return epochs_clean

        except Exception as e:
            message("error", f"Error during AutoReject: {str(e)}")
            raise RuntimeError(f"Failed to apply AutoReject: {str(e)}") from e
