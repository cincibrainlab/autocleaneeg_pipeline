"""Source localization mixin for autoclean tasks.

This module provides functionality for estimating cortical sources from sensor-space
EEG data using minimum norm estimation (MNE). Source localization projects EEG sensor
data to the cortical surface, enabling region-of-interest (ROI) analyses and
connectivity studies.

The SourceLocalizationMixin class implements methods for applying MNE source
localization to both continuous (Raw) and epoched (Epochs) EEG data. The method
uses the fsaverage template brain and an identity noise covariance matrix, making
it suitable for resting-state and task-based analyses.

Source localization is a fundamental step for:
- ROI-based power spectral density analysis
- Functional connectivity analysis
- Network analysis of brain dynamics
- Localization of evoked responses
"""

from pathlib import Path
from typing import Optional, Union

import mne

# Import algorithm functions from calc.source
from autoclean.calc.source import (
    estimate_source_function_epochs,
    estimate_source_function_raw,
    convert_stc_to_eeg,
    convert_stc_list_to_eeg,
)


class SourceLocalizationMixin:
    """Mixin class providing functionality for EEG source localization.

    This mixin provides methods for estimating cortical sources from sensor-space
    EEG data using minimum norm estimation (MNE). The implementation uses the
    fsaverage template brain with ico-5 source space (10,242 vertices) and an
    identity noise covariance matrix.

    The mixin supports both continuous (Raw) and epoched (Epochs) data, automatically
    detecting the input type and applying the appropriate source localization function.
    Results are stored as SourceEstimate objects (STCs) that can be used for downstream
    analyses such as ROI PSD calculation and connectivity analysis.

    The mixin respects configuration settings from the task config, allowing users to
    customize parameters and enable/disable the step.

    References
    ----------
    Hämäläinen MS & Ilmoniemi RJ (1994). Interpreting magnetic fields of the brain:
    minimum norm estimates. Medical & Biological Engineering & Computing, 32(1), 35-42.
    """

    def apply_source_localization(
        self,
        data: Union[mne.io.Raw, mne.Epochs, None] = None,
        method: str = "MNE",
        lambda2: float = 1.0 / 9.0,
        pick_ori: str = "normal",
        n_jobs: int = 10,
        stage_name: str = "apply_source_localization",
    ) -> Union[mne.SourceEstimate, list]:
        """Apply MNE source localization to estimate cortical sources.

        This method applies minimum norm estimation (MNE) to project sensor-space EEG
        data to the cortical surface. The method automatically detects whether the input
        is continuous (Raw) or epoched (Epochs) data and applies the appropriate
        source localization function.

        The method uses the fsaverage template brain with an identity noise covariance
        matrix, making it suitable for resting-state analyses without requiring
        subject-specific anatomical data or empty-room recordings.

        Source estimates are returned as SourceEstimate objects (STCs) containing:
        - data: (n_vertices x n_times) array of source activations
        - vertices: Vertex indices for left and right hemispheres
        - tmin: Start time
        - tstep: Time step (1/sfreq)

        Args:
            data: Optional Raw or Epochs object. If None, uses self.raw or self.epochs
            method: Source estimation method (default: "MNE", others: "dSPM", "sLORETA")
            lambda2: Regularization parameter (default: 1/9 = 0.111)
            pick_ori: Source orientation constraint (default: "normal")
            n_jobs: Number of parallel jobs for forward solution (default: 10)
            stage_name: Name for saving and metadata tracking

        Returns:
            SourceEstimate or list: Single STC for Raw input, list of STCs for Epochs

        Raises:
            AttributeError: If no input data found (no self.raw or self.epochs)
            TypeError: If input is not Raw or Epochs
            RuntimeError: If source localization fails
            ValueError: If fsaverage data is not available

        Example:
            ```python
            # Apply source localization with default parameters
            stc = task.apply_source_localization()

            # Apply with custom parameters
            stc_list = task.apply_source_localization(
                data=epochs,
                method="dSPM",
                lambda2=0.05,
                n_jobs=4
            )
            ```

        Notes
        -----
        - First run will download fsaverage data (~200MB) if not present
        - Source localization is computationally expensive (~2-5 min per subject)
        - Results are automatically saved to derivatives/source_localization/
        - For Epochs input, returns a list of STCs (one per epoch)
        - Sets EEG reference to average automatically
        """
        # Check if this step is enabled in the configuration
        config_value = None
        if hasattr(self, "_check_step_enabled"):
            is_enabled, config_value = self._check_step_enabled(
                "apply_source_localization"
            )

            if not is_enabled:
                if hasattr(self, "message"):
                    self.message("info", "Source localization step is disabled")
                else:
                    print("INFO: Source localization step is disabled")
                return None

            # Get parameters from config if available
            if config_value and isinstance(config_value, dict):
                method = config_value.get("method", method)
                lambda2 = config_value.get("lambda2", lambda2)
                pick_ori = config_value.get("pick_ori", pick_ori)
                n_jobs = config_value.get("n_jobs", n_jobs)

        # Determine which data to use
        if data is None:
            # Try to get data from task object
            if hasattr(self, "epochs") and self.epochs is not None:
                data = self.epochs
            elif hasattr(self, "raw") and self.raw is not None:
                data = self.raw
            else:
                raise AttributeError(
                    "No input data found. Provide data parameter or ensure "
                    "self.raw or self.epochs exists."
                )

        # Type checking
        is_raw = isinstance(data, mne.io.BaseRaw)
        is_epochs = isinstance(data, mne.BaseEpochs)

        if not (is_raw or is_epochs):
            raise TypeError(
                f"Data must be mne.io.Raw or mne.Epochs, got {type(data)}"
            )

        try:
            # Log start
            if hasattr(self, "message"):
                self.message("header", "Applying MNE source localization")
                self.message("info", f"Method: {method}, lambda2: {lambda2}")
                self.message(
                    "info",
                    f"Input type: {'Raw' if is_raw else 'Epochs'}, n_jobs: {n_jobs}",
                )
            else:
                print("=== Applying MNE Source Localization ===")
                print(f"Method: {method}, lambda2: {lambda2}")
                print(f"Input type: {'Raw' if is_raw else 'Epochs'}, n_jobs: {n_jobs}")

            # Prepare config dict for save_stc_to_file
            save_config = None
            if hasattr(self, "config"):
                save_config = self.config

            # Call appropriate algorithm function
            if is_raw:
                stc = estimate_source_function_raw(data, config=save_config)
                stc_type = "continuous"
                n_sources = stc.data.shape[0]
                n_times = stc.data.shape[1]
                duration = stc.times[-1] - stc.times[0]
            else:  # is_epochs
                stc_list = estimate_source_function_epochs(data, config=save_config)
                stc = stc_list  # For consistency
                stc_type = "epoched"
                n_sources = stc_list[0].data.shape[0]
                n_times = stc_list[0].data.shape[1]
                n_epochs = len(stc_list)
                epoch_duration = stc_list[0].times[-1] - stc_list[0].times[0]

            # Log completion
            if is_raw:
                if hasattr(self, "message"):
                    self.message(
                        "success",
                        f"Source localization complete: {n_sources} vertices, "
                        f"{duration:.1f}s duration",
                    )
                else:
                    print(
                        f"SUCCESS: Source localization complete: {n_sources} vertices, "
                        f"{duration:.1f}s duration"
                    )
            else:
                if hasattr(self, "message"):
                    self.message(
                        "success",
                        f"Source localization complete: {n_epochs} epochs, "
                        f"{n_sources} vertices, {epoch_duration:.1f}s per epoch",
                    )
                else:
                    print(
                        f"SUCCESS: Source localization complete: {n_epochs} epochs, "
                        f"{n_sources} vertices, {epoch_duration:.1f}s per epoch"
                    )

            # Update metadata
            if hasattr(self, "_update_metadata"):
                metadata = {
                    "method": method,
                    "lambda2": lambda2,
                    "pick_ori": pick_ori,
                    "n_jobs": n_jobs,
                    "stc_type": stc_type,
                    "n_vertices": n_sources,
                }

                if is_raw:
                    metadata.update(
                        {
                            "n_times": n_times,
                            "duration_sec": duration,
                            "sfreq": stc.sfreq,
                        }
                    )
                else:
                    metadata.update(
                        {
                            "n_epochs": n_epochs,
                            "n_times_per_epoch": n_times,
                            "epoch_duration_sec": epoch_duration,
                            "sfreq": stc_list[0].sfreq,
                        }
                    )

                self._update_metadata("step_apply_source_localization", metadata)

            # Store in task object for downstream use
            if is_raw:
                self.stc = stc
            else:
                self.stc_list = stc

            # Optional: Convert STC to EEG format for BIDS derivatives
            convert_to_eeg = False
            if config_value and isinstance(config_value, dict):
                convert_to_eeg = config_value.get("convert_to_eeg", False)

            if convert_to_eeg:
                try:
                    # Set up output directory
                    if hasattr(self, "config") and "derivatives_dir" in self.config:
                        base_dir = Path(self.config["derivatives_dir"])
                    elif hasattr(self, "file_path"):
                        base_dir = Path(self.file_path).parent / "derivatives"
                    else:
                        base_dir = Path.cwd() / "derivatives"

                    output_dir = base_dir / "source_localization_eeg"
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Get subject ID
                    subject_id = "unknown"
                    if hasattr(self, "config") and "subject_id" in self.config:
                        subject_id = self.config["subject_id"]
                    elif hasattr(self, "file_path"):
                        subject_id = Path(self.file_path).stem

                    if hasattr(self, "message"):
                        self.message("info", "Converting source estimates to EEG format (68 ROI channels)...")

                    if is_raw:
                        raw_eeg, eeg_file = convert_stc_to_eeg(
                            stc=stc,
                            subject="fsaverage",
                            subjects_dir=None,
                            output_dir=str(output_dir),
                            subject_id=subject_id
                        )
                        self.source_eeg = raw_eeg
                        self.source_eeg_file = eeg_file

                        if hasattr(self, "message"):
                            self.message("success", f"Converted to EEG: {eeg_file}")
                    else:
                        epochs_eeg, eeg_file = convert_stc_list_to_eeg(
                            stc_list=stc,
                            subject="fsaverage",
                            subjects_dir=None,
                            output_dir=str(output_dir),
                            subject_id=subject_id,
                            events=None,
                            event_id=None
                        )
                        self.source_eeg = epochs_eeg
                        self.source_eeg_file = eeg_file

                        if hasattr(self, "message"):
                            self.message("success", f"Converted {len(stc)} epochs to EEG: {eeg_file}")

                    # Update metadata
                    if hasattr(self, "_update_metadata"):
                        conversion_metadata = {
                            "converted_to_eeg": True,
                            "n_roi_channels": 68,
                            "atlas": "Desikan-Killiany (aparc)",
                            "eeg_file": eeg_file,
                        }
                        self._update_metadata("step_source_eeg_conversion", conversion_metadata)

                except Exception as conv_error:
                    error_msg = f"Warning: STC→EEG conversion failed: {str(conv_error)}"
                    if hasattr(self, "message"):
                        self.message("warning", error_msg)
                    else:
                        print(f"WARNING: {error_msg}")
                    # Don't fail the whole step if conversion fails

            return stc

        except Exception as e:
            error_msg = f"Error during source localization: {str(e)}"
            if hasattr(self, "message"):
                self.message("error", error_msg)
            else:
                print(f"ERROR: {error_msg}")
            raise RuntimeError(f"Failed to apply source localization: {str(e)}") from e