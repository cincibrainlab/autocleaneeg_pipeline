"""Statistical Learning epochs creation mixin for autoclean tasks.

This module provides functionality for creating fixed-length epochs from
Statistical Learning continuous EEG data. Statistical Learning fixed-length epochs
are time segments of equal duration that are created at fixed intervals throughout the recording, based on event markers specific to the SL paradigm.

The StatisticalLearningEpochsMixin class implements methods for creating these epochs and
handling annotations, allowing users to either automatically reject epochs that
overlap with bad annotations or just mark them in the metadata for later processing.

This epoching is particularly useful for Statistical Learning data analysis, where
the data needs to be segmented into manageable chunks for further processing and analysis.

"""

from typing import Dict, Optional, Union

import mne
import numpy as np
import pandas as pd

from autoclean.functions.epoching.statistical import create_statistical_learning_epochs
from autoclean.utils.logging import message


class StatisticalLearningEpochsMixin:
    """Mixin class for creating syllable-based epochs (SL_epochs) from EEG data."""

    def create_sl_epochs(
        self,
        data: Union[mne.io.Raw, None] = None,
        tmin: float = 0,
        num_syllables: int = 30,
        volt_threshold: Optional[Dict[str, float]] = None,
        stage_name: str = "post_epochs",
        reject_by_annotation: bool = False,
        subject_id: Optional[str] = None,
        baseline: Optional[bool] = True,
    ) -> mne.Epochs:
        """Create syllable-based epochs (SL_epochs) from raw EEG data.

        This is a mixin wrapper that handles configuration, metadata tracking,
        and result saving. The core epoching logic is implemented in the
        standalone function create_statistical_learning_epochs().

        Parameters
        ----------
        data : mne.io.Raw, Optional
            The raw EEG data. If None, uses self.raw.
        tmin : float, Optional
            Start time of the epoch in seconds. Default is 0.
        num_syllables : int, Optional
            Number of syllables per epoch. Default is 30.
        volt_threshold : dict, Optional
            Dictionary of channel types and thresholds for rejection. Default is None.
        stage_name : str, Optional
            Name for saving and metadata tracking. Default is "post_epochs".
        reject_by_annotation : bool, Optional
            Whether to reject epochs overlapping bad annotations or mark them in metadata. Default is False.
        subject_id : str, Optional
            Subject ID to handle specific event codes (e.g., for subject 2310). Default is None.
        baseline : bool, Optional
            Whether to apply baseline correction. Default is True.

        Returns
        -------
        epochs_clean : mne.Epochs
            The created epochs object with bad epochs marked (and dropped if reject_by_annotation=True).
        """
        # Check if this step is enabled in the configuration
        is_enabled, config_value = self._check_step_enabled("epoch_settings")

        if not is_enabled:
            message("info", "SL epoch creation step is disabled in configuration")
            return None

        # Get parameters from config if available
        if config_value and isinstance(config_value, dict):
            epoch_value = config_value.get("value", {})
            if isinstance(epoch_value, dict):
                tmin = epoch_value.get("tmin", tmin)
                num_syllables = epoch_value.get("num_syllables", num_syllables)

            threshold_settings = config_value.get("threshold_rejection", {})
            if isinstance(threshold_settings, dict) and threshold_settings.get(
                "enabled", False
            ):
                threshold_config = threshold_settings.get("volt_threshold", {})
                if isinstance(threshold_config, (int, float)):
                    volt_threshold = {"eeg": float(threshold_config)}
                elif isinstance(threshold_config, dict):
                    volt_threshold = {k: float(v) for k, v in threshold_config.items()}

        # Determine which data to use
        data = self._get_data_object(data)
        if not isinstance(data, (mne.io.Raw, mne.io.base.BaseRaw)):
            raise TypeError("Data must be an MNE Raw object for SL epoch creation")

        try:
            # Call the standalone epoching function with verbose=False (use autoclean logging)
            message(
                "header",
                f"Creating statistical learning epochs with {num_syllables} syllables...",
            )
            epochs, epochs_dropped = create_statistical_learning_epochs(
                data=data,
                tmin=tmin,
                num_syllables=num_syllables,
                volt_threshold=volt_threshold,
                reject_by_annotation=reject_by_annotation,
                subject_id=subject_id,
                baseline=baseline,
                verbose=True,  # Use autoclean logging instead
            )

            self._save_epochs_result(result_data=epochs, stage_name=stage_name)
            if epochs_dropped is not None:
                self._save_epochs_result(
                    result_data=epochs_dropped, stage_name="post_drop_bad_epochs"
                )
            else:
                epochs_dropped = epochs

            self.epochs = epochs_dropped

            # Analyze drop log to tally different annotation types
            tmax = num_syllables * 0.3
            drop_log = epochs_dropped.drop_log
            total_epochs = len(drop_log)
            good_epochs = sum(1 for log in drop_log if len(log) == 0)

            # Dynamically collect all unique annotation types
            annotation_types = {}
            for log in drop_log:
                if len(log) > 0:  # If epoch was dropped
                    for annotation in log:
                        # Convert numpy string to regular string if needed
                        annotation = str(annotation)
                        annotation_types[annotation] = (
                            annotation_types.get(annotation, 0) + 1
                        )

            message("info", "\nEpoch Drop Log Summary:")
            message("info", f"Total epochs: {total_epochs}")
            message("info", f"Good epochs: {good_epochs}")
            for annotation, count in annotation_types.items():
                message("info", f"Epochs with {annotation}: {count}")

            # Flag low retention
            if (good_epochs / total_epochs) < self.EPOCH_RETENTION_THRESHOLD:
                flagged_reason = f"WARNING: Only {good_epochs / total_epochs * 100:.1f}% of epochs were kept"
                self._update_flagged_status(flagged=True, reason=flagged_reason)

            # Update metadata
            annotation_types["KEEP"] = good_epochs
            annotation_types["TOTAL"] = total_epochs
            metadata = {
                "duration": tmax - tmin,
                "reject_by_annotation": reject_by_annotation,
                "initial_epoch_count": total_epochs,  # Approximation since we don't have pre-drop count
                "final_epoch_count": good_epochs,
                "single_epoch_duration": epochs.times[-1] - epochs.times[0],
                "single_epoch_samples": epochs.times.shape[0],
                "initial_duration": (epochs.times[-1] - epochs.times[0]) * good_epochs,
                "numberSamples": epochs.times.shape[0] * good_epochs,
                "channelCount": len(epochs.ch_names),
                "annotation_types": annotation_types,
                "marked_epochs_file": stage_name,
                "cleaned_epochs_file": "post_drop_bad_sl_epochs",
                "tmin": tmin,
                "tmax": tmax,
                "num_syllables": num_syllables,
            }
            self._update_metadata("step_create_sl_epochs", metadata)

            return epochs_dropped

        except Exception as e:
            message("error", f"Error during SL epoch creation: {str(e)}")
            raise RuntimeError(f"Failed to create SL epochs: {str(e)}") from e
