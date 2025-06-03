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
from autoclean.utils.logging import message


class StatisticalLearningEpochsMixin:
    """Mixin class for creating syllable-based epochs (SL_epochs) from EEG data."""

    def create_sl_epochs(
        self,
        data: Union[mne.io.Raw, None] = None,
        tmin: float = 0,
        tmax: float = 5.4,
        volt_threshold: Optional[Dict[str, float]] = None,
        stage_name: str = "post_epochs",
        reject_by_annotation: bool = False,
        subject_id: Optional[str] = None,
    ) -> mne.Epochs:
        """Create syllable-based epochs (SL_epochs) from raw EEG data.

        Parameters
        ----------
        data : mne.io.Raw, Optional
            The raw EEG data. If None, uses self.raw.
        tmin : float, Optional
            Start time of the epoch in seconds. Default is 0.
        tmax : float, Optional
            End time of the epoch in seconds. Default is 5.4 (18 syllables * 300ms).
        volt_threshold : dict, Optional
            Dictionary of channel types and thresholds for rejection. Default is None.
        stage_name : str, Optional
            Name for saving and metadata tracking. Default is "sl_epochs".
        reject_by_annotation : bool, Optional
            Whether to reject epochs overlapping bad annotations or mark them in metadata. Default is False.
        subject_id : str, Optional
            Subject ID to handle specific event codes (e.g., for subject 2310). Default is None.

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
                tmax = epoch_value.get("tmax", tmax)

            threshold_settings = config_value.get("threshold_rejection", {})
            if isinstance(threshold_settings, dict) and threshold_settings.get("enabled", False):
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
            # Define event codes
            syllable_codes = [
                'DIN1', 'DIN2', 'DIN3', 'DIN4', 'DIN5', 'DIN6', 'DIN7', 'DIN8', 'DIN9', 'DI10', 'DI11', 'DI12'
            ]
            events_to_remove = ['DI64']
            word_onset_codes = ['DIN1', 'DIN8', 'DIN9', 'DI11']
            if subject_id == '2310':
                syllable_codes = [f'D1{i:02d}' for i in range(1, 13)]
                word_onset_codes = ['D101', 'D108', 'D109', 'D111']
        
            # Remove DI64 events from annotations before extracting events
            message("header", "Removing DI64 events from annotations...")
            if data.annotations is not None:
                # Get indices of DI64 annotations
                di64_indices = [i for i, desc in enumerate(data.annotations.description) if desc == 'DI64']
                if di64_indices:
                    # Create new annotations without DI64
                    new_annotations = data.annotations.copy()
                    new_annotations.delete(di64_indices)
                    data.set_annotations(new_annotations)
                    message("debug", f"Removed {len(di64_indices)} DI64 events from annotations")
            
            # Extract all events from cleaned annotations
            message("header", "Extracting events from annotations...")
            events_all, event_id_all = mne.events_from_annotations(data)

            # Get the event IDs that correspond to our word onset codes
            word_onset_ids = [event_id_all[code] for code in word_onset_codes if code in event_id_all]
            if not word_onset_ids:
                raise ValueError("No word onset events found in annotations")
            word_onset_events = events_all[np.isin(events_all[:, 2], word_onset_ids)]

            # Validate epochs for 18 syllable events
            message("info", "Validating epochs for 18 syllable events...")
            syllable_code_ids = [event_id_all[code] for code in syllable_codes if code in event_id_all]

            #Working up to here
            valid_events = []
            num_syllables_per_epoch = 18

            for i, onset_event in enumerate(word_onset_events):
                # MATLAB: # Skip first 4 events (3 start codes + 1st syllable)
                # Python: # Skip first 1 event of word onset events (1st syllable)
                if i < 1:  
                    continue

                candidate_sample = onset_event[0]
                syllable_count = 0
                current_idx = np.where(events_all[:, 0] == candidate_sample)[0]
                if current_idx.size == 0:
                    continue
                current_idx = current_idx[0]

                # Count syllables from candidate onset
                #Events are invalidated because of DI64
                for j in range(current_idx, min(current_idx + num_syllables_per_epoch, len(events_all))):
                    event_code = events_all[j, 2]
                    event_label = event_id_all.get(event_code, f"code_{event_code}")
                    if event_code in syllable_code_ids:
                        syllable_count += 1
                    else:
                        # Non-syllable event (e.g., boundary), reset and skip
                        message("debug", f"Non-syllable event found: {event_label}")
                        syllable_count = 0
                        break

                    if syllable_count == num_syllables_per_epoch:
                        valid_events.append(onset_event)
                        message("debug", f"Valid epoch found at sample {candidate_sample}")
                        break

                if syllable_count < num_syllables_per_epoch - 1:  # Allow 17 syllables
                    message("info", f"Epoch at sample {candidate_sample} has only {syllable_count} syllables, skipping")

            valid_events = np.array(valid_events, dtype=int)
            if valid_events.size == 0:
                raise ValueError("No valid epochs found with 18 syllables")

            # Create epochs
            message("header", f"Creating SL epochs from {tmin}s to {tmax}s...")
            epochs = mne.Epochs(
                data,
                valid_events,
                tmin=tmin,
                tmax=tmax,
                baseline=None,  # No baseline correction
                reject=volt_threshold,
                preload=True,
                reject_by_annotation=reject_by_annotation,
            )

            # Step 5: Filter other events to keep only those that fall *within the kept epochs*
            sfreq = data.info["sfreq"]
            epoch_samples = epochs.events[:, 0]  # sample indices of epoch triggers

            # Compute valid ranges for each epoch (in raw sample indices)
            start_offsets = int(tmin * sfreq)
            end_offsets = int(tmax * sfreq)
            epoch_sample_ranges = [
                (s + start_offsets, s + end_offsets) for s in epoch_samples
            ]

            # Filter events_all for events that fall inside any of those ranges
            events_in_epochs = []
            for sample, prev, code in events_all:
                for i, (start, end) in enumerate(epoch_sample_ranges):
                    if start <= sample <= end:
                        events_in_epochs.append([sample, prev, code])
                        break  # prevent double counting
                    elif sample < start:
                        break

            events_in_epochs = np.array(events_in_epochs, dtype=int)
            event_descriptions = {v: k for k, v in event_id_all.items()}

            # Build metadata rows
            metadata_rows = []
            for i, (start, end) in enumerate(epoch_sample_ranges):
                epoch_events = []
                for sample, _, code in events_in_epochs:
                    if start <= sample <= end:
                        relative_time = (sample - epoch_samples[i]) / sfreq
                        label = event_descriptions.get(code, f"code_{code}")
                        epoch_events.append((label, relative_time))
                metadata_rows.append({"additional_events": epoch_events})

            # Add the metadata column
            if epochs.metadata is not None:
                epochs.metadata["additional_events"] = [
                    row["additional_events"] for row in metadata_rows
                ]
            else:
                epochs.metadata = pd.DataFrame(metadata_rows)

            # Create a copy for potential dropping
            epochs_clean = epochs.copy()

            # If not using reject_by_annotation or keeping all epochs, manually track bad annotations
            if not reject_by_annotation:
                # Find epochs that overlap with any "bad" or "BAD" annotations
                bad_epochs = []
                bad_annotations = {}  # To track which annotation affected each epoch

                for ann in data.annotations:
                    # Check if annotation description starts with "bad" or "BAD"
                    if ann["description"].lower().startswith("bad"):
                        ann_start = ann["onset"]
                        ann_end = ann["onset"] + ann["duration"]

                        # Check each epoch
                        for idx, event in enumerate(epochs.events):
                            epoch_start = (
                                event[0] / epochs.info["sfreq"]
                            )  # Convert to seconds
                            epoch_end = epoch_start + (tmax - tmin)

                            # Check for overlap
                            if (epoch_start <= ann_end) and (epoch_end >= ann_start):
                                bad_epochs.append(idx)

                                # Track which annotation affected this epoch
                                if idx not in bad_annotations:
                                    bad_annotations[idx] = []
                                bad_annotations[idx].append(ann["description"])

                # Remove duplicates and sort
                bad_epochs = sorted(list(set(bad_epochs)))

                # Mark bad epochs in metadata
                epochs.metadata["BAD_ANNOTATION"] = [
                    idx in bad_epochs for idx in range(len(epochs))
                ]

                # Add specific annotation types to metadata
                for idx, annotations in bad_annotations.items():
                    for annotation in annotations:
                        col_name = annotation.upper()
                        if col_name not in epochs.metadata.columns:
                            epochs.metadata[col_name] = False
                        epochs.metadata.loc[idx, col_name] = True

                message(
                    "info",
                    f"Marked {len(bad_epochs)} epochs with bad annotations (not dropped)",
                )

                # Save epochs with bad epochs marked but not dropped
                self._save_epochs_result(result_data=epochs, stage_name=stage_name)

                # Drop bad epochs only if not keeping all epochs
                epochs_clean.drop(bad_epochs, reason="BAD_ANNOTATION")

                message("debug", "reordering metadata after dropping")
                # After epochs_clean.drop(), epochs_clean.events contains the actual surviving events.
                # epochs.metadata contains the fully augmented metadata for the original set of epochs
                # (before this manual annotation-based drop).
                # We need to select rows from epochs.metadata that correspond to the events
                # actually remaining in epochs_clean.

                if (
                    epochs_clean.metadata is not None
                ):  # Should always be true as it's copied
                    # Get sample times of events that survived in epochs_clean
                    surviving_event_samples = epochs_clean.events[:, 0]

                    # Get sample times of the events in the original 'epochs' object
                    # (from which epochs.metadata was derived)
                    original_event_samples = epochs.events[:, 0]

                    # Find the indices in 'original_event_samples' that match 'surviving_event_samples'.
                    # This effectively maps the surviving events in epochs_clean back to their
                    # corresponding rows in the original (and fully augmented) epochs.metadata.
                    # np.isin creates a boolean mask, np.where converts it to indices.
                    kept_original_indices = np.where(
                        np.isin(original_event_samples, surviving_event_samples)
                    )[0]

                    if len(kept_original_indices) != len(epochs_clean.events):
                        message(
                            "error",
                            f"Mismatch when aligning surviving events to original metadata. "
                            f"Expected {len(epochs_clean.events)} matches, found {len(kept_original_indices)}. "
                            f"Metadata might be incorrect.",
                        )
                        # If there's a mismatch, it indicates a deeper issue, perhaps non-unique event samples
                        # or an unexpected state. For now, we proceed with potentially incorrect metadata
                        # or let MNE raise an error if lengths still don't match later.
                        # A more robust solution might involve raising an error here.

                    # Slice the augmented epochs.metadata using these derived indices.
                    # The resulting DataFrame will have the same number of rows as len(epochs_clean.events).
                    epochs_clean.metadata = epochs.metadata.iloc[
                        kept_original_indices
                    ].reset_index(drop=True)
                else:
                    message(
                        "warning",
                        "epochs_clean.metadata was None before assignment, which is unexpected.",
                    )

            # Analyze drop log to tally different annotation types
            drop_log = epochs_clean.drop_log
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
                "initial_epoch_count": len(epochs),
                "final_epoch_count": len(epochs_clean),
                "single_epoch_duration": epochs.times[-1] - epochs.times[0],
                "single_epoch_samples": epochs.times.shape[0],
                "initial_duration": (epochs.times[-1] - epochs.times[0]) * len(epochs_clean),
                "numberSamples": epochs.times.shape[0] * len(epochs_clean),
                "channelCount": len(epochs.ch_names),
                "annotation_types": annotation_types,
                "marked_epochs_file": stage_name,
                "cleaned_epochs_file": "post_drop_bad_sl_epochs",
                "tmin": tmin,
                "tmax": tmax,
            }
            self._update_metadata("step_create_sl_epochs", metadata)

            # Store and save epochs
            if hasattr(self, "config") and self.config.get("run_id"):
                self.epochs = epochs_clean
            self._save_epochs_result(result_data=epochs_clean, stage_name="post_drop_bad_epochs")

            return epochs_clean

        except Exception as e:
            message("error", f"Error during SL epoch creation: {str(e)}")
            raise RuntimeError(f"Failed to create SL epochs: {str(e)}") from e