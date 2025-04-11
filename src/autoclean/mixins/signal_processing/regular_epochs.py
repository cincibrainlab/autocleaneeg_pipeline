"""Regular epochs creation mixin for autoclean tasks.

This module provides functionality for creating regular fixed-length epochs from
continuous EEG data. Regular epochs are time segments of equal duration that are
created at fixed intervals throughout the recording, regardless of event markers.

The RegularEpochsMixin class implements methods for creating these epochs and
handling annotations, allowing users to either automatically reject epochs that
overlap with bad annotations or just mark them in the metadata for later processing.

Regular epoching is particularly useful for resting-state data analysis, where
there are no specific events of interest, but the data needs to be segmented
into manageable chunks for further processing and analysis.

Updates:
- Added reject_by_annotation parameter to create_regular_epochs to control how bad annotations are handled
- Added functionality to detect muscle artifacts in continuous Raw data using a sliding window approach
"""

from typing import Union, Dict, Optional, List
import mne
import numpy as np
import pandas as pd
from datetime import datetime

from autoclean.utils.logging import message

class RegularEpochsMixin:
    """Mixin class providing regular (fixed-length) epochs creation functionality for EEG data.
    """
    
    def create_regular_epochs(self, data: Union[mne.io.Raw, None] = None,
                             tmin: float = -1,
                             tmax: float = 1,
                             baseline: Optional[tuple] = None,
                             volt_threshold: Optional[Dict[str, float]] = None,
                             stage_name: str = "post_epochs",
                             reject_by_annotation: bool = False) -> mne.Epochs:
        """Create regular fixed-length epochs from raw data.

        Parameters
        ----------
        data : mne.io.Raw, Optional
            The raw data to create epochs from. If None, uses self.raw.
        tmin : float, Optional
            The start time of the epoch in seconds. Default is -1.
        tmax : float, Optional
            The end time of the epoch in seconds. Default is 1.
        baseline : tuple of float, Optional
            The time interval to apply baseline correction. Default is None.
        volt_threshold : dict, Optional
            Dictionary of channel types and thresholds for rejection, by default None.
        stage_name : str, Optional
            Name for saving and metadata tracking. Default is "post_epochs".
        reject_by_annotation : bool, Optional
            Whether to automatically reject epochs that overlap with bad annotations,
            or just mark them in the metadata for later processing. Default is False.

        Returns
        -------
        epochs_clean: mne.Epochs
            The created epochs object with bad epochs marked (and dropped if reject_by_annotation=True)
        
        Notes
        -----
        If reject_by_annotation is False, an intermediate file with bad epochs marked but not dropped is saved. 
        
        The epoching parameters can be customized through the configuration file
        (autoclean_config.yaml) under the "epoch_settings" section. If enabled, the
        configuration values will override the default parameters.

        See Also
        --------
        create_eventid_epochs : For creating epochs based on specific event markers.
        """
        # Check if this step is enabled in the configuration
        is_enabled, config_value = self._check_step_enabled("epoch_settings")
            
        if not is_enabled:
            message("info", "Epoch creation step is disabled in configuration")
            return None
            
        # Get parameters from config if available
        if config_value and isinstance(config_value, dict):
            # Get epoch settings
            epoch_value = config_value.get("value", {})
            if isinstance(epoch_value, dict):
                tmin = epoch_value.get("tmin", tmin)
                tmax = epoch_value.get("tmax", tmax)
            
            # Get baseline settings
            baseline_settings = config_value.get("remove_baseline", {})
            if isinstance(baseline_settings, dict) and baseline_settings.get("enabled", False):
                baseline = baseline_settings.get("window", baseline)
            
            # Get threshold settings
            threshold_settings = config_value.get("threshold_rejection", {})
            if isinstance(threshold_settings, dict) and threshold_settings.get("enabled", False):
                threshold_config = threshold_settings.get("volt_threshold", {})
                if isinstance(threshold_config, (int, float)):
                    volt_threshold = {"eeg": float(threshold_config)}
                elif isinstance(threshold_config, dict):
                    volt_threshold = {k: float(v) for k, v in threshold_config.items()}
        
        # Determine which data to use
        data = self._get_data_object(data)
        
        # Type checking
        if not isinstance(data, mne.io.Raw) and not isinstance(data, mne.io.base.BaseRaw):
            raise TypeError("Data must be an MNE Raw object for epoch creation")
            
        try:
            # Create initial epochs with reject_by_annotation parameter
            message("header", f"Creating regular epochs from {tmin}s to {tmax}s...")
            events = mne.make_fixed_length_events(
                data, duration=tmax - tmin, overlap=0, start=abs(tmin)
            )

            # Get all events from annotations
            try:
                events_all, event_id_all = mne.events_from_annotations(data)
            except Exception as e:
                message("warning", f"No annotations found in data, skipping event extraction from annotations")
                events_all = None
                event_id_all = None
            

            epochs = mne.Epochs(
                data,
                events,
                tmin=tmin,
                tmax=tmax,
                baseline=baseline,
                reject=volt_threshold,
                preload=True,
                reject_by_annotation=reject_by_annotation,
            )
            
            if events_all is not None:
                # Step 5: Filter other events to keep only those that fall *within the kept epochs*
                sfreq = data.info['sfreq']
                epoch_samples = epochs.events[:, 0]  # sample indices of epoch triggers

                # Compute valid ranges for each epoch (in raw sample indices)
                start_offsets = int(tmin * sfreq)
                end_offsets = int(tmax * sfreq)
                epoch_sample_ranges = [(s + start_offsets, s + end_offsets) for s in epoch_samples]

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
                    metadata_rows.append({'additional_events': epoch_events})

                # Add the metadata column
                if epochs.metadata is not None:
                    epochs.metadata['additional_events'] = [row['additional_events'] for row in metadata_rows]
                else:
                    epochs.metadata = pd.DataFrame(metadata_rows)
            else:
                epochs.metadata = pd.DataFrame(index=range(len(epochs)))

            # Create a copy for dropping if using amplitude thresholds
            epochs_clean = epochs.copy()
            
            # If not using reject_by_annotation, manually track bad annotations
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
                            epoch_start = event[0] / epochs.info["sfreq"]  # Convert to seconds
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
                
                message("info", f"Marked {len(bad_epochs)} epochs with bad annotations (not dropped)")
            
                # Save epochs with bad epochs marked but not dropped
                self._save_epochs_result(result_data = epochs_clean, stage_name = stage_name)

                epochs_clean.drop(bad_epochs, reason="BAD_ANNOTATION")

                if events_all is not None:
                    # Reorder metadata after dropping bad epochs
                    message('debug', 'reordering metadata after dropping')
                    if epochs_clean.metadata is not None:
                        kept_indices = epochs_clean.selection
                        max_index = epochs.metadata.shape[0] - 1
                        if kept_indices.max() > max_index:
                            print("Metadata shape:", epochs.metadata.shape)
                            print("Regular indices:", kept_indices)
                            kept_indices = kept_indices - 1
                            print("Adjusted indices:", kept_indices)

                        epochs_clean.metadata = epochs.metadata.iloc[kept_indices].reset_index(drop=True)
        
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
                        annotation_types[annotation] = annotation_types.get(annotation, 0) + 1
            
            message("info", "\nEpoch Drop Log Summary:")
            message("info", f"Total epochs: {total_epochs}")
            message("info", f"Good epochs: {good_epochs}")
            for annotation, count in annotation_types.items():
                message("info", f"Epochs with {annotation}: {count}")

            # Add flags if needed
            if (good_epochs/total_epochs) < self.EPOCH_RETENTION_THRESHOLD:
                flagged_reason = f"WARNING: Only {good_epochs/total_epochs*100}% of epochs were kept"
                self._update_flagged_status(flagged = True, reason = flagged_reason)
            
            # Add good and total to the annotation_types dictionary
            annotation_types["KEEP"] = good_epochs
            annotation_types["TOTAL"] = total_epochs
            
            # Update metadata
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
                "marked_epochs_file": "post_epochs",
                "cleaned_epochs_file": "post_drop_bads",
                "tmin": tmin,
                "tmax": tmax,
            }
            
            self._update_metadata("step_create_regular_epochs", metadata)
            
            # Store epochs
            if hasattr(self, 'config') and self.config.get("run_id"):
                self.epochs = epochs_clean

            # Save epochs
            self._save_epochs_result(result_data = epochs_clean, stage_name = "post_drop_bad_epochs")
                
            return epochs_clean
            
        except Exception as e:
            message("error", f"Error during regular epoch creation: {str(e)}")
            raise RuntimeError(f"Failed to create regular epochs: {str(e)}") from e
            
    