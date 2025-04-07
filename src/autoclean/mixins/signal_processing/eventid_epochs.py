"""Event ID epochs creation mixin for autoclean tasks.

This module provides functionality for creating epochs based on event markers in
EEG data. Event-based epochs are time segments centered around specific event markers
that represent stimuli, responses, or other experimental events of interest.

The EventIDEpochsMixin class implements methods for creating these epochs and
detecting artifacts within them, particularly focusing on reference and muscle
artifacts that can contaminate the data.

Event-based epoching is particularly useful for task-based EEG analysis, where
the data needs to be segmented around specific events of interest for further
processing and analysis, such as event-related potentials (ERPs) or time-frequency
analysis.
"""

from typing import Union, Dict, Optional, List, Any
import mne
import numpy as np
import pandas as pd
from datetime import datetime

from autoclean.utils.logging import message

class EventIDEpochsMixin:
    """Mixin class providing event ID based epochs creation functionality for EEG data.
    """
    def create_eventid_epochs(self, data: Union[mne.io.Raw, None] = None,
                             event_id: Optional[Dict[str, int]] = None,
                             tmin: float = -0.5,
                             tmax: float = 2,
                             baseline: Optional[tuple] = (None, 0),
                             volt_threshold: Optional[Dict[str, float]] = None,
                             reject_by_annotation: bool = False,
                             stage_name: str = "post_epochs") -> Optional[mne.Epochs]:
        """Create epochs based on event IDs from raw data.

        Parameters
        ----------
        data : mne.io.Raw, Optional
            The raw data to create epochs from. If None, uses self.raw.
        event_id : dict, Optional
            Dictionary mapping event names to event IDs (e.g., {"target": 1, "standard": 2}).
        tmin : float, Optional
            Start time of the epoch relative to the event in seconds, by default -0.5.
        tmax : float, Optional
            End time of the epoch relative to the event in seconds, by default 2.
        baseline : tuple, Optional
            Baseline correction (tuple of start, end), by default (None, 0).
        volt_threshold : dict, Optional
            Dictionary of channel types and thresholds for rejection, by default None.
        reject_by_annotation : bool, Optional
            Whether to reject epochs by annotation, by default False.
        stage_name : str, Optional
            Name for saving and metadata, by default "post_epochs".

        Returns
        -------
        epochs_clean : instance of mne.Epochs | None
            The created epochs or None if epoching is disabled. 

        Notes
        -----
        This method creates epochs centered around specific event IDs in the raw data.
        It is useful for event-related potential (ERP) analysis where you want to
        extract segments of data time-locked to specific events.
        """
        # Check if epoch_settings is enabled in the configuration
        is_enabled, epoch_config = self._check_step_enabled("epoch_settings")
            
        if not is_enabled:
            message("info", "Epoch settings step is disabled in configuration")
            return None
            
                
        # Get epoch settings
        if epoch_config and isinstance(epoch_config, dict):
            epoch_value = epoch_config.get("value", {})
            if isinstance(epoch_value, dict):
                tmin = epoch_value.get("tmin", tmin)
                tmax = epoch_value.get("tmax", tmax)

            event_id = epoch_config.get("event_id", {})
            
            # Get baseline settings
            baseline_settings = epoch_config.get("remove_baseline", {})
            if isinstance(baseline_settings, dict) and baseline_settings.get("enabled", False):
                baseline = baseline_settings.get("window", baseline)
            
            # Get threshold settings
            threshold_settings = epoch_config.get("threshold_rejection", {})
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
            # Check if event_id is provided
            if event_id is None:
                message("warning", "No event_id provided for event-based epoching")
                return None
                
            message("header", f"Creating epochs based on event IDs: {event_id}")
            
            # Get all events from annotations
            events_all, event_id_all = mne.events_from_annotations(data)
            
            # Find all event types that match our event_id keys
            event_patterns = {} #Name and code of events to epoch by
            for event_key in event_id.keys():
                #Could lead to undesired results if event_key is a substring of another event
                matching_events = [k for k in event_id_all.keys() if event_key in k]
                for match in matching_events:
                    event_patterns[match] = event_id_all[match]
            
            message("info", f"Looking for events matching patterns: {list(event_patterns.keys())}")
            
            # Filter events to include only those with matching trigger codes
            trigger_codes = list(event_patterns.values())
            events_trig = events_all[np.isin(events_all[:, 2], trigger_codes)]
            
            if len(events_trig) == 0:
                message("warning", "No matching events found")
                return None
                
            message("info", f"Found {len(events_trig)} events matching the patterns")
            
            # Create epochs with the filtered events
            epochs = mne.Epochs(
                data,
                events_trig,
                event_id=event_patterns,
                tmin=tmin,
                tmax=tmax,
                baseline=baseline,
                reject=volt_threshold,
                preload=True,
                reject_by_annotation=reject_by_annotation,
            )
            
            # Step 5: Filter other events to keep only those that fall *within the kept epochs*
            sfreq = data.info['sfreq']
            epoch_samples = epochs.events[:, 0]  # sample indices of epoch triggers
            n_epochs = len(epochs.events)
            n_samples = len(epochs.times)             # total samples per epoch
            offset = int(round(-epochs.tmin * sfreq))  # Number of samples from epoch start to time 0

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
                    # if sample < start:
                    #     break
                metadata_rows.append({'additional_events': epoch_events})

            # Add the metadata column
            if epochs.metadata is not None:
                epochs.metadata['additional_events'] = [row['additional_events'] for row in metadata_rows]
            else:
                epochs.metadata = pd.DataFrame(metadata_rows)

            # Create a copy for dropping
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
            self._save_epochs_result(result_data = epochs_clean, stage_name = "post_epochs")

            epochs_clean.drop(bad_epochs, reason="BAD_ANNOTATION")
            if epochs_clean.metadata is not None:
                kept_indices = epochs_clean.selection
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
                "initial_epoch_count": len(events_trig),
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
                "event_id": event_id,
            }
            
            self._update_metadata("step_create_eventid_epochs", metadata)
            
            # Store epochs
            if hasattr(self, 'config') and self.config.get("run_id"):
                self.epochs = epochs_clean

            # Save epochs
            self._save_epochs_result(result_data = epochs_clean, stage_name = "post_drop_bad_epochs")
                
            return epochs_clean
            
        except Exception as e:
            message("error", f"Error during event ID epoch creation: {str(e)}")
            raise RuntimeError(f"Failed to create event ID epochs: {str(e)}") from e
            