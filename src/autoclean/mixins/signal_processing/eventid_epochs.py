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
    
    This mixin provides methods for creating epochs based on event markers in EEG data.
    It includes functionality for artifact detection, specifically focusing on reference
    and muscle artifacts that can contaminate the data.
    
    Event-based epochs are time segments centered around specific event markers that
    represent stimuli, responses, or other experimental events of interest. This approach
    is particularly useful for task-based EEG analysis, where the data needs to be
    segmented around specific events for further processing.
    
    The mixin respects configuration settings from the autoclean_config.yaml file,
    allowing users to customize the epoching parameters, event IDs, and artifact
    detection thresholds.
    """
    
    def create_eventid_epochs(self, data: Union[mne.io.BaseRaw, None] = None,
                             event_id: Optional[Dict[str, int]] = None,
                             tmin: float = -0.5,
                             tmax: float = 2,
                             baseline: Optional[tuple] = (None, 0),
                             volt_threshold: Optional[Dict[str, float]] = None,
                             reject_by_annotation: bool = False,
                             stage_name: str = "post_epochs") -> Optional[mne.Epochs]:
        """Create epochs based on event IDs from raw data.
        
        This method creates epochs centered around specific event IDs in the raw data.
        It is useful for event-related potential (ERP) analysis where you want to
        extract segments of data time-locked to specific events.
        
        If no event_id is provided, the method will attempt to extract event IDs from
        the configuration file.
        
        Args:
            data: Optional MNE Raw object. If None, uses self.raw
            event_id: Dictionary mapping event names to event IDs (e.g., {"target": 1, "standard": 2})
            tmin: Start time of the epoch relative to the event in seconds
            tmax: End time of the epoch relative to the event in seconds
            baseline: Baseline correction (tuple of start, end)
            volt_threshold: Dictionary of channel types and thresholds for rejection
            stage_name: Name for saving and metadata
            
        Returns:
            The created epochs object or None if disabled
            
        Example:
            ```python
            # Create epochs around target and standard events
            epochs = self.create_eventid_epochs(
                event_id={"target": 1, "standard": 2},
                tmin=-0.2,
                tmax=0.5
            )
            
            # Access specific event types
            target_epochs = epochs["target"]
            ```
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
        if not isinstance(data, mne.io.BaseRaw):
            raise TypeError("Data must be an MNE Raw object for epoch creation")
            
        try:
            # Check if event_id is provided
            if event_id is None:
                message("warning", "No event_id provided for event-based epoching")
                return None
                
            message("header", f"Creating epochs based on event IDs: {event_id}")
            
             # Create regexp pattern for all event types
            target_event_type = list(event_id.keys())[0]
            reg_exp = f".*{target_event_type}.*"
            message("info", f"Looking for events matching pattern: {reg_exp}")

            # Get events using regexp
            events, event_id = mne.events_from_annotations(data, regexp=reg_exp)

            if len(events) == 0:
                message("warning", "No matching events found")
                return None

            message("info", f"Found {len(events)} events matching the patterns:")
            for event_name, event_num in event_id.items():
                message("info", f"  {event_name}: {event_num}")
            # Find events in the data
            events = mne.events_from_annotations(data, verbose=True)
            
            # Check if any events were found
            if len(events) == 0:
                message("warning", "No events found in the data")
                return None
                
            # Extract the events array and event_id dictionary from the tuple
            events_array, events_id_dict = events
                
            # Filter events to only include those in event_id
            event_ids_to_keep = list(event_id.values())
            filtered_events = [evt for evt in events_array if evt[2] in event_ids_to_keep]
            
            if len(filtered_events) == 0:
                message("warning", f"No events matching event_id {event_id} found in the data")
                return None
            # Create epochs with reject_by_annotation=False to handle annotations manually
            epochs = mne.Epochs(
                data,
                filtered_events,
                event_id=event_id,
                tmin=tmin,
                tmax=tmax,
                baseline=baseline,
                reject=volt_threshold,
                preload=True,
                reject_by_annotation=reject_by_annotation,
            )
            
            # Initialize metadata DataFrame
            epochs.metadata = pd.DataFrame(index=range(len(epochs)))

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

                epochs_clean.drop(bad_epochs, reason="BAD_ANNOTATION")
            
            
            # Save epochs with bad epochs marked but not dropped
            from autoclean.step_functions.io import save_epochs_to_set
            if hasattr(self, 'config'):
                save_epochs_to_set(epochs, self.config, stage_name)
                
            
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
            
            # Add good and total to the annotation_types dictionary
            annotation_types["KEEP"] = good_epochs
            annotation_types["TOTAL"] = total_epochs
            
            # Update metadata
            metadata = {
                "duration": tmax - tmin,
                "reject_by_annotation": reject_by_annotation,
                "initial_epoch_count": len(filtered_events),
                "final_epoch_count": len(epochs_clean),
                "single_epoch_duration": epochs.times[-1] - epochs.times[0],
                "single_epoch_samples": epochs.times.shape[0],
                "durationSec": (epochs.times[-1] - epochs.times[0]) * len(epochs_clean),
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
            if hasattr(self, 'config'):
                save_epochs_to_set(epochs_clean, self.config, "post_drop_bad_epochs")
                
            return epochs_clean
            
        except Exception as e:
            message("error", f"Error during event ID epoch creation: {str(e)}")
            raise RuntimeError(f"Failed to create event ID epochs: {str(e)}") from e
            