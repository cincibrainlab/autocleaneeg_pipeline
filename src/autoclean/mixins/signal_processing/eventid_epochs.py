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

import json
from typing import Dict, Optional, Union

import mne
import numpy as np
import pandas as pd

from autoclean.functions.epoching import create_eventid_epochs as _create_eventid_epochs
from autoclean.utils.logging import message


class EventIDEpochsMixin:
    """Mixin class providing event ID based epochs creation functionality for EEG data."""

    def print_discovered_events(
        self,
        data: Union[mne.io.Raw, None] = None,
        show_config_example: bool = True,
    ) -> Optional[Dict[str, int]]:
        """Discover and print all events found in the raw data annotations.

        This is a helper method to assist users in understanding what events are
        available in their EEG data and how to configure the event_id parameter
        for epoching. It prints a formatted table of events with counts and
        provides example JSON configuration snippets.

        Parameters
        ----------
        data : mne.io.Raw, Optional
            The raw data to extract events from. If None, uses self.raw.
        show_config_example : bool, Optional
            Whether to print example JSON config snippets, by default True.

        Returns
        -------
        event_id_all : dict | None
            Dictionary mapping event descriptions to event codes found in the data.
            Returns None if no events are found or an error occurs.

        Examples
        --------
        >>> # Discover events in your data
        >>> processor.print_discovered_events()
        
        >>> # Use in a script to see available events
        >>> available_events = processor.print_discovered_events(show_config_example=False)
        >>> if available_events:
        >>>     # Select specific events for your analysis
        >>>     my_event_id = {"target": available_events["DIN4"]}

        Notes
        -----
        This method is particularly useful when:
        - You're working with new EEG data and don't know the event codes
        - Your epoching is producing empty results (no matching events)
        - You need to update your configuration file with correct event mappings
        """
        # Determine which data to use
        data = self._get_data_object(data)

        # Type checking
        if not isinstance(data, mne.io.Raw) and not isinstance(
            data, mne.io.base.BaseRaw
        ):
            message("error", "Data must be an MNE Raw object for event discovery")
            return None

        try:
            message("header", "Discovering Events in EEG Data")
            message("info", "Extracting events from annotations...")

            # Get all events from annotations
            try:
                events_all, event_id_all = mne.events_from_annotations(data)
            except Exception as e:
                message(
                    "error",
                    f"Failed to extract events from annotations: {str(e)}. "
                    "Check that your data has valid annotations.",
                )
                return None

            if not event_id_all:
                message("warning", "No events found in the data annotations.")
                message(
                    "info",
                    "Tip: Events might need to be added manually using mne.Annotations or mne.add_events",
                )
                return None

            # Count occurrences of each event
            event_counts = {}
            for event_desc, event_code in event_id_all.items():
                count = np.sum(events_all[:, 2] == event_code)
                event_counts[event_desc] = {
                    "code": event_code,
                    "count": count,
                    "percentage": (count / len(events_all) * 100) if len(events_all) > 0 else 0,
                }

            # Sort by count (most frequent first)
            sorted_events = sorted(
                event_counts.items(), key=lambda x: x[1]["count"], reverse=True
            )

            # Print formatted table
            message("info", f"\nFound {len(event_id_all)} unique event types:")
            message("info", "=" * 80)
            
            # Header
            header = f"{'Event Name':<25} {'Code':<10} {'Count':<12} {'% of Total':<15}"
            message("info", header)
            message("info", "-" * 80)

            # Rows
            for event_name, event_info in sorted_events:
                row = (
                    f"{event_name:<25} "
                    f"{event_info['code']:<10} "
                    f"{event_info['count']:<12} "
                    f"{event_info['percentage']:>6.1f}%"
                )
                message("info", row)

            message("info", "=" * 80)
            message("info", f"Total events: {len(events_all)}")

            # Provide configuration guidance
            if show_config_example:
                message("header", "\nConfiguration Guide")
                
                # Filter out likely artifact markers (BAD, etc.)
                likely_stimuli = {
                    k: v["code"]
                    for k, v in event_counts.items()
                    if not k.upper().startswith("BAD")
                    and v["count"] >= 5  # Exclude rare events
                }

                if likely_stimuli:
                    message(
                        "info",
                        "\nTo use these events for epoching, add to your config.json:",
                    )
                    
                    # Example 1: Simple config
                    simple_config = {
                        "epoch_settings": {
                            "enabled": True,
                            "value": {"tmin": -0.2, "tmax": 0.8},
                            "event_id": likely_stimuli,
                        }
                    }
                    
                    message("info", "\nBasic Example (uses all available events):")
                    message("info", json.dumps(simple_config, indent=2))
                    
                    # Example 2: Custom naming
                    if len(likely_stimuli) >= 2:
                        first_event = list(likely_stimuli.items())[0]
                        second_event = list(likely_stimuli.items())[1]
                        
                        custom_config = {
                            "epoch_settings": {
                                "enabled": True,
                                "value": {"tmin": -0.5, "tmax": 2.0},
                                "event_id": {
                                    "target": first_event[1],  # Use descriptive names
                                    "standard": second_event[1],
                                },
                            }
                        }
                        
                        message("info", "\nCustom Naming Example (recommended for clarity):")
                        message("info", json.dumps(custom_config, indent=2))
                        message(
                            "info",
                            f"\nNote: Replace 'target' and 'standard' with meaningful names for your experiment",
                        )
                    
                    message("info", "\n" + "=" * 80)
                    message("info", "Pro Tips:")
                    message("info", "  • Event codes must match EXACTLY (integers)")
                    message("info", "  • Use descriptive names instead of 'DIN4' for better readability")
                    message("info", "  • Adjust tmin/tmax based on your experimental design")
                    message("info", "  • If epochs are empty, verify codes match what's shown above")
                    message("info", "=" * 80)
                else:
                    message(
                        "warning",
                        "Only artifact markers found. Check if stimulus events are present.",
                    )

            return event_id_all

        except Exception as e:
            message("error", f"Error during event discovery: {str(e)}")
            return None

    def create_eventid_epochs(
        self,
        data: Union[mne.io.Raw, None] = None,
        event_id: Optional[Dict[str, int]] = None,
        tmin: float = -0.5,
        tmax: float = 2,
        baseline: Optional[tuple] = (None, 0),
        volt_threshold: Optional[Dict[str, float]] = None,
        reject_by_annotation: bool = False,
        keep_all_epochs: bool = False,
        stage_name: str = "post_epochs",
    ) -> Optional[mne.Epochs]:
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
        keep_all_epochs : bool, Optional
            If True, no epochs will be dropped - bad epochs will only be marked in metadata, by default False.
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
            epoch_value = epoch_config.get("value") or {}
            if isinstance(epoch_value, dict):
                tmin = epoch_value.get("tmin", tmin)
                tmax = epoch_value.get("tmax", tmax)

            event_id = epoch_config.get("event_id", {})

            # Get keep_all_epochs setting if available
            keep_all_epochs = epoch_config.get("keep_all_epochs", keep_all_epochs)

            # Get baseline settings
            baseline_settings = epoch_config.get("remove_baseline", {})
            if isinstance(baseline_settings, dict) and baseline_settings.get(
                "enabled", False
            ):
                baseline = baseline_settings.get("window", baseline)

            # Get threshold settings
            threshold_settings = epoch_config.get("threshold_rejection", {})
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

        # Type checking
        if not isinstance(data, mne.io.Raw) and not isinstance(
            data, mne.io.base.BaseRaw
        ):
            raise TypeError("Data must be an MNE Raw object for epoch creation")

        try:
            # Check if event_id is provided
            if event_id is None or len(event_id) == 0:
                message("warning", "No event_id provided for event-based epoching")
                message(
                    "info",
                    "Tip: Call print_discovered_events() to see available events in your data",
                )
                # Automatically show available events to help the user
                message("info", "Automatically discovering events to help you configure...")
                self.print_discovered_events(data=data)
                return None

            message("header", f"Creating epochs based on event IDs: {event_id}")

            # Always show discovered events to help users verify their configuration
            message("info", "\nDiscovering available events in data for verification...")
            discovered_events = self.print_discovered_events(data=data, show_config_example=True)
            
            if discovered_events is None:
                message("error", "Failed to discover events in data")
                return None

            # Get all events from annotations for epoching
            try:
                events_all, event_id_all = mne.events_from_annotations(data)
            except ValueError as e:
                message(
                    "error",
                    f"Failed to parse annotations from data: {str(e)}. "
                    "Check that annotations are properly formatted.",
                )
                return None
            except RuntimeError as e:
                message(
                    "error",
                    f"MNE runtime error while extracting events from annotations: {str(e)}",
                )
                return None
            except Exception as e:
                message(
                    "error",
                    f"Unexpected error extracting events from annotations: {str(e)}",
                )
                return None

            # Find all event types that match our event_id values
            event_patterns = {}  # Name and code of events to epoch by
            for event_key in event_id.keys():
                # Match the event_id values with the actual event codes in the file
                matching_events = [
                    k for k in event_id_all if event_id_all[k] == event_id[event_key]
                ]
                for match in matching_events:
                    event_patterns[match] = event_id_all[match]

            message(
                "info",
                f"Looking for events matching patterns: {list(event_patterns.keys())}",
            )
            
            # If no patterns matched, provide helpful debugging info
            if len(event_patterns) == 0:
                message(
                    "error",
                    f"No events in data match the configured event_id codes: {event_id}",
                )
                message(
                    "info",
                    f"Available events in data: {event_id_all}",
                )
                message(
                    "info",
                    "Showing all discovered events to help you fix the configuration...",
                )
                self.print_discovered_events(data=data)
                return None
            
            # Filter events to include only those with matching trigger codes
            trigger_codes = list(event_patterns.values())

            events_trig = events_all[np.isin(events_all[:, 2], trigger_codes)]

            if len(events_trig) == 0:
                message("warning", "No matching events found in the data")
                return None

            message("info", f"Found {len(events_trig)} events matching the patterns")

            # Create epochs with the filtered events

            # Use standalone function for core epoch creation
            epochs = _create_eventid_epochs(
                data=data,
                event_id=event_patterns,
                tmin=tmin,
                tmax=tmax,
                baseline=baseline,
                reject=(None if keep_all_epochs else volt_threshold),
                reject_by_annotation=(reject_by_annotation and not keep_all_epochs),
                preload=True,
                on_missing="ignore",  # Don't error if no events
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

            # If we're keeping all epochs but still want to mark them, we need to apply additional logic
            if keep_all_epochs:
                # 1. Mark epochs that would have been rejected by voltage threshold
                if volt_threshold is not None:
                    # Manually compute threshold violations since MNE has no built-in
                    # dry-run threshold detection
                    bad_epochs_thresh = []
                    
                    # Map channel names to their types
                    ch_types_list = epochs.get_channel_types()
                    ch_type_map = {}
                    for ch_name, ch_type in zip(epochs.ch_names, ch_types_list):
                        if ch_type not in ch_type_map:
                            ch_type_map[ch_type] = []
                        ch_type_map[ch_type].append(ch_name)
                    
                    # Check each epoch for threshold violations
                    for idx in range(len(epochs)):
                        this_epoch = epochs[idx].get_data()[0]  # Shape: (n_chans, n_times)
                        drop_reasons = []
                        
                        for ch_type, thresh in volt_threshold.items():
                            if ch_type in ch_type_map:
                                # Get indices of channels of this type
                                ch_names_of_type = ch_type_map[ch_type]
                                mask = np.isin(epochs.ch_names, ch_names_of_type)
                                epoch_type_data = this_epoch[mask]
                                
                                # Check if any sample exceeds threshold
                                if np.any(np.abs(epoch_type_data) > thresh):
                                    drop_reasons.append(ch_type)
                        
                        if drop_reasons:
                            bad_epochs_thresh.append(idx)
                            # Mark which channel types exceeded threshold
                            for ch_type in drop_reasons:
                                col_name = f"THRESHOLD_{ch_type.upper()}"
                                if col_name not in epochs.metadata.columns:
                                    epochs.metadata[col_name] = False
                                epochs.metadata.loc[idx, col_name] = True
                    
                    message(
                        "info",
                        f"Marked {len(bad_epochs_thresh)} epochs exceeding voltage thresholds (not dropped)",
                    )

            # If not using reject_by_annotation or keeping all epochs, manually track bad annotations
            if not reject_by_annotation or keep_all_epochs:
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
                if not keep_all_epochs:
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

                        # Check for duplicate event sample indices upfront
                        if len(np.unique(original_event_samples)) != len(
                            original_event_samples
                        ):
                            message(
                                "warning",
                                "Duplicate event sample indices detected; using sorted matching",
                            )

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
                                f"Attempting fallback with sorted matching.",
                            )

                            # Fallback: use searchsorted for 1:1 matching
                            # This handles duplicate event samples by matching in order
                            surviving_sorted_idx = np.argsort(surviving_event_samples)
                            original_sorted_idx = np.argsort(original_event_samples)

                            surviving_sorted = surviving_event_samples[
                                surviving_sorted_idx
                            ]
                            original_sorted = original_event_samples[original_sorted_idx]

                            # Find positions of surviving events in the sorted original array
                            positions = np.searchsorted(original_sorted, surviving_sorted)

                            # Map back to original indices
                            kept_original_indices = original_sorted_idx[positions]

                            # Validate alignment
                            if len(kept_original_indices) != len(epochs_clean.events):
                                raise ValueError(
                                    f"Cannot align metadata after fallback. "
                                    f"Expected {len(epochs_clean.events)} matches, "
                                    f"found {len(kept_original_indices)}. "
                                    f"Duplicate event samples may have caused alignment failure."
                                )

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

            # If keeping all epochs, use the original epochs for subsequent processing
            if keep_all_epochs:
                epochs_clean = epochs
                message(
                    "info", "Keeping all epochs as requested (keep_all_epochs=True)"
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

            # Add flags if needed (only if not keeping all epochs)
            if (
                not keep_all_epochs
                and (good_epochs / total_epochs) < self.EPOCH_RETENTION_THRESHOLD
            ):
                flagged_reason = (
                    f"WARNING: Only {good_epochs / total_epochs * 100}% "
                    "of epochs were kept"
                )
                self._update_flagged_status(flagged=True, reason=flagged_reason)

            # Add good and total to the annotation_types dictionary
            annotation_types["KEEP"] = good_epochs
            annotation_types["TOTAL"] = total_epochs

            # Update metadata
            metadata = {
                "duration": tmax - tmin,
                "reject_by_annotation": reject_by_annotation,
                "keep_all_epochs": keep_all_epochs,
                "initial_epoch_count": len(events_trig),
                "final_epoch_count": len(epochs_clean),
                "single_epoch_duration": epochs.times[-1] - epochs.times[0],
                "single_epoch_samples": epochs.times.shape[0],
                "initial_duration": (epochs.times[-1] - epochs.times[0])
                * len(epochs_clean),
                "numberSamples": epochs.times.shape[0] * len(epochs_clean),
                "channelCount": len(epochs.ch_names),
                "annotation_types": annotation_types,
                "marked_epochs_file": "post_epochs",
                "cleaned_epochs_file": (
                    "post_drop_bads" if not keep_all_epochs else "post_epochs"
                ),
                "tmin": tmin,
                "tmax": tmax,
                "event_id": event_id,
            }

            self._update_metadata("step_create_eventid_epochs", metadata)

            # Store epochs
            if hasattr(self, "config") and self.config.get("run_id"):
                self.epochs = epochs_clean

            # Save epochs
            if not keep_all_epochs:
                self._save_epochs_result(
                    result_data=epochs_clean, stage_name="post_drop_bad_epochs"
                )

            return epochs_clean

        except Exception as e:
            message("error", f"Error during event ID epoch creation: {str(e)}")
            raise RuntimeError(f"Failed to create event ID epochs: {str(e)}") from e
