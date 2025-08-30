"""Statistical learning randomized epochs creation functions for EEG data.

This module provides standalone functions for creating epochs based on statistical
learning paradigm event patterns, specifically for creating baseline epochs every
30 syllables regardless of word structure.
"""

from typing import Dict, Optional

import mne
import numpy as np
import pandas as pd

from autoclean.utils.logging import message


def create_statistical_learning_randomized_epochs(
    data: mne.io.Raw,
    tmin: float = 0,
    num_syllables: int = 30,
    volt_threshold: Optional[Dict[str, float]] = None,
    reject_by_annotation: bool = False,
    subject_id: Optional[str] = None,
    baseline: Optional[bool] = True,
    verbose: bool = True,
) -> mne.Epochs:
    """Create randomized syllable-based epochs (SL_epochs) from raw EEG data.

    This function implements the core logic for creating randomized statistical learning epochs,
    following MATLAB logic for event skipping and syllable validation but selecting every 30th
    syllable as epoch starts instead of word onsets. It can be used independently of the
    AutoClean pipeline.

    Parameters
    ----------
    data : mne.io.Raw
        The raw EEG data.
    tmin : float, optional
        Start time of the epoch in seconds. Default is 0.
    num_syllables : int, optional
        Number of syllables per epoch. Default is 30.
    volt_threshold : dict, optional
        Dictionary of channel types and thresholds for rejection. Default is None.
    reject_by_annotation : bool, optional
        Whether to reject epochs overlapping bad annotations or mark them in metadata. Default is False.
    subject_id : str, optional
        Subject ID to handle specific event codes (e.g., for subject 2310). Default is None.
    baseline : bool, optional
        Whether to apply baseline correction. Default is True.
    verbose : bool, optional
        Whether to print progress messages. Default is True.

    Returns
    -------
    epochs_clean : mne.Epochs
        The created epochs object with bad epochs marked (and dropped if reject_by_annotation=True).

    Raises
    ------
    TypeError
        If data is not an MNE Raw object.
    ValueError
        If no valid epochs are found or insufficient events exist.

    Notes
    -----
    This function implements MATLAB-compatible logic including:
    - Skipping first 5 events to avoid auditory onset responses
    - Validating 30-syllable sequences without interruption
    - Handling subject-specific event codes
    - Adding metadata for events within epochs
    - Creating epochs every 30 syllables regardless of word boundaries (randomized baseline)
    """

    if not isinstance(data, (mne.io.Raw, mne.io.base.BaseRaw)):
        raise TypeError(
            "Data must be an MNE Raw object for SL randomized epoch creation"
        )

    # Calculate tmax from num_syllables
    tmax = num_syllables * 0.3

    # Define event codes
    syllable_codes = [
        "DIN1",
        "DIN2",
        "DIN3",
        "DIN4",
        "DIN5",
        "DIN6",
        "DIN7",
        "DIN8",
        "DIN9",
        "DI10",
        "DI11",
        "DI12",
    ]
    if subject_id == "2310":
        syllable_codes = [f"D1{i:02d}" for i in range(1, 13)]

    # Remove DI64 events from annotations before extracting events
    if verbose:
        message("info", "Removing DI64 events from annotations...")
    if data.annotations is not None:
        # Get indices of DI64 annotations
        di64_indices = [
            i for i, desc in enumerate(data.annotations.description) if desc == "DI64"
        ]
        if di64_indices:
            # Create new annotations without DI64
            new_annotations = data.annotations.copy()
            new_annotations.delete(di64_indices)
            data.set_annotations(new_annotations)
            if verbose:
                message(
                    "debug", f"Removed {len(di64_indices)} DI64 events from annotations"
                )

    # Extract all events from cleaned annotations
    if verbose:
        message("info", "Extracting events from annotations...")
    events_all, event_id_all = mne.events_from_annotations(data)

    # Create an array of [sample, event_id_label] for all events
    # This will help map each event's sample index to its string label
    # The event_id_all dict maps label -> int, so we need to invert it for int -> label
    event_id_label_map = {v: k for k, v in event_id_all.items()}
    # Build array: each row is [sample, event_id_label]
    events_with_labels = np.array(
        [
            (sample, event_id_label_map.get(event_id, "UNKNOWN"))
            for sample, _, event_id in events_all
        ],
        dtype=object,
    )
    # Now events_with_labels is an array of [sample, label] for all events

    # Skip first 5 events to match MATLAB's y = 5 logic
    # MATLAB comment: "don't include the very first syllable due to sharp auditory onset response; skip over four 'start codes' plus first syllable"
    if verbose:
        message("info", "Skipping first 5 events to match MATLAB logic...")
    if len(events_all) > 5:
        events_all = events_all[5:]  # Skip first 5 events
        events_with_labels = events_with_labels[5:]  # Also skip in the label array
        if verbose:
            message(
                "debug",
                f"Skipped first 5 events, now processing {len(events_all)} events",
            )
    else:
        raise ValueError(
            f"Not enough events to skip initial 5 events. Found only {len(events_all)} events."
        )

    # Get all syllable events (including word onsets) for proper spacing calculation
    syllable_code_ids = [
        event_id_all[code] for code in syllable_codes if code in event_id_all
    ]
    all_syllable_events = events_all[np.isin(events_all[:, 2], syllable_code_ids)]

    # Select every 30th syllable as epoch starts (randomized approach)
    # This ignores word boundaries and creates epochs every num_syllables syllables
    non_overlapping_events = []
    for i in range(0, len(all_syllable_events), num_syllables):
        if i + num_syllables <= len(all_syllable_events):
            non_overlapping_events.append(all_syllable_events[i])

    non_overlapping_events = np.array(non_overlapping_events, dtype=int)
    if verbose:
        message(
            "info",
            f"Selected {len(non_overlapping_events)} randomized syllable onsets (every {num_syllables} syllables)",
        )

    # Validate epochs for num_syllables syllable events
    if verbose:
        message("info", f"Validating epochs for {num_syllables} syllable events...")
    valid_events = []

    for i, onset_event in enumerate(non_overlapping_events):
        candidate_sample = onset_event[0]
        syllable_count = 0
        current_idx = np.where(events_all[:, 0] == candidate_sample)[0]
        if current_idx.size == 0:
            continue
        current_idx = current_idx[0]

        # Count syllables from candidate onset
        for j in range(
            current_idx,
            min(current_idx + num_syllables, len(events_all)),
        ):
            event_code = events_all[j, 2]
            event_label = event_id_all.get(event_code, f"code_{event_code}")
            if event_code in syllable_code_ids:
                syllable_count += 1
            else:
                # Non-syllable event (e.g., boundary), reset and skip
                if verbose:
                    message("debug", f"Non-syllable event found: {event_label}")
                syllable_count = 0
                break

            if syllable_count == num_syllables:
                valid_events.append(onset_event)
                if verbose:
                    message(
                        "debug",
                        f"Valid randomized epoch found at sample {candidate_sample}",
                    )
                break

        if syllable_count < num_syllables - 1:  # Allow tolerance
            if verbose:
                message(
                    "info",
                    f"Randomized epoch at sample {candidate_sample} has only {syllable_count} syllables, skipping",
                )

    valid_events = np.array(valid_events, dtype=int)
    if valid_events.size == 0:
        raise ValueError(
            f"No valid randomized epochs found with {num_syllables} syllables"
        )

    # Create epochs
    if verbose:
        message("info", f"Creating randomized SL epochs from {tmin}s to {tmax}s...")
    epochs = mne.Epochs(
        data,
        valid_events,
        tmin=tmin,
        tmax=tmax,
        baseline=(None, tmax) if baseline else None,
        reject=volt_threshold,
        preload=True,
        reject_by_annotation=reject_by_annotation,
    )

    # Add metadata for events that fall within the kept epochs
    sfreq = data.info["sfreq"]
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

        if verbose:
            message(
                "info",
                f"Marked {len(bad_epochs)} epochs with bad annotations (not dropped)",
            )

        # Drop bad epochs from the cleaned version
        epochs_clean.drop(bad_epochs, reason="BAD_ANNOTATION")

        # Align metadata with surviving epochs
        if epochs_clean.metadata is not None:
            # Get sample times of events that survived in epochs_clean
            surviving_event_samples = epochs_clean.events[:, 0]

            # Get sample times of the events in the original 'epochs' object
            original_event_samples = epochs.events[:, 0]

            # Find the indices that match
            kept_original_indices = np.where(
                np.isin(original_event_samples, surviving_event_samples)
            )[0]

            if len(kept_original_indices) != len(epochs_clean.events):
                if verbose:
                    message(
                        "warning",
                        f"Mismatch when aligning surviving events to original metadata. "
                        f"Expected {len(epochs_clean.events)} matches, found {len(kept_original_indices)}. "
                        f"Metadata might be incorrect.",
                    )

            # Slice the metadata using these indices
            epochs_clean.metadata = epochs.metadata.iloc[
                kept_original_indices
            ].reset_index(drop=True)

    else:
        epochs_clean = None

    return epochs, epochs_clean
