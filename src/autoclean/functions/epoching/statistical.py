"""Statistical learning epochs creation functions for EEG data.

This module provides standalone functions for creating epochs based on statistical
learning paradigm event patterns, specifically for validating 18-syllable sequences.
"""

from typing import Dict, List, Optional

import mne
import numpy as np
import pandas as pd


def create_sl_epochs(
    data: mne.io.BaseRaw,
    tmin: float = 0.0,
    tmax: float = 5.4,
    baseline: Optional[tuple] = None,
    reject: Optional[Dict[str, float]] = None,
    flat: Optional[Dict[str, float]] = None,
    reject_by_annotation: bool = True,
    subject_id: Optional[str] = None,
    syllable_codes: Optional[List[str]] = None,
    word_onset_codes: Optional[List[str]] = None,
    num_syllables_per_epoch: int = 18,
    preload: bool = True,
    verbose: Optional[bool] = None,
) -> mne.Epochs:
    """Create statistical learning epochs based on syllable event patterns.

    This function creates epochs for statistical learning experiments by identifying
    valid word onset events followed by the expected number of syllable events.
    It validates that each epoch contains exactly the specified number of syllables
    and removes problematic DI64 events that can interfere with the analysis.

    Statistical learning paradigms typically present sequences of syllables where
    participants learn statistical regularities. This function identifies valid
    epochs by ensuring each epoch contains a complete syllable sequence.

    Parameters
    ----------
    data : mne.io.BaseRaw
        The continuous EEG data containing statistical learning events.
    tmin : float, default 0.0
        Start time of the epoch relative to the word onset event in seconds.
    tmax : float, default 5.4
        End time of the epoch relative to the word onset event in seconds.
        Default corresponds to 18 syllables * 300ms duration.
    baseline : tuple or None, default None
        Time interval for baseline correction. None applies no baseline correction.
        Statistical learning epochs typically don't use baseline correction.
    reject : dict or None, default None
        Rejection thresholds for different channel types in volts.
        Example: {'eeg': 100e-6, 'eog': 200e-6}.
    flat : dict or None, default None
        Rejection thresholds for flat channels in volts.
        Example: {'eeg': 1e-6}.
    reject_by_annotation : bool, default True
        Whether to automatically reject epochs that overlap with 'bad' annotations.
    subject_id : str or None, default None
        Subject ID for handling special event code mappings (e.g., '2310').
        If None, uses standard event codes.
    syllable_codes : list of str or None, default None
        List of event codes representing syllables. If None, uses default codes:
        ['DIN1', 'DIN2', ..., 'DIN9', 'DI10', 'DI11', 'DI12']
    word_onset_codes : list of str or None, default None
        List of event codes representing word onsets. If None, uses default:
        ['DIN1', 'DIN8', 'DIN9', 'DI11']
    num_syllables_per_epoch : int, default 18
        Expected number of syllables per valid epoch.
    preload : bool, default True
        Whether to preload epoch data into memory.
    verbose : bool or None, default None
        Control verbosity of output.

    Returns
    -------
    epochs : mne.Epochs
        The created epochs object containing valid statistical learning sequences.

    Examples
    --------
    >>> epochs = create_sl_epochs(raw, tmin=0, tmax=5.4)
    >>> epochs = create_sl_epochs(raw, subject_id='2310', num_syllables_per_epoch=16)

    See Also
    --------
    create_regular_epochs : Create fixed-length epochs
    create_eventid_epochs : Create event-based epochs
    mne.events_from_annotations : Extract events from annotations
    mne.Epochs : MNE epochs class
    """
    # Input validation
    if not isinstance(data, mne.io.BaseRaw):
        raise TypeError(f"Data must be an MNE Raw object, got {type(data).__name__}")

    if tmin >= tmax:
        raise ValueError(f"tmin ({tmin}) must be less than tmax ({tmax})")

    if num_syllables_per_epoch <= 0:
        raise ValueError(
            f"num_syllables_per_epoch must be positive, got {num_syllables_per_epoch}"
        )

    try:
        # Set up event codes based on subject or defaults
        if syllable_codes is None:
            if subject_id == "2310":
                syllable_codes = [f"D1{i:02d}" for i in range(1, 13)]
            else:
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

        if word_onset_codes is None:
            if subject_id == "2310":
                word_onset_codes = ["D101", "D108", "D109", "D111"]
            else:
                word_onset_codes = ["DIN1", "DIN8", "DIN9", "DI11"]

        # Create a copy of data to avoid modifying the original
        data_copy = data.copy()

        # Remove DI64 events from annotations
        if data_copy.annotations is not None:
            di64_indices = [
                i
                for i, desc in enumerate(data_copy.annotations.description)
                if desc == "DI64"
            ]
            if di64_indices:
                new_annotations = data_copy.annotations.copy()
                new_annotations.delete(di64_indices)
                data_copy.set_annotations(new_annotations)

        # Extract all events from cleaned annotations
        try:
            events_all, event_id_all = mne.events_from_annotations(
                data_copy, verbose=verbose
            )
        except Exception as e:
            raise ValueError(f"No events found in data: {str(e)}") from e

        # Get word onset events
        word_onset_ids = [
            event_id_all[code] for code in word_onset_codes if code in event_id_all
        ]
        if not word_onset_ids:
            raise ValueError(
                f"No word onset events found. Expected: {word_onset_codes}, Available: {list(event_id_all.keys())}"
            )

        word_onset_events = events_all[np.isin(events_all[:, 2], word_onset_ids)]

        # Get syllable event IDs
        syllable_code_ids = [
            event_id_all[code] for code in syllable_codes if code in event_id_all
        ]
        if not syllable_code_ids:
            raise ValueError(
                f"No syllable events found. Expected: {syllable_codes}, Available: {list(event_id_all.keys())}"
            )

        # Validate epochs for required syllable count
        valid_events = []

        for i, onset_event in enumerate(word_onset_events):
            # Skip first event as per original implementation
            if i < 1:
                continue

            candidate_sample = onset_event[0]
            syllable_count = 0
            current_idx = np.where(events_all[:, 0] == candidate_sample)[0]
            if current_idx.size == 0:
                continue
            current_idx = current_idx[0]

            # Count syllables from candidate onset
            for j in range(
                current_idx, min(current_idx + num_syllables_per_epoch, len(events_all))
            ):
                event_code = events_all[j, 2]
                if event_code in syllable_code_ids:
                    syllable_count += 1
                else:
                    # Non-syllable event breaks the sequence
                    syllable_count = 0
                    break

                if syllable_count == num_syllables_per_epoch:
                    valid_events.append(onset_event)
                    break

            # Allow slight flexibility (17-18 syllables)
            if syllable_count >= num_syllables_per_epoch - 1:
                if onset_event.tolist() not in [v.tolist() for v in valid_events]:
                    valid_events.append(onset_event)

        valid_events = np.array(valid_events, dtype=int)
        if valid_events.size == 0:
            raise ValueError(
                f"No valid epochs found with {num_syllables_per_epoch} syllables"
            )

        # Create epochs using valid events (match original mixin exactly)
        epochs = mne.Epochs(
            data_copy,
            valid_events,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            reject=reject,
            preload=preload,
            reject_by_annotation=reject_by_annotation,
        )

        # Add metadata about syllable events within epochs
        epochs = _add_sl_metadata(epochs, data_copy, events_all, event_id_all)

        return epochs

    except Exception as e:
        if "No events found" in str(e) or "No valid epochs" in str(e):
            # Let validation errors bubble up
            raise
        raise RuntimeError(
            f"Failed to create statistical learning epochs: {str(e)}"
        ) from e


def _add_sl_metadata(
    epochs: mne.Epochs, raw: mne.io.BaseRaw, events_all: np.ndarray, event_id_all: Dict
) -> mne.Epochs:
    """Add metadata about syllable events within each statistical learning epoch.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs object to add metadata to.
    raw : mne.io.BaseRaw
        The raw data containing events.
    events_all : np.ndarray
        Array of all events from the data.
    event_id_all : dict
        Mapping of event descriptions to event codes.

    Returns
    -------
    epochs : mne.Epochs
        Epochs object with added metadata.
    """
    try:
        # Get epoch timing information
        sfreq = raw.info["sfreq"]
        epoch_samples = epochs.events[:, 0]  # Sample indices of epoch triggers
        tmin_samples = int(epochs.tmin * sfreq)
        tmax_samples = int(epochs.tmax * sfreq)

        # Build metadata for each epoch
        metadata_rows = []
        event_descriptions = {v: k for k, v in event_id_all.items()}

        for i, epoch_start_sample in enumerate(epoch_samples):
            # Calculate sample range for this epoch
            epoch_start = epoch_start_sample + tmin_samples
            epoch_end = epoch_start_sample + tmax_samples

            # Find syllable events within this epoch
            epoch_events = []
            syllable_count = 0

            for sample, _, code in events_all:
                if epoch_start <= sample <= epoch_end:
                    # Calculate relative time within epoch
                    relative_time = (sample - epoch_start_sample) / sfreq
                    label = event_descriptions.get(code, f"code_{code}")
                    epoch_events.append((label, relative_time))

                    # Count syllables (assuming syllable codes contain 'DIN' or 'D1')
                    if "DIN" in label or label.startswith("D1"):
                        syllable_count += 1

            metadata_rows.append(
                {
                    "epoch_number": i,
                    "epoch_start_sample": epoch_start_sample,
                    "epoch_duration": epochs.tmax - epochs.tmin,
                    "syllable_events": epoch_events,
                    "syllable_count": syllable_count,
                }
            )

        # Create metadata DataFrame
        metadata_df = pd.DataFrame(metadata_rows)

        if epochs.metadata is not None:
            # Merge with existing metadata
            epochs.metadata = pd.concat([epochs.metadata, metadata_df], axis=1)
        else:
            # Create new metadata
            epochs.metadata = metadata_df

        return epochs

    except Exception:
        # If metadata creation fails, return epochs without metadata
        return epochs
