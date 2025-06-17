"""Regular epochs creation functions for EEG data.

This module provides standalone functions for creating fixed-length epochs from
continuous EEG data without relying on specific event markers.
"""

from typing import Dict, Optional, Tuple

import mne
import pandas as pd


def create_regular_epochs(
    data: mne.io.BaseRaw,
    tmin: float = -1.0,
    tmax: float = 1.0,
    duration: Optional[float] = None,
    overlap: float = 0.0,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    reject: Optional[Dict[str, float]] = None,
    flat: Optional[Dict[str, float]] = None,
    reject_by_annotation: bool = True,
    include_metadata: bool = True,
    preload: bool = True,
    verbose: Optional[bool] = None,
) -> mne.Epochs:
    """Create regular fixed-length epochs from continuous EEG data.

    This function creates epochs of fixed length at regular intervals throughout
    the continuous EEG recording. This approach is particularly useful for
    resting-state data or when analyzing ongoing brain activity without specific
    event markers.

    The function automatically generates events at regular intervals and creates
    epochs around these synthetic events. Optionally, it can include information
    about annotations that fall within each epoch as metadata.

    Parameters
    ----------
    data : mne.io.BaseRaw
        The continuous EEG data to create epochs from.
    tmin : float, default -1.0
        Start time of the epoch relative to the synthetic event in seconds.
        Negative values start before the event.
    tmax : float, default 1.0
        End time of the epoch relative to the synthetic event in seconds.
        Positive values extend after the event.
    duration : float or None, default None
        Duration of each epoch in seconds. If None, calculated as tmax - tmin.
        This parameter provides an alternative way to specify epoch length.
    overlap : float, default 0.0
        Overlap between consecutive epochs in seconds. Zero means no overlap.
        Positive values create overlapping epochs for increased data yield.
    baseline : tuple of (float, float) or None, default None
        Time interval for baseline correction in seconds relative to epoch start.
        For example, (None, 0) uses the entire pre-stimulus period, (-0.2, 0)
        uses 200ms before stimulus. None applies no baseline correction.
    reject : dict or None, default None
        Rejection thresholds for different channel types in volts.
        Example: {'eeg': 100e-6, 'eog': 200e-6}. Epochs exceeding these
        thresholds will be marked as bad and potentially dropped.
    flat : dict or None, default None
        Rejection thresholds for flat channels in volts (minimum required range).
        Example: {'eeg': 1e-6}. Channels with signal range below threshold
        in any epoch will cause epoch rejection.
    reject_by_annotation : bool, default True
        Whether to automatically reject epochs that overlap with 'bad' annotations.
        If False, epochs are marked but not dropped automatically.
    include_metadata : bool, default True
        Whether to include metadata about annotations and events that fall
        within each epoch. Useful for post-hoc analysis and quality control.
    preload : bool, default True
        Whether to preload epoch data into memory. Recommended for most use cases
        to enable all epoch manipulation functions.
    verbose : bool or None, default None
        Control verbosity of output. If None, uses MNE default.

    Returns
    -------
    epochs : mne.Epochs
        The created epochs object with metadata about contained events and
        annotations (if include_metadata=True).

    Examples
    --------
    >>> epochs = create_regular_epochs(raw, tmin=-1.0, tmax=1.0)
    >>> epochs = create_regular_epochs(raw, overlap=1.0, reject={'eeg': 100e-6})

    See Also
    --------
    create_eventid_epochs : Create epochs based on specific events
    create_sl_epochs : Create statistical learning epochs
    mne.make_fixed_length_events : Generate events for fixed-length epochs
    mne.Epochs : MNE epochs class
    """
    # Input validation
    if not isinstance(data, mne.io.BaseRaw):
        raise TypeError(f"Data must be an MNE Raw object, got {type(data).__name__}")

    if tmin >= tmax:
        raise ValueError(f"tmin ({tmin}) must be less than tmax ({tmax})")

    # Calculate epoch duration if not provided
    if duration is None:
        duration = tmax - tmin

    # Validate overlap
    if overlap < 0:
        raise ValueError(f"Overlap must be non-negative, got {overlap}")

    if overlap >= duration:
        raise ValueError(
            f"Overlap ({overlap}s) must be less than epoch duration ({duration}s)"
        )

    try:
        # Create fixed-length events
        events = mne.make_fixed_length_events(
            data,
            duration=duration,
            overlap=overlap,
            start=abs(tmin),  # Start after the negative tmin offset
        )

        if len(events) == 0:
            raise RuntimeError("No events could be created - data may be too short")

        # Create epochs from the synthetic events
        epochs = mne.Epochs(
            data,
            events,
            event_id=None,  # Single event type for regular epochs
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            reject=reject,
            flat=flat,
            reject_by_annotation=reject_by_annotation,
            preload=preload,
            verbose=verbose,
        )

        # Add metadata about annotations if requested
        if include_metadata:
            epochs = _add_annotation_metadata(epochs, data)

        return epochs

    except Exception as e:
        raise RuntimeError(f"Failed to create regular epochs: {str(e)}") from e


def _add_annotation_metadata(epochs: mne.Epochs, raw: mne.io.BaseRaw) -> mne.Epochs:
    """Add metadata about annotations that fall within each epoch.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs object to add metadata to.
    raw : mne.io.BaseRaw
        The raw data containing annotations.

    Returns
    -------
    epochs : mne.Epochs
        Epochs object with added metadata.
    """
    try:
        # Extract all events/annotations from raw data
        events_all, event_id_all = mne.events_from_annotations(raw)
        event_descriptions = {v: k for k, v in event_id_all.items()}
    except Exception:
        # No annotations found
        events_all = None
        event_descriptions = {}

    # Get epoch timing information
    sfreq = raw.info["sfreq"]
    epoch_samples = epochs.events[:, 0]  # Sample indices of epoch triggers
    tmin_samples = int(epochs.tmin * sfreq)
    tmax_samples = int(epochs.tmax * sfreq)

    # Build metadata for each epoch
    metadata_rows = []

    for i, epoch_start_sample in enumerate(epoch_samples):
        # Calculate sample range for this epoch
        epoch_start = epoch_start_sample + tmin_samples
        epoch_end = epoch_start_sample + tmax_samples

        # Start with the fixed epoch marker
        epoch_events = [("fixed_marker", 0.0)]

        # Find annotations that fall within this epoch
        if events_all is not None and len(events_all) > 0:
            for sample, _, code in events_all:
                if epoch_start <= sample <= epoch_end:
                    # Calculate relative time within epoch
                    relative_time = (sample - epoch_start_sample) / sfreq
                    # Get description for this event code
                    label = event_descriptions.get(code, f"code_{code}")
                    epoch_events.append((label, relative_time))

        metadata_rows.append(
            {
                "epoch_number": i,
                "epoch_start_sample": epoch_start_sample,
                "epoch_duration": epochs.tmax - epochs.tmin,
                "additional_events": epoch_events,
            }
        )

    # Create or update metadata DataFrame
    metadata_df = pd.DataFrame(metadata_rows)

    if epochs.metadata is not None:
        # Merge with existing metadata
        epochs.metadata = pd.concat([epochs.metadata, metadata_df], axis=1)
    else:
        # Create new metadata
        epochs.metadata = metadata_df

    return epochs
