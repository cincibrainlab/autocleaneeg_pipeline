"""Event-based epochs creation functions for EEG data.

This module provides standalone functions for creating epochs based on specific
event markers in continuous EEG data.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np


def create_eventid_epochs(
    data: mne.io.BaseRaw,
    event_id: Union[Dict[str, int], List[int], int],
    tmin: float = -0.5,
    tmax: float = 2.0,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = (None, 0),
    reject: Optional[Dict[str, float]] = None,
    flat: Optional[Dict[str, float]] = None,
    reject_by_annotation: bool = True,
    decim: int = 1,
    detrend: Optional[Union[int, str]] = None,
    picks: Optional[Union[str, List[str]]] = None,
    preload: bool = True,
    on_missing: str = "raise",
    verbose: Optional[bool] = None,
) -> mne.Epochs:
    """Create epochs based on specific event IDs from continuous EEG data.

    This function creates epochs centered around specific event markers in the
    continuous EEG recording. This approach is fundamental for event-related
    potential (ERP) analysis and other time-locked analyses where brain responses
    to specific stimuli or events need to be examined.

    The function extracts events from the raw data annotations or event channel,
    filters for the specified event IDs, and creates epochs around these events
    with the specified time window and preprocessing parameters.

    Parameters
    ----------
    data : mne.io.BaseRaw
        The continuous EEG data containing events to create epochs from.
    event_id : dict, list of int, or int
        Event identifier(s) to create epochs for. Can be:
        - Dict mapping event names to event codes: {'target': 1, 'standard': 2}
        - List of event codes: [1, 2, 3]
        - Single event code: 1
    tmin : float, default -0.5
        Start time of the epoch relative to the event in seconds.
        Negative values start before the event occurrence.
    tmax : float, default 2.0
        End time of the epoch relative to the event in seconds.
        Positive values extend after the event occurrence.
    baseline : tuple of (float, float) or None, default (None, 0)
        Time interval for baseline correction in seconds relative to event.
        (None, 0) uses the entire pre-event period, (-0.2, 0) uses 200ms
        before event. None applies no baseline correction.
    reject : dict or None, default None
        Rejection thresholds for different channel types in volts.
        Example: {'eeg': 100e-6, 'eog': 200e-6}. Epochs exceeding these
        thresholds will be marked as bad and potentially dropped.
    flat : dict or None, default None
        Rejection thresholds for flat channels in volts (minimum required range).
        Example: {'eeg': 1e-6}. Channels with insufficient signal range
        will cause epoch rejection.
    reject_by_annotation : bool, default True
        Whether to automatically reject epochs that overlap with 'bad' annotations.
        If False, epochs are marked but not dropped automatically.
    decim : int, default 1
        Decimation factor for downsampling. 1 means no decimation, 2 means
        every other sample, etc. Useful for reducing memory usage.
    detrend : int, str, or None, default None
        Detrending method. None (no detrending), 0 (remove DC), 1 (remove
        linear trend), or 'linear' (same as 1).
    picks : str, list of str, or None, default None
        Channels to include. None includes all data channels. Can be channel
        names, channel types ('eeg', 'meg'), or channel indices.
    preload : bool, default True
        Whether to preload epoch data into memory. Recommended for most use cases.
    on_missing : str, default 'raise'
        What to do if no events are found for the specified event_id.
        Options: 'raise', 'warn', 'ignore'.
    verbose : bool or None, default None
        Control verbosity of output. If None, uses MNE default.

    Returns
    -------
    epochs : mne.Epochs
        The created epochs object containing data segments around specified events.

    Raises
    ------
    TypeError
        If data is not an MNE Raw object.
    ValueError
        If event_id format is invalid, tmin >= tmax, or no events found when
        on_missing='raise'.
    RuntimeError
        If epoch creation fails due to data issues or processing errors.

    Notes
    -----
    Event-based epoching is fundamental to EEG analysis for experimental paradigms
    where brain responses to specific stimuli or events are of interest. The function
    handles the complete workflow from event detection to epoch creation.

    **Event Detection:**
    The function first extracts events from the raw data using MNE's
    `events_from_annotations` function, which converts annotations to discrete
    events. This works with various data formats that store events as annotations.

    **Epoch Timing:**
    Each epoch spans from tmin to tmax seconds relative to the event occurrence.
    The choice of time window should consider:
    - Pre-stimulus baseline period (typically 100-500ms before event)
    - Expected response duration (varies by component, e.g., P300 ~300-600ms)
    - Post-stimulus recovery period

    **Baseline Correction:**
    Baseline correction removes pre-stimulus activity to isolate event-related
    responses. Common baseline periods:
    - (None, 0): Entire pre-stimulus period
    - (-0.2, 0): 200ms before stimulus
    - (-0.1, -0.05): Specific pre-stimulus window

    **Artifact Rejection:**
    Multiple rejection mechanisms help ensure data quality:
    - Amplitude-based: Reject epochs with excessive voltage deflections
    - Flatline detection: Reject epochs with insufficient signal variation
    - Annotation-based: Reject epochs overlapping with marked bad segments

    **Memory Considerations:**
    For large datasets, consider using decim parameter to reduce sampling rate
    and memory usage, especially if high temporal resolution is not critical.

    Examples
    --------
    Create epochs for target and standard stimuli:

    >>> from autoclean import create_eventid_epochs
    >>> event_id = {'target': 1, 'standard': 2}
    >>> epochs = create_eventid_epochs(
    ...     raw,
    ...     event_id=event_id,
    ...     tmin=-0.2,
    ...     tmax=0.8,
    ...     baseline=(-0.2, 0)
    ... )

    Create epochs with artifact rejection:

    >>> epochs = create_eventid_epochs(
    ...     raw,
    ...     event_id={'stimulus': 1},
    ...     tmin=-0.1,
    ...     tmax=0.6,
    ...     reject={'eeg': 100e-6, 'eog': 200e-6},
    ...     flat={'eeg': 1e-6}
    ... )

    Create epochs with decimation for memory efficiency:

    >>> epochs = create_eventid_epochs(
    ...     raw,
    ...     event_id=[1, 2, 3],
    ...     tmin=-0.5,
    ...     tmax=1.0,
    ...     decim=2,  # Downsample by factor of 2
    ...     detrend=1  # Remove linear trend
    ... )

    See Also
    --------
    mne.events_from_annotations : Extract events from annotations
    mne.Epochs : MNE epochs class
    autoclean.create_regular_epochs : Create fixed-length epochs
    """

    # Input validation
    if not isinstance(data, mne.io.BaseRaw):
        raise TypeError(f"Data must be an MNE Raw object, got {type(data).__name__}")

    if tmin >= tmax:
        raise ValueError(f"tmin ({tmin}) must be less than tmax ({tmax})")

    # Normalize event_id to dictionary format
    if isinstance(event_id, int):
        event_id = {f"event_{event_id}": event_id}
    elif isinstance(event_id, (list, tuple)):
        event_id = {f"event_{id}": id for id in event_id}
    elif not isinstance(event_id, dict):
        raise ValueError(
            f"event_id must be dict, list, or int, got {type(event_id).__name__}"
        )

    try:
        # Extract events from raw data
        try:
            events, event_id_all = mne.events_from_annotations(
                data, event_id=event_id, verbose=verbose
            )
        except Exception as e:
            if on_missing == "raise":
                raise ValueError(f"No events found in data: {str(e)}") from e
            elif on_missing == "warn":
                warnings.warn(f"No events found in data: {str(e)}")
                # Create empty epochs object - return early
                n_samples = int((tmax - tmin) * data.info["sfreq"])
                empty_epochs = mne.EpochsArray(
                    np.empty((0, data.info["nchan"], n_samples)), data.info, tmin=tmin
                )
                return empty_epochs
            else:  # on_missing == 'ignore'
                # Create empty epochs object - return early
                n_samples = int((tmax - tmin) * data.info["sfreq"])
                empty_epochs = mne.EpochsArray(
                    np.empty((0, data.info["nchan"], n_samples)), data.info, tmin=tmin
                )
                return empty_epochs

        # Filter events for requested event IDs
        requested_events = []
        final_event_id = {}

        for event_name, event_code in event_id.items():
            if event_code in event_id_all.values():
                # Find matching events
                matching_events = events[events[:, 2] == event_code]
                requested_events.append(matching_events)
                final_event_id[event_name] = event_code

        if requested_events:
            # Combine all requested events
            filtered_events = np.vstack(requested_events)
            # Sort by time
            filtered_events = filtered_events[filtered_events[:, 0].argsort()]
        else:
            filtered_events = np.empty((0, 3), dtype=int)
            final_event_id = {}

        # Check if any events were found
        if len(filtered_events) == 0:
            if on_missing == "raise":
                raise ValueError(
                    f"No events found for specified event_id: {event_id}. "
                    f"Available events: {list(event_id_all.keys())}"
                )
            elif on_missing == "warn":
                warnings.warn(
                    f"No events found for specified event_id: {event_id}. "
                    f"Available events: {list(event_id_all.keys())}"
                )
                # Create empty epochs object - return early
                n_samples = int((tmax - tmin) * data.info["sfreq"])
                empty_epochs = mne.EpochsArray(
                    np.empty((0, data.info["nchan"], n_samples)), data.info, tmin=tmin
                )
                return empty_epochs
            else:  # on_missing == 'ignore'
                # Create empty epochs object - return early
                n_samples = int((tmax - tmin) * data.info["sfreq"])
                empty_epochs = mne.EpochsArray(
                    np.empty((0, data.info["nchan"], n_samples)), data.info, tmin=tmin
                )
                return empty_epochs

        # Create epochs
        epochs = mne.Epochs(
            data,
            filtered_events,
            event_id=final_event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            reject=reject,
            flat=flat,
            reject_by_annotation=reject_by_annotation,
            decim=decim,
            detrend=detrend,
            picks=picks,
            preload=preload,
            verbose=verbose,
        )

        return epochs

    except Exception as e:
        if "No events found" in str(e):
            # Let the ValueError bubble up for proper test handling
            raise
        raise RuntimeError(f"Failed to create event-based epochs: {str(e)}") from e
