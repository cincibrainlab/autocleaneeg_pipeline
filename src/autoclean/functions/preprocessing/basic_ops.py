"""Basic operations for EEG data preprocessing.

This module provides standalone functions for common EEG data operations including
channel dropping, time cropping, edge trimming, and channel type assignment.
"""

import warnings
from typing import Dict, List, Optional, Union

import mne


def drop_channels(
    data: Union[mne.io.BaseRaw, mne.Epochs],
    ch_names: Union[str, List[str]],
    on_missing: str = "raise",
) -> Union[mne.io.BaseRaw, mne.Epochs]:
    """Drop channels from EEG data.

    This function removes specified channels from continuous (Raw) or epoched EEG data.
    Useful for removing bad channels, artifact channels, or channels not needed for analysis.

    Parameters
    ----------
    data : mne.io.BaseRaw or mne.Epochs
        The EEG data from which to drop channels.
    ch_names : str or list of str
        Name(s) of channel(s) to drop. Can be a single channel name or list of names.
    on_missing : str, default 'raise'
        What to do if a channel name is not found. Options:
        - 'raise': Raise an error
        - 'warn': Issue a warning and continue
        - 'ignore': Silently ignore missing channels

    Returns
    -------
    data_dropped : mne.io.BaseRaw or mne.Epochs
        Data with specified channels removed.

    Raises
    ------
    TypeError
        If data is not an MNE Raw or Epochs object.
    ValueError
        If on_missing='raise' and channels are not found.

    Examples
    --------
    Drop single channel:

    >>> from autoclean import drop_channels
    >>> data_clean = drop_channels(raw, 'E125')  # Drop channel E125

    Drop multiple channels:

    >>> data_clean = drop_channels(raw, ['E125', 'E126', 'E127'])

    Drop channels with warning for missing:

    >>> data_clean = drop_channels(raw, ['E125', 'NonExistent'], on_missing='warn')
    """
    if not isinstance(data, (mne.io.BaseRaw, mne.Epochs)):
        raise TypeError(
            f"Data must be an MNE Raw or Epochs object, got {type(data).__name__}"
        )

    if isinstance(ch_names, str):
        ch_names = [ch_names]

    # Filter out channels that don't exist based on on_missing parameter
    existing_channels = [ch for ch in ch_names if ch in data.ch_names]
    missing_channels = [ch for ch in ch_names if ch not in data.ch_names]

    if missing_channels:
        if on_missing == "raise":
            raise ValueError(f"Channels not found in data: {missing_channels}")
        elif on_missing == "warn":
            warnings.warn(f"Channels not found in data: {missing_channels}")

    if not existing_channels:
        return data.copy()  # No channels to drop

    result = data.copy()
    result.drop_channels(existing_channels)
    return result


def crop_data(
    data: Union[mne.io.BaseRaw, mne.Epochs],
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    include_tmax: bool = True,
) -> Union[mne.io.BaseRaw, mne.Epochs]:
    """Crop EEG data to a specific time range.

    This function crops continuous (Raw) or epoched EEG data to a specified time window.
    Useful for focusing analysis on specific time periods or removing unwanted segments.

    Parameters
    ----------
    data : mne.io.BaseRaw or mne.Epochs
        The EEG data to crop.
    tmin : float or None, default None
        Start time in seconds. If None, uses the start of the data.
    tmax : float or None, default None
        End time in seconds. If None, uses the end of the data.
    include_tmax : bool, default True
        Whether to include the tmax time point.

    Returns
    -------
    cropped_data : mne.io.BaseRaw or mne.Epochs
        Data cropped to the specified time range.

    Raises
    ------
    TypeError
        If data is not an MNE Raw or Epochs object.
    ValueError
        If tmin >= tmax or times are outside data range.

    Examples
    --------
    Crop to specific time window:

    >>> from autoclean import crop_data
    >>> cropped = crop_data(raw, tmin=10.0, tmax=60.0)  # Keep 10-60 seconds

    Crop from beginning:

    >>> cropped = crop_data(raw, tmax=30.0)  # Keep first 30 seconds

    Crop to end:

    >>> cropped = crop_data(raw, tmin=5.0)  # Remove first 5 seconds
    """
    if not isinstance(data, (mne.io.BaseRaw, mne.Epochs)):
        raise TypeError(
            f"Data must be an MNE Raw or Epochs object, got {type(data).__name__}"
        )

    # Get data time bounds
    data_tmin = data.times[0]
    data_tmax = data.times[-1]

    # Use data bounds if not specified
    if tmin is None:
        tmin = data_tmin
    if tmax is None:
        tmax = data_tmax

    # Validate time range
    if tmin >= tmax:
        raise ValueError(f"tmin ({tmin}) must be less than tmax ({tmax})")

    if tmin < data_tmin or tmax > data_tmax:
        raise ValueError(
            f"Requested time range ({tmin}-{tmax}) extends beyond data range "
            f"({data_tmin}-{data_tmax})"
        )

    result = data.copy()
    result.crop(tmin=tmin, tmax=tmax, include_tmax=include_tmax)
    return result


def trim_edges(
    data: Union[mne.io.BaseRaw, mne.Epochs], duration: float
) -> Union[mne.io.BaseRaw, mne.Epochs]:
    """Trim specified duration from both edges of EEG data.

    This function removes the specified duration from both the beginning and end
    of continuous (Raw) or epoched EEG data. Useful for removing edge artifacts
    from filtering or other processing steps.

    Parameters
    ----------
    data : mne.io.BaseRaw or mne.Epochs
        The EEG data to trim.
    duration : float
        Duration in seconds to remove from each edge.

    Returns
    -------
    trimmed_data : mne.io.BaseRaw or mne.Epochs
        Data with edges trimmed.

    Raises
    ------
    TypeError
        If data is not an MNE Raw or Epochs object.
    ValueError
        If duration is negative or too large for the data.

    Examples
    --------
    Trim 1 second from each edge:

    >>> from autoclean import trim_edges
    >>> trimmed = trim_edges(raw, duration=1.0)

    Remove filter edge artifacts:

    >>> trimmed = trim_edges(filtered_raw, duration=0.5)
    """
    if not isinstance(data, (mne.io.BaseRaw, mne.Epochs)):
        raise TypeError(
            f"Data must be an MNE Raw or Epochs object, got {type(data).__name__}"
        )

    if duration < 0:
        raise ValueError(f"Duration must be non-negative, got {duration}")

    data_duration = data.times[-1] - data.times[0]

    if 2 * duration >= data_duration:
        raise ValueError(
            f"Total trim duration ({2 * duration}s) is greater than or equal to "
            f"data duration ({data_duration}s)"
        )

    tmin = data.times[0] + duration
    tmax = data.times[-1] - duration

    result = data.copy()
    result.crop(tmin=tmin, tmax=tmax)
    return result


def assign_channel_types(
    data: Union[mne.io.BaseRaw, mne.Epochs], channel_types: Dict[str, str]
) -> Union[mne.io.BaseRaw, mne.Epochs]:
    """Assign channel types to EEG data channels.

    This function sets the channel types (e.g., 'eeg', 'eog', 'ecg', 'emg') for
    specified channels. Proper channel typing is important for many MNE functions
    and affects how channels are processed and visualized.

    Parameters
    ----------
    data : mne.io.BaseRaw or mne.Epochs
        The EEG data to modify channel types for.
    channel_types : dict
        Dictionary mapping channel names to channel types.
        Common types: 'eeg', 'eog', 'ecg', 'emg', 'misc', 'stim', 'bad'.

    Returns
    -------
    retyped_data : mne.io.BaseRaw or mne.Epochs
        Data with updated channel types.

    Raises
    ------
    TypeError
        If data is not an MNE Raw or Epochs object.
    ValueError
        If channel names are not found in data.

    Examples
    --------
    Assign EOG channels:

    >>> from autoclean import assign_channel_types
    >>> types = {'E125': 'eog', 'E126': 'eog'}
    >>> retyped = assign_channel_types(raw, types)

    Mix of channel types:

    >>> types = {
    ...     'E125': 'eog',
    ...     'E126': 'eog',
    ...     'E127': 'ecg',
    ...     'E128': 'emg'
    ... }
    >>> retyped = assign_channel_types(raw, types)
    """
    if not isinstance(data, (mne.io.BaseRaw, mne.Epochs)):
        raise TypeError(
            f"Data must be an MNE Raw or Epochs object, got {type(data).__name__}"
        )

    # Validate that all channels exist
    missing_channels = [ch for ch in channel_types.keys() if ch not in data.ch_names]
    if missing_channels:
        raise ValueError(
            f"Channels not found in data: {missing_channels}. "
            f"Available channels: {data.ch_names}"
        )

    result = data.copy()
    result.set_channel_types(channel_types)
    return result
