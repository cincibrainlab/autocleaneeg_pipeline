"""Segment-based rejection functions for EEG data.

This module provides standalone functions for identifying and annotating
noisy or problematic segments in continuous EEG data using various statistical
and correlation-based methods.
"""

from typing import Dict, List, Optional, Union

import mne
import numpy as np
import pandas as pd
import scipy.stats
import xarray as xr
from scipy.spatial import distance_matrix


def annotate_noisy_segments(
    raw: mne.io.Raw,
    epoch_duration: float = 2.0,
    epoch_overlap: float = 0.0,
    picks: Optional[Union[List[str], str]] = None,
    quantile_k: float = 3.0,
    quantile_flag_crit: float = 0.2,
    annotation_description: str = "BAD_noisy_segment",
    verbose: Optional[bool] = None
) -> mne.io.Raw:
    """Identify and annotate noisy segments in continuous EEG data.
    
    This function temporarily epochs the continuous data, calculates channel-wise
    standard deviations for each epoch, and then identifies epochs where a
    significant proportion of channels exhibit outlier standard deviations.
    The outlier detection is based on the interquartile range (IQR) method,
    similar to what's used in the pylossless pipeline.
    
    The method works by:
    1. Creating temporary fixed-length epochs from continuous data
    2. Calculating standard deviation for each channel in each epoch
    3. Using IQR-based outlier detection to identify abnormal standard deviations
    4. Flagging epochs where too many channels show outlier behavior
    5. Adding annotations to mark these problematic time periods
    
    Parameters
    ----------
    raw : mne.io.Raw
        The continuous EEG data to analyze for noisy segments.
    epoch_duration : float, default 2.0
        Duration of epochs in seconds for noise detection. Shorter epochs
        provide finer temporal resolution but may be less stable for
        outlier detection.
    epoch_overlap : float, default 0.0
        Overlap between epochs in seconds. Non-zero overlap provides
        smoother detection but increases computation time.
    picks : list of str, str, or None, default None
        Channels to include in analysis. If None, defaults to 'eeg'.
        Can be channel names (e.g., ['EEG 001', 'EEG 002']) or channel
        types (e.g., 'eeg', 'grad').
    quantile_k : float, default 3.0
        Multiplier for the IQR when defining outlier thresholds for channel
        standard deviations. A channel's std in an epoch is an outlier if it's
        k IQRs above Q3 or below Q1 relative to its own distribution of stds
        across all epochs. Higher values = more conservative detection.
    quantile_flag_crit : float, default 0.2
        Proportion threshold (0.0-1.0). If more than this proportion of picked
        channels are marked as outliers (having outlier std) within an epoch,
        that epoch is flagged as noisy. Lower values = more sensitive detection.
    annotation_description : str, default "BAD_noisy_segment"
        The description to use for MNE annotations marking noisy segments.
        Should start with "BAD_" to be recognized by MNE as artifact annotations.
    verbose : bool or None, default None
        Control verbosity of output during processing.
        
    Returns
    -------
    raw_annotated : mne.io.Raw
        Copy of input Raw object with added annotations for noisy segments.
        Original data is not modified.
        
    Raises
    ------
    TypeError
        If raw is not an MNE Raw object.
    ValueError
        If parameters are outside valid ranges or no epochs can be created.
    RuntimeError
        If processing fails due to insufficient data or other errors.
        
    Notes
    -----
    **Detection Algorithm:**
    1. For each channel, its standard deviation is calculated within each epoch
    2. For each channel, the distribution of its standard deviations across all
       epochs is analyzed using quartiles (Q1, Q3) and IQR
    3. Outlier thresholds are: Q1 - k*IQR and Q3 + k*IQR
    4. An epoch is marked as noisy if the proportion of channels whose standard
       deviation falls outside their respective outlier bounds exceeds the
       quantile_flag_crit threshold
    
    **Parameter Guidelines:**
    - epoch_duration: 1-4 seconds typical. Shorter for transient artifacts,
      longer for stable outlier detection
    - quantile_k: 2-4 typical. Higher values = fewer false positives
    - quantile_flag_crit: 0.1-0.3 typical. Lower = more sensitive
    
    **Performance Considerations:**
    - Processing time scales with (data_length / epoch_duration)
    - Memory usage depends on number of epochs and channels
    - Overlap increases computation but may improve detection continuity
    
    Examples
    --------
    Basic noise detection with default parameters:
    
    >>> from autoclean import annotate_noisy_segments
    >>> raw_clean = annotate_noisy_segments(raw)
    >>> noisy_annotations = [ann for ann in raw_clean.annotations 
    ...                     if 'noisy' in ann['description']]
    >>> print(f"Found {len(noisy_annotations)} noisy segments")
    
    Conservative detection for high-quality data:
    
    >>> raw_clean = annotate_noisy_segments(
    ...     raw,
    ...     epoch_duration=3.0,
    ...     quantile_k=4.0,
    ...     quantile_flag_crit=0.3
    ... )
    
    Sensitive detection for noisy data:
    
    >>> raw_clean = annotate_noisy_segments(
    ...     raw,
    ...     epoch_duration=1.0,
    ...     quantile_k=2.0,
    ...     quantile_flag_crit=0.1,
    ...     annotation_description="BAD_very_noisy"
    ... )
    
    EEG-only detection with channel selection:
    
    >>> raw_clean = annotate_noisy_segments(
    ...     raw,
    ...     picks=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4'],
    ...     epoch_duration=2.0
    ... )
    
    See Also
    --------
    annotate_uncorrelated_segments : Detect segments with poor channel correlations
    mne.Annotations : MNE annotations system
    autoclean.detect_outlier_epochs : Statistical outlier detection for epochs
    
    References
    ----------
    This implementation adapts concepts from the PREP pipeline and pylossless:
    
    Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K. M., & Robbins, K. A. (2015).
    The PREP pipeline: standardized preprocessing for large-scale EEG analysis.
    Frontiers in neuroinformatics, 9, 16.
    """
    # Input validation
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(
            f"Data must be an MNE Raw object, got {type(raw).__name__}"
        )
    
    if epoch_duration <= 0:
        raise ValueError(f"epoch_duration must be positive, got {epoch_duration}")
    
    if epoch_overlap < 0:
        raise ValueError(f"epoch_overlap must be non-negative, got {epoch_overlap}")
    
    if not 0 <= quantile_flag_crit <= 1:
        raise ValueError(f"quantile_flag_crit must be between 0 and 1, got {quantile_flag_crit}")
    
    if quantile_k <= 0:
        raise ValueError(f"quantile_k must be positive, got {quantile_k}")
    
    # Set default picks
    if picks is None:
        picks = "eeg"
    
    try:
        # Create fixed-length epochs for analysis
        events = mne.make_fixed_length_events(
            raw, duration=epoch_duration, overlap=epoch_overlap
        )
        
        # Ensure events are within data boundaries
        if events.shape[0] == 0:
            raise ValueError("No epochs could be created with the given parameters")
        
        max_event_time = events[-1, 0] + int(epoch_duration * raw.info["sfreq"])
        if max_event_time > len(raw.times):
            # Prune events that would lead to epochs exceeding data length
            valid_events_mask = (
                events[:, 0] + int(epoch_duration * raw.info["sfreq"]) <= len(raw.times)
            )
            events = events[valid_events_mask]
            if events.shape[0] == 0:
                raise ValueError("No valid epochs after boundary check")
        
        # Create epochs
        epochs = mne.Epochs(
            raw,
            events,
            tmin=0.0,
            tmax=epoch_duration - 1.0 / raw.info["sfreq"],
            picks=picks,
            preload=True,
            baseline=None,  # No baseline correction for std calculation
            reject=None,  # We are detecting bads, not rejecting yet
            verbose=verbose
        )
        
        if len(epochs) == 0:
            raise ValueError(
                f"No epochs left after picking channels: {picks}"
            )
        
        # Convert epochs to xarray DataArray (channels, epochs, time)
        epochs_xr = _epochs_to_xr(epochs)
        
        # Calculate standard deviation for each channel within each epoch
        data_sd = epochs_xr.std("time")  # Shape: (channels, epochs)
        
        # Detect noisy epochs using outlier detection logic
        outliers_kwargs = {"k": quantile_k}
        
        bad_epoch_indices = _detect_outliers(
            data_sd,
            flag_dim="epoch",  # We want to flag epochs
            outlier_method="quantile",
            flag_crit=quantile_flag_crit,
            init_dir="pos",  # Interested in high std_dev for noise
            outliers_kwargs=outliers_kwargs,
        )
        
        if len(bad_epoch_indices) == 0:
            # No noisy segments found, return copy of original
            return raw.copy()
        
        # Add annotations to the original raw object
        relative_onsets = epochs.events[bad_epoch_indices, 0] / raw.info["sfreq"]
        onsets = relative_onsets - raw.first_samp / raw.info["sfreq"]
        
        # Duration of each annotation matches epoch_duration
        annotation_durations = np.full_like(onsets, fill_value=epoch_duration)
        descriptions = [annotation_description] * len(bad_epoch_indices)
        
        # Create new annotations
        new_annotations = mne.Annotations(
            onset=onsets,
            duration=annotation_durations,
            description=descriptions,
            orig_time=raw.annotations.orig_time,
        )
        
        # Make a copy and add annotations
        raw_annotated = raw.copy()
        raw_annotated.set_annotations(raw_annotated.annotations + new_annotations)
        
        return raw_annotated
        
    except Exception as e:
        raise RuntimeError(f"Failed to annotate noisy segments: {str(e)}") from e


def annotate_uncorrelated_segments(
    raw: mne.io.Raw,
    epoch_duration: float = 2.0,
    epoch_overlap: float = 0.0,
    picks: Optional[Union[List[str], str]] = None,
    n_nearest_neighbors: int = 5,
    corr_method: str = "max",
    corr_trim_percent: float = 10.0,
    outlier_k: float = 4.0,
    outlier_flag_crit: float = 0.2,
    annotation_description: str = "BAD_uncorrelated_segment",
    verbose: Optional[bool] = None
) -> mne.io.Raw:
    """Identify and annotate segments with poor channel-neighbor correlations.
    
    This function temporarily epochs data, calculates correlations between each
    channel and its spatial neighbors for each epoch, and then flags epochs
    where a significant proportion of channels show unusually low correlations.
    Outlier detection for low correlations is based on the IQR method.
    
    The method works by:
    1. Creating temporary fixed-length epochs from continuous data
    2. Finding spatial neighbors for each channel based on montage positions
    3. Calculating correlations between each channel and its neighbors in each epoch
    4. Using IQR-based outlier detection to identify abnormally low correlations
    5. Flagging epochs where too many channels show poor neighbor correlations
    6. Adding annotations to mark these problematic time periods
    
    Parameters
    ----------
    raw : mne.io.Raw
        The continuous EEG data to analyze. Must have a montage set for
        channel positions.
    epoch_duration : float, default 2.0
        Duration of epochs in seconds for correlation analysis.
    epoch_overlap : float, default 0.0
        Overlap between epochs in seconds.
    picks : list of str, str, or None, default None
        Channels to include. If None, defaults to 'eeg'.
    n_nearest_neighbors : int, default 5
        Number of nearest spatial neighbors to consider for correlation.
        Higher values use more neighbors but may dilute local effects.
    corr_method : str, default "max"
        Method to aggregate correlations with neighbors:
        - "max": Maximum absolute correlation with any neighbor
        - "mean": Mean absolute correlation across all neighbors  
        - "trimmean": Trimmed mean (removes extreme values)
    corr_trim_percent : float, default 10.0
        If corr_method is "trimmean", percentage to trim from each end
        before averaging (e.g., 10.0 = trim 10% from each end).
    outlier_k : float, default 4.0
        Multiplier for the IQR when defining outlier thresholds for low
        correlations. A channel's correlation in an epoch is an outlier if it's
        k IQRs below Q1 of its own distribution across all epochs.
    outlier_flag_crit : float, default 0.2
        Proportion threshold (0.0-1.0). If more than this proportion of picked
        channels are marked as outliers (having low correlations) within an epoch,
        that epoch is flagged as uncorrelated.
    annotation_description : str, default "BAD_uncorrelated_segment"
        Description for MNE annotations marking these segments.
    verbose : bool or None, default None
        Control verbosity of output during processing.
        
    Returns
    -------
    raw_annotated : mne.io.Raw
        Copy of input Raw object with added annotations for uncorrelated segments.
        
    Raises
    ------
    TypeError
        If raw is not an MNE Raw object.
    ValueError
        If parameters are invalid or no montage is set.
    RuntimeError
        If processing fails due to insufficient data or other errors.
        
    Notes
    -----
    **Requirements:**
    - Raw object must have a montage set for channel positions
    - Sufficient good channels around each channel for meaningful correlations
    - At least n_nearest_neighbors + 1 channels total
    
    **Detection Algorithm:**
    1. For each epoch and channel, calculate correlations with spatial neighbors
    2. Aggregate neighbor correlations using specified method (max/mean/trimmean)
    3. For each channel, analyze distribution of correlations across epochs
    4. Use IQR-based outlier detection to identify abnormally low correlations
    5. Flag epochs where proportion of low-correlation channels exceeds threshold
    
    **Parameter Guidelines:**
    - n_nearest_neighbors: 3-8 typical. Higher for dense arrays, lower for sparse
    - corr_method: "max" most sensitive, "mean" most stable, "trimmean" robust
    - outlier_k: 3-5 typical. Higher values = fewer false positives
    - outlier_flag_crit: 0.1-0.3 typical. Lower = more sensitive
    
    Examples
    --------
    Basic correlation-based detection:
    
    >>> from autoclean import annotate_uncorrelated_segments
    >>> # Ensure montage is set
    >>> raw.set_montage('standard_1020')
    >>> raw_clean = annotate_uncorrelated_segments(raw)
    
    Conservative detection with more neighbors:
    
    >>> raw_clean = annotate_uncorrelated_segments(
    ...     raw,
    ...     n_nearest_neighbors=8,
    ...     corr_method="mean",
    ...     outlier_k=5.0,
    ...     outlier_flag_crit=0.3
    ... )
    
    Sensitive detection for artifact-prone channels:
    
    >>> raw_clean = annotate_uncorrelated_segments(
    ...     raw,
    ...     n_nearest_neighbors=3,
    ...     corr_method="max",
    ...     outlier_k=3.0,
    ...     outlier_flag_crit=0.15
    ... )
    
    Robust detection with trimmed mean:
    
    >>> raw_clean = annotate_uncorrelated_segments(
    ...     raw,
    ...     corr_method="trimmean",
    ...     corr_trim_percent=20.0,
    ...     epoch_duration=3.0
    ... )
    
    See Also
    --------
    annotate_noisy_segments : Detect segments with high noise levels
    mne.channels.find_ch_adjacency : Find channel adjacency for correlation analysis
    scipy.spatial.distance_matrix : Spatial distance calculations
    
    References
    ----------
    This implementation adapts concepts from the PREP pipeline:
    
    Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K. M., & Robbins, K. A. (2015).
    The PREP pipeline: standardized preprocessing for large-scale EEG analysis.
    Frontiers in neuroinformatics, 9, 16.
    """
    # Input validation
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(
            f"Data must be an MNE Raw object, got {type(raw).__name__}"
        )
    
    if epoch_duration <= 0:
        raise ValueError(f"epoch_duration must be positive, got {epoch_duration}")
    
    if epoch_overlap < 0:
        raise ValueError(f"epoch_overlap must be non-negative, got {epoch_overlap}")
    
    if n_nearest_neighbors <= 0:
        raise ValueError(f"n_nearest_neighbors must be positive, got {n_nearest_neighbors}")
    
    if corr_method not in ["max", "mean", "trimmean"]:
        raise ValueError(f"corr_method must be 'max', 'mean', or 'trimmean', got '{corr_method}'")
    
    if not 0 <= corr_trim_percent <= 50:
        raise ValueError(f"corr_trim_percent must be between 0 and 50, got {corr_trim_percent}")
    
    if not 0 <= outlier_flag_crit <= 1:
        raise ValueError(f"outlier_flag_crit must be between 0 and 1, got {outlier_flag_crit}")
    
    if outlier_k <= 0:
        raise ValueError(f"outlier_k must be positive, got {outlier_k}")
    
    # Set default picks
    if picks is None:
        picks = "eeg"
    
    try:
        # Create fixed-length epochs
        events = mne.make_fixed_length_events(
            raw, duration=epoch_duration, overlap=epoch_overlap
        )
        
        if events.shape[0] == 0:
            raise ValueError("No epochs could be created with the given parameters")
        
        max_event_time = events[-1, 0] + int(epoch_duration * raw.info["sfreq"])
        if max_event_time > len(raw.times):
            valid_events_mask = (
                events[:, 0] + int(epoch_duration * raw.info["sfreq"]) <= len(raw.times)
            )
            events = events[valid_events_mask]
            if events.shape[0] == 0:
                raise ValueError("No valid epochs after boundary check")
        
        epochs = mne.Epochs(
            raw,
            events,
            tmin=0.0,
            tmax=epoch_duration - 1.0 / raw.info["sfreq"],
            picks=picks,
            preload=True,
            baseline=None,
            reject=None,
            verbose=verbose
        )
        
        if len(epochs) == 0:
            raise ValueError(f"No epochs left after picking channels: {picks}")
        
        if epochs.get_montage() is None:
            raise ValueError(
                "The raw object must have a montage set for spatial neighbor analysis. "
                "Use raw.set_montage() to add channel positions."
            )
        
        # Calculate nearest neighbor correlations for channels within epochs
        data_r_ch = _calculate_neighbor_correlations(
            epochs,
            n_nearest_neighbors=n_nearest_neighbors,
            corr_method=corr_method,
            corr_trim_percent=corr_trim_percent,
        )
        
        # Detect epochs with too many uncorrelated channels
        # Looking for *low* correlations, so init_dir="neg"
        outliers_kwargs = {"k": outlier_k}
        bad_epoch_indices = _detect_outliers(
            data_r_ch,
            flag_dim="epoch",
            outlier_method="quantile",
            flag_crit=outlier_flag_crit,
            init_dir="neg",  # Flagging based on *low* correlation values
            outliers_kwargs=outliers_kwargs,
        )
        
        if len(bad_epoch_indices) == 0:
            # No uncorrelated segments found
            return raw.copy()
        
        # Add annotations to the original raw object
        absolute_onsets = (
            epochs.events[bad_epoch_indices, 0] - raw.first_samp
        ) / raw.info["sfreq"]
        
        annotation_durations = np.full_like(absolute_onsets, fill_value=epoch_duration)
        descriptions = [annotation_description] * len(bad_epoch_indices)
        
        new_annotations = mne.Annotations(
            onset=absolute_onsets,
            duration=annotation_durations,
            description=descriptions,
            orig_time=raw.annotations.orig_time,
        )
        
        raw_annotated = raw.copy()
        raw_annotated.set_annotations(raw_annotated.annotations + new_annotations)
        
        return raw_annotated
        
    except Exception as e:
        raise RuntimeError(f"Failed to annotate uncorrelated segments: {str(e)}") from e


# Helper functions
def _epochs_to_xr(epochs: mne.Epochs) -> xr.DataArray:
    """Create an Xarray DataArray from MNE Epochs.
    
    Converts epochs data to xarray format for easier manipulation
    with dimensions (channels, epochs, time).
    """
    data = epochs.get_data()  # n_epochs, n_channels, n_times
    ch_names = epochs.ch_names
    # Transpose to (n_channels, n_epochs, n_times)
    data_transposed = data.transpose(1, 0, 2)
    return xr.DataArray(
        data_transposed,
        coords={
            "ch": ch_names,
            "epoch": np.arange(data_transposed.shape[1]),
            "time": epochs.times,
        },
        dims=("ch", "epoch", "time"),
    )


def _get_outliers_quantile(
    array: xr.DataArray, 
    dim: str, 
    lower: float = 0.25, 
    upper: float = 0.75, 
    mid: float = 0.5, 
    k: float = 3.0
) -> tuple:
    """Calculate outlier bounds based on the IQR method."""
    lower_val, mid_val, upper_val = array.quantile([lower, mid, upper], dim=dim)
    
    lower_dist = mid_val - lower_val
    upper_dist = upper_val - mid_val
    return mid_val - lower_dist * k, mid_val + upper_dist * k


def _detect_outliers(
    array: xr.DataArray,
    flag_dim: str,
    outlier_method: str = "quantile",
    flag_crit: float = 0.2,
    init_dir: str = "pos",
    outliers_kwargs: Optional[Dict] = None,
) -> np.ndarray:
    """Mark items along flag_dim as flagged for artifact."""
    if outliers_kwargs is None:
        outliers_kwargs = {}
    
    # Determine the dimension to operate across
    dims = list(array.dims)
    if flag_dim not in dims:
        raise ValueError(f"flag_dim '{flag_dim}' not in array dimensions: {dims}")
    dims.remove(flag_dim)
    if not dims:
        raise ValueError("Array must have at least two dimensions")
    operate_dim = dims[0]
    
    if outlier_method == "quantile":
        l_out, u_out = _get_outliers_quantile(array, dim=flag_dim, **outliers_kwargs)
    else:
        raise ValueError(f"outlier_method '{outlier_method}' not supported. Use 'quantile'")
    
    outlier_mask = xr.zeros_like(array, dtype=bool)
    
    if init_dir == "pos" or init_dir == "both":
        outlier_mask = outlier_mask | (array > u_out)
    if init_dir == "neg" or init_dir == "both":
        outlier_mask = outlier_mask | (array < l_out)
    
    # Calculate proportion of outliers along operate_dim
    prop_outliers = outlier_mask.astype(float).mean(operate_dim)
    
    if "quantile" in list(prop_outliers.coords.keys()):
        prop_outliers = prop_outliers.drop_vars("quantile")
    
    flagged_indices = (
        prop_outliers[prop_outliers > flag_crit].coords[flag_dim].values
    )
    return flagged_indices


def _calculate_neighbor_correlations(
    epochs: mne.Epochs,
    n_nearest_neighbors: int,
    corr_method: str = "max",
    corr_trim_percent: float = 10.0,
) -> xr.DataArray:
    """Compute nearest neighbor correlations for channels within epochs."""
    montage = epochs.get_montage()
    ch_positions = montage.get_positions()["ch_pos"]
    valid_chs = [ch for ch in epochs.ch_names if ch in ch_positions]
    
    if not valid_chs:
        raise ValueError("No channel positions found for any channels in the epochs object")
    
    if len(valid_chs) <= n_nearest_neighbors:
        actual_n_neighbors = max(0, len(valid_chs) - 1)
    else:
        actual_n_neighbors = n_nearest_neighbors
    
    # Create distance matrix and find nearest neighbors
    ch_locs_df = pd.DataFrame(ch_positions).T.loc[valid_chs]
    dist_matrix_val = distance_matrix(ch_locs_df.values, ch_locs_df.values)
    chan_dist_df = pd.DataFrame(
        dist_matrix_val, columns=ch_locs_df.index, index=ch_locs_df.index
    )
    
    rank = chan_dist_df.rank(axis="columns", method="first", ascending=True) - 1
    rank[rank == 0] = np.nan
    
    nearest_neighbor_df = pd.DataFrame(
        index=ch_locs_df.index, columns=range(actual_n_neighbors), dtype=object
    )
    for ch_name_iter in ch_locs_df.index:
        sorted_neighbors = rank.loc[ch_name_iter].dropna().sort_values()
        nearest_neighbor_df.loc[ch_name_iter] = sorted_neighbors.index[
            :actual_n_neighbors
        ].values
    
    # Pick only valid channels for epochs_xr
    epochs_xr = _epochs_to_xr(epochs.copy().pick(valid_chs))
    
    all_channel_corrs = []
    
    for _, ch_name in enumerate(valid_chs):
        neighbor_names_for_ch = [
            n
            for n in nearest_neighbor_df.loc[ch_name].values.tolist()
            if pd.notna(n) and n != ch_name
        ]
        
        if not neighbor_names_for_ch:
            # No valid neighbors
            ch_neighbor_corr_aggregated = xr.DataArray(
                np.full(epochs_xr.sizes["epoch"], np.nan),
                coords={"epoch": epochs_xr.coords["epoch"]},
                dims=["epoch"],
            )
        else:
            # Data for the current reference channel
            this_ch_data = epochs_xr.sel(ch=ch_name)
            
            # Data for its neighbors
            neighbor_chs_data = epochs_xr.sel(ch=neighbor_names_for_ch)
            
            # Calculate Pearson correlation along the 'time' dimension
            ch_to_neighbors_corr = xr.corr(this_ch_data, neighbor_chs_data, dim="time")
            
            # Aggregate correlations based on corr_method
            if corr_method == "max":
                ch_neighbor_corr_aggregated = np.abs(ch_to_neighbors_corr).max(dim="ch")
            elif corr_method == "mean":
                ch_neighbor_corr_aggregated = np.abs(ch_to_neighbors_corr).mean(dim="ch")
            elif corr_method == "trimmean":
                proportion_to_cut = corr_trim_percent / 100.0
                np_data = np.abs(ch_to_neighbors_corr).transpose("epoch", "ch").data
                
                trimmed_means_per_epoch = [
                    (
                        scipy.stats.trim_mean(
                            epoch_data_for_trim, proportiontocut=proportion_to_cut
                        )
                        if not np.all(np.isnan(epoch_data_for_trim)) and len(epoch_data_for_trim) > 0
                        else np.nan
                    )
                    for epoch_data_for_trim in np_data
                ]
                ch_neighbor_corr_aggregated = xr.DataArray(
                    trimmed_means_per_epoch,
                    coords={"epoch": ch_to_neighbors_corr.coords["epoch"]},
                    dims=["epoch"],
                )
            else:
                raise ValueError(f"Unknown corr_method: {corr_method}")
        
        # Expand to add the reference channel's name as a new 'ch' dimension
        expanded_corr = ch_neighbor_corr_aggregated.expand_dims(dim={"ch": [ch_name]})
        all_channel_corrs.append(expanded_corr)
    
    if not all_channel_corrs:
        return xr.DataArray(
            np.empty((0, epochs_xr.sizes.get("epoch", 0))),
            coords={"ch": [], "epoch": epochs_xr.coords.get("epoch", [])},
            dims=("ch", "epoch"),
        )
    
    # Concatenate results for all reference channels
    concatenated_corrs = xr.concat(all_channel_corrs, dim="ch")
    return concatenated_corrs