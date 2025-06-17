"""Channel operations functions for EEG data.

This module provides standalone functions for detecting and handling bad channels
in EEG data using various statistical and correlation-based methods.
"""

from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np
from pyprep.find_noisy_channels import NoisyChannels


def detect_bad_channels(
    data: mne.io.BaseRaw,
    correlation_thresh: float = 0.35,
    deviation_thresh: float = 2.5,
    ransac_sample_prop: float = 0.35,
    ransac_corr_thresh: float = 0.65,
    ransac_frac_bad: float = 0.25,
    ransac_channel_wise: bool = False,
    random_state: int = 1337,
    exclude_channels: Optional[List[str]] = None,
    return_by_method: bool = False,
    verbose: Optional[bool] = None
) -> Union[List[str], Dict[str, List[str]]]:
    """Detect bad channels using multiple statistical methods.
    
    This function uses the PyPREP NoisyChannels implementation to detect bad
    channels through correlation, deviation, and RANSAC-based methods. These
    methods are based on established preprocessing pipelines and provide
    robust detection of various types of channel artifacts.
    
    The function implements multiple complementary detection approaches:
    - Correlation-based: Identifies channels poorly correlated with neighbors
    - Deviation-based: Finds channels with excessive amplitude deviations
    - RANSAC-based: Uses robust regression to detect outlier channels
    
    Parameters
    ----------
    data : mne.io.BaseRaw
        The continuous EEG data to analyze for bad channels.
    correlation_thresh : float, default 0.35
        Threshold for correlation-based bad channel detection. Channels with
        correlation below this value with their neighbors are marked as bad.
        Lower values are more stringent.
    deviation_thresh : float, default 2.5
        Threshold for deviation-based detection in standard deviations.
        Channels exceeding this many SDs from the mean are marked as bad.
    ransac_sample_prop : float, default 0.35
        Proportion of samples to use for RANSAC detection (0.0-1.0).
        Higher values use more data but are computationally more expensive.
    ransac_corr_thresh : float, default 0.65
        Correlation threshold for RANSAC-based detection. Set to 0 to disable
        RANSAC detection entirely.
    ransac_frac_bad : float, default 0.25
        Expected fraction of bad channels for RANSAC algorithm (0.0-1.0).
        Should reflect prior knowledge about data quality.
    ransac_channel_wise : bool, default False
        Whether to perform RANSAC detection channel-wise rather than globally.
        Channel-wise detection can be more sensitive but slower.
    random_state : int, default 1337
        Random seed for reproducible RANSAC results.
    exclude_channels : list of str or None, default None
        Channel names to exclude from bad channel detection (e.g., reference
        channels). These channels will never be marked as bad.
    return_by_method : bool, default False
        If True, returns a dictionary with bad channels separated by detection
        method. If False, returns a combined list of all bad channels.
    verbose : bool or None, default None
        Control verbosity of output during detection.
        
    Returns
    -------
    bad_channels : list of str or dict
        If return_by_method=False: List of channel names detected as bad.
        If return_by_method=True: Dictionary with keys 'correlation', 'deviation',
        'ransac', and 'combined' containing lists of bad channels for each method.
        
    Examples
    --------
    >>> bad_channels = detect_bad_channels(raw)
    >>> bad_channels = detect_bad_channels(raw, correlation_thresh=0.4, return_by_method=True)
    
    See Also
    --------
    pyprep.find_noisy_channels.NoisyChannels : Underlying detection implementation
    mne.io.Raw.info : Access channel information and bad channel lists
    Autoreject: Automated artifact rejection for MEG and EEG data. NeuroImage, 159, 417-429.
    """
    # Input validation
    if not isinstance(data, mne.io.BaseRaw):
        raise TypeError(
            f"Data must be an MNE Raw object, got {type(data).__name__}"
        )
    
    if not 0.0 <= correlation_thresh <= 1.0:
        raise ValueError(f"correlation_thresh must be between 0 and 1, got {correlation_thresh}")
    
    if deviation_thresh <= 0:
        raise ValueError(f"deviation_thresh must be positive, got {deviation_thresh}")
    
    if not 0.0 <= ransac_sample_prop <= 1.0:
        raise ValueError(f"ransac_sample_prop must be between 0 and 1, got {ransac_sample_prop}")
    
    if not 0.0 <= ransac_corr_thresh <= 1.0:
        raise ValueError(f"ransac_corr_thresh must be between 0 and 1, got {ransac_corr_thresh}")
    
    if not 0.0 <= ransac_frac_bad <= 1.0:
        raise ValueError(f"ransac_frac_bad must be between 0 and 1, got {ransac_frac_bad}")
    
    if exclude_channels is None:
        exclude_channels = []
    
    try:
        # Create a copy to avoid modifying original data
        data_copy = data.copy()
        
        # Initialize NoisyChannels detector
        noisy_detector = NoisyChannels(data_copy, random_state=random_state)
        
        # Run correlation-based detection
        noisy_detector.find_bad_by_correlation(
            correlation_secs=5.0,
            correlation_threshold=correlation_thresh,
            frac_bad=0.01
        )
        
        # Run deviation-based detection
        noisy_detector.find_bad_by_deviation(
            deviation_threshold=deviation_thresh
        )
        
        # Run RANSAC-based detection if enabled
        if ransac_corr_thresh > 0:
            noisy_detector.find_bad_by_ransac(
                n_samples=100,
                sample_prop=ransac_sample_prop,
                corr_thresh=ransac_corr_thresh,
                frac_bad=ransac_frac_bad,
                corr_window_secs=5.0,
                channel_wise=ransac_channel_wise,
                max_chunk_size=None
            )
        
        # Get results by method
        all_bad_channels = noisy_detector.get_bads(as_dict=True)
        
        correlation_bads = [
            ch for ch in all_bad_channels.get("bad_by_correlation", [])
            if ch not in exclude_channels
        ]
        deviation_bads = [
            ch for ch in all_bad_channels.get("bad_by_deviation", [])
            if ch not in exclude_channels
        ]
        ransac_bads = [
            ch for ch in all_bad_channels.get("bad_by_ransac", [])
            if ch not in exclude_channels
        ]
        
        # Combine all bad channels (remove duplicates)
        combined_bads = list(set(correlation_bads + deviation_bads + ransac_bads))
        
        if return_by_method:
            return {
                "correlation": correlation_bads,
                "deviation": deviation_bads,
                "ransac": ransac_bads,
                "combined": combined_bads
            }
        else:
            return combined_bads
            
    except Exception as e:
        raise RuntimeError(f"Failed to detect bad channels: {str(e)}") from e


def interpolate_bad_channels(
    data: Union[mne.io.BaseRaw, mne.Epochs],
    bad_channels: Optional[List[str]] = None,
    reset_bads: bool = True,
    mode: str = "accurate",
    origin: Union[str, Tuple[float, float, float]] = "auto",
    verbose: Optional[bool] = None
) -> Union[mne.io.BaseRaw, mne.Epochs]:
    """Interpolate bad channels using spherical spline interpolation.
    
    This function interpolates bad channels using spherical spline interpolation,
    which estimates the signal at bad channel locations based on the signals
    recorded at nearby good channels. This method preserves the spatial
    relationships in the data and maintains the original number of channels.
    
    Spherical spline interpolation is the gold standard for EEG channel
    interpolation as it accounts for the spherical geometry of the head
    and provides smooth, realistic interpolated signals.
    
    Parameters
    ----------
    data : mne.io.BaseRaw or mne.Epochs
        The EEG data containing bad channels to interpolate.
    bad_channels : list of str or None, default None
        List of channel names to interpolate. If None, uses channels marked
        as bad in data.info['bads']. Channels not present in the data are
        ignored with a warning.
    reset_bads : bool, default True
        Whether to reset the bad channels list after interpolation.
        If True, interpolated channels are removed from data.info['bads'].
        If False, they remain marked as bad for reference.
    mode : str, default 'accurate'
        Interpolation mode to use. Options:
        - 'accurate': More precise but slower interpolation
        - 'fast': Faster but less precise interpolation
    origin : str or tuple, default 'auto'
        Origin for spherical spline interpolation. Options:
        - 'auto': Automatically determine origin (recommended)
        - tuple: (x, y, z) coordinates in meters
        - 'head': Use head origin from digitization
    verbose : bool or None, default None
        Control verbosity of interpolation output.
        
    Returns
    -------
    data_interpolated : mne.io.BaseRaw or mne.Epochs
        Copy of input data with bad channels interpolated.
        
    Raises
    ------
    TypeError
        If data is not an MNE Raw or Epochs object.
    ValueError
        If bad_channels contains invalid channel names or parameters are invalid.
    RuntimeError
        If interpolation fails due to insufficient good channels or other errors.
        
    Notes
    -----
    **Interpolation Method:**
    Uses spherical spline interpolation which fits smooth functions on the
    sphere to the scalp potential distribution. This method:
    - Preserves spatial relationships in the EEG data
    - Provides realistic interpolated signals
    - Maintains the rank of the data (important for ICA and other analyses)
    
    **Requirements:**
    - Requires channel positions (montage) to be set in the data
    - Needs sufficient good channels around bad channels for accurate interpolation
    - Interpolation quality decreases with increasing number of bad channels
    
    **Best Practices:**
    - Interpolate no more than 10-15% of total channels
    - Ensure bad channels are not clustered in one region
    - Always visually inspect interpolated channels
    - Consider excluding channels rather than interpolating if >20% are bad
    
    **Performance Notes:**
    - 'accurate' mode provides better results but is slower
    - 'fast' mode suitable for real-time processing or large datasets
    - Interpolation time scales with number of channels and bad channels
    
    Examples
    --------
    Basic interpolation using channels marked as bad:
    
    >>> from autoclean import interpolate_bad_channels
    >>> # Assume raw.info['bads'] = ['Fp1', 'T7']
    >>> raw_interp = interpolate_bad_channels(raw)
    >>> print(f"Interpolated {len(raw.info['bads'])} bad channels")
    
    Interpolate specific channels:
    
    >>> raw_interp = interpolate_bad_channels(
    ...     raw, 
    ...     bad_channels=['Fp1', 'Fp2', 'F7'],
    ...     reset_bads=False  # Keep them marked as bad
    ... )
    
    Fast interpolation for large datasets:
    
    >>> raw_interp = interpolate_bad_channels(
    ...     raw,
    ...     mode='fast',
    ...     reset_bads=True
    ... )
    
    Interpolate epochs data:
    
    >>> epochs_interp = interpolate_bad_channels(epochs)
    
    See Also
    --------
    mne.io.Raw.interpolate_bads : MNE's raw data interpolation method
    mne.Epochs.interpolate_bads : MNE's epochs interpolation method  
    autoclean.detect_bad_channels : Detect bad channels automatically
    
    References
    ----------
    Perrin, F., Pernier, J., Bertrand, O., & Echallier, J. F. (1989).
    Spherical splines for scalp potential and current density mapping.
    Electroencephalography and clinical neurophysiology, 72(2), 184-187.
    
    Ferree, T. C., Luu, P., Russell, G. S., & Tucker, D. M. (2001).
    Scalp electrode impedance, infection risk, and EEG data quality.
    Clinical Neurophysiology, 112(3), 536-544.
    """
    # Input validation
    if not isinstance(data, (mne.io.BaseRaw, mne.Epochs)):
        raise TypeError(
            f"Data must be an MNE Raw or Epochs object, got {type(data).__name__}"
        )
    
    if mode not in ['accurate', 'fast']:
        raise ValueError(f"mode must be 'accurate' or 'fast', got '{mode}'")
    
    # Determine bad channels to interpolate
    if bad_channels is None:
        bad_channels = list(data.info['bads'])
    
    if not bad_channels:
        # No bad channels to interpolate
        return data.copy()
    
    # Validate that bad channels exist in the data
    available_channels = data.ch_names
    invalid_channels = [ch for ch in bad_channels if ch not in available_channels]
    if invalid_channels:
        raise ValueError(
            f"Bad channels not found in data: {invalid_channels}. "
            f"Available channels: {available_channels[:10]}..." if len(available_channels) > 10 
            else f"Available channels: {available_channels}"
        )
    
    # Check if montage is available
    if data.get_montage() is None:
        raise RuntimeError(
            "Channel positions (montage) must be set before interpolation. "
            "Use data.set_montage() to add channel positions."
        )
    
    try:
        # Create a copy to avoid modifying original data
        data_copy = data.copy()
        
        # Set the bad channels in the copy
        data_copy.info['bads'] = list(set(data_copy.info['bads'] + bad_channels))
        
        # Perform interpolation
        data_copy.interpolate_bads(
            reset_bads=reset_bads,
            mode=mode,
            origin=origin,
            verbose=verbose
        )
        
        return data_copy
        
    except Exception as e:
        raise RuntimeError(f"Failed to interpolate bad channels: {str(e)}") from e