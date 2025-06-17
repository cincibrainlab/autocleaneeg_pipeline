"""Resampling functions for EEG data.

This module provides standalone functions for changing the sampling frequency 
of EEG data through resampling operations.
"""

from typing import Optional, Union

import mne


def resample_data(
    data: Union[mne.io.BaseRaw, mne.Epochs],
    sfreq: float,
    npad: str = 'auto',
    window: str = 'auto',
    n_jobs: int = 1,
    pad: str = 'auto',
    verbose: Optional[bool] = None
) -> Union[mne.io.BaseRaw, mne.Epochs]:
    """Resample EEG data to a new sampling frequency.
    
    This function changes the sampling frequency of continuous (Raw) or epoched 
    EEG data using anti-aliasing filtering and interpolation. The function 
    automatically applies appropriate anti-aliasing filters and handles edge 
    effects to preserve data quality during resampling.
    
    The function automatically detects the input data type and preserves it,
    returning the same type (Raw or Epochs) with identical structure but 
    modified sampling frequency and adjusted time points.
    
    Parameters
    ----------
    data : mne.io.BaseRaw or mne.Epochs
        The EEG data to resample. Can be any MNE Raw object (e.g., RawFIF, 
        RawEEGLAB, etc.) or Epochs object.
    sfreq : float
        The target sampling frequency in Hz. Must be positive and typically
        should follow the Nyquist criterion relative to the highest frequency
        content in the data.
    npad : str or int, default 'auto'
        Amount of padding to use. 'auto' uses a heuristic to determine optimal
        padding length. Can also be an integer specifying exact padding samples.
    window : str, default 'auto'
        Windowing function for resampling. 'auto' selects appropriate window
        based on resampling parameters. Other options include 'boxcar', 'triang',
        'blackman', 'hamming', 'hann', 'bartlett', 'flattop', 'parzen', 'bohman'.
    n_jobs : int, default 1
        Number of parallel jobs to run. Use -1 for all available cores.
        Parallel processing can speed up resampling for large datasets.
    pad : str, default 'auto'
        Padding mode. 'auto' selects appropriate padding. Other options include
        'reflect_limited', 'zero', 'constant', 'edge', 'wrap'.
    verbose : bool or None, default None
        Control verbosity of output. If None, uses MNE default.
        
    Returns
    -------
    resampled_data : mne.io.BaseRaw or mne.Epochs
        The resampled data object, same type as input. Contains identical
        structure and metadata but with modified sampling frequency and 
        adjusted time axis.
        
    Raises
    ------
    TypeError
        If data is not an MNE Raw or Epochs object.
    ValueError
        If target sampling frequency is not positive or is invalid.
    RuntimeError
        If resampling fails due to insufficient data or other processing errors.
        
    Notes
    -----
    Resampling modifies the sampling frequency and time axis but preserves all 
    metadata including channel information, events, and annotations. The resampling
    is applied in-place on a copy of the data to avoid modifying the original.
    
    When downsampling (reducing sampling frequency), anti-aliasing filtering is 
    automatically applied to prevent aliasing artifacts. When upsampling (increasing
    sampling frequency), interpolation is used to estimate intermediate sample points.
    
    For continuous data (Raw), edge effects may occur at the beginning and end
    of the recording due to filtering operations. For epoched data, edge effects
    occur at epoch boundaries and may affect short epochs more severely.
    
    The function uses MNE's resampling implementation, which applies appropriate
    anti-aliasing filters and windowing to minimize artifacts. For very long
    recordings, consider processing in chunks to manage memory usage.
    
    Resampling can significantly affect subsequent processing steps:
    - Filtering parameters may need adjustment for new sampling frequency
    - Epoch length in samples will change proportionally
    - Event timing remains in seconds but sample indices will change
    
    Examples
    --------
    Downsample to reduce file size and processing time:
    
    >>> from autoclean import resample_data
    >>> downsampled_raw = resample_data(raw, sfreq=250)  # From 1000 Hz to 250 Hz
    
    Upsample for higher temporal resolution:
    
    >>> upsampled_epochs = resample_data(epochs, sfreq=500)  # From 250 Hz to 500 Hz
    
    Resample with custom parameters for better quality:
    
    >>> resampled_data = resample_data(
    ...     raw, 
    ...     sfreq=250,
    ...     window='hamming',
    ...     n_jobs=4
    ... )
    
    Check if resampling is needed before applying:
    
    >>> current_sfreq = raw.info['sfreq']
    >>> if abs(current_sfreq - target_sfreq) > 0.01:
    ...     resampled_raw = resample_data(raw, sfreq=target_sfreq)
    ... else:
    ...     resampled_raw = raw.copy()  # No resampling needed
    
    See Also
    --------
    mne.io.Raw.resample : MNE's raw data resampling method
    mne.Epochs.resample : MNE's epochs resampling method
    """
    # Input validation
    if not isinstance(data, (mne.io.BaseRaw, mne.Epochs)):
        raise TypeError(
            f"Data must be an MNE Raw or Epochs object, got {type(data).__name__}"
        )
    
    if not isinstance(sfreq, (int, float)) or sfreq <= 0:
        raise ValueError(f"Target sampling frequency must be positive, got {sfreq}")
    
    # Check if resampling is actually needed
    current_sfreq = data.info['sfreq']
    if abs(current_sfreq - sfreq) < 0.01:  # Tolerance for floating point comparison
        # No resampling needed, return copy
        return data.copy()
    
    # Create a copy to avoid modifying the original
    resampled_data = data.copy()
    
    try:
        # Perform resampling using MNE's built-in method
        resampled_data.resample(
            sfreq=sfreq,
            npad=npad,
            window=window,
            n_jobs=n_jobs,
            pad=pad,
            verbose=verbose
        )
        
        return resampled_data
        
    except Exception as e:
        raise RuntimeError(f"Failed to resample data from {current_sfreq} Hz to {sfreq} Hz: {str(e)}") from e