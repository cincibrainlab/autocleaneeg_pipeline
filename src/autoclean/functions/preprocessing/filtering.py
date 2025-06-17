"""Digital filtering functions for EEG data.

This module provides standalone functions for applying digital filters to EEG data,
including highpass, lowpass, and notch filtering capabilities.
"""

from typing import List, Optional, Union

import mne


def filter_data(
    data: Union[mne.io.BaseRaw, mne.Epochs],
    l_freq: Optional[float] = None,
    h_freq: Optional[float] = None,
    notch_freqs: Optional[List[float]] = None,
    notch_widths: Union[float, List[float]] = 0.5,
    method: str = 'fir',
    phase: str = 'zero',
    fir_window: str = 'hamming',
    verbose: Optional[bool] = None
) -> Union[mne.io.base.BaseRaw, mne.Epochs]:
    """Filter EEG data using highpass, lowpass, and/or notch filtering.
    
    This function applies digital filtering to continuous (Raw) or epoched EEG data.
    It supports highpass filtering to remove slow drifts, lowpass filtering to 
    remove high-frequency noise, and notch filtering to remove line noise at 
    specific frequencies (e.g., 50/60 Hz power line interference).
    
    The function automatically detects the input data type and preserves it,
    returning the same type (Raw or Epochs) with identical structure but 
    filtered time series data.
    
    Parameters
    ----------
    data : mne.io.BaseRaw or mne.Epochs
        The EEG data to filter. Can be any MNE Raw object (e.g., RawFIF, 
        RawEEGLAB, etc.) or Epochs object.
    l_freq : float or None, default None
        Low cutoff frequency for highpass filtering in Hz. If None, no 
        highpass filtering is applied. Typical values: 0.1-1.0 Hz for 
        removing slow drifts.
    h_freq : float or None, default None
        High cutoff frequency for lowpass filtering in Hz. If None, no 
        lowpass filtering is applied. Typical values: 30-100 Hz for 
        removing high-frequency noise.
    notch_freqs : list of float or None, default None
        Frequencies to notch filter in Hz. If None, no notch filtering 
        is applied. Common values: [50] or [60] for power line noise,
        [50, 100, 150] for harmonics.
    notch_widths : float or list of float, default 0.5
        Width of the notch filter in Hz. If float, same width applied to 
        all frequencies in notch_freqs. If list, must match length of 
        notch_freqs. Wider notches remove more surrounding frequencies.
    method : str, default 'fir'
        Filtering method. 'fir' uses finite impulse response (linear phase),
        'iir' uses infinite impulse response (faster but can introduce artifacts).
    phase : str, default 'zero'
        Phase of the filter. 'zero' for zero-phase (no delay), 'zero-double'
        for zero-phase with double filtering, 'minimum' for minimum phase.
    fir_window : str, default 'hamming'
        Window function for FIR filter design. Options: 'hamming', 'hann', 
        'blackman'. Affects filter characteristics and artifacts.
    verbose : bool or None, default None
        Control verbosity of output. If None, uses MNE default.
        
    Returns
    -------
    filtered_data : mne.io.BaseRaw or mne.Epochs
        The filtered data object, same type as input. Contains identical
        structure and metadata but with filtered time series data.
        
    Examples
    --------
    >>> filtered_raw = filter_data(raw, l_freq=1.0, h_freq=40.0)
    >>> filtered_raw = filter_data(raw, notch_freqs=[60])
    >>> bandpass_data = filter_data(raw, l_freq=8.0, h_freq=12.0)
    
    See Also
    --------
    mne.io.Raw.filter : MNE's raw data filtering method
    mne.Epochs.filter : MNE's epochs filtering method
    mne.io.Raw.notch_filter : MNE's notch filtering for raw data
    mne.Epochs.notch_filter : MNE's notch filtering for epochs
    """
    # Input validation
    if not isinstance(data, (mne.io.BaseRaw, mne.Epochs)):
        raise TypeError(
            f"Data must be an MNE Raw or Epochs object, got {type(data).__name__}"
        )
    
    # Validate frequency parameters
    if l_freq is not None and l_freq < 0:
        raise ValueError(f"l_freq must be positive, got {l_freq}")
    
    if h_freq is not None and h_freq < 0:
        raise ValueError(f"h_freq must be positive, got {h_freq}")
    
    if l_freq is not None and h_freq is not None and l_freq >= h_freq:
        raise ValueError(
            f"l_freq ({l_freq}) must be less than h_freq ({h_freq})"
        )
    
    # Check if any filtering is requested
    if l_freq is None and h_freq is None and notch_freqs is None:
        # No filtering requested, return copy of original data
        return data.copy()
    
    # Create a copy of the data to avoid modifying the original
    filtered_data = data.copy()
    
    # Apply highpass and/or lowpass filtering
    if l_freq is not None or h_freq is not None:
        filtered_data.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            method=method,
            phase=phase,
            fir_window=fir_window,
            verbose=verbose
        )
    
    # Apply notch filtering
    if notch_freqs is not None:
        if not isinstance(notch_freqs, (list, tuple)):
            notch_freqs = [notch_freqs]
        
        # Validate notch frequencies
        for freq in notch_freqs:
            if freq <= 0:
                raise ValueError(f"Notch frequencies must be positive, got {freq}")
        
        filtered_data.notch_filter(
            freqs=notch_freqs,
            notch_widths=notch_widths,
            method=method,
            phase=phase,
            fir_window=fir_window,
            verbose=verbose
        )
    
    return filtered_data