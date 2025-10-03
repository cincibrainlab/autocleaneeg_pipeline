"""Zapline DSS-based line noise removal algorithm.

This module implements the Zapline algorithm using Denoising Source Separation (DSS)
to remove power line artifacts from EEG/MEG data.

Based on:
- de Cheveigné, A. (2020). ZapLine: A simple and effective method to remove
  power line artifacts. NeuroImage, 207, 116356.
- MEEGkit Python toolbox: https://github.com/nbara/python-meegkit
"""

import numpy as np
from typing import Tuple, Optional


def apply_zapline_dss(
    raw,
    fline: float = 60.0,
    nkeep: int = 1,
    use_iter: bool = False,
    max_iter: int = 10,
) -> Tuple:
    """Apply Zapline DSS-based line noise removal to Raw data.

    Uses Denoising Source Separation (DSS) to remove power line artifacts.
    The method combines spectral and spatial filtering to effectively remove
    line noise while preserving signal characteristics.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object containing EEG/MEG data
    fline : float, default=60.0
        Line noise frequency in Hz (60 for US/Americas, 50 for Europe/Asia)
    nkeep : int, default=1
        Number of noise components to remove. Typically 1 removes the strongest
        line noise component. Higher values (2-3) can remove harmonics.
    use_iter : bool, default=False
        If True, use iterative removal (dss_line_iter) for more thorough
        noise reduction. Automatically determines when noise is sufficiently removed.
    max_iter : int, default=10
        Maximum number of iterations for iterative mode (only used if use_iter=True)

    Returns
    -------
    raw_clean : mne.io.Raw
        Raw object with line noise removed
    info : dict
        Dictionary containing:
        - 'iterations': Number of iterations used (only for iterative mode)
        - 'method': 'dss_line' or 'dss_line_iter'
        - 'fline': Line frequency used
        - 'nkeep': Number of components removed

    Raises
    ------
    ImportError
        If meegkit is not installed
    ValueError
        If data has insufficient channels or samples

    Notes
    -----
    - Works best with multi-channel data (>32 channels recommended)
    - Assumes spatial consistency of line noise across channels
    - Non-iterative mode (use_iter=False) is faster and usually sufficient
    - Iterative mode may provide better noise removal for severe contamination

    Examples
    --------
    >>> # Basic usage (remove 60 Hz)
    >>> raw_clean, info = apply_zapline_dss(raw, fline=60, nkeep=1)

    >>> # Remove 50 Hz with iterative refinement
    >>> raw_clean, info = apply_zapline_dss(raw, fline=50, use_iter=True)

    >>> # Remove main component and first harmonic
    >>> raw_clean, info = apply_zapline_dss(raw, fline=60, nkeep=2)

    References
    ----------
    .. [1] de Cheveigné, A. (2020). ZapLine: A simple and effective method to
       remove power line artifacts. NeuroImage, 207, 116356.
    .. [2] de Cheveigné, A., & Simon, J. Z. (2008). Denoising based on spatial
       filtering. Journal of Neuroscience Methods, 171(2), 331-339.
    """
    # Import meegkit (raise user-friendly error if not installed)
    try:
        from meegkit import dss
    except ImportError:
        from autoclean.utils.block_errors import raise_dependency_error

        raise_dependency_error(
            block_name="zapline",
            missing_packages=[("meegkit", ">=0.1.9")],
            what_it_does=(
                "Zapline removes electrical noise (like the hum from power outlets) "
                "from your EEG recordings. This makes your brain signals cleaner "
                "and easier to analyze."
            )
        )

    # Validate input
    if raw.info['nchan'] < 2:
        raise ValueError(
            f"Zapline requires at least 2 channels, got {raw.info['nchan']}. "
            "DSS exploits spatial structure of noise."
        )

    # Extract data and parameters
    data = raw.get_data().T  # Convert to (n_samples, n_channels)
    sfreq = raw.info['sfreq']

    # Defensive validation of all parameters before passing to meegkit
    if fline is None:
        raise ValueError("fline cannot be None")
    if nkeep is None:
        raise ValueError("nkeep cannot be None")
    if sfreq is None:
        raise ValueError("sfreq cannot be None (check raw.info['sfreq'])")

    fline = float(fline)
    nkeep = int(nkeep)
    sfreq = float(sfreq)

    # Validate frequency parameters
    nyquist = sfreq / 2
    if fline >= nyquist:
        raise ValueError(
            f"Line frequency ({fline} Hz) must be below Nyquist frequency "
            f"({nyquist} Hz) for sampling rate {sfreq} Hz"
        )

    # Apply appropriate DSS method
    info = {
        'method': 'dss_line_iter' if use_iter else 'dss_line',
        'fline': fline,
        'nkeep': nkeep,
    }

    # Calculate appropriate nfft (should be power of 2, at least 1 second of data)
    # Using None can cause issues in some meegkit versions
    nfft = int(2 ** np.ceil(np.log2(sfreq)))  # Next power of 2 >= sfreq

    if use_iter:
        # Iterative removal - automatically determines convergence
        out, iterations = dss.dss_line_iter(
            data,
            fline=fline,
            sfreq=sfreq,
            nfft=nfft,
            n_iter_max=max_iter,
        )
        info['iterations'] = iterations
    else:
        # Single-pass removal
        out, _ = dss.dss_line(
            data,
            fline=fline,
            sfreq=sfreq,
            nkeep=nkeep,
            nfft=nfft,
        )
        info['iterations'] = 1

    # Create cleaned Raw object
    raw_clean = raw.copy()
    raw_clean._data = out.T  # Convert back to (n_channels, n_samples)

    return raw_clean, info


def compute_line_noise_power(
    raw,
    fline: float = 60.0,
    bandwidth: float = 2.0
) -> Tuple[float, float]:
    """Compute power at line frequency before and after Zapline.

    Useful for quantifying noise reduction effectiveness.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data
    fline : float, default=60.0
        Line noise frequency in Hz
    bandwidth : float, default=2.0
        Bandwidth around fline to integrate power (Hz)

    Returns
    -------
    power_db : float
        Power at line frequency in dB
    snr : float
        Signal-to-noise ratio: power at line freq / average power elsewhere
    """
    from scipy import signal

    data = raw.get_data()
    sfreq = raw.info['sfreq']

    # Compute power spectrum
    freqs, psd = signal.welch(
        data,
        fs=sfreq,
        nperseg=int(4 * sfreq),  # 4-second windows
        scaling='density'
    )

    # Find indices for line frequency band
    line_band = (freqs >= fline - bandwidth/2) & (freqs <= fline + bandwidth/2)

    # Average across channels
    psd_mean = psd.mean(axis=0)

    # Power at line frequency
    line_power = psd_mean[line_band].mean()

    # Background power (excluding line frequency ±5 Hz)
    background_mask = (freqs < fline - 5) | (freqs > fline + 5)
    background_power = psd_mean[background_mask].mean()

    # Convert to dB
    power_db = 10 * np.log10(line_power)
    snr = line_power / background_power if background_power > 0 else 0

    return power_db, snr


def validate_zapline_effectiveness(
    raw_before,
    raw_after,
    fline: float = 60.0
) -> dict:
    """Validate Zapline effectiveness by comparing spectra.

    Parameters
    ----------
    raw_before : mne.io.Raw
        Raw data before Zapline
    raw_after : mne.io.Raw
        Raw data after Zapline
    fline : float, default=60.0
        Line noise frequency

    Returns
    -------
    results : dict
        Validation metrics including:
        - 'reduction_db': Power reduction at line frequency (dB)
        - 'snr_before': SNR before Zapline
        - 'snr_after': SNR after Zapline
        - 'success': True if reduction >= 10 dB
    """
    power_before, snr_before = compute_line_noise_power(raw_before, fline)
    power_after, snr_after = compute_line_noise_power(raw_after, fline)

    reduction_db = power_before - power_after

    results = {
        'power_before_db': power_before,
        'power_after_db': power_after,
        'reduction_db': reduction_db,
        'snr_before': snr_before,
        'snr_after': snr_after,
        'success': reduction_db >= 10.0,  # At least 10 dB reduction
        'fline': fline
    }

    return results
