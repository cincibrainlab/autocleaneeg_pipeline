"""Wavelet thresholding for EEG data.

This module implements wavelet-based denoising identical in spirit to the
HAPPE MATLAB pipeline. It performs a discrete wavelet transform on each channel
and applies universal soft-thresholding to attenuate high-amplitude
transients.
"""

from __future__ import annotations

from typing import Union

import mne
import numpy as np
import pywt


def _resolve_decomposition_level(
    signal_length: int,
    wavelet: str,
    level: int,
) -> int:
    """Return a safe decomposition level for the requested wavelet.

    PyWavelets raises an error when the requested level exceeds the
    maximum supported for the given signal length and wavelet filter.
    This helper determines the highest valid level and clamps the
    requested value accordingly.
    """
    wavelet_obj = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(signal_length, wavelet_obj.dec_len)
    if max_level <= 0:
        return 0
    return min(level, max_level)


def _denoise_signal(
    signal: np.ndarray,
    wavelet: str,
    level: int,
) -> np.ndarray:
    """Denoise a 1D signal using wavelet thresholding.

    Parameters
    ----------
    signal : ndarray
        The time series to denoise.
    wavelet : str
        Wavelet name.
    level : int
        Decomposition level.

    Returns
    -------
    ndarray
        Denoised signal.
    """
    signal_array = np.asarray(signal)
    effective_level = _resolve_decomposition_level(signal_array.size, wavelet, level)
    if effective_level == 0:
        return signal_array.copy()

    coeffs = pywt.wavedec(signal_array, wavelet, level=effective_level)
    # Estimate noise sigma using median absolute deviation of detail coeffs at level 1
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if coeffs[-1].size else 0.0
    uthresh = sigma * np.sqrt(2 * np.log(signal_array.size)) if sigma else 0.0
    coeffs_thresh = [coeffs[0]]
    for c in coeffs[1:]:
        coeffs_thresh.append(pywt.threshold(c, uthresh, mode="soft"))
    denoised = pywt.waverec(coeffs_thresh, wavelet)
    if denoised.shape[0] != signal_array.shape[0]:
        denoised = denoised[: signal_array.shape[0]]
    return denoised.astype(signal_array.dtype, copy=False)


def wavelet_threshold(
    data: Union[mne.io.BaseRaw, mne.Epochs],
    wavelet: str = "sym4",
    level: int = 5,
) -> Union[mne.io.BaseRaw, mne.Epochs]:
    """Apply wavelet thresholding to EEG data.

    Parameters
    ----------
    data : Raw or Epochs
        MNE object containing EEG data. The data is copied before processing.
    wavelet : str, default 'sym4'
        Mother wavelet used for the discrete wavelet transform.
    level : int, default 5
        Requested decomposition level. The actual level is clamped to the
        maximum supported for each channel and wavelet.

    Returns
    -------
    Raw or Epochs
        A copy of the input with wavelet thresholding applied channel-wise.
    """
    cleaned = data.copy()
    if isinstance(cleaned, mne.io.BaseRaw):
        arr = cleaned.get_data()
        for idx in range(arr.shape[0]):
            arr[idx] = _denoise_signal(arr[idx], wavelet, level)
        cleaned._data = arr
    elif isinstance(cleaned, mne.Epochs):
        arr = cleaned.get_data()
        for ep in range(arr.shape[0]):
            for ch in range(arr.shape[1]):
                arr[ep, ch] = _denoise_signal(arr[ep, ch], wavelet, level)
        cleaned._data = arr
    else:
        raise TypeError("data must be mne.io.BaseRaw or mne.Epochs")
    return cleaned
