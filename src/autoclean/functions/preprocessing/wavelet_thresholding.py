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
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # Estimate noise sigma using median absolute deviation of detail coeffs at level 1
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(signal.size))
    coeffs_thresh = [coeffs[0]]
    for c in coeffs[1:]:
        coeffs_thresh.append(pywt.threshold(c, uthresh, mode="soft"))
    return pywt.waverec(coeffs_thresh, wavelet)


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
        Decomposition level.

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
