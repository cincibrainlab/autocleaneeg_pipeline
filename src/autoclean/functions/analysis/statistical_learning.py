"""Statistical learning analysis functions for EEG data.

This module provides functions for analyzing statistical learning epochs,
including inter-trial coherence (ITC) and spectral analysis optimized for
neural entrainment research.

Key Features:
- Inter-trial coherence (ITC) analysis using 0.6-5 Hz frequency range
- Word Learning Index (WLI) calculation for statistical learning assessment
- Significance testing using Rayleigh test for phase consistency
- Frequency-specific analysis targeting syllable (3.33 Hz) and word (1.11 Hz) rates
- Modern MNE-Python API integration with comprehensive validation

Functions:
- compute_statistical_learning_itc: Main ITC analysis with modern MNE API
- calculate_word_learning_index: WLI = ITC(word_freq) / ITC(syllable_freq)
- analyze_itc_bands: Frequency band analysis for statistical learning
- extract_itc_at_frequencies: Precise frequency extraction utility
- validate_itc_significance: Rayleigh test for statistical significance
- compute_itc_confidence_intervals: Bootstrap confidence intervals

Research Protocol:
This implementation follows established statistical learning research protocols
using neural entrainment analysis in the 0.6-5 Hz range to capture rhythmic
brain responses to syllable and word presentation rates.
"""

from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np

from autoclean.utils.logging import message


def _validate_wavelet_parameters(
    freqs: np.ndarray, n_cycles: Union[float, np.ndarray], epochs: mne.Epochs
) -> None:
    """Validate that wavelet length doesn't exceed epoch duration.

    Parameters
    ----------
    freqs : np.ndarray
        Frequencies of interest.
    n_cycles : float or np.ndarray
        Number of cycles for wavelets.
    epochs : mne.Epochs
        The epoched data.

    Raises
    ------
    ValueError
        If wavelet length exceeds epoch duration.
    """
    sfreq = epochs.info["sfreq"]
    epoch_length = len(epochs.times) / sfreq

    # Calculate minimum wavelet length: (5/π) * (n_cycles * sfreq) / freqs - 1
    if isinstance(n_cycles, (int, float)):
        n_cycles_array = np.full_like(freqs, n_cycles, dtype=float)
    else:
        n_cycles_array = n_cycles

    min_wavelet_length = (5 / np.pi) * (n_cycles_array * sfreq) / freqs - 1
    max_wavelet_length = np.max(min_wavelet_length) / sfreq

    if max_wavelet_length > epoch_length:
        problematic_freqs = freqs[min_wavelet_length / sfreq > epoch_length]
        raise ValueError(
            f"Wavelet length ({max_wavelet_length:.3f}s) exceeds epoch duration ({epoch_length:.3f}s). "
            f"Problematic frequencies: {problematic_freqs} Hz. "
            f"Consider reducing n_cycles or increasing epoch length."
        )


def _validate_epoch_requirements(epochs: mne.Epochs, min_trials: int = 10) -> None:
    """Validate sufficient trials for stable ITC estimates.

    Parameters
    ----------
    epochs : mne.Epochs
        The epoched data.
    min_trials : int, optional
        Minimum number of trials required. Default is 10.

    Raises
    ------
    ValueError
        If insufficient trials available.
    """
    n_trials = len(epochs)
    if n_trials < min_trials:
        raise ValueError(
            f"Insufficient trials for stable ITC estimation. "
            f"Found {n_trials} trials, minimum recommended: {min_trials}. "
            f"ITC estimates become unreliable with very few trials."
        )


def _validate_frequency_range(freqs: np.ndarray, sfreq: float) -> None:
    """Validate frequency range against Nyquist limit.

    Parameters
    ----------
    freqs : np.ndarray
        Frequencies of interest.
    sfreq : float
        Sampling frequency.

    Raises
    ------
    ValueError
        If frequencies exceed Nyquist limit or are invalid.
    """
    nyquist = sfreq / 2

    if np.any(freqs <= 0):
        raise ValueError("All frequencies must be positive")

    if np.any(freqs >= nyquist):
        problematic_freqs = freqs[freqs >= nyquist]
        raise ValueError(
            f"Frequencies {problematic_freqs} Hz exceed Nyquist limit ({nyquist} Hz). "
            f"Maximum allowable frequency is {nyquist - 1} Hz."
        )

    if not np.all(np.diff(freqs) > 0):
        raise ValueError("Frequencies must be in ascending order")


def compute_statistical_learning_itc(
    epochs: mne.Epochs,
    freqs: Optional[np.ndarray] = None,
    n_cycles: Union[float, np.ndarray] = 7.0,
    time_bandwidth: float = 4.0,
    use_multitaper: bool = False,
    decim: int = 1,
    n_jobs: int = 1,
    picks: Optional[Union[str, List[str]]] = None,
    baseline: Optional[Tuple[float, float]] = None,
    mode: str = "mean",
    verbose: bool = True,
) -> Tuple[mne.time_frequency.AverageTFR, mne.time_frequency.AverageTFR]:
    """Compute inter-trial coherence (ITC) for statistical learning epochs.

    This function computes both power and inter-trial coherence (ITC) from
    statistical learning epochs using time-frequency analysis. ITC measures
    the phase consistency across trials, which is particularly relevant for
    statistical learning paradigms where neural entrainment is expected.

    Parameters
    ----------
    epochs : mne.Epochs
        The epoched data from statistical learning paradigm.
    freqs : np.ndarray, optional
        Frequencies of interest. If None, uses 50 logarithmically spaced frequencies
        from 0.6 to 5 Hz (statistical learning protocol). Default is None.
    n_cycles : float or np.ndarray, optional
        Number of cycles for Morlet wavelets. Can be a single value or array
        matching freqs length. Default is 7.0.
    time_bandwidth : float, optional
        Time-bandwidth product for multitaper method. Only used if use_multitaper=True.
        Default is 4.0.
    use_multitaper : bool, optional
        Whether to use multitaper method instead of Morlet wavelets. Default is False.
    decim : int, optional
        Decimation factor to reduce temporal resolution. Default is 1 (no decimation).
    n_jobs : int, optional
        Number of jobs for parallel processing. Default is 1.
    picks : str or list of str, optional
        Channels to include. If None, uses all EEG channels. Default is None.
    baseline : tuple of float, optional
        Baseline period for correction (tmin, tmax) in seconds. Default is None.
    mode : str, optional
        Baseline correction mode. One of 'mean', 'ratio', 'logratio', 'percent', 'zscore', 'zlogratio'.
        Default is 'mean'.
    verbose : bool, optional
        Whether to print progress messages. Default is True.

    Returns
    -------
    power : mne.time_frequency.AverageTFR
        Time-frequency representation of power.
    itc : mne.time_frequency.AverageTFR
        Time-frequency representation of inter-trial coherence.

    Raises
    ------
    ValueError
        If epochs is empty or invalid parameters are provided.
    TypeError
        If epochs is not an MNE Epochs object.

    Notes
    -----
    Inter-trial coherence (ITC) measures the consistency of phase across trials.
    Values range from 0 (no consistency) to 1 (perfect consistency). For statistical
    learning paradigms, higher ITC values at syllable frequencies (~3.3 Hz) may
    indicate neural entrainment to the rhythmic structure.

    Examples
    --------
    >>> from autoclean.functions.analysis import compute_statistical_learning_itc
    >>> power, itc = compute_statistical_learning_itc(epochs)
    >>> # Plot ITC for frontal channels
    >>> itc.plot_topo(picks='frontal', title='Inter-Trial Coherence')
    """
    if verbose:
        message(
            "info", "Computing inter-trial coherence for statistical learning epochs..."
        )

    # Validate input
    if not isinstance(epochs, mne.Epochs):
        raise TypeError("epochs must be an MNE Epochs object")

    if len(epochs) == 0:
        raise ValueError("epochs object is empty")

    # Validate epoch requirements
    _validate_epoch_requirements(epochs, min_trials=10)

    # Set default frequencies if not provided
    if freqs is None:
        # Statistical learning paradigm: neural responses to syllables and words
        # Frequency range 0.6-5 Hz using Fourier transform as per research protocol
        # Key frequencies: syllable presentation (3.33 Hz), word presentation (1.11 Hz)
        freqs = np.logspace(np.log10(0.6), np.log10(5.0), 50)
        if verbose:
            message(
                "debug",
                f"Using statistical learning frequencies: {freqs[0]:.2f} to {freqs[-1]:.2f} Hz ({len(freqs)} freqs)",
            )
            message("debug", "Key frequencies: word (1.11 Hz), syllable (3.33 Hz)")
            message(
                "debug",
                "Protocol: Fourier transform from 0.6-5 Hz for neural entrainment",
            )

    # Validate frequency range
    _validate_frequency_range(freqs, epochs.info["sfreq"])

    # Set channel picks
    if picks is None:
        picks = "eeg"

    # Validate n_cycles
    if isinstance(n_cycles, (int, float)):
        n_cycles = float(n_cycles)
    elif isinstance(n_cycles, np.ndarray):
        if len(n_cycles) != len(freqs):
            raise ValueError("n_cycles array must have same length as freqs")
    else:
        raise TypeError("n_cycles must be float or numpy array")

    # Validate wavelet parameters
    _validate_wavelet_parameters(freqs, n_cycles, epochs)

    if verbose:
        message("info", f"Computing time-frequency analysis on {len(epochs)} epochs...")
        message("debug", f"Frequency range: {freqs[0]:.1f}-{freqs[-1]:.1f} Hz")
        message(
            "debug",
            f"Using {'multitaper' if use_multitaper else 'Morlet wavelet'} method",
        )

    try:
        # Use modern MNE API - epochs.compute_tfr() (replaces deprecated tfr_morlet/tfr_multitaper)
        if use_multitaper:
            method = "multitaper"
            method_kw = {"time_bandwidth": time_bandwidth}
            if verbose:
                message(
                    "debug", f"Using multitaper method: time_bandwidth={time_bandwidth}"
                )
        else:
            method = "morlet"
            method_kw = {}
            if verbose:
                message("debug", f"Using Morlet wavelet method: n_cycles={n_cycles}")

        if verbose:
            message(
                "debug",
                f"Computing TFR with modern MNE API: epochs.compute_tfr(method='{method}')",
            )

        power, itc = epochs.compute_tfr(
            method=method,
            freqs=freqs,
            n_cycles=n_cycles,
            return_itc=True,
            average=True,
            picks=picks,
            decim=decim,
            n_jobs=n_jobs,
            verbose=False,  # Use autoclean logging
            **method_kw,
        )

        # Apply baseline correction to power only (ITC is inherently normalized 0-1)
        if baseline is not None:
            if verbose:
                message(
                    "info",
                    f"Applying baseline correction to power: {baseline} s, mode: {mode}",
                )
                message(
                    "debug",
                    "ITC is not baseline corrected as it's inherently normalized (0-1 range)",
                )

            power = power.apply_baseline(baseline, mode=mode)
            # NOTE: ITC is never baseline corrected as it measures phase consistency (0-1 range)
            # and is already normalized across trials

        # Log summary statistics
        if verbose:
            power_mean = np.mean(power.data)
            itc_mean = np.mean(itc.data)
            itc_max = np.max(itc.data)
            message("info", "Analysis complete:")
            message("debug", f"  Power mean: {power_mean:.3e}")
            message("debug", f"  ITC mean: {itc_mean:.3f}, max: {itc_max:.3f}")
            message("debug", f"  Time samples: {power.data.shape[-1]}")
            message("debug", f"  Channels: {power.data.shape[0]}")
            message("debug", f"  Frequencies: {power.data.shape[1]}")

        # Add metadata
        power.comment = f"Statistical Learning Power (n={len(epochs)} epochs)"
        itc.comment = f"Statistical Learning ITC (n={len(epochs)} epochs)"

        return power, itc

    except Exception as e:
        if verbose:
            message("error", f"Error computing ITC: {str(e)}")
        raise RuntimeError(f"Failed to compute inter-trial coherence: {str(e)}") from e


def analyze_itc_bands(
    itc: mne.time_frequency.AverageTFR,
    frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    time_window: Optional[Tuple[float, float]] = None,
    picks: Optional[Union[str, List[str]]] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Analyze ITC values within specific frequency bands.

    Parameters
    ----------
    itc : mne.time_frequency.AverageTFR
        Inter-trial coherence data from compute_statistical_learning_itc.
    frequency_bands : dict, optional
        Dictionary mapping band names to (fmin, fmax) tuples. If None, uses
        default bands relevant to statistical learning. Default is None.
    time_window : tuple of float, optional
        Time window for analysis (tmin, tmax) in seconds. If None, uses entire epoch.
    picks : str or list of str, optional
        Channels to analyze. If None, uses all channels in itc. Default is None.
    verbose : bool, optional
        Whether to print progress messages. Default is True.

    Returns
    -------
    band_values : dict
        Dictionary mapping band names to averaged ITC values (float) across the specified
        time window and channels.
    """
    if verbose:
        message("info", "Analyzing ITC by frequency bands...")

    # Statistical learning paradigm frequency bands
    # Updated for research protocol: 0.6-5 Hz range targeting neural entrainment
    if frequency_bands is None:
        frequency_bands = {
            "sub_word": (0.6, 1.0),  # Below word frequency
            "word_frequency": (1.0, 1.3),  # Word presentation frequency (1.11 Hz)
            "intermediate": (1.5, 2.8),  # Between word and syllable rates
            "syllable_frequency": (
                3.0,
                3.7,
            ),  # Syllable presentation frequency (3.33 Hz)
            "higher_harmonics": (3.8, 5.0),  # Higher frequency components and harmonics
        }

    # Set time window
    if time_window is None:
        time_indices = slice(None)
    else:
        time_mask = (itc.times >= time_window[0]) & (itc.times <= time_window[1])
        time_indices = np.where(time_mask)[0]

    # Set channel picks
    if picks is not None:
        pick_indices = mne.pick_channels(itc.ch_names, picks)
    else:
        pick_indices = slice(None)

    band_values = {}

    for band_name, (fmin, fmax) in frequency_bands.items():
        # Find frequency indices
        freq_mask = (itc.freqs >= fmin) & (itc.freqs <= fmax)
        freq_indices = np.where(freq_mask)[0]

        if len(freq_indices) == 0:
            if verbose:
                message(
                    "warning",
                    f"No frequencies found in {band_name} band ({fmin}-{fmax} Hz)",
                )
            band_values[band_name] = np.nan
            continue

        # Extract and average ITC values
        band_data = itc.data[pick_indices, :, :][:, freq_indices, :]
        if time_window is not None:
            band_data = band_data[:, :, time_indices]

        band_values[band_name] = np.mean(band_data)

        if verbose:
            message(
                "debug",
                f"{band_name} ({fmin}-{fmax} Hz): ITC = {band_values[band_name]:.3f}",
            )

    return band_values


def validate_itc_significance(
    itc_values: np.ndarray, n_trials: int, alpha: float = 0.05, verbose: bool = True
) -> Tuple[np.ndarray, float]:
    """Test ITC significance using Rayleigh test approximation.

    Parameters
    ----------
    itc_values : np.ndarray
        ITC values to test (any shape).
    n_trials : int
        Number of trials used to compute ITC.
    alpha : float, optional
        Significance level. Default is 0.05.
    verbose : bool, optional
        Whether to print progress messages. Default is True.

    Returns
    -------
    significant_mask : np.ndarray
        Boolean array indicating significant ITC values.
    threshold : float
        Significance threshold used.

    Notes
    -----
    Uses the Rayleigh test approximation for circular uniformity.
    For large n_trials, the critical value is approximately sqrt(-ln(alpha)/n_trials).

    References
    ----------
    Fisher, N.I. (1993). Statistical analysis of circular data. Cambridge University Press.
    """
    if n_trials < 5:
        if verbose:
            message(
                "warning",
                f"Very few trials (n={n_trials}) for significance testing. Results may be unreliable.",
            )

    # Rayleigh test approximation for large n
    # Critical value: sqrt(-ln(alpha) / n_trials)
    threshold = np.sqrt(-np.log(alpha) / n_trials)

    # Test significance
    significant_mask = itc_values > threshold
    n_significant = np.sum(significant_mask)
    total_values = itc_values.size

    if verbose:
        message("info", f"ITC significance testing (α = {alpha}):")
        message("debug", f"  Threshold: {threshold:.4f}")
        message(
            "debug",
            f"  Significant values: {n_significant}/{total_values} ({100*n_significant/total_values:.1f}%)",
        )
        message("debug", f"  Max ITC: {np.max(itc_values):.4f}")
        message("debug", f"  Mean ITC: {np.mean(itc_values):.4f}")

    return significant_mask, threshold


def compute_itc_confidence_intervals(
    itc_values: np.ndarray,
    n_trials: int,
    confidence_level: float = 0.95,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute confidence intervals for ITC values.

    Parameters
    ----------
    itc_values : np.ndarray
        ITC values (any shape).
    n_trials : int
        Number of trials used to compute ITC.
    confidence_level : float, optional
        Confidence level (e.g., 0.95 for 95% CI). Default is 0.95.
    verbose : bool, optional
        Whether to print progress messages. Default is True.

    Returns
    -------
    ci_lower : np.ndarray
        Lower confidence interval bounds.
    ci_upper : np.ndarray
        Upper confidence interval bounds.

    Notes
    -----
    Computes approximate confidence intervals for ITC based on
    the circular statistics literature.
    """
    if n_trials < 10:
        if verbose:
            message(
                "warning",
                f"Few trials (n={n_trials}) for CI estimation. Results may be inaccurate.",
            )

    # Approximate standard error for ITC
    # This is a simplified approximation - more sophisticated methods exist
    se_approx = 1.0 / np.sqrt(2 * n_trials)

    # Z-score for confidence level
    from scipy import stats

    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)

    # Confidence intervals (bounded by [0, 1])
    ci_lower = np.maximum(0, itc_values - z_score * se_approx)
    ci_upper = np.minimum(1, itc_values + z_score * se_approx)

    if verbose:
        mean_width = np.mean(ci_upper - ci_lower)
        message(
            "info", f"ITC confidence intervals ({confidence_level*100:.1f}% level):"
        )
        message("debug", f"  Mean CI width: {mean_width:.4f}")
        message("debug", f"  Approximate SE: {se_approx:.4f}")

    return ci_lower, ci_upper


def calculate_word_learning_index(
    itc: mne.time_frequency.AverageTFR,
    word_freq: float = 1.11,
    syllable_freq: float = 3.33,
    freq_tolerance: float = 0.1,
    time_window: Optional[Tuple[float, float]] = None,
    picks: Optional[Union[str, List[str]]] = None,
    verbose: bool = True,
) -> Dict[str, Union[float, np.ndarray]]:
    """Calculate Word Learning Index (WLI) for statistical learning.

    The Word Learning Index provides an estimate of statistical learning and is
    defined as ITC at the word frequency divided by ITC at the syllable frequency.

    Parameters
    ----------
    itc : mne.time_frequency.AverageTFR
        Inter-trial coherence data from statistical learning paradigm.
    word_freq : float, optional
        Word presentation frequency in Hz. Default is 1.11 Hz.
    syllable_freq : float, optional
        Syllable presentation frequency in Hz. Default is 3.33 Hz.
    freq_tolerance : float, optional
        Tolerance for frequency matching in Hz. Default is 0.1 Hz.
    time_window : tuple of float, optional
        Time window for analysis (tmin, tmax) in seconds. If None, uses entire epoch.
    picks : str or list of str, optional
        Channels to analyze. If None, uses all channels. Default is None.
    verbose : bool, optional
        Whether to print progress messages. Default is True.

    Returns
    -------
    wli_results : dict
        Dictionary containing:
        - 'wli': Word Learning Index values (per channel or averaged)
        - 'itc_word': ITC at word frequency
        - 'itc_syllable': ITC at syllable frequency
        - 'word_freq_actual': Actual word frequency used (closest match)
        - 'syllable_freq_actual': Actual syllable frequency used (closest match)
        - 'channel_names': Channel names (if per-channel analysis)

    Notes
    -----
    WLI = ITC(word_frequency) / ITC(syllable_frequency)

    Higher WLI values indicate stronger neural entrainment to word-level structure
    relative to syllable-level structure, suggesting better statistical learning.

    References
    ----------
    Statistical learning paradigm as described in the research protocol.
    """
    if verbose:
        message("info", "Computing Word Learning Index (WLI)...")
        message(
            "debug",
            f"Target frequencies: word={word_freq} Hz, syllable={syllable_freq} Hz",
        )

    # Find closest frequency indices
    word_freq_idx = np.argmin(np.abs(itc.freqs - word_freq))
    syllable_freq_idx = np.argmin(np.abs(itc.freqs - syllable_freq))

    word_freq_actual = itc.freqs[word_freq_idx]
    syllable_freq_actual = itc.freqs[syllable_freq_idx]

    # Check if frequencies are within tolerance
    if abs(word_freq_actual - word_freq) > freq_tolerance:
        message(
            "warning",
            f"Word frequency mismatch: requested {word_freq} Hz, closest available {word_freq_actual:.3f} Hz",
        )

    if abs(syllable_freq_actual - syllable_freq) > freq_tolerance:
        message(
            "warning",
            f"Syllable frequency mismatch: requested {syllable_freq} Hz, closest available {syllable_freq_actual:.3f} Hz",
        )

    if verbose:
        message(
            "debug",
            f"Using frequencies: word={word_freq_actual:.3f} Hz, syllable={syllable_freq_actual:.3f} Hz",
        )

    # Set time window
    if time_window is not None:
        time_mask = (itc.times >= time_window[0]) & (itc.times <= time_window[1])
        time_indices = np.where(time_mask)[0]
        if len(time_indices) == 0:
            raise ValueError(f"No time points found in window {time_window}")
    else:
        time_indices = slice(None)

    # Set channel picks
    if picks is not None:
        pick_indices = mne.pick_channels(itc.ch_names, picks)
        channel_names = [itc.ch_names[i] for i in pick_indices]
    else:
        pick_indices = slice(None)
        channel_names = itc.ch_names.copy()

    # Extract ITC values at target frequencies
    itc_word = itc.data[pick_indices, word_freq_idx, time_indices]
    itc_syllable = itc.data[pick_indices, syllable_freq_idx, time_indices]

    # Average over time if needed
    if itc_word.ndim > 1:
        itc_word_avg = np.mean(itc_word, axis=-1)
        itc_syllable_avg = np.mean(itc_syllable, axis=-1)
    else:
        itc_word_avg = itc_word
        itc_syllable_avg = itc_syllable

    # Calculate Word Learning Index
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        wli = itc_word_avg / itc_syllable_avg
        wli[itc_syllable_avg == 0] = np.nan

    # Calculate summary statistics
    wli_mean = np.nanmean(wli)
    wli_std = np.nanstd(wli)

    if verbose:
        message("info", "Word Learning Index Results:")
        message("debug", f"  Mean WLI: {wli_mean:.4f} ± {wli_std:.4f}")
        message(
            "debug",
            f"  Word ITC: {np.mean(itc_word_avg):.4f} ± {np.std(itc_word_avg):.4f}",
        )
        message(
            "debug",
            f"  Syllable ITC: {np.mean(itc_syllable_avg):.4f} ± {np.std(itc_syllable_avg):.4f}",
        )
        message("debug", f"  Channels analyzed: {len(channel_names)}")

        # Check for potential issues
        if np.any(np.isnan(wli)):
            n_nan = np.sum(np.isnan(wli))
            message(
                "warning",
                f"{n_nan}/{len(wli)} channels have NaN WLI (zero syllable ITC)",
            )

        if wli_mean < 0.1:
            message(
                "warning", "Very low WLI values detected - check frequency extraction"
            )
        elif wli_mean > 10:
            message(
                "warning", "Very high WLI values detected - check frequency extraction"
            )

    return {
        "wli": wli,
        "wli_mean": wli_mean,
        "wli_std": wli_std,
        "itc_word": itc_word_avg,
        "itc_syllable": itc_syllable_avg,
        "word_freq_actual": word_freq_actual,
        "syllable_freq_actual": syllable_freq_actual,
        "channel_names": channel_names,
        "time_window": time_window,
    }


def extract_itc_at_frequencies(
    itc: mne.time_frequency.AverageTFR,
    target_freqs: List[float],
    freq_tolerance: float = 0.1,
    time_window: Optional[Tuple[float, float]] = None,
    picks: Optional[Union[str, List[str]]] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
    """Extract ITC values at specific target frequencies.

    Utility function for extracting ITC at precise frequencies of interest
    in statistical learning paradigms.

    Parameters
    ----------
    itc : mne.time_frequency.AverageTFR
        Inter-trial coherence data.
    target_freqs : list of float
        List of target frequencies to extract.
    freq_tolerance : float, optional
        Tolerance for frequency matching in Hz. Default is 0.1 Hz.
    time_window : tuple of float, optional
        Time window for analysis. If None, uses entire epoch.
    picks : str or list of str, optional
        Channels to analyze. If None, uses all channels.
    verbose : bool, optional
        Whether to print progress messages. Default is True.

    Returns
    -------
    freq_results : dict
        Dictionary with target frequencies as keys, each containing:
        - 'itc_values': ITC values at this frequency
        - 'freq_actual': Actual frequency used (closest match)
        - 'freq_index': Index in frequency axis
    """
    if verbose:
        message("info", f"Extracting ITC at {len(target_freqs)} target frequencies...")

    # Set channel picks
    if picks is not None:
        pick_indices = mne.pick_channels(itc.ch_names, picks)
    else:
        pick_indices = slice(None)

    # Set time window
    if time_window is not None:
        time_mask = (itc.times >= time_window[0]) & (itc.times <= time_window[1])
        time_indices = np.where(time_mask)[0]
    else:
        time_indices = slice(None)

    freq_results = {}

    for target_freq in target_freqs:
        # Find closest frequency
        freq_idx = np.argmin(np.abs(itc.freqs - target_freq))
        freq_actual = itc.freqs[freq_idx]

        # Check tolerance
        if abs(freq_actual - target_freq) > freq_tolerance:
            message(
                "warning",
                f"Frequency {target_freq} Hz: closest available is {freq_actual:.3f} Hz",
            )

        # Extract ITC values
        itc_values = itc.data[pick_indices, freq_idx, time_indices]

        # Average over time if needed
        if itc_values.ndim > 1:
            itc_values = np.mean(itc_values, axis=-1)

        freq_results[f"{target_freq:.2f}Hz"] = {
            "itc_values": itc_values,
            "itc_mean": np.mean(itc_values),
            "itc_std": np.std(itc_values),
            "freq_actual": freq_actual,
            "freq_index": freq_idx,
            "target_freq": target_freq,
        }

        if verbose:
            message(
                "debug",
                f"  {target_freq:.2f} Hz (actual: {freq_actual:.3f} Hz): "
                f"ITC = {np.mean(itc_values):.4f} ± {np.std(itc_values):.4f}",
            )

    return freq_results
