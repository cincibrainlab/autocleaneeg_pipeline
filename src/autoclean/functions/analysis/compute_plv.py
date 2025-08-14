import numpy as np
import mne
import matplotlib.pyplot as plt


def compute_plv_spectrum(
    epochs: mne.BaseEpochs,
    roi: list[str] | None = None,
    fmin: float = 0.6,
    fmax: float = 5.0,
    df: float = 0.01,
    tmin: float | None = None,
    tmax: float | None = None,
    syllable_interval: float = 0.3,
    word_length_syllables: int = 3,
):
    """Compute an ROI-averaged PLV spectrum across a frequency range.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epoched EEG data. Montage should already be set if needed for other plots.
    roi : list[str] | None
        Channel names to average across. If None or empty, all EEG channels are used.
    fmin, fmax : float
        Frequency range in Hz for the spectrum.
    df : float
        Frequency step (grid spacing). Effective spectral resolution is ~1/T where
        T is the epoch/window duration; df only controls sampling density.
    tmin, tmax : float | None
        Optional time window in seconds within each epoch. If either is None, the
        full epoch is used.
    syllable_interval : float
        Syllable duration in seconds (e.g., 0.3s → 3.33 Hz).
    word_length_syllables : int
        Number of syllables per word (e.g., 3 → 1.11 Hz).

    Returns
    -------
    freqs : np.ndarray
        Frequency grid in Hz (includes exact word/syllable targets if within range).
    plv_roi : np.ndarray
        ROI-averaged PLV at each frequency, values in [0, 1].
    info : dict
        Dictionary with keys ``word_frequency`` and ``syllable_frequency``.

    Notes
    -----
    PLV is computed per channel as the magnitude of the across-trial mean of unit
    phasors at each frequency; the ROI spectrum is the channel-average of those PLVs.
    """
    # Target frequencies commonly used in statistical learning
    syllable_frequency = 1.0 / syllable_interval
    word_frequency = syllable_frequency / float(word_length_syllables)

    # Extract EEG data (n_epochs, n_channels, n_times)
    data = epochs.get_data(picks="eeg")
    sfreq = epochs.info["sfreq"]
    _, n_ch, n_times = data.shape
    t = np.arange(n_times) / sfreq

    # Resolve ROI channel indices; default to all channels if none provided
    if roi:
        roi_idx = mne.pick_channels(epochs.ch_names, roi, ordered=False)
    else:
        roi_idx = np.arange(n_ch)
    if len(roi_idx) == 0:
        roi_idx = np.arange(n_ch)

    # Build frequency grid and ensure exact target frequencies are included
    freqs = np.unique(
        np.sort(np.r_[np.arange(fmin, fmax, df), word_frequency, syllable_frequency])
    )

    # Optional central time window to mitigate edge effects; defaults to full epoch
    if tmin is not None and tmax is not None:
        tmask = (t >= tmin) & (t <= tmax)
        data_win = data[..., tmask]
        t_win = t[tmask]
    else:
        data_win = data
        t_win = t

    # Rectangular-window DFT via projection onto complex exponentials
    exps = np.exp(-2j * np.pi * np.outer(freqs, t_win))
    Z = np.tensordot(data_win, exps, axes=([-1], [-1]))  # (n_epochs, n_channels, n_freqs)

    # Normalize to unit phasors to discard amplitude, then average across trials
    phases = Z / np.maximum(np.abs(Z), 1e-12)
    plv = np.abs(phases.mean(axis=0))  # (n_channels, n_freqs)

    # Average PLV across ROI channels to obtain a single spectrum
    plv_roi = plv[roi_idx].mean(axis=0)

    return freqs, plv_roi, {"word_frequency": word_frequency, "syllable_frequency": syllable_frequency}


def plot_plv_spectrum(
    freqs: np.ndarray,
    plv: np.ndarray,
    word_frequency: float | None = None,
    syllable_frequency: float | None = None,
    xlim: tuple[float, float] = (0.6, 5.0),
):
    """Plot a PLV spectrum with optional markers at target frequencies.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency grid in Hz.
    plv : np.ndarray
        PLV values per frequency (typically ROI-averaged).
    word_frequency, syllable_frequency : float | None
        Optional target frequencies to annotate on the plot.
    xlim : tuple[float, float]
        X-axis limits in Hz.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure instance.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, plv, "k-", lw=2)
    if word_frequency is not None:
        plt.axvline(word_frequency, color="b", ls="--", label=f"Word {word_frequency:.3f} Hz")
    if syllable_frequency is not None:
        plt.axvline(syllable_frequency, color="r", ls="--", label=f"Syllable {syllable_frequency:.3f} Hz")
    plt.xlim(*xlim)
    yl = (float(plv.min()), float(plv.max()))
    pad = max(1e-3, 0.1 * (yl[1] - yl[0]))
    plt.ylim(yl[0] - pad, yl[1] + pad)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PLV")
    plt.title("PLV Spectrum")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()


if __name__ == "__main__":
    epochs = mne.read_epochs_eeglab("C:/Users/Gam9LG/Documents/Autoclean-EEG/output/Statistical_Learning/bids/derivatives/autoclean-v2.1.0/intermediate/FLAGGED_08_drop_bad_epochs/1037_slstructured_drop_bad_epochs_epo.set")
    freqs, plv, freq_info = compute_plv_spectrum(epochs, roi=[f"E{idx + 1}" for idx in [27,19,11,4,117,116,28,12,5,111,110,35,29,6,105,104,103,30,79,54,36,86,40,102]])
    plot_plv_spectrum(freqs, plv, **freq_info)
    plt.show()