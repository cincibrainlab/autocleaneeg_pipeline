import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd

 


def compute_itc(
    epochs: mne.BaseEpochs,
    roi: list[str] | None = None,
    fmin: float = 0.6,
    fmax: float = 5.0,
    df: float = 0.01,
    tmin: float | None = None,
    tmax: float | None = None,
    target_frequencies: list[float] | None = None,
):
    """Compute ROI-averaged ITC/PLV and power spectra on one frequency grid.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epoched EEG data. Montage is not required for these computations.
    roi : list[str] | None
        Channel names to average across. If None or empty, all EEG channels are used.
    fmin, fmax : float
        Frequency range in Hz to evaluate.
    df : float
        Frequency step used to sample the grid (does not change true resolution ~ 1/T).
    tmin, tmax : float | None
        Optional time window in seconds within each epoch. If either is None, the full epoch is used.
    target_frequencies : list[float] | None
        Specific frequencies to include on the grid (e.g., 1.111, 3.333). Values outside [fmin, fmax]
        are ignored with a warning. Included targets are reflected back in ``info``.

    Returns
    -------
    freqs : np.ndarray
        Frequency grid in Hz (includes any valid ``target_frequencies``).
    plv_roi : np.ndarray
        ROI-averaged PLV (ITC) at each frequency, in [0, 1].
    power_roi : np.ndarray
        ROI- and trial-averaged power at each frequency (periodogram-style units).
    info : dict
        Contains ``target_frequencies`` (np.ndarray of those included on the grid).
    """
    data = epochs.get_data(picks="eeg")
    sfreq = epochs.info["sfreq"]
    _, n_ch, n_times = data.shape
    t = np.arange(n_times) / sfreq

    if roi:
        roi_idx = mne.pick_channels(epochs.ch_names, roi, ordered=False)
    else:
        roi_idx = np.arange(n_ch)
    if len(roi_idx) == 0:
        roi_idx = np.arange(n_ch)

    if tmin is not None and tmax is not None:
        tmask = (t >= tmin) & (t <= tmax)
        data_win = data[..., tmask]
        t_win = t[tmask]
    else:
        data_win = data
        t_win = t

    # Build frequency grid and ensure requested target frequencies are included
    base_grid = np.arange(fmin, fmax, df)
    if target_frequencies is None or len(target_frequencies) == 0:
        extra = np.empty(0)
    else:
        extra = np.asarray(target_frequencies, dtype=float)
        # keep only those inside [fmin, fmax]
        for freq in extra:
            if freq < fmin or freq > fmax:
                print(f"Target frequency {freq} is out of range [{fmin}, {fmax}]")
                extra = np.delete(extra, np.where(extra == freq))
    freqs = np.unique(np.sort(np.r_[base_grid, extra]))

    # Rectangular-window DFT via projection onto complex exponentials
    exps = np.exp(-2j * np.pi * np.outer(freqs, t_win))
    Z = np.tensordot(data_win, exps, axes=([-1], [-1]))  # (n_epochs, n_channels, n_freqs)

    # Periodogram-style power averaged over trials and ROI channels
    power_roi = (np.abs(Z[:, roi_idx, :]) ** 2).mean(axis=(0, 1)) / data_win.shape[-1]

    # PLV: normalize to unit phasors, average across trials, then ROI-average
    phases = Z / np.maximum(np.abs(Z), 1e-12)
    plv = np.abs(phases.mean(axis=0))  # (n_channels, n_freqs)
    plv_roi = plv[roi_idx].mean(axis=0)
    info = {"target_frequencies": freqs[np.isin(freqs, extra)] if extra.size else np.array([])}
    return freqs, plv_roi, power_roi, info


def plot_itc(
    freqs: np.ndarray,
    plv: np.ndarray,
    target_frequencies: list[float] | None = None,
    xlim: tuple[float, float] = (0.6, 5.0),
    output_path: str | None = None,
):
    """Plot an ITC/PLV spectrum with optional vertical lines at target frequencies.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency grid in Hz.
    plv : np.ndarray
        PLV values per frequency (typically ROI-averaged).
    target_frequencies : list[float] | None
        Frequencies to annotate with vertical lines on the plot.
    xlim : tuple[float, float]
        X-axis limits in Hz.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure instance.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, plv, "k-", lw=2)
    if target_frequencies is not None:
        for freq in target_frequencies:
            plt.axvline(freq, color="r", ls="--", label=f"Target {freq:.3f} Hz")
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
    if output_path:
        plt.savefig(output_path)
    return plt.gcf()


def export_itc_csv(
    output_path: str,
    file_name: str,
    freqs: np.ndarray,
    plv_roi: np.ndarray,
    power_roi: np.ndarray,
) -> str:
    """Export a CSV with columns: file_name, frequency, phase_locking_value, power.

    Parameters
    ----------
    output_path : str
        Destination CSV file. Appends if the file exists; writes header only once.
    file_name : str
        Identifier to store in the file_name column (e.g., subject or dataset name).
    freqs : np.ndarray
        Frequency grid in Hz.
    plv_roi : np.ndarray
        ROI-averaged PLV values per frequency.
    power_roi : np.ndarray
        ROI- and trial-averaged power values per frequency.

    Returns
    -------
    str
        The path to the written CSV file.
    """

    import os

    df_out = pd.DataFrame(
        {
            "file_name": [file_name] * len(freqs),
            "frequency": np.round(freqs.astype(float), 3),
            "phase_locking_value": plv_roi.astype(float),
            "power": power_roi.astype(float),
        }
    )

    file_exists = os.path.isfile(output_path)
    df_out.to_csv(
        output_path,
        mode="a",
        header=not file_exists, # Only write header if file doesn't exist
        index=False,
    )
    return output_path
