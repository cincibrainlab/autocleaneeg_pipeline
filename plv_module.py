import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd

 


def compute_plv_power_spectrum(
    epochs: mne.BaseEpochs,
    roi: list[str] | None = None,
    fmin: float = 0.6,
    fmax: float = 5.0,
    df: float = 0.01,
    tmin: float | None = None,
    tmax: float | None = None,
    target_frequencies: list[float] | None = None,
):
    """Compute ROI-averaged PLV and power spectra on the same frequency grid.

    Power is estimated via a simple periodogram: mean over trials and ROI of
    |DFT|^2 divided by the number of time samples in the window.
    Returns
    -------
    freqs : np.ndarray
        Frequency grid in Hz (includes ``target_frequencies`` if provided and within range).
    plv_roi : np.ndarray
        ROI-averaged PLV at each frequency, values in [0, 1].
    power_roi : np.ndarray
        ROI- and trial-averaged power at each frequency (periodogram units).
    info : dict
        Dictionary with key ``target_frequencies`` (np.ndarray or empty list).
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


def plot_plv_spectrum(
    freqs: np.ndarray,
    plv: np.ndarray,
    target_frequencies: list[float] | None = None,
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
    return plt.gcf()


def export_plv_power_csv(
    output_path: str,
    file_name: str,
    freqs: np.ndarray,
    plv_roi: np.ndarray,
    power_roi: np.ndarray,
) -> str:
    """Export a CSV with columns file_name, frequency, phase_locking_value, power.

    Returns the path to the written CSV.
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


if __name__ == "__main__":
    epochs = mne.read_epochs_eeglab("C:/Users/Gam9LG/Documents/Autoclean-EEG/output/Statistical_Learning/bids/derivatives/autoclean-v2.1.0/intermediate/FLAGGED_08_drop_bad_epochs/1037_slstructured_drop_bad_epochs_epo.set")
    freqs, plv_roi, power_roi, info = compute_plv_power_spectrum(epochs, roi=[f"E{idx + 1}" for idx in [27,19,11,4,117,116,28,12,5,111,110,35,29,6,105,104,103,30,79,54,36,86,40,102]], target_frequencies=[10/9, 30/9])
    plot_plv_spectrum(freqs, plv_roi, target_frequencies=info["target_frequencies"])
    plt.show()
    export_plv_power_csv(
        output_path="./plv_power_spectrum.csv",
        file_name="glibglob",
        freqs=freqs,
        plv_roi=plv_roi,
        power_roi=power_roi,
    )