from pathlib import Path

import mne
import numpy as np
import pandas as pd


def load_epochs(file_path):
    """
    Load epochs from an EEGLAB .set file

    Parameters:
    -----------
    file_path : str or Path
        Path to the EEGLAB .set file

    Returns:
    --------
    epochs : mne.Epochs
        The loaded epochs object
    """
    file_path = Path(file_path)
    epochs = mne.io.read_epochs_eeglab(file_path)
    print(f"Loaded {len(epochs)} epochs with {len(epochs.ch_names)} channels")
    print(f"Epoch duration: {epochs.times[0]:.3f}s to {epochs.times[-1]:.3f}s")
    return epochs


def compute_time_frequency(epochs, freqs=None, n_cycles=None, baseline=(-0.5, 0)):
    """
    Compute time-frequency representations (power, ITC, ERSP, single trial power)

    Parameters:
    -----------
    epochs : mne.Epochs
        Epochs object to analyze
    freqs : array, optional
        Frequencies to analyze. If None, will use optimized frequency array for 40 Hz
    n_cycles : array, optional
        Number of cycles for Morlet wavelets. If None, will optimize for 40 Hz
    baseline : tuple, optional
        Baseline period for ERSP calculation

    Returns:
    --------
    dict : Dictionary containing time-frequency results
        - 'power': Average power
        - 'itc': ITC values (tuple with power and itc)
        - 'ersp': Event-related spectral perturbation
        - 'single_trial_power': Single trial power
        - 'freqs': Frequency array used
    """
    # Create optimized frequency array if not provided
    if freqs is None:
        freqs_low = np.arange(1, 30, 2)  # Lower frequencies with coarser resolution
        freqs_mid = np.arange(30, 50, 0.5)  # Finer resolution around 40 Hz
        freqs_high = np.arange(50, 101, 2)  # Higher frequencies with coarser resolution
        freqs = np.concatenate([freqs_low, freqs_mid, freqs_high])

    # Optimize wavelet cycles if not provided
    if n_cycles is None:
        n_cycles_base = freqs / 2.0  # Base cycles
        # Increase cycles around 40 Hz for better frequency resolution
        n_cycles = n_cycles_base.copy()
        freq_mask = (freqs >= 35) & (freqs <= 45)
        n_cycles[freq_mask] = freqs[freq_mask] / 1.5  # More cycles around 40 Hz

    # Compute average power
    power = epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        decim=3,
        n_jobs=1,
        average=True,
    )

    # Compute ITC (inter-trial coherence)
    itc = epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=True,
        decim=3,
        n_jobs=1,
        average=True,
    )

    # Compute single trial power (non-baseline corrected)
    single_trial_power = epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        decim=3,
        n_jobs=1,
        average=False,
    )

    # Compute ERSP (baseline corrected power)
    ersp = epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        decim=3,
        n_jobs=1,
        average=True,
    )
    # Apply baseline correction after computing TFR
    ersp.apply_baseline(baseline, mode="mean")

    return {
        "power": power,
        "itc": itc,
        "ersp": ersp,
        "single_trial_power": single_trial_power,
        "freqs": freqs,
    }


def compute_metrics(tf_data, epochs, freq_bands=None, time_windows=None):
    """
    Compute various metrics from time-frequency data

    Parameters:
    -----------
    tf_data : dict
        Output from compute_time_frequency function
    epochs : mne.Epochs
        Epochs object used for computation
    freq_bands : dict, optional
        Dictionary of frequency bands to analyze
    time_windows : dict, optional
        Dictionary of time windows to analyze

    Returns:
    --------
    results_df : pandas.DataFrame
        DataFrame containing computed metrics for each channel
    """
    # Unpack time-frequency data
    itc = tf_data["itc"]
    single_trial_power = tf_data["single_trial_power"]
    ersp = tf_data["ersp"]
    freqs = tf_data["freqs"]

    # Define frequency bands if not provided
    if freq_bands is None:
        freq_bands = {
            "alpha": (8, 13),
            "theta": (4, 7),
            "gamma1": (30, 55),
            "gamma2": (65, 80),
            "itc40": (35, 45),
            "itc80": (75, 85),
            "itc_onset": (2, 13),
        }

    # Define time windows if not provided
    if time_windows is None:
        time_windows = {
            "all": (0, 3.0),
            "itc_onset": (0.092, 0.308),
            "itc_offset": (2.8, 3.0),
        }

    # Helper functions to find indices
    def find_freq_indices(freqs, fmin, fmax):
        return np.where((freqs >= fmin) & (freqs <= fmax))[0]

    def find_time_indices(times, tmin, tmax):
        return np.where((times >= tmin) & (times <= tmax))[0]

    # Calculate file info and rejected trials
    if hasattr(epochs, "filename") and epochs.filename is not None:
        file_path = Path(epochs.filename)
        file_basename = file_path.stem
    else:
        # For synthetic/in-memory epochs without a filename
        file_path = Path("synthetic_data")
        file_basename = "synthetic_epochs"

    n_total_trials = len(epochs.drop_log)
    n_used_trials = len(epochs)
    n_rejected_trials = sum(
        1 for log in epochs.drop_log if log
    )  # Count non-empty drop logs

    # Create a list to store results
    results = []

    # Process each channel
    for ch_idx, ch_name in enumerate(epochs.ch_names):
        # Calculate ITC40 (mean ITC from 0-3s in 35-45 Hz)
        itc40_idx = find_freq_indices(freqs, *freq_bands["itc40"])
        time_all_idx = find_time_indices(
            itc[1].times, *time_windows["all"]
        )  # itc[1] for actual ITC values
        itc40 = np.mean(itc[1].data[ch_idx, itc40_idx][:, time_all_idx])

        # Calculate ITC80 (mean ITC from 0-3s in 75-85 Hz)
        itc80_idx = find_freq_indices(freqs, *freq_bands["itc80"])
        itc80 = np.mean(itc[1].data[ch_idx, itc80_idx][:, time_all_idx])

        # Calculate ITC onset (2-13 Hz, 92-308 ms)
        itc_onset_freq_idx = find_freq_indices(freqs, *freq_bands["itc_onset"])
        itc_onset_time_idx = find_time_indices(itc[1].times, *time_windows["itc_onset"])
        itc_onset = np.mean(
            itc[1].data[ch_idx, itc_onset_freq_idx][:, itc_onset_time_idx]
        )

        # Calculate ITC offset (2-13 Hz, 2800-3000 ms)
        itc_offset_freq_idx = find_freq_indices(
            freqs, *freq_bands["itc_onset"]
        )  # Same frequency band as onset
        itc_offset_time_idx = find_time_indices(
            itc[1].times, *time_windows["itc_offset"]
        )
        itc_offset = np.mean(
            itc[1].data[ch_idx, itc_offset_freq_idx][:, itc_offset_time_idx]
        )

        # Calculate single trial power (STP) for different frequency bands
        stp_gamma1_idx = find_freq_indices(freqs, *freq_bands["gamma1"])
        stp_gamma1 = np.mean(single_trial_power.data[:, ch_idx, stp_gamma1_idx, :])

        stp_gamma2_idx = find_freq_indices(freqs, *freq_bands["gamma2"])
        stp_gamma2 = np.mean(single_trial_power.data[:, ch_idx, stp_gamma2_idx, :])

        stp_alpha_idx = find_freq_indices(freqs, *freq_bands["alpha"])
        stp_alpha = np.mean(single_trial_power.data[:, ch_idx, stp_alpha_idx, :])

        stp_theta_idx = find_freq_indices(freqs, *freq_bands["theta"])
        stp_theta = np.mean(single_trial_power.data[:, ch_idx, stp_theta_idx, :])

        # Calculate combined gamma (30-80 Hz)
        stp_gamma_idx = find_freq_indices(freqs, 30, 80)
        stp_gamma = np.mean(single_trial_power.data[:, ch_idx, stp_gamma_idx, :])

        # Calculate ERSP for different frequency bands
        ersp_gamma1_idx = find_freq_indices(freqs, *freq_bands["gamma1"])
        ersp_gamma1 = np.mean(ersp.data[ch_idx, ersp_gamma1_idx, :])

        ersp_gamma2_idx = find_freq_indices(freqs, *freq_bands["gamma2"])
        ersp_gamma2 = np.mean(ersp.data[ch_idx, ersp_gamma2_idx, :])

        ersp_alpha_idx = find_freq_indices(freqs, *freq_bands["alpha"])
        ersp_alpha = np.mean(ersp.data[ch_idx, ersp_alpha_idx, :])

        ersp_theta_idx = find_freq_indices(freqs, *freq_bands["theta"])
        ersp_theta = np.mean(ersp.data[ch_idx, ersp_theta_idx, :])

        # Calculate combined gamma ERSP (30-80 Hz)
        ersp_gamma_idx = find_freq_indices(freqs, 30, 80)
        ersp_gamma = np.mean(ersp.data[ch_idx, ersp_gamma_idx, :])

        # Store results for this channel
        results.append(
            {
                "eegid": file_basename,
                "trials": n_total_trials,
                "chan": ch_name,
                "rejtrials": n_rejected_trials,
                "stp_gamma": stp_gamma,
                "stp_gamma1": stp_gamma1,
                "stp_gamma2": stp_gamma2,
                "stp_alpha": stp_alpha,
                "stp_theta": stp_theta,
                "ersp_gamma": ersp_gamma,
                "ersp_gamma1": ersp_gamma1,
                "ersp_gamma2": ersp_gamma2,
                "ersp_alpha": ersp_alpha,
                "ersp_theta": ersp_theta,
                "itc40": itc40,
                "itc80": itc80,
                "itconset": itc_onset,
                "itcoffset": itc_offset,
            }
        )

    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    return results_df


def analyze_assr(
    file_path=None, output_dir=None, save_results=True, epochs=None, file_basename=None
):
    """
    Main function to analyze ASSR data

    Parameters:
    -----------
    file_path : str or Path, optional
        Path to the EEGLAB .set file. Not required if epochs are provided directly.
    output_dir : str or Path, optional
        Directory to save results
    save_results : bool, optional
        Whether to save results to disk
    epochs : mne.Epochs, optional
        Pre-loaded MNE Epochs object. If provided, file_path is ignored.
    file_basename : str, optional
        Base filename to use for saving results when epochs don't have a filename.
        Takes precedence over automatically extracted filenames.

    Returns:
    --------
    dict : Dictionary containing analysis results
        - 'results_df': DataFrame with computed metrics
        - 'tf_data': Time-frequency data
        - 'epochs': MNE Epochs object
        - 'file_basename': Basename used for saving files
    """
    # Set up output directory
    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load epochs if not provided
    if epochs is None:
        if file_path is None:
            raise ValueError("Either file_path or epochs must be provided")
        epochs = load_epochs(file_path)

    # Compute time-frequency representations
    tf_data = compute_time_frequency(epochs)

    # Compute metrics
    results_df = compute_metrics(tf_data, epochs)

    # Print summary information
    freq_idx = np.argmin(np.abs(tf_data["freqs"] - 40))
    print(
        f"Maximum ITC value at 40 Hz: {np.max(tf_data['itc'][1].data[:, freq_idx, :]):.3f}"
    )

    # Find midrange frequencies around 40 Hz to compute resolution
    freqs_mid = [f for f in tf_data["freqs"] if 30 <= f <= 50]
    if len(freqs_mid) > 1:
        print(
            f"Frequency resolution around 40 Hz: {freqs_mid[1] - freqs_mid[0]:.2f} Hz"
        )

    # Save results if requested
    if save_results:
        # Determine file basename with explicit parameter taking precedence
        if file_basename is not None:
            # Use provided basename
            pass
        elif file_path is not None:
            file_basename = Path(file_path).stem
        elif hasattr(epochs, "filename") and epochs.filename is not None:
            file_basename = Path(epochs.filename).stem
        else:
            # Use a default name if no file path or filename is provided
            file_basename = "assr_analysis"

        # Create data subdirectory
        data_dir = output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save with descriptive filename including basename
        output_filename = data_dir / f"{file_basename}_metrics_assr.csv"
        results_df.to_csv(output_filename, index=False)
        print(f"Saved analysis results to {output_filename}")

    # Return results as dictionary
    return {
        "results_df": results_df,
        "tf_data": tf_data,
        "epochs": epochs,
        "file_basename": file_basename,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze ASSR data from EEGLAB .set files"
    )
    parser.add_argument("file_path", type=str, help="Path to the EEGLAB .set file")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Directory to save results"
    )
    parser.add_argument(
        "--no_save_results",
        action="store_false",
        dest="save_results",
        help="Do not save results to disk",
    )

    args = parser.parse_args()

    analyze_assr(args.file_path, args.output_dir, args.save_results)
