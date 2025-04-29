# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mne",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "scipy",
#     "tqdm",
# ]
# ///

import argparse
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from tqdm import tqdm

# Configure logging to output to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[logging.FileHandler("eeg_processing.log"), logging.StreamHandler()],
)

# Define the folder containing your .set EEG files
eeg_folder = "/mnt/srv2/RAWDATA/HBCD/For_Ernie_MMN_artifact_data_check/SET"


def get_set_files(folder_path: str) -> List[str]:
    """
    Retrieve a list of EEG .set files from the specified folder.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing .set files.

    Returns
    -------
    List[str]
        List of full file paths to .set files.
    """
    logging.info(f"Retrieving .set files from folder: {folder_path}")
    if not os.path.isdir(folder_path):
        logging.error(f"Folder does not exist: {folder_path}")
        return []
    set_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".set")
    ]
    logging.info(f"Found {len(set_files)} .set files.")
    return set_files


def preprocess_raw(
    raw, apply_notch: bool = False, notch_freqs: List[float] = [50.0, 60.0]
):
    """
    Preprocess the raw EEG data with standard parameters and optional notch filtering.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data to preprocess.
    apply_notch : bool, optional
        Whether to apply a notch filter to remove powerline noise. Default is False.
    notch_freqs : List[float], optional
        List of frequencies to remove using the notch filter. Default is [50.0, 60.0].

    Returns
    -------
    mne.io.Raw
        Preprocessed EEG data.
    """
    logging.info("Preprocessing raw EEG data...")

    # Resample to 250 Hz
    raw.resample(250, npad="auto")
    logging.info("Resampled to 250 Hz.")

    # Apply a band-pass filter between 1 Hz and 80 Hz
    raw.filter(l_freq=1.0, h_freq=80.0, fir_design="firwin")
    logging.info("Applied band-pass filter between 1 Hz and 80 Hz.")

    # Apply notch filter if enabled
    if apply_notch:
        raw.notch_filter(freqs=notch_freqs, fir_design="firwin")
        logging.info(f"Applied notch filter at frequencies: {notch_freqs} Hz.")

    return raw


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to be used as a valid CSV column name.

    Parameters
    ----------
    filename : str
        Original filename.

    Returns
    -------
    str
        Sanitized filename.
    """
    return re.sub(r"[^\w\-]", "_", filename)


def extract_numerical_id(filename: str) -> str:
    """
    Extract the first numerical ID from the filename.

    Parameters
    ----------
    filename : str
        Filename from which to extract the numerical ID.

    Returns
    -------
    str
        Extracted numerical ID or the original filename if no ID is found.
    """
    match = re.search(r"\d+", filename)
    if match:
        return match.group()
    else:
        return filename  # Fallback to filename if no numerical ID is found


def compute_power_spectrum(
    data: np.ndarray, sfreq: float, fmin: float = 1.0, fmax: float = 80.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the power spectrum of an EEG time series using Welch's method.

    Parameters
    ----------
    data : np.ndarray
        EEG time series data.
    sfreq : float
        Sampling frequency in Hz.
    fmin : float, optional
        Minimum frequency of interest. Default is 1.0 Hz.
    fmax : float, optional
        Maximum frequency of interest. Default is 80.0 Hz.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing frequency bins and corresponding power spectral density.
    """
    freqs, power = welch(data, fs=sfreq, nperseg=1024)
    # Restrict to desired frequency range
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freq_mask]
    power = power[freq_mask]
    return freqs, power


def load_eeg_data(
    set_file_path: str,
    electrode: str = "E55",
    apply_notch: bool = False,
    notch_freqs: List[float] = [50.0, 60.0],
) -> Tuple[np.ndarray, float]:
    """
    Load EEG data from a .set file and extract the time series for a specific electrode.

    Parameters
    ----------
    set_file_path : str
        Path to the EEG .set file.
    electrode : str, optional
        Name of the electrode to extract. Default is 'E55'.
    apply_notch : bool, optional
        Whether to apply a notch filter to remove powerline noise. Default is False.
    notch_freqs : List[float], optional
        List of frequencies to remove using the notch filter. Default is [50.0, 60.0].

    Returns
    -------
    Tuple[np.ndarray, float]
        Tuple containing the time series data (in microvolts) and the sampling frequency.
    """
    logging.info(f"Loading EEG data from file: {set_file_path}")
    try:
        raw = mne.io.read_raw_eeglab(set_file_path, preload=True, verbose=False)
        raw = preprocess_raw(raw, apply_notch=apply_notch, notch_freqs=notch_freqs)
    except Exception as e:
        logging.error(f"Error loading {set_file_path}: {e}")
        raise e

    if electrode not in raw.ch_names:
        logging.error(
            f"Electrode '{electrode}' not found in {set_file_path}. Available electrodes: {raw.ch_names}"
        )
        raise ValueError(f"Electrode '{electrode}' not found.")

    logging.info(f"Extracting data for electrode: {electrode}")
    data, times = raw.copy().pick_channels([electrode]).get_data(return_times=True)
    sfreq = raw.info["sfreq"]  # Sampling frequency in Hz

    # Convert data to microvolts (assuming data is in Volts)
    data_microvolts = data.flatten() * 1e6  # Convert to µV
    logging.info(f"Data loaded: {len(data_microvolts)} samples at {sfreq} Hz.")
    return data_microvolts, sfreq


def truncate_time_series(
    data: np.ndarray, sfreq: float, max_seconds: float = 180.0
) -> np.ndarray:
    """
    Truncate the time series data to a maximum duration.

    Parameters
    ----------
    data : np.ndarray
        The EEG time series data.
    sfreq : float
        Sampling frequency in Hz.
    max_seconds : float, optional
        Maximum duration in seconds. Default is 180 seconds.

    Returns
    -------
    np.ndarray
        Truncated EEG time series data.
    """
    max_samples = int(max_seconds * sfreq)
    if len(data) > max_samples:
        logging.info(
            f"Truncating data from {len(data)} samples to {max_samples} samples ({max_seconds} seconds)."
        )
        return data[:max_samples]
    else:
        logging.info(
            f"Data length {len(data)} samples is within the {max_seconds} seconds limit."
        )
        # Pad with NaNs to maintain uniform length
        padding = np.full(max_samples - len(data), np.nan)
        return np.concatenate((data, padding))


def calculate_noisy_metrics(data: np.ndarray, sfreq: float) -> Dict[str, float]:
    """
    Calculate noisy metrics for a given EEG time series.

    Parameters
    ----------
    data : np.ndarray
        EEG time series data.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    Dict[str, float]
        Dictionary containing noisy metrics.
    """
    metrics = {}
    # Standard Deviation
    metrics["std_dev"] = np.std(data)
    # Variance
    metrics["variance"] = np.var(data)
    # Root Mean Square (RMS)
    metrics["rms"] = np.sqrt(np.mean(data**2))
    # Number of Peaks above 100 µV
    peak_threshold = 100.0  # µV
    metrics["num_peaks_100uV"] = np.sum(np.abs(data) > peak_threshold)
    # Signal-to-Noise Ratio (SNR) estimation
    metrics["snr"] = (
        np.mean(np.abs(data)) / metrics["std_dev"]
        if metrics["std_dev"] != 0
        else np.nan
    )
    return metrics


def save_time_series_to_csv(
    time_vector: np.ndarray,
    data_list: List[np.ndarray],
    file_ids: List[str],
    output_csv_path: str,
):
    """
    Save multiple EEG time series to a CSV file with time as first column and data from each file in subsequent columns.

    Parameters
    ----------
    time_vector : np.ndarray
        Array of time points in seconds.
    data_list : List[np.ndarray]
        List of EEG time series arrays.
    file_ids : List[str]
        List of numerical IDs corresponding to each file.
    output_csv_path : str
        Path to save the CSV file.
    """
    logging.info(f"Saving time series data to CSV: {output_csv_path}")

    # Create DataFrame with time as first column
    df = pd.DataFrame({"Time (s)": time_vector})

    # Add data columns with sanitized numerical IDs as headers
    for data, file_id in zip(data_list, file_ids):
        sanitized_name = sanitize_filename(file_id)
        df[sanitized_name] = data

    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    logging.info(f"Time series data saved to {output_csv_path}")


def save_noisy_metrics_to_csv(
    metrics_list: List[Dict[str, float]], output_csv_path: str
):
    """
    Save noisy metrics for multiple EEG time series to a CSV file.

    Parameters
    ----------
    metrics_list : List[Dict[str, float]]
        List of dictionaries containing noisy metrics for each file.
    output_csv_path : str
        Path to save the CSV file.
    """
    logging.info(f"Saving noisy metrics summary to CSV: {output_csv_path}")
    # Convert list of dicts to DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    # Reorder columns to place 'File' first
    if "File" in metrics_df.columns:
        cols = ["File", "std_dev", "variance", "rms", "num_peaks_100uV", "snr"]
        metrics_df = metrics_df[cols]
    metrics_df.to_csv(output_csv_path, index=False)
    logging.info(f"Noisy metrics summary saved to {output_csv_path}")


def save_power_spectrum_to_csv(
    freqs: np.ndarray, power: np.ndarray, output_csv_path: str
):
    """
    Save the power spectrum data to a CSV file.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency bins.
    power : np.ndarray
        Power spectral density values.
    output_csv_path : str
        Path to save the CSV file.
    """
    logging.info(f"Saving power spectrum to CSV: {output_csv_path}")
    df = pd.DataFrame({"Frequency (Hz)": freqs, "Power (µV²/Hz)": power})
    df.to_csv(output_csv_path, index=False)
    logging.info(f"Power spectrum data saved to {output_csv_path}")


def plot_power_spectrum(
    freqs: np.ndarray,
    power: np.ndarray,
    file_id: str,
    output_plot_path: str,
    color: str = None,
):
    """
    Plot the power spectrum and save the plot to a file.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency bins.
    power : np.ndarray
        Power spectral density values.
    file_id : str
        Numerical ID of the EEG file (for plot title).
    output_plot_path : str
        Path to save the plot image.
    color : str, optional
        Color of the plot line. If None, a default color is used.
    """
    logging.info(
        f"Plotting power spectrum for {file_id} and saving to {output_plot_path}"
    )
    plt.figure(figsize=(10, 6))
    plt.semilogy(freqs, power, label=file_id, color=color)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (µV²/Hz)")
    plt.title(f"Power Spectrum for {file_id} at Electrode E55")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.close()
    logging.info(f"Power spectrum plot saved to {output_plot_path}")


def plot_summary_power_spectrum(
    power_spectra: List[Dict[str, np.ndarray]], output_plot_path: str
):
    """
    Plot a summary power spectrum with multiple files' spectra in different colors, labeled by numerical IDs.

    Parameters
    ----------
    power_spectra : List[Dict[str, np.ndarray]]
        List of dictionaries containing 'File' (Numerical_ID), 'Frequency (Hz)', and 'Power (µV²/Hz)'.
    output_plot_path : str
        Path to save the summary plot image.
    """
    logging.info(f"Plotting summary power spectrum and saving to {output_plot_path}")
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10.colors  # Up to 10 distinct colors

    for idx, spectrum in enumerate(power_spectra):
        file_id = spectrum["File"]
        freqs = spectrum["Frequency (Hz)"]
        power = spectrum["Power (µV²/Hz)"]
        color = colors[idx % len(colors)]
        plt.semilogy(freqs, power, label=file_id, color=color)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (µV²/Hz)")
    plt.title("Summary Power Spectrum for Electrode E55")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.close()
    logging.info(f"Summary power spectrum plot saved to {output_plot_path}")


def plot_summary_jitter(metrics_df: pd.DataFrame, output_plot_path: str):
    """
    Create summary jitter plots for noisy channel metrics with elegant subplots and annotations.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame containing noisy metrics with 'File' and 'Numerical_ID' columns.
    output_plot_path : str
        Path to save the summary jitter plot image.
    """
    logging.info(f"Creating summary jitter plots and saving to {output_plot_path}")

    # List of metrics to plot
    metrics = ["std_dev", "variance", "rms", "num_peaks_100uV", "snr"]

    # Number of metrics
    num_metrics = len(metrics)

    # Determine subplot grid size
    cols = 2
    rows = (num_metrics + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()  # Flatten in case of multiple rows

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        # Extract metric values and file identifiers
        values = metrics_df[metric].values
        files = metrics_df["Numerical_ID"].values

        # Generate jitter
        jitter_strength = 0.1  # Adjust as needed
        x_positions = np.random.normal(1, jitter_strength, size=len(values))

        # Create scatter plot with jitter
        ax.scatter(x_positions, values, alpha=0.7, edgecolors="w", s=100)

        # Annotate each point with its numerical ID
        for i, txt in enumerate(files):
            ax.annotate(
                txt,
                (x_positions[i], values[i]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=8,
            )

        # Set labels and title
        ax.set_xlabel("")  # No x-axis label for jitter plot
        ax.set_ylabel(metric.replace("_", " ").capitalize())
        ax.set_title(f"Jitter Plot of {metric.replace('_', ' ').capitalize()}")
        ax.set_xlim(0.5, 1.5)
        ax.set_xticks([])  # Hide x-axis ticks

    # Remove any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.close()
    logging.info(f"Summary jitter plots saved to {output_plot_path}")


def save_power_spectra_summary_to_csv(
    power_spectra_summary: List[Dict[str, float]], output_csv_path: str
):
    """
    Save power spectra summary for multiple EEG time series to a CSV file.

    Parameters
    ----------
    power_spectra_summary : List[Dict[str, float]]
        List of dictionaries containing power spectra summary for each file.
    output_csv_path : str
        Path to save the CSV file.
    """
    logging.info(f"Saving power spectra summary to CSV: {output_csv_path}")
    df = pd.DataFrame(power_spectra_summary)
    df.to_csv(output_csv_path, index=False)
    logging.info(f"Power spectra summary saved to {output_csv_path}")


def process_single_file(
    file_path: str, electrode: str, apply_notch: bool, notch_freqs: List[float]
) -> Tuple[
    str, np.ndarray, float, Dict[str, float], Dict[str, float], Dict[str, np.ndarray]
]:
    """
    Process a single EEG .set file: load data, preprocess, extract metrics, compute power spectrum.

    Parameters
    ----------
    file_path : str
        Path to the EEG .set file.
    electrode : str
        Name of the electrode to extract.
    apply_notch : bool
        Whether to apply a notch filter to remove powerline noise.
    notch_freqs : List[float]
        Frequencies to remove using the notch filter.

    Returns
    -------
    Tuple[str, np.ndarray, float, Dict[str, float], Dict[str, float], Dict[str, np.ndarray]]
        A tuple containing:
        - file_id: Numerical ID extracted from the filename.
        - truncated_e55_data: Truncated EEG data array.
        - sampling_freq: Sampling frequency in Hz.
        - metrics: Dictionary of noisy metrics.
        - power_summary: Dictionary of power spectrum summary (peak freq and power).
        - power_spectrum: Dictionary containing 'Frequency (Hz)' and 'Power (µV²/Hz)'.
    """
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    file_id = extract_numerical_id(file_name)
    logging.info(f"Processing file: {file_name} (ID: {file_id})")
    try:
        # Load EEG data and extract E55 time series
        e55_data, sampling_freq = load_eeg_data(
            set_file_path=file_path,
            electrode=electrode,
            apply_notch=apply_notch,
            notch_freqs=notch_freqs,
        )
        # Truncate the E55 data to 180 seconds
        truncated_e55_data = truncate_time_series(
            e55_data, sampling_freq, max_seconds=180.0
        )

        logging.info(
            f"Data loaded: {len(truncated_e55_data)} samples at {sampling_freq} Hz."
        )

        # Calculate noisy metrics
        metrics = calculate_noisy_metrics(truncated_e55_data, sampling_freq)
        metrics["File"] = file_id  # Use numerical ID

        # Compute power spectrum
        freqs, power = compute_power_spectrum(
            truncated_e55_data, sampling_freq, fmin=1.0, fmax=80.0
        )

        # Store power spectrum summary
        peak_freq = freqs[np.argmax(power)]
        peak_power = np.max(power)
        power_summary = {
            "File": file_id,
            "Peak Frequency (Hz)": peak_freq,
            "Peak Power (µV²/Hz)": peak_power,
        }

        # Prepare power spectrum data for summary plot
        power_spectrum = {
            "File": file_id,
            "Frequency (Hz)": freqs,
            "Power (µV²/Hz)": power,
        }

        return (
            file_id,
            truncated_e55_data,
            sampling_freq,
            metrics,
            power_summary,
            power_spectrum,
        )

    except Exception as e:
        logging.error(f"Error processing {file_name}: {e}")
        return (file_id, None, None, None, None, None)


def process_eeg_files_async(
    file_paths: List[str],
    electrode: str = "E55",
    apply_notch: bool = False,
    notch_freqs: List[float] = [50.0, 60.0],
    generate_individual_plots: bool = True,
    generate_individual_csvs: bool = True,
):
    """
    Process multiple EEG .set files asynchronously using multiprocessing.

    Parameters
    ----------
    file_paths : List[str]
        List of EEG .set file paths to process.
    electrode : str, optional
        Name of the electrode to extract. Default is 'E55'.
    apply_notch : bool, optional
        Whether to apply a notch filter to remove powerline noise. Default is False.
    notch_freqs : List[float], optional
        Frequencies to remove using the notch filter. Default is [50.0, 60.0].
    generate_individual_plots : bool, optional
        Whether to generate and save individual power spectrum plots. Default is True.
    generate_individual_csvs : bool, optional
        Whether to generate and save individual power spectrum CSVs. Default is True.
    """
    # Initialize lists to collect results
    all_time_series = []
    all_file_ids = []
    noisy_metrics_list = []
    power_spectra_summary = []
    individual_power_spectra = []

    # Sanitize electrode name for file prefixes
    sanitized_electrode = sanitize_filename(electrode)

    # Ensure power spectra folder exists
    power_spectra_folder = f"{sanitized_electrode}_power_spectra_plots"
    if generate_individual_plots or generate_individual_csvs:
        os.makedirs(power_spectra_folder, exist_ok=True)

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        # Submit all file processing tasks
        futures = [
            executor.submit(
                process_single_file,
                file_path=file_path,
                electrode=electrode,
                apply_notch=apply_notch,
                notch_freqs=notch_freqs,
            )
            for file_path in file_paths
        ]

        # Iterate over completed futures with a progress bar
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing EEG files"
        ):
            result = future.result()
            (
                file_id,
                truncated_e55_data,
                sampling_freq,
                metrics,
                power_summary,
                power_spectrum,
            ) = result

            if truncated_e55_data is not None:
                # Collect time series data
                all_time_series.append(truncated_e55_data)
                all_file_ids.append(file_id)

                # Collect noisy metrics
                noisy_metrics_list.append(metrics)

                # Collect power spectra summary
                power_spectra_summary.append(power_summary)

                # Collect power spectrum data for summary plot
                individual_power_spectra.append(power_spectrum)

                # Optionally, save individual plots and CSVs
                if generate_individual_plots:
                    plot_path = os.path.join(
                        power_spectra_folder,
                        f"{sanitized_electrode}_{file_id}_power_spectrum.png",
                    )
                    plot_power_spectrum(
                        freqs=power_spectrum["Frequency (Hz)"],
                        power=power_spectrum["Power (µV²/Hz)"],
                        file_id=file_id,
                        output_plot_path=plot_path,
                    )

                if generate_individual_csvs:
                    power_csv_path = os.path.join(
                        power_spectra_folder,
                        f"{sanitized_electrode}_{file_id}_power_spectrum.csv",
                    )
                    save_power_spectrum_to_csv(
                        freqs=power_spectrum["Frequency (Hz)"],
                        power=power_spectrum["Power (µV²/Hz)"],
                        output_csv_path=power_csv_path,
                    )

    if not all_time_series:
        logging.error("No files were successfully processed.")
        return

    # Create time vector based on the first file's length
    num_samples = len(all_time_series[0])
    time_vector = np.linspace(0, 180.0, num_samples)  # 180 seconds

    # Save all time series to CSV with electrode prefix
    time_series_output = f"{sanitized_electrode}_eeg_time_series_all.csv"
    save_time_series_to_csv(
        time_vector, all_time_series, all_file_ids, time_series_output
    )

    # Save noisy metrics summary to CSV with electrode prefix
    noisy_metrics_output = f"{sanitized_electrode}_noisy_metrics_summary.csv"
    save_noisy_metrics_to_csv(noisy_metrics_list, noisy_metrics_output)

    # Save power spectra summary to CSV with electrode prefix
    power_spectra_summary_output = f"{sanitized_electrode}_power_spectra_summary.csv"
    save_power_spectra_summary_to_csv(
        power_spectra_summary, power_spectra_summary_output
    )

    # Plot and save summary power spectrum with electrode prefix
    summary_plot_path = f"{sanitized_electrode}_summary_power_spectrum.png"
    plot_summary_power_spectrum(individual_power_spectra, summary_plot_path)

    # Create and save summary jitter plots for noisy metrics with electrode prefix
    try:
        metrics_df = pd.DataFrame(noisy_metrics_list)
        metrics_df["Numerical_ID"] = metrics_df["File"].apply(extract_numerical_id)
        jitter_plot_path = f"{sanitized_electrode}_summary_jitter_plots.png"
        plot_summary_jitter(metrics_df, jitter_plot_path)
    except Exception as e:
        logging.error(f"Error creating summary jitter plots: {e}")

    logging.info("EEG folder processing completed.")


def main():
    """
    Main function to parse arguments and initiate EEG processing.
    """
    parser = argparse.ArgumentParser(
        description="Process EEG .set files asynchronously with optional notch filtering and output controls."
    )
    parser.add_argument(
        "--apply_notch",
        action="store_true",
        help="Enable notch filter to remove powerline noise.",
    )
    parser.add_argument(
        "--notch_freqs",
        nargs="+",
        type=float,
        default=[50.0, 60.0],
        help="Frequencies to remove using the notch filter (e.g., --notch_freqs 50 60). Default is [50.0, 60.0].",
    )
    parser.add_argument(
        "--skip_individual_plots",
        action="store_true",
        help="Skip generating individual power spectrum plots.",
    )
    parser.add_argument(
        "--skip_individual_csvs",
        action="store_true",
        help="Skip generating individual power spectrum CSVs.",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Enable test mode to process only a subset of files.",
    )
    parser.add_argument(
        "--max_test_files",
        type=int,
        default=3,
        help="Number of files to process in test mode. Default is 3.",
    )
    args = parser.parse_args()

    set_files = get_set_files(eeg_folder)
    if not set_files:
        logging.error("No .set files to process.")
        return

    # Handle test mode
    if args.test_mode:
        set_files = set_files[: args.max_test_files]
        logging.info(
            f"Test mode enabled: Processing only the first {args.max_test_files} files."
        )

    # Process EEG files asynchronously with the specified options
    process_eeg_files_async(
        file_paths=set_files,
        electrode="E24",
        apply_notch=args.apply_notch,
        notch_freqs=args.notch_freqs,
        generate_individual_plots=not args.skip_individual_plots,
        generate_individual_csvs=not args.skip_individual_csvs,
    )


if __name__ == "__main__":
    main()
