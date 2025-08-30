import gc
import itertools
import logging
import os
import tempfile
import time
import traceback
import warnings

import h5py
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.gridspec import GridSpec
from mne.datasets import fetch_fsaverage
from mne.filter import filter_data
from mne.source_estimate import SourceEstimate
from mne_connectivity import spectral_connectivity_time
from scipy import signal, stats
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, hilbert, savgol_filter
from tqdm import tqdm

# Optional imports with availability flags
try:
    from networkx.algorithms.community import louvain_communities, modularity

    NETWORK_ANALYSIS_AVAILABLE = True
except ImportError:
    NETWORK_ANALYSIS_AVAILABLE = False

try:
    from fooof.analysis import get_band_peak_fm

    FOOOF_AVAILABLE = True
except ImportError:
    FOOOF_AVAILABLE = False

from autoclean.io.export import save_stc_to_file


def estimate_source_function_raw(raw: mne.io.Raw, config: dict = None):
    """
    Perform source localization on continuous resting-state EEG data using an identity matrix
    for noise covariance, keeping it as raw data.
    """
    # --------------------------------------------------------------------------
    # Preprocessing for Source Localization
    # --------------------------------------------------------------------------
    matplotlib.use("Qt5Agg")

    raw.set_eeg_reference("average", projection=True)
    print("Set EEG reference to average")

    noise_cov = mne.make_ad_hoc_cov(raw.info)
    print("Using an identity matrix for noise covariance")

    # --------------------------------------------------------------------------
    # Source Localization Setup
    # --------------------------------------------------------------------------
    fs_dir = fetch_fsaverage()
    trans = "fsaverage"
    src = mne.read_source_spaces(f"{fs_dir}/bem/fsaverage-ico-5-src.fif")
    bem = mne.read_bem_solution(f"{fs_dir}/bem/fsaverage-5120-5120-5120-bem-sol.fif")

    fwd = mne.make_forward_solution(
        raw.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=10
    )
    print("Created forward solution")

    inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov)
    print("Created inverse operator with identity noise covariance")

    stc = mne.minimum_norm.apply_inverse_raw(
        raw, inv, lambda2=1.0 / 9.0, method="MNE", pick_ori="normal", verbose=True
    )

    # mne.viz.plot_alignment(
    #     raw.info,
    #     src=src,
    #     eeg=["original", "projected"],
    #     trans='fsaverage',
    #     show_axes=True,
    #     mri_fiducials=True,
    #     dig="fiducials",
    # )
    print(
        "Computed continuous source estimates using MNE with identity noise covariance"
    )

    if config is not None:
        save_stc_to_file(stc, config, stage="post_source_localization")

    matplotlib.use("Agg")
    return stc


def estimate_source_function_epochs(epochs: mne.Epochs, config: dict = None):
    """
    Perform source localization on epoched EEG data using an identity matrix
    for noise covariance.
    """
    # --------------------------------------------------------------------------
    # Preprocessing for Source Localization
    # --------------------------------------------------------------------------
    matplotlib.use("Qt5Agg")

    epochs.set_eeg_reference("average", projection=True)
    print("Set EEG reference to average")

    noise_cov = mne.make_ad_hoc_cov(epochs.info)
    print("Using an identity matrix for noise covariance")

    # --------------------------------------------------------------------------
    # Source Localization Setup
    # --------------------------------------------------------------------------
    fs_dir = fetch_fsaverage()
    trans = "fsaverage"
    src = mne.read_source_spaces(f"{fs_dir}/bem/fsaverage-ico-5-src.fif")
    bem = mne.read_bem_solution(f"{fs_dir}/bem/fsaverage-5120-5120-5120-bem-sol.fif")

    fwd = mne.make_forward_solution(
        epochs.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=10
    )
    print("Created forward solution")

    inv = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov)
    print("Created inverse operator with identity noise covariance")

    stc = mne.minimum_norm.apply_inverse_epochs(
        epochs, inv, lambda2=1.0 / 9.0, method="MNE", pick_ori="normal", verbose=True
    )

    print(
        "Computed source estimates for epochs using MNE with identity noise covariance"
    )

    if config is not None:
        # For epochs, we get a list of STCs, so we might need to handle this differently
        # or save each epoch separately
        if isinstance(stc, list):
            print(f"Generated {len(stc)} source estimates for epochs")
            # Optionally save the first few as examples
            for i, stc_epoch in enumerate(stc[: min(3, len(stc))]):
                save_stc_to_file(
                    stc_epoch, config, stage=f"post_source_localization_epoch_{i}"
                )
        else:
            save_stc_to_file(stc, config, stage="post_source_localization")

    return stc


def calculate_source_psd(
    stc,
    subjects_dir=None,
    subject="fsaverage",
    n_jobs=4,
    output_dir=None,
    subject_id=None,
):
    """
    Calculate power spectral density (PSD) from resting-state source estimates using Welch's method,
    average into anatomical ROIs using the Desikan-Killiany atlas, and save to structured files.

    Parameters
    ----------
    stc : instance of SourceEstimate
        The source time course to calculate PSD from
    subjects_dir : str | None
        Path to the freesurfer subjects directory. If None, uses the environment variable
    subject : str
        Subject name in the subjects_dir (default: 'fsaverage')
    n_jobs : int
        Number of parallel jobs to use for computation
    output_dir : str | None
        Directory to save output files. If None, saves in current directory
    subject_id : str | None
        Subject identifier for file naming

    Returns
    -------
    psd_df : DataFrame
        DataFrame containing ROI-averaged PSD values
    file_path : str
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    if subject_id is None:
        subject_id = "unknown_subject"

    print(f"Calculating PSD for {subject_id}...")

    # Define frequency parameters
    fmin = 1.0
    fmax = 45.0

    # Get data array and sampling frequency
    data = stc.data
    sfreq = stc.sfreq

    print(f"Source data shape: {data.shape}")
    print(f"Computing source PSD using Welch's method for {len(data)} vertices...")

    # Welch parameters for resting-state data
    window_length = int(4 * sfreq)  # 4-second windows
    n_overlap = window_length // 2  # 50% overlap

    # For multitaper, we'll use mne's implementation directly which handles this correctly

    # First, calculate frequencies that will be generated
    # We'll use a helper call to get the frequency axis
    freqs = np.fft.rfftfreq(window_length, 1 / sfreq)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freq_mask]

    # Initialize PSD array
    n_vertices = data.shape[0]
    psd = np.zeros((n_vertices, len(freqs)))

    # Function to process a batch of vertices
    def process_batch(vertices_idx):
        batch_psd = np.zeros((len(vertices_idx), len(freqs)))

        for i, vertex_idx in enumerate(vertices_idx):
            # Use standard Welch method for stability
            f, Pxx = signal.welch(
                data[vertex_idx],
                fs=sfreq,
                window="hann",  # Hann window is more stable than multitaper for this
                nperseg=window_length,
                noverlap=n_overlap,
                nfft=None,
                scaling="density",
            )

            # Store PSD for frequencies in our range
            batch_psd[i] = Pxx[freq_mask]

        return batch_psd

    # Process in batches to avoid memory issues with large source spaces
    batch_size = 1000
    n_batches = int(np.ceil(n_vertices / batch_size))

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_vertices)

        print(
            f"Processing batch {batch_idx + 1}/{n_batches}, vertices {start_idx}-{end_idx}"
        )

        # Get vertices for this batch
        vertices_idx = range(start_idx, end_idx)

        # Process batch
        batch_psd = process_batch(vertices_idx)

        # Store in main PSD array
        psd[start_idx:end_idx] = batch_psd

    print(f"PSD calculation complete. Shape: {psd.shape}, frequencies: {freqs.shape}")
    print(f"Frequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz")

    # Load Desikan-Killiany atlas labels
    print("Loading Desikan-Killiany atlas labels...")
    labels = mne.read_labels_from_annot(
        subject, parc="aparc", subjects_dir=subjects_dir
    )

    # Remove 'unknown' labels
    labels = [label for label in labels if "unknown" not in label.name]

    # Initialize dataframe for ROI-averaged PSDs
    roi_psds = []

    # Process each label/ROI
    print("Averaging PSD within anatomical ROIs...")
    for label in labels:
        # Get vertices in this label
        label_verts = label.get_vertices_used()

        # Find indices of these vertices in the stc
        if label.hemi == "lh":
            # Left hemisphere
            stc_idx = np.where(np.in1d(stc.vertices[0], label_verts))[0]
        else:
            # Right hemisphere
            stc_idx = np.where(np.in1d(stc.vertices[1], label_verts))[0] + len(
                stc.vertices[0]
            )

        # Skip if no vertices found
        if len(stc_idx) == 0:
            print(f"Warning: No vertices found for label {label.name}")
            continue

        # Calculate mean PSD across vertices in this ROI
        roi_psd = np.mean(psd[stc_idx, :], axis=0)

        # Add to dataframe
        for freq_idx, freq in enumerate(freqs):
            roi_psds.append(
                {
                    "subject": subject_id,
                    "roi": label.name,
                    "hemisphere": label.hemi,
                    "frequency": freq,
                    "psd": roi_psd[freq_idx],
                }
            )

    # Create dataframe
    psd_df = pd.DataFrame(roi_psds)

    # Save to file
    file_path = os.path.join(output_dir, f"{subject_id}_roi_psd.parquet")
    psd_df.to_parquet(file_path)
    print(f"Saved ROI-averaged PSD to {file_path}")

    # Also save a summary CSV with band averages
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
    }

    band_psds = []

    # For each ROI, calculate band power
    for roi in psd_df["roi"].unique():
        roi_data = psd_df[psd_df["roi"] == roi]
        for band_name, (band_min, band_max) in bands.items():
            # Get data in this frequency band
            band_data = roi_data[
                (roi_data["frequency"] >= band_min) & (roi_data["frequency"] < band_max)
            ]

            # Calculate mean band power
            band_power = band_data["psd"].mean()

            # Add to band_psds
            band_psds.append(
                {
                    "subject": subject_id,
                    "roi": roi,
                    "hemisphere": roi_data["hemisphere"].iloc[0],
                    "band": band_name,
                    "band_start_hz": band_min,
                    "band_end_hz": band_max,
                    "power": band_power,
                }
            )

    # Create dataframe for band averages
    band_df = pd.DataFrame(band_psds)

    # Save to CSV
    csv_path = os.path.join(output_dir, f"{subject_id}_roi_bands.csv")
    band_df.to_csv(csv_path, index=False)
    print(f"Saved frequency band summary to {csv_path}")

    return psd_df, file_path


def calculate_source_psd_list(
    stc_list,
    subjects_dir=None,
    subject="fsaverage",
    n_jobs=4,
    output_dir=None,
    subject_id=None,
    generate_plots=True,
    segment_duration=80,
):
    """
    Optimized function to calculate power spectral density (PSD) from source estimates.
    Processes a limited time segment to improve performance while maintaining spectral accuracy.

    Parameters
    ----------
    stc_list : instance of SourceEstimate or list of SourceEstimates
        The source time course(s) to calculate PSD from.
    subjects_dir : str | None
        Path to the freesurfer subjects directory. If None, uses the environment variable
    subject : str
        Subject name in the subjects_dir (default: 'fsaverage')
    n_jobs : int
        Number of parallel jobs to use for computation
    output_dir : str | None
        Directory to save output files. If None, saves in current directory
    subject_id : str | None
        Subject identifier for file naming
    generate_plots : bool
        Whether to generate diagnostic PSD plots (default: True)
    segment_duration : float or None
        Duration in seconds to process. If None, processes the entire data.
        Default is 80 seconds for optimal balance of accuracy and performance.

    Returns
    -------
    psd_df : DataFrame
        DataFrame containing ROI-averaged PSD values
    file_path : str
        Path to the saved file
    """

    import mne
    import pandas as pd

    # Start timing
    start_time = time.time()

    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    if generate_plots:
        os.makedirs(os.path.join(output_dir, "psd_plots"), exist_ok=True)

    if subject_id is None:
        subject_id = "unknown_subject"

    # Convert single stc to list for consistency
    if not isinstance(stc_list, list):
        stc_list = [stc_list]

    # Determine the total available signal duration
    epoch_duration = stc_list[0].times[-1] - stc_list[0].times[0]
    total_duration = epoch_duration * len(stc_list)
    sfreq = stc_list[0].sfreq

    print(
        f"Total available signal: {total_duration:.1f} seconds ({len(stc_list)} epochs of {epoch_duration:.1f}s each)"
    )

    # If segment_duration is specified and less than total, select a subset of epochs
    selected_stcs = stc_list
    if segment_duration is not None and segment_duration < total_duration:
        # Calculate how many epochs we need
        n_epochs_needed = int(np.ceil(segment_duration / epoch_duration))

        # Make sure we don't exceed the available epochs
        n_epochs_needed = min(n_epochs_needed, len(stc_list))

        # Select epochs from the middle of the recording for better stationarity
        start_idx = (len(stc_list) - n_epochs_needed) // 2
        selected_stcs = stc_list[start_idx : start_idx + n_epochs_needed]

        actual_duration = len(selected_stcs) * epoch_duration
        print(
            f"Using {len(selected_stcs)} epochs ({actual_duration:.1f}s) from the middle of the recording"
        )
    else:
        print(f"Using all {len(stc_list)} epochs ({total_duration:.1f}s)")

    # Define frequency parameters
    fmin = 0.5
    fmax = 45.0

    # Get data shape and sampling frequency
    n_vertices = selected_stcs[0].data.shape[0]

    # Determine optimal window length - adaptive based on available data
    available_duration = len(selected_stcs) * epoch_duration

    # For spectral analysis, prefer 4-second windows for good frequency resolution
    # but ensure we have at least 8 windows for reliable averaging
    window_length = int(min(4 * sfreq, available_duration * sfreq / 8))
    n_overlap = window_length // 2  # 50% overlap

    print(f"Using {window_length / sfreq:.2f}s windows with 50% overlap")

    # Calculate frequencies that will be generated
    freqs = np.fft.rfftfreq(window_length, 1 / sfreq)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freq_mask]

    # Define bands for direct computation from PSD
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "lowalpha": (8, 10),
        "highalpha": (10, 13),
        "lowbeta": (13, 20),
        "highbeta": (20, 30),
        "gamma": (30, 45),
    }

    # Calculate band indices for faster computation
    band_indices = {}
    for band_name, (band_min, band_max) in bands.items():
        band_indices[band_name] = np.where((freqs >= band_min) & (freqs < band_max))[0]

    # Sample a few vertices to check variance
    print("Sampling vertex variance to set threshold...")
    sample_size = min(1000, n_vertices)
    sample_indices = np.linspace(0, n_vertices - 1, sample_size, dtype=int)

    vertex_variance = np.zeros(sample_size)
    for i, vertex_idx in enumerate(sample_indices):
        vertex_data = np.concatenate([stc.data[vertex_idx] for stc in selected_stcs])
        vertex_variance[i] = np.var(vertex_data)

    # Set threshold at 10th percentile of non-zero variances
    non_zero_vars = vertex_variance[vertex_variance > 0]
    if len(non_zero_vars) > 0:
        var_threshold = np.percentile(non_zero_vars, 10)
    else:
        var_threshold = 1e-12

    print(f"Variance threshold set to {var_threshold:.3e}")

    # Define batch processing function
    def process_vertex_batch(batch_indices):
        n_batch_vertices = len(batch_indices)
        batch_psd = np.zeros((n_batch_vertices, len(freqs)))

        # For visualization
        viz_vertices = []
        viz_psds = []

        for i, vertex_idx in enumerate(batch_indices):
            # Quick check for very large batches
            if i % 1000 == 0 and i > 0:
                print(f"  Processed {i}/{n_batch_vertices} vertices in current batch")

            try:
                # Concatenate data from all selected epochs for this vertex
                vertex_data = np.concatenate(
                    [stc.data[vertex_idx] for stc in selected_stcs]
                )

                # Skip if variance is below threshold
                if np.var(vertex_data) < var_threshold:
                    continue

                # Detrend and apply window in-place to save memory
                vertex_data = signal.detrend(vertex_data)
                vertex_data *= np.hanning(len(vertex_data))

                # Calculate PSD using Welch's method
                f, Pxx = signal.welch(
                    vertex_data,
                    fs=sfreq,
                    window="hann",
                    nperseg=window_length,
                    noverlap=n_overlap,
                    nfft=None,
                    scaling="density",
                    detrend=False,  # Already detrended
                )

                # Store PSD for frequencies in our range
                batch_psd[i] = Pxx[freq_mask]

                # Store data for visualization (only for a few vertices)
                if generate_plots and vertex_idx % 5000 == 0:
                    viz_vertices.append(vertex_idx)
                    viz_psds.append((f, Pxx))

            except Exception as e:
                print(f"Error processing vertex {vertex_idx}: {str(e)}")

        return batch_psd, viz_vertices, viz_psds

    # Determine optimal batch size based on available memory and number of jobs
    batch_size = min(5000, max(1000, n_vertices // (n_jobs * 2)))
    n_batches = int(np.ceil(n_vertices / batch_size))

    print(
        f"Processing {n_vertices} vertices in {n_batches} batches of size {batch_size}..."
    )

    # Create batches
    batch_indices = [
        range(i * batch_size, min((i + 1) * batch_size, n_vertices))
        for i in range(n_batches)
    ]

    # Process batches in parallel
    batch_start = time.time()
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_vertex_batch)(indices) for indices in batch_indices
    )
    print(f"Batch processing completed in {time.time() - batch_start:.1f} seconds")

    # Initialize PSD array
    psd = np.zeros((n_vertices, len(freqs)))

    # Collect visualization data
    all_viz_vertices = []
    all_viz_psds = []

    # Combine results
    for batch_idx, (batch_psd, viz_vertices, viz_psds) in enumerate(results):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_vertices)
        psd[start_idx:end_idx] = batch_psd

        all_viz_vertices.extend(viz_vertices)
        all_viz_psds.extend(viz_psds)

    print(f"PSD calculation complete in {time.time() - start_time:.1f} seconds")

    # Rest of the function remains the same...
    # ... [ROI processing, plotting, saving]

    # Generate visualization plots in parallel
    if generate_plots and all_viz_vertices:
        print(f"Generating {len(all_viz_vertices)} PSD plots...")

        def create_psd_plot(vertex_idx, f_pxx):
            f, Pxx = f_pxx
            plt.figure(figsize=(10, 6))
            plt.semilogy(f, Pxx, "b")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("PSD (µV²/Hz)")
            plt.title(f"Vertex {vertex_idx} PSD")
            plt.xlim([0, 50])
            plt.grid(True)
            plt.savefig(
                os.path.join(
                    output_dir, "psd_plots", f"{subject_id}_vertex{vertex_idx}_psd.png"
                )
            )
            plt.close()

        # Generate plots in parallel
        Parallel(n_jobs=min(n_jobs, len(all_viz_vertices)))(
            delayed(create_psd_plot)(vertex_idx, f_pxx)
            for vertex_idx, f_pxx in zip(all_viz_vertices, all_viz_psds)
        )

    # Load Desikan-Killiany atlas labels
    print("Loading Desikan-Killiany atlas labels...")
    labels = mne.read_labels_from_annot(
        subject, parc="aparc", subjects_dir=subjects_dir
    )
    labels = [label for label in labels if "unknown" not in label.name]

    # Function to process a single ROI/label
    def process_label(label_idx):
        label = labels[label_idx]

        # Get vertices in this label
        label_verts = label.get_vertices_used()

        # Find indices of these vertices in the stc
        if label.hemi == "lh":
            # Left hemisphere
            stc_idx = np.where(np.in1d(selected_stcs[0].vertices[0], label_verts))[0]
        else:
            # Right hemisphere
            stc_idx = np.where(np.in1d(selected_stcs[0].vertices[1], label_verts))[
                0
            ] + len(selected_stcs[0].vertices[0])

        # Skip if no vertices found
        if len(stc_idx) == 0:
            print(f"Warning: No vertices found for label {label.name}")
            return None

        # Calculate mean PSD across vertices in this ROI
        roi_psd = np.mean(psd[stc_idx, :], axis=0)

        # Directly calculate band powers
        band_powers = {}
        for band_name, indices in band_indices.items():
            if len(indices) > 0:
                band_powers[band_name] = np.mean(roi_psd[indices])
            else:
                band_powers[band_name] = 0

        # Create row for PSD dataframe
        roi_psd_data = []
        for freq_idx, freq in enumerate(freqs):
            roi_psd_data.append(
                {
                    "subject": subject_id,
                    "roi": label.name,
                    "hemisphere": label.hemi,
                    "frequency": freq,
                    "psd": roi_psd[freq_idx],
                }
            )

        # Create rows for band power dataframe
        band_data = []
        for band_name, power in band_powers.items():
            band_min, band_max = bands[band_name]
            band_data.append(
                {
                    "subject": subject_id,
                    "roi": label.name,
                    "hemisphere": label.hemi,
                    "band": band_name,
                    "band_start_hz": band_min,
                    "band_end_hz": band_max,
                    "power": power,
                }
            )

        return roi_psd_data, band_data

    # Process all labels in parallel
    print(f"Processing {len(labels)} ROIs in parallel...")
    roi_results = Parallel(n_jobs=n_jobs)(
        delayed(process_label)(i) for i in range(len(labels))
    )

    # Combine results
    roi_psds = []
    band_psds = []

    for result in roi_results:
        if result is not None:
            roi_psd_data, band_data = result
            roi_psds.extend(roi_psd_data)
            band_psds.extend(band_data)

    # Create dataframes
    psd_df = pd.DataFrame(roi_psds)
    band_df = pd.DataFrame(band_psds)

    # Save to files
    file_path = os.path.join(output_dir, f"{subject_id}_roi_psd.parquet")
    psd_df.to_parquet(file_path)

    csv_path = os.path.join(output_dir, f"{subject_id}_roi_bands.csv")
    band_df.to_csv(csv_path, index=False)

    print(f"Saved ROI-averaged PSD to {file_path}")
    print(f"Saved frequency band summary to {csv_path}")

    # Create visualization of average band power per hemisphere
    if generate_plots:
        print("Creating summary visualizations...")

        # Group by band and hemisphere, then calculate mean power
        band_summary = (
            band_df.groupby(["band", "hemisphere"])["power"].mean().reset_index()
        )

        # Create a pivot table for easier plotting
        pivot_df = band_summary.pivot(
            index="band", columns="hemisphere", values="power"
        )

        # Sort bands in frequency order
        band_order = [
            "delta",
            "theta",
            "lowalpha",
            "highalpha",
            "alpha",
            "lowbeta",
            "highbeta",
            "gamma",
        ]
        pivot_df = pivot_df.reindex(band_order)

        # Create bar plot
        plt.figure(figsize=(12, 8))
        pivot_df.plot(kind="bar", ax=plt.gca())
        plt.title(f"Mean Band Power by Hemisphere - {subject_id}")
        plt.ylabel("Power (µV²/Hz)")
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{subject_id}_hemisphere_bands.png"))

        # Also create log scale version for better visualization of 1/f pattern
        plt.figure(figsize=(12, 8))
        np.log10(pivot_df).plot(kind="bar", ax=plt.gca())
        plt.title(f"Log10 Mean Band Power by Hemisphere - {subject_id}")
        plt.ylabel("Log10 Power")
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{subject_id}_hemisphere_bands_log.png"))

    total_time = time.time() - start_time
    print(
        f"Total processing time: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)"
    )

    return psd_df, file_path


def visualize_psd_results(psd_df, output_dir=None, subject_id=None):
    """
    Create visualization plots for PSD data to confirm spectral analysis results.

    Parameters
    ----------
    psd_df : DataFrame
        DataFrame containing ROI-averaged PSD values, with columns:
        subject, roi, hemisphere, frequency, psd
    output_dir : str or None
        Directory to save output files. If None, current directory is used.
    subject_id : str or None
        Subject identifier for plot titles and filenames.
        If None, extracted from the data.

    Returns
    -------
    fig : matplotlib Figure
        Figure containing the visualization
    """

    import pandas as pd

    # Set up plotting style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("notebook", font_scale=1.1)

    if output_dir is None:
        output_dir = os.getcwd()

    if subject_id is None:
        subject_id = psd_df["subject"].iloc[0]

    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig)

    # Define frequency bands
    bands = {
        "delta": (1, 4, "#1f77b4"),
        "theta": (4, 8, "#ff7f0e"),
        "alpha": (8, 13, "#2ca02c"),
        "beta": (13, 30, "#d62728"),
        "gamma": (30, 45, "#9467bd"),
    }

    # 1. Plot PSD for selected regions
    ax1 = fig.add_subplot(gs[0, 0])

    # Select a subset of interesting regions to plot
    regions_to_plot = [
        "precentral-lh",
        "postcentral-lh",
        "superiorparietal-lh",
        "lateraloccipital-lh",
        "superiorfrontal-lh",
    ]

    # If some regions aren't in the data, use what's available
    available_rois = psd_df["roi"].unique()
    regions_to_plot = [r for r in regions_to_plot if r in available_rois]

    # If none of the selected regions are available, use the first 5 available
    if not regions_to_plot:
        regions_to_plot = list(available_rois)[:5]

    # Plot each region with linear scale
    for roi in regions_to_plot:
        roi_data = psd_df[psd_df["roi"] == roi]
        ax1.plot(
            roi_data["frequency"],
            roi_data["psd"],
            linewidth=2,
            alpha=0.8,
            label=roi.split("-")[0],
        )

    # Add frequency band backgrounds
    y_min, y_max = ax1.get_ylim()
    for band_name, (fmin, fmax, color) in bands.items():
        ax1.axvspan(fmin, fmax, color=color, alpha=0.1)

    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Power Spectral Density")
    ax1.set_title("PSD for Selected Regions (Left Hemisphere)")
    ax1.legend(loc="upper right")
    ax1.set_xlim(1, 45)
    ax1.grid(True, which="both", ls="--", alpha=0.3)

    # 2. Plot left vs right hemisphere comparison
    ax2 = fig.add_subplot(gs[0, 1])

    # Average across all regions in each hemisphere
    for hemi, color, label in zip(
        ["lh", "rh"], ["#1f77b4", "#d62728"], ["Left Hemisphere", "Right Hemisphere"]
    ):
        hemi_data = (
            psd_df[psd_df["hemisphere"] == hemi]
            .groupby("frequency")["psd"]
            .mean()
            .reset_index()
        )
        ax2.plot(
            hemi_data["frequency"],
            hemi_data["psd"],
            linewidth=2.5,
            color=color,
            label=label,
        )

    # Add frequency band backgrounds
    for band_name, (fmin, fmax, color) in bands.items():
        ax2.axvspan(fmin, fmax, color=color, alpha=0.1)

    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power Spectral Density")
    ax2.set_title("Left vs Right Hemisphere Average")
    ax2.legend(loc="upper right")
    ax2.set_xlim(1, 45)
    ax2.grid(True, which="both", ls="--", alpha=0.3)

    # 3. Frequency band power comparison across regions
    ax3 = fig.add_subplot(gs[1, 0])

    # Calculate average power in each frequency band for each ROI
    band_powers = []
    for roi in psd_df["roi"].unique():
        roi_base = roi.split("-")[0]
        hemi = roi.split("-")[1]

        roi_data = psd_df[psd_df["roi"] == roi]

        for band_name, (fmin, fmax, _) in bands.items():
            band_data = roi_data[
                (roi_data["frequency"] >= fmin) & (roi_data["frequency"] <= fmax)
            ]
            band_power = band_data["psd"].mean()

            band_powers.append(
                {
                    "ROI": roi_base,
                    "Hemisphere": "Left" if hemi == "lh" else "Right",
                    "Band": band_name.capitalize(),
                    "Power": band_power,
                }
            )

    band_power_df = pd.DataFrame(band_powers)

    # Select a subset of regions for clarity
    regions_for_bands = [
        "precentral",
        "postcentral",
        "superiorfrontal",
        "lateraloccipital",
    ]
    available_base_rois = band_power_df["ROI"].unique()
    regions_for_bands = [r for r in regions_for_bands if r in available_base_rois]

    if not regions_for_bands:
        regions_for_bands = list(available_base_rois)[:4]

    plot_data = band_power_df[
        band_power_df["ROI"].isin(regions_for_bands)
        & (band_power_df["Hemisphere"] == "Left")
    ]

    # Normalize powers within each region for better visualization
    for roi in plot_data["ROI"].unique():
        roi_mask = plot_data["ROI"] == roi
        max_power = plot_data.loc[roi_mask, "Power"].max()
        plot_data.loc[roi_mask, "Normalized Power"] = (
            plot_data.loc[roi_mask, "Power"] / max_power
        )

    # Plot band powers
    sns.barplot(
        x="ROI",
        y="Normalized Power",
        hue="Band",
        data=plot_data,
        ax=ax3,
        palette=[bands[b.lower()][2] for b in plot_data["Band"].unique()],
    )

    ax3.set_xlabel("Brain Region")
    ax3.set_ylabel("Normalized Band Power")
    ax3.set_title("Frequency Band Distribution Across Regions")
    ax3.legend(title="Frequency Band")

    # 4. Alpha/Beta ratio across regions
    ax4 = fig.add_subplot(gs[1, 1])

    # Calculate alpha/beta ratio for each ROI
    alpha_beta_data = []

    for roi_base in band_power_df["ROI"].unique():
        for hemi in ["Left", "Right"]:
            alpha_power = band_power_df[
                (band_power_df["ROI"] == roi_base)
                & (band_power_df["Hemisphere"] == hemi)
                & (band_power_df["Band"] == "Alpha")
            ]["Power"].values

            beta_power = band_power_df[
                (band_power_df["ROI"] == roi_base)
                & (band_power_df["Hemisphere"] == hemi)
                & (band_power_df["Band"] == "Beta")
            ]["Power"].values

            if len(alpha_power) > 0 and len(beta_power) > 0 and beta_power[0] > 0:
                ratio = alpha_power[0] / beta_power[0]

                alpha_beta_data.append(
                    {"ROI": roi_base, "Hemisphere": hemi, "Alpha/Beta Ratio": ratio}
                )

    ratio_df = pd.DataFrame(alpha_beta_data)

    # Select regions for visualization
    if len(regions_for_bands) > 0:
        ratio_plot = ratio_df[ratio_df["ROI"].isin(regions_for_bands)]
    else:
        # If no specific regions, use top 4 by ratio
        ratio_plot = ratio_df.sort_values("Alpha/Beta Ratio", ascending=False).head(8)

    sns.barplot(
        x="ROI",
        y="Alpha/Beta Ratio",
        hue="Hemisphere",
        data=ratio_plot,
        ax=ax4,
        palette=["#1f77b4", "#d62728"],
    )

    ax4.set_xlabel("Brain Region")
    ax4.set_ylabel("Alpha/Beta Power Ratio")
    ax4.set_title("Alpha/Beta Ratio by Region")
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha="right")
    ax4.legend(title="Hemisphere")

    # Final adjustments
    plt.suptitle(f"Power Spectral Density Analysis: {subject_id}", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle

    # Save figure
    output_path = os.path.join(output_dir, f"{subject_id}_psd_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved PSD visualization to {output_path}")

    return fig


def calculate_source_connectivity(
    stc,
    labels=None,
    subjects_dir=None,
    subject="fsaverage",
    n_jobs=4,
    output_dir=None,
    subject_id=None,
    sfreq=None,
    epoch_length=4.0,
    n_epochs=40,
):
    """
    Calculate connectivity metrics between brain regions from source-localized data.

    Parameters
    ----------
    stc : instance of SourceEstimate
        The source time course to calculate connectivity from
    labels : list of Labels | None
        List of ROI labels to use. If None, will load Desikan-Killiany atlas
    subjects_dir : str | None
        Path to the freesurfer subjects directory
    subject : str
        Subject name in the subjects_dir (default: 'fsaverage')
    n_jobs : int
        Number of parallel jobs to use for computation
    output_dir : str | None
        Directory to save output files. If None, saves in current directory
    subject_id : str | None
        Subject identifier for file naming
    sfreq : float | None
        Sampling frequency. If None, will use stc.sfreq
    epoch_length : float
        Length of epochs in seconds for connectivity calculation
    n_epochs : int
        Number of epochs to use for connectivity calculation

    Returns
    -------
    conn_df : DataFrame
        DataFrame containing connectivity values
    summary_path : str
        Path to the saved summary file
    """
    import traceback

    import pandas as pd

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("connectivity")

    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "connectivity"), exist_ok=True)

    # Set up a log file
    log_file = os.path.join(output_dir, f"{subject_id}_connectivity_log.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    if subject_id is None:
        subject_id = "unknown_subject"
    if sfreq is None:
        sfreq = stc.sfreq

    logger.info(
        f"Calculating connectivity for {subject_id} with {n_epochs} {epoch_length}-second epochs (sfreq={sfreq} Hz)..."
    )

    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
    }

    # Updated connectivity methods list - use only methods supported by mne_connectivity
    # According to the logs, 'imcoh' and 'dpli' are causing KeyErrors, so removing them
    conn_methods = ["wpli", "plv", "coh", "pli"]

    # For AEC we'll need to handle it separately since it's not part of spectral_connectivity_time
    include_aec = True

    if labels is None:
        logger.info("Loading Desikan-Killiany atlas labels...")
        labels = mne.read_labels_from_annot(
            subject, parc="aparc", subjects_dir=subjects_dir
        )
        labels = [label for label in labels if "unknown" not in label.name]

    selected_rois = [
        "precentral-lh",
        "precentral-rh",
        "postcentral-lh",
        "postcentral-rh",
        "paracentral-lh",
        "paracentral-rh",
        "caudalmiddlefrontal-lh",
        "caudalmiddlefrontal-rh",
    ]
    label_names = [label.name for label in labels]
    selected_labels = [
        labels[label_names.index(roi)] for roi in selected_rois if roi in label_names
    ]
    if not selected_labels:
        logger.warning("No selected ROIs found, using all available labels")
        selected_labels = labels
        selected_rois = label_names
    logger.info(f"Using {len(selected_labels)} selected ROIs: {selected_rois}")

    roi_pairs = list(itertools.combinations(range(len(selected_rois)), 2))

    logger.info("Extracting ROI time courses...")
    roi_time_courses = [
        stc.extract_label_time_course(label, src=None, mode="mean", verbose=False)[0]
        for label in selected_labels
    ]
    roi_data = np.array(roi_time_courses)
    logger.info(f"ROI data shape: {roi_data.shape}")

    n_times = roi_data.shape[1]
    samples_per_epoch = int(epoch_length * sfreq)
    max_epochs = n_times // samples_per_epoch
    if max_epochs < n_epochs:
        logger.warning(
            f"Requested {n_epochs} epochs, but only {max_epochs} possible. Using {max_epochs}."
        )
        n_epochs = max_epochs

    epoch_starts = (
        np.random.choice(max_epochs, size=n_epochs, replace=False) * samples_per_epoch
    )
    epoched_data = np.stack(
        [roi_data[:, start : start + samples_per_epoch] for start in epoch_starts],
        axis=0,
    )
    logger.info(f"Epoched data shape: {epoched_data.shape}")

    connectivity_data = []
    logger.info("Calculating connectivity metrics...")

    # Function to calculate AEC
    def calculate_aec(data, band_range, sfreq):
        import numpy as np
        from mne.filter import filter_data
        from scipy.signal import hilbert

        # Filter the data for the specific frequency band
        filtered_data = filter_data(
            data, sfreq=sfreq, l_freq=band_range[0], h_freq=band_range[1], verbose=False
        )

        # Get the amplitude envelope using Hilbert transform
        analytic_signal = hilbert(filtered_data, axis=-1)
        amplitude_envelope = np.abs(analytic_signal)

        # Compute correlation between envelopes
        n_signals = amplitude_envelope.shape[0]
        aec_matrix = np.zeros((n_signals, n_signals))

        for i in range(n_signals):
            for j in range(n_signals):
                if i != j:
                    corr = np.corrcoef(amplitude_envelope[i], amplitude_envelope[j])[
                        0, 1
                    ]
                    aec_matrix[i, j] = corr

        return aec_matrix

    # Calculate spectral connectivity methods
    for method in conn_methods:
        for band_name, band_range in bands.items():
            logger.info(f"Computing {method} connectivity in {band_name} band...")
            try:
                # Log detailed parameters for troubleshooting
                logger.info(
                    f"  Parameters: freqs={np.arange(band_range[0], band_range[1] + 1)}, "
                    f"sfreq={sfreq}, n_jobs={n_jobs}, n_cycles=2"
                )

                con = spectral_connectivity_time(
                    epoched_data,
                    freqs=np.arange(band_range[0], band_range[1] + 1),
                    method=method,
                    sfreq=sfreq,
                    mode="multitaper",
                    faverage=True,
                    average=True,
                    n_jobs=n_jobs,
                    verbose=False,
                    n_cycles=2,
                )
                con_matrix = con.get_data(output="dense").squeeze()
                if con_matrix.shape != (len(selected_rois), len(selected_rois)):
                    error_msg = f"Unexpected con_matrix shape: {con_matrix.shape}, expected {(len(selected_rois), len(selected_rois))}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                # Debug: Print con_matrix to verify
                logger.info(
                    f"{method} {band_name} con_matrix shape: {con_matrix.shape}"
                )
                logger.debug(
                    f"Matrix values range: min={np.min(con_matrix)}, max={np.max(con_matrix)}"
                )

                con_df = pd.DataFrame(
                    con_matrix, columns=selected_rois, index=selected_rois
                )
                matrix_filename = os.path.join(
                    output_dir, f"{subject_id}_{method}_{band_name}_matrix.csv"
                )
                con_df.to_csv(matrix_filename)
                logger.info(f"Saved connectivity matrix to {matrix_filename}")

                for i, (roi1_idx, roi2_idx) in enumerate(roi_pairs):
                    # Use lower triangle by swapping indices
                    value = con_matrix[
                        roi2_idx, roi1_idx
                    ]  # Changed from [roi1_idx, roi2_idx]
                    connectivity_data.append(
                        {
                            "subject": subject_id,
                            "method": method,
                            "band": band_name,
                            "roi1": selected_rois[roi1_idx],
                            "roi2": selected_rois[roi2_idx],
                            "connectivity": value,
                        }
                    )
            except Exception as e:
                error_msg = f"Error computing {method} in {band_name} band: {str(e)}"
                logger.error(error_msg)
                logger.error(f"Traceback: {traceback.format_exc()}")
                logger.error(
                    f"Data shape: {epoched_data.shape}, freqs: {np.arange(band_range[0], band_range[1] + 1)}"
                )
                continue

    # Calculate AEC separately
    if include_aec:
        method = "aec"
        for band_name, band_range in bands.items():
            logger.info(f"Computing {method} connectivity in {band_name} band...")
            try:
                # For each epoch, calculate AEC, then average
                aec_matrices = []
                for epoch_idx in range(epoched_data.shape[0]):
                    epoch_data = epoched_data[epoch_idx]
                    aec_matrix = calculate_aec(epoch_data, band_range, sfreq)
                    aec_matrices.append(aec_matrix)

                # Average across epochs
                con_matrix = np.mean(aec_matrices, axis=0)

                # Debug: Print con_matrix to verify
                logger.info(
                    f"{method} {band_name} con_matrix shape: {con_matrix.shape}"
                )
                logger.debug(
                    f"Matrix values range: min={np.min(con_matrix)}, max={np.max(con_matrix)}"
                )

                con_df = pd.DataFrame(
                    con_matrix, columns=selected_rois, index=selected_rois
                )
                matrix_filename = os.path.join(
                    output_dir, f"{subject_id}_{method}_{band_name}_matrix.csv"
                )
                con_df.to_csv(matrix_filename)
                logger.info(f"Saved connectivity matrix to {matrix_filename}")

                for i, (roi1_idx, roi2_idx) in enumerate(roi_pairs):
                    value = con_matrix[roi2_idx, roi1_idx]
                    connectivity_data.append(
                        {
                            "subject": subject_id,
                            "method": method,
                            "band": band_name,
                            "roi1": selected_rois[roi1_idx],
                            "roi2": selected_rois[roi2_idx],
                            "connectivity": value,
                        }
                    )
            except Exception as e:
                error_msg = f"Error computing {method} in {band_name} band: {str(e)}"
                logger.error(error_msg)
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue

    # Debug: Print sample connectivity_data
    if connectivity_data:
        logger.info(f"Sample connectivity_data entries: {connectivity_data[:5]}")
        conn_df = pd.DataFrame(connectivity_data)
        summary_path = os.path.join(
            output_dir, f"{subject_id}_connectivity_summary.csv"
        )
        conn_df.to_csv(summary_path, index=False)
        logger.info(f"Saved connectivity summary to {summary_path}")
    else:
        logger.warning("No connectivity data was generated")
        conn_df = pd.DataFrame()
        summary_path = None

    # Calculate graph metrics for each connectivity method and band
    logger.info("Calculating graph metrics...")

    from networkx.algorithms.community import louvain_communities, modularity

    graph_metrics_data = []
    for method in list(conn_methods) + (["aec"] if include_aec else []):
        for band_name in bands.keys():
            subset_df = conn_df[
                (conn_df["method"] == method) & (conn_df["band"] == band_name)
            ]
            if subset_df.empty:
                logger.warning(
                    f"No data for {method} in {band_name} band. Skipping graph metrics."
                )
                continue

            if not NETWORK_ANALYSIS_AVAILABLE:
                logger.warning(
                    "Network analysis libraries not available. Skipping graph metrics."
                )
                continue

            G = nx.Graph()
            for _, row in subset_df.iterrows():
                G.add_edge(row["roi1"], row["roi2"], weight=row["connectivity"])

            adj_matrix = nx.to_numpy_array(G, nodelist=selected_rois, weight="weight")

            # Check for NaNs or negative values in the adjacency matrix
            if np.isnan(adj_matrix).any():
                logger.warning(
                    f"NaN values in {method} {band_name} adjacency matrix. Skipping graph metrics."
                )
                continue

            # For some metrics like clustering coefficient, we need positive weights
            if (adj_matrix < 0).any():
                logger.warning(
                    f"Negative values in {method} {band_name} matrix. Using absolute values for graph metrics."
                )
                adj_matrix = np.abs(adj_matrix)

            try:
                clustering = np.mean(clustering_coef_wu(adj_matrix))
                global_efficiency = efficiency_wei(adj_matrix)
                char_path_length, _, _, _, _ = charpath(adj_matrix)
                communities = louvain_communities(G, resolution=1.0)
                modularity_score = modularity(G, communities, weight="weight")
                strength = np.mean(np.sum(adj_matrix, axis=1))

                # Additional graph metrics
                # Assortativity measures how similar connected nodes are
                assortativity = nx.degree_assortativity_coefficient(G, weight="weight")

                # Small-worldness
                # (requires computing random networks for comparison - simplified version)
                small_worldness = (
                    clustering * char_path_length if char_path_length > 0 else 0
                )

                graph_metrics_data.append(
                    {
                        "subject": subject_id,
                        "method": method,
                        "band": band_name,
                        "clustering": clustering,
                        "global_efficiency": global_efficiency,
                        "char_path_length": char_path_length,
                        "modularity": modularity_score,
                        "strength": strength,
                        "assortativity": assortativity,
                        "small_worldness": small_worldness,
                    }
                )
                logger.info(
                    f"Calculated graph metrics for {method} in {band_name} band"
                )
            except Exception as e:
                error_msg = f"Error calculating graph metrics for {method} in {band_name} band: {str(e)}"
                logger.error(error_msg)
                logger.error(f"Traceback: {traceback.format_exc()}")

    if graph_metrics_data:
        graph_metrics_df = pd.DataFrame(graph_metrics_data)
        metrics_path = os.path.join(output_dir, f"{subject_id}_graph_metrics.csv")
        graph_metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Saved graph metrics to {metrics_path}")
    else:
        logger.warning("No graph metrics were calculated")
        graph_metrics_df = pd.DataFrame()

    logger.info(f"Connectivity analysis complete for {subject_id}")
    logger.info(f"Log file saved to: {log_file}")

    return conn_df, summary_path


def test_connectivity_function_list():
    """
    Test the calculate_source_connectivity function with simulated data.
    This function creates a synthetic source time course, simulates connectivity,
    and runs the analysis pipeline.

    Returns:
        bool: True if test passes, False otherwise
    """

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("Running connectivity function test with simulated data...")

    # Create temporary directory for outputs
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory for test outputs: {temp_dir}")

    # Simulation parameters
    n_sources = 8  # Number of sources/ROIs
    n_times = 2000  # Number of time points per epoch
    sfreq = 250.0  # Sample frequency in Hz
    n_epochs = 5  # Number of epochs to simulate

    # Simulated data parameters
    signal_freq = 10  # 10 Hz oscillation (alpha band)
    noise_level = 0.1
    coupling_strength = 0.6  # Strength of coupling between regions

    # Create time vector
    times = np.arange(n_times) / sfreq

    # Create coupled oscillations for our simulated sources
    source_data = np.zeros((n_sources, n_times))

    # Create a list of SourceEstimates (simulating epochs)
    stc_list = []

    for epoch in range(n_epochs):
        # Base oscillation (seed) with small random variation per epoch
        epoch_freq = signal_freq + 0.2 * np.random.randn()
        seed_oscillation = np.sin(2 * np.pi * epoch_freq * times)

        # Create coupled sources with different phase shifts and noise
        for i in range(n_sources):
            # Add phase shift based on source index (more shift for distant sources)
            phase_shift = i * np.pi / (2 * n_sources)
            source_data[i, :] = seed_oscillation * coupling_strength * np.exp(-i * 0.2)

            # Add phase shift
            source_data[i, :] = np.sin(2 * np.pi * epoch_freq * times + phase_shift)

            # Add noise
            source_data[i, :] += noise_level * np.random.randn(n_times)

        # Create vertices arrays for left and right hemispheres
        vertices = [
            np.array([i for i in range(n_sources) if i % 2 == 0]),  # Left hemisphere
            np.array([i for i in range(n_sources) if i % 2 == 1]),  # Right hemisphere
        ]

        # Create a SourceEstimate for this epoch
        stc = SourceEstimate(
            source_data.copy(), vertices, tmin=0, tstep=1 / sfreq, subject="simulated"
        )
        stc_list.append(stc)

    # Create simulated labels
    labels = []
    label_names = [f"simulated_roi_{i}" for i in range(n_sources)]

    # Create labels compatible with different MNE versions
    try:
        # Try newer MNE API first
        for i, name in enumerate(label_names):
            vertices = np.array([i])
            pos = np.random.rand(1, 3)
            values = np.ones(1)

            label = mne.Label(
                vertices=vertices,
                pos=pos,
                values=values,
                hemi="lh" if i % 2 == 0 else "rh",
                name=name,
                subject="simulated",
            )
            labels.append(label)
    except TypeError:
        # Fall back to alternative construction method
        for i, name in enumerate(label_names):
            if i % 2 == 0:  # Left hemisphere
                vertices = [i]
                pos = np.random.rand(1, 3)
                hemi = "lh"
            else:  # Right hemisphere
                vertices = [i]
                pos = np.random.rand(1, 3)
                hemi = "rh"

            label = mne.Label(vertices, pos, hemi=hemi, name=name, subject="simulated")
            labels.append(label)

    # Run our connectivity function
    try:
        conn_df, summary_path = calculate_source_connectivity(
            stc_list=stc_list,  # Now passing a list of STCs
            labels=labels,
            subject_id="test_subject",
            sfreq=sfreq,
            epoch_length=2.0,  # Shorter epochs for the test
            n_epochs=10,  # Fewer epochs for speed
            max_duration=80,  # Limit to 80 seconds
            output_dir=temp_dir,
        )

        # Verify outputs
        test_passed = os.path.exists(summary_path)
        if test_passed:
            print(f"Test PASSED - output file created at {summary_path}")

            # Optionally examine some results
            print("\nSample connectivity results:")
            if not conn_df.empty:
                print(conn_df.head())

                # Compare with ground truth
                print("\nConnectivity measures summary:")
                for method in conn_df["method"].unique():
                    for band in conn_df["band"].unique():
                        subset = conn_df[
                            (conn_df["method"] == method) & (conn_df["band"] == band)
                        ]
                        if not subset.empty:
                            avg_conn = subset["connectivity"].mean()
                            print(
                                f"  {method} ({band}): Mean connectivity = {avg_conn:.4f}"
                            )

        else:
            print("Test FAILED - output file not created")

    except Exception as e:
        import traceback

        print(f"Test FAILED with error: {e}")
        print(traceback.format_exc())
        test_passed = False

    print(f"Test complete. Results in {temp_dir}")
    return test_passed


def calculate_source_connectivity_list(
    stc_list,
    labels=None,
    subjects_dir=None,
    subject="fsaverage",
    n_jobs=4,
    output_dir=None,
    subject_id=None,
    sfreq=None,
    epoch_length=4.0,
    n_epochs=20,
    max_duration=80,
):
    """
    Calculate connectivity metrics between brain regions from source-localized data.

    Parameters
    ----------
    stc_list : instance of SourceEstimate or list of SourceEstimates
        The source time course(s) to calculate connectivity from
    labels : list of Labels | None
        List of ROI labels to use. If None, will load Desikan-Killiany atlas
    subjects_dir : str | None
        Path to the freesurfer subjects directory
    subject : str
        Subject name in the subjects_dir (default: 'fsaverage')
    n_jobs : int
        Number of parallel jobs to use for computation
    output_dir : str | None
        Directory to save output files. If None, saves in current directory
    subject_id : str | None
        Subject identifier for file naming
    sfreq : float | None
        Sampling frequency. If None, will use stc.sfreq
    epoch_length : float
        Length of epochs in seconds for connectivity calculation
    n_epochs : int
        Number of epochs to use for connectivity calculation
    max_duration : float
        Maximum duration in seconds to process. Default is 80 seconds.

    Returns
    -------
    conn_df : DataFrame
        DataFrame containing connectivity values
    summary_path : str
        Path to the saved summary file
    """
    import time

    import pandas as pd

    start_time = time.time()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("connectivity")

    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "connectivity"), exist_ok=True)

    # Set up a log file
    log_file = os.path.join(output_dir, f"{subject_id}_connectivity_log.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    if subject_id is None:
        subject_id = "unknown_subject"

    # Convert stc to list if it's a single SourceEstimate
    if not isinstance(stc_list, list):
        stc_list = [stc_list]

    if sfreq is None:
        sfreq = stc_list[0].sfreq

    # Calculate available data duration
    single_epoch_duration = stc_list[0].times[-1] - stc_list[0].times[0]
    total_duration = single_epoch_duration * len(stc_list)
    logger.info(
        f"Total available data: {total_duration:.1f} seconds ({len(stc_list)} epochs of {single_epoch_duration:.1f}s each)"
    )

    # Limit to max_duration if specified
    if max_duration and total_duration > max_duration:
        epochs_needed = int(max_duration / single_epoch_duration)
        # Take epochs from the middle for better signal quality
        middle_idx = len(stc_list) // 2
        start_idx = max(0, middle_idx - epochs_needed // 2)
        end_idx = min(len(stc_list), start_idx + epochs_needed)
        stc_list = stc_list[start_idx:end_idx]
        actual_duration = len(stc_list) * single_epoch_duration
        logger.info(
            f"Limited to {actual_duration:.1f} seconds ({len(stc_list)} epochs)"
        )

    logger.info(
        f"Calculating connectivity for {subject_id} with {n_epochs} {epoch_length}-second epochs (sfreq={sfreq} Hz)..."
    )

    # bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
    bands = {
        "theta": (4, 8),
        "lowalpha": (8, 10),
        "highalpha": (10, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
    }

    # Updated connectivity methods list - use only methods supported by mne_connectivity
    # conn_methods = ['wpli', 'plv', 'coh', 'pli']
    conn_methods = ["wpli", "coh"]

    # For AEC we'll need to handle it separately since it's not part of spectral_connectivity_time
    include_aec = True

    if labels is None:
        logger.info("Loading Desikan-Killiany atlas labels...")
        labels = mne.read_labels_from_annot(
            subject, parc="aparc", subjects_dir=subjects_dir
        )
        labels = [label for label in labels if "unknown" not in label.name]

    selected_rois = [
        "precentral-lh",
        "precentral-rh",
        "postcentral-lh",
        "postcentral-rh",
        "paracentral-lh",
        "paracentral-rh",
        "caudalmiddlefrontal-lh",
        "caudalmiddlefrontal-rh",
    ]
    label_names = [label.name for label in labels]
    selected_labels = [
        labels[label_names.index(roi)] for roi in selected_rois if roi in label_names
    ]
    if not selected_labels:
        logger.warning("No selected ROIs found, using all available labels")
        selected_labels = labels
        selected_rois = label_names
    logger.info(f"Using {len(selected_labels)} selected ROIs: {selected_rois}")

    roi_pairs = list(itertools.combinations(range(len(selected_rois)), 2))

    logger.info("Extracting ROI time courses...")

    # Extract time courses from each stc in the list
    roi_data_list = []
    for stc in stc_list:
        roi_time_courses = [
            stc.extract_label_time_course(label, src=None, mode="mean", verbose=False)[
                0
            ]
            for label in selected_labels
        ]
        roi_data_list.append(np.array(roi_time_courses))

    # Concatenate all time courses
    roi_data = np.hstack(roi_data_list) if len(roi_data_list) > 1 else roi_data_list[0]
    logger.info(f"ROI data shape after concatenation: {roi_data.shape}")

    # Create epochs for connectivity calculation
    n_times = roi_data.shape[1]
    samples_per_epoch = int(epoch_length * sfreq)
    max_epochs = n_times // samples_per_epoch
    if max_epochs < n_epochs:
        logger.warning(
            f"Requested {n_epochs} epochs, but only {max_epochs} possible. Using {max_epochs}."
        )
        n_epochs = max_epochs

    # Use non-overlapping epochs from the concatenated data
    epoch_starts = np.arange(0, n_epochs) * samples_per_epoch
    epoched_data = np.stack(
        [
            roi_data[:, start : start + samples_per_epoch]
            for start in epoch_starts
            if start + samples_per_epoch <= n_times
        ],
        axis=0,
    )
    logger.info(f"Epoched data shape: {epoched_data.shape}")

    connectivity_data = []
    logger.info("Calculating connectivity metrics...")

    # Function to calculate AEC
    def calculate_aec(data, band_range, sfreq):
        import numpy as np
        from mne.filter import filter_data
        from scipy.signal import hilbert

        # Filter the data for the specific frequency band
        filtered_data = filter_data(
            data, sfreq=sfreq, l_freq=band_range[0], h_freq=band_range[1], verbose=False
        )

        # Get the amplitude envelope using Hilbert transform
        analytic_signal = hilbert(filtered_data, axis=-1)
        amplitude_envelope = np.abs(analytic_signal)

        # Compute correlation between envelopes
        n_signals = amplitude_envelope.shape[0]
        aec_matrix = np.zeros((n_signals, n_signals))

        for i in range(n_signals):
            for j in range(n_signals):
                if i != j:
                    corr = np.corrcoef(amplitude_envelope[i], amplitude_envelope[j])[
                        0, 1
                    ]
                    aec_matrix[i, j] = corr

        return aec_matrix

    # Calculate spectral connectivity methods
    for method in conn_methods:
        for band_name, band_range in bands.items():
            logger.info(f"Computing {method} connectivity in {band_name} band...")
            try:
                # Log detailed parameters for troubleshooting
                logger.info(
                    f"  Parameters: freqs={np.arange(band_range[0], band_range[1] + 1)}, "
                    f"sfreq={sfreq}, n_jobs={n_jobs}, n_cycles=2"
                )

                con = spectral_connectivity_time(
                    epoched_data,
                    freqs=np.arange(band_range[0], band_range[1] + 1),
                    method=method,
                    sfreq=sfreq,
                    mode="multitaper",
                    faverage=True,
                    average=True,
                    n_jobs=n_jobs,
                    verbose=False,
                    n_cycles=2,
                )
                con_matrix = con.get_data(output="dense").squeeze()
                if con_matrix.shape != (len(selected_rois), len(selected_rois)):
                    error_msg = f"Unexpected con_matrix shape: {con_matrix.shape}, expected {(len(selected_rois), len(selected_rois))}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                # Debug: Print con_matrix to verify
                logger.info(
                    f"{method} {band_name} con_matrix shape: {con_matrix.shape}"
                )
                logger.debug(
                    f"Matrix values range: min={np.min(con_matrix)}, max={np.max(con_matrix)}"
                )

                con_df = pd.DataFrame(
                    con_matrix, columns=selected_rois, index=selected_rois
                )
                matrix_filename = os.path.join(
                    output_dir, f"{subject_id}_{method}_{band_name}_matrix.csv"
                )
                con_df.to_csv(matrix_filename)
                logger.info(f"Saved connectivity matrix to {matrix_filename}")

                for i, (roi1_idx, roi2_idx) in enumerate(roi_pairs):
                    # Use lower triangle by swapping indices
                    value = con_matrix[
                        roi2_idx, roi1_idx
                    ]  # Changed from [roi1_idx, roi2_idx]
                    connectivity_data.append(
                        {
                            "subject": subject_id,
                            "method": method,
                            "band": band_name,
                            "roi1": selected_rois[roi1_idx],
                            "roi2": selected_rois[roi2_idx],
                            "connectivity": value,
                        }
                    )
            except Exception as e:
                error_msg = f"Error computing {method} in {band_name} band: {str(e)}"
                logger.error(error_msg)
                logger.error(f"Traceback: {traceback.format_exc()}")
                logger.error(
                    f"Data shape: {epoched_data.shape}, freqs: {np.arange(band_range[0], band_range[1] + 1)}"
                )
                continue

    # Calculate AEC separately
    if include_aec:
        method = "aec"
        for band_name, band_range in bands.items():
            logger.info(f"Computing {method} connectivity in {band_name} band...")
            try:
                # For each epoch, calculate AEC, then average
                aec_matrices = []
                for epoch_idx in range(epoched_data.shape[0]):
                    epoch_data = epoched_data[epoch_idx]
                    aec_matrix = calculate_aec(epoch_data, band_range, sfreq)
                    aec_matrices.append(aec_matrix)

                # Average across epochs
                con_matrix = np.mean(aec_matrices, axis=0)

                # Debug: Print con_matrix to verify
                logger.info(
                    f"{method} {band_name} con_matrix shape: {con_matrix.shape}"
                )
                logger.debug(
                    f"Matrix values range: min={np.min(con_matrix)}, max={np.max(con_matrix)}"
                )

                con_df = pd.DataFrame(
                    con_matrix, columns=selected_rois, index=selected_rois
                )
                matrix_filename = os.path.join(
                    output_dir, f"{subject_id}_{method}_{band_name}_matrix.csv"
                )
                con_df.to_csv(matrix_filename)
                logger.info(f"Saved connectivity matrix to {matrix_filename}")

                for i, (roi1_idx, roi2_idx) in enumerate(roi_pairs):
                    value = con_matrix[roi2_idx, roi1_idx]
                    connectivity_data.append(
                        {
                            "subject": subject_id,
                            "method": method,
                            "band": band_name,
                            "roi1": selected_rois[roi1_idx],
                            "roi2": selected_rois[roi2_idx],
                            "connectivity": value,
                        }
                    )
            except Exception as e:
                error_msg = f"Error computing {method} in {band_name} band: {str(e)}"
                logger.error(error_msg)
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue

    # Debug: Print sample connectivity_data
    if connectivity_data:
        logger.info(f"Sample connectivity_data entries: {connectivity_data[:5]}")
        conn_df = pd.DataFrame(connectivity_data)
        summary_path = os.path.join(
            output_dir, f"{subject_id}_connectivity_summary.csv"
        )
        conn_df.to_csv(summary_path, index=False)
        logger.info(f"Saved connectivity summary to {summary_path}")
    else:
        logger.warning("No connectivity data was generated")
        conn_df = pd.DataFrame()
        summary_path = None

    # Calculate graph metrics for each connectivity method and band
    logger.info("Calculating graph metrics...")

    from networkx.algorithms.community import louvain_communities, modularity

    graph_metrics_data = []
    for method in list(conn_methods) + (["aec"] if include_aec else []):
        for band_name in bands.keys():
            subset_df = conn_df[
                (conn_df["method"] == method) & (conn_df["band"] == band_name)
            ]
            if subset_df.empty:
                logger.warning(
                    f"No data for {method} in {band_name} band. Skipping graph metrics."
                )
                continue

            if not NETWORK_ANALYSIS_AVAILABLE:
                logger.warning(
                    "Network analysis libraries not available. Skipping graph metrics."
                )
                continue

            G = nx.Graph()
            for _, row in subset_df.iterrows():
                G.add_edge(row["roi1"], row["roi2"], weight=row["connectivity"])

            adj_matrix = nx.to_numpy_array(G, nodelist=selected_rois, weight="weight")

            # Check for NaNs or negative values in the adjacency matrix
            if np.isnan(adj_matrix).any():
                logger.warning(
                    f"NaN values in {method} {band_name} adjacency matrix. Skipping graph metrics."
                )
                continue

            # For some metrics like clustering coefficient, we need positive weights
            if (adj_matrix < 0).any():
                logger.warning(
                    f"Negative values in {method} {band_name} matrix. Using absolute values for graph metrics."
                )
                adj_matrix = np.abs(adj_matrix)

            try:
                clustering = np.mean(clustering_coef_wu(adj_matrix))
                global_efficiency = efficiency_wei(adj_matrix)
                char_path_length, _, _, _, _ = charpath(adj_matrix)
                communities = louvain_communities(G, resolution=1.0)
                modularity_score = modularity(G, communities, weight="weight")
                strength = np.mean(np.sum(adj_matrix, axis=1))

                # Additional graph metrics
                # Assortativity measures how similar connected nodes are
                assortativity = nx.degree_assortativity_coefficient(G, weight="weight")

                # Small-worldness
                # (requires computing random networks for comparison - simplified version)
                small_worldness = (
                    clustering * char_path_length if char_path_length > 0 else 0
                )

                graph_metrics_data.append(
                    {
                        "subject": subject_id,
                        "method": method,
                        "band": band_name,
                        "clustering": clustering,
                        "global_efficiency": global_efficiency,
                        "char_path_length": char_path_length,
                        "modularity": modularity_score,
                        "strength": strength,
                        "assortativity": assortativity,
                        "small_worldness": small_worldness,
                    }
                )
                logger.info(
                    f"Calculated graph metrics for {method} in {band_name} band"
                )
            except Exception as e:
                error_msg = f"Error calculating graph metrics for {method} in {band_name} band: {str(e)}"
                logger.error(error_msg)
                logger.error(f"Traceback: {traceback.format_exc()}")

    if graph_metrics_data:
        graph_metrics_df = pd.DataFrame(graph_metrics_data)
        metrics_path = os.path.join(output_dir, f"{subject_id}_graph_metrics.csv")
        graph_metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Saved graph metrics to {metrics_path}")
    else:
        logger.warning("No graph metrics were calculated")
        graph_metrics_df = pd.DataFrame()

    total_time = time.time() - start_time
    logger.info(
        f"Connectivity analysis complete for {subject_id} in {total_time:.1f} seconds ({total_time / 60:.1f} minutes)"
    )
    logger.info(f"Log file saved to: {log_file}")

    return conn_df, summary_path


def test_connectivity_function():
    """
    Test the calculate_source_connectivity function with simulated data.
    This function creates a synthetic source time course, simulates connectivity,
    and runs the analysis pipeline.

    Returns:
        bool: True if test passes, False otherwise
    """

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("Running connectivity function test with simulated data...")

    # Create temporary directory for outputs
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory for test outputs: {temp_dir}")

    # Simulation parameters
    n_sources = 8  # Number of sources/ROIs
    n_times = 10000  # Number of time points
    sfreq = 250.0  # Sample frequency in Hz

    # Simulated data parameters
    signal_freq = 10  # 10 Hz oscillation (alpha band)
    noise_level = 0.1
    coupling_strength = 0.6  # Strength of coupling between regions

    # Create time vector
    times = np.arange(n_times) / sfreq

    # Create coupled oscillations for our simulated sources
    source_data = np.zeros((n_sources, n_times))

    # Base oscillation (seed)
    seed_oscillation = np.sin(2 * np.pi * signal_freq * times)

    # Create coupled sources with different phase shifts and noise
    for i in range(n_sources):
        # Add phase shift based on source index (more shift for distant sources)
        phase_shift = i * np.pi / (2 * n_sources)
        source_data[i, :] = seed_oscillation * coupling_strength * np.exp(-i * 0.2)

        # Add phase shift
        source_data[i, :] = np.sin(2 * np.pi * signal_freq * times + phase_shift)

        # Add noise
        source_data[i, :] += noise_level * np.random.randn(n_times)

    # Create a coupling pattern - sources closer in index are more coupled
    true_coupling = np.zeros((n_sources, n_sources))
    for i in range(n_sources):
        for j in range(n_sources):
            # Coupling decreases with distance between ROIs
            true_coupling[i, j] = np.exp(-abs(i - j) * 0.5)

    # Set self-connections to zero
    np.fill_diagonal(true_coupling, 0)

    print("True coupling pattern:")
    print(true_coupling)

    # Create simulated labels
    labels = []
    label_names = [f"simulated_roi_{i}" for i in range(n_sources)]

    # Create labels compatible with different MNE versions
    try:
        # Try newer MNE API first
        for i, name in enumerate(label_names):
            vertices = np.array([i])
            pos = np.random.rand(1, 3)
            values = np.ones(1)

            label = mne.Label(
                vertices=vertices,
                pos=pos,
                values=values,
                hemi="lh" if i % 2 == 0 else "rh",
                name=name,
                subject="simulated",
            )
            labels.append(label)
    except TypeError:
        # Fall back to alternative construction method
        for i, name in enumerate(label_names):
            if i % 2 == 0:  # Left hemisphere
                vertices = [i]
                pos = np.random.rand(1, 3)
                hemi = "lh"
            else:  # Right hemisphere
                vertices = [i]
                pos = np.random.rand(1, 3)
                hemi = "rh"

            label = mne.Label(vertices, pos, hemi=hemi, name=name, subject="simulated")
            labels.append(label)

    # Create a SourceEstimate object
    vertices = [
        np.array([i for i in range(n_sources) if i % 2 == 0]),  # Left hemisphere
        np.array([i for i in range(n_sources) if i % 2 == 1]),  # Right hemisphere
    ]

    # Create a dummy stc object
    stc = SourceEstimate(
        source_data, vertices, tmin=0, tstep=1 / sfreq, subject="simulated"
    )

    # Run our connectivity function
    try:
        conn_df, summary_path = calculate_source_connectivity(
            stc=stc,
            labels=labels,
            subject_id="test_subject",
            sfreq=sfreq,
            epoch_length=2.0,  # Shorter epochs for the test
            n_epochs=10,  # Fewer epochs for speed
            output_dir=temp_dir,
        )

        # Verify outputs
        test_passed = os.path.exists(summary_path)
        if test_passed:
            print(f"Test PASSED - output file created at {summary_path}")

            # Optionally examine some results
            print("\nSample connectivity results:")
            if not conn_df.empty:
                print(conn_df.head())

                # Compare with ground truth
                print("\nConnectivity measures summary:")
                for method in conn_df["method"].unique():
                    for band in conn_df["band"].unique():
                        subset = conn_df[
                            (conn_df["method"] == method) & (conn_df["band"] == band)
                        ]
                        if not subset.empty:
                            avg_conn = subset["connectivity"].mean()
                            print(
                                f"  {method} ({band}): Mean connectivity = {avg_conn:.4f}"
                            )

        else:
            print("Test FAILED - output file not created")

    except Exception as e:
        import traceback

        print(f"Test FAILED with error: {e}")
        print(traceback.format_exc())
        test_passed = False

    print(f"Test complete. Results in {temp_dir}")
    return test_passed


def calculate_aec_connectivity(
    stc_list,
    labels=None,
    subjects_dir=None,
    subject="fsaverage",
    sfreq=None,
    output_dir=None,
    subject_id=None,
    epoch_length=2.0,
    n_epochs=40,
):
    """
    Calculate Amplitude Envelope Correlation (AEC) between all brain region labels.

    Parameters
    ----------
    stc_list : instance of SourceEstimate or list of SourceEstimates
        The source time course(s) to calculate connectivity from
    labels : list of Labels | None
        List of ROI labels to use. If None, will load Desikan-Killiany atlas
    subjects_dir : str | None
        Path to the freesurfer subjects directory
    subject : str
        Subject name in the subjects_dir (default: 'fsaverage')
    sfreq : float | None
        Sampling frequency. If None, will use stc.sfreq
    output_dir : str | None
        Directory to save output files. If None, saves in current directory
    subject_id : str | None
        Subject identifier for file naming
    epoch_length : float
        Length of epochs in seconds for connectivity calculation
    n_epochs : int
        Number of epochs to use for connectivity calculation

    Returns
    -------
    conn_df : DataFrame
        DataFrame containing AEC connectivity values
    conn_matrices : dict
        Dictionary containing connectivity matrices for each frequency band
    """
    import time

    import pandas as pd
    from scipy.signal import hilbert

    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("aec_connectivity")

    # Set up output directory
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    if subject_id is None:
        subject_id = "unknown_subject"

    # Convert stc to list if it's a single SourceEstimate
    if not isinstance(stc_list, list):
        stc_list = [stc_list]

    if sfreq is None:
        sfreq = stc_list[0].sfreq

    # Calculate available data duration
    single_epoch_duration = stc_list[0].times[-1] - stc_list[0].times[0]
    total_duration = single_epoch_duration * len(stc_list)
    logger.info(
        f"Total available data: {total_duration:.1f} seconds ({len(stc_list)} epochs of {single_epoch_duration:.1f}s each)"
    )

    # Define frequency bands
    bands = {
        "theta": (4, 8),
        "lowalpha": (8, 10),
        "highalpha": (10, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
    }

    # Load all labels if not provided
    if labels is None:
        logger.info("Loading Desikan-Killiany atlas labels...")
        labels = mne.read_labels_from_annot(
            subject, parc="aparc", subjects_dir=subjects_dir
        )
        labels = [label for label in labels if "unknown" not in label.name]

    label_names = [label.name for label in labels]
    logger.info(f"Using {len(labels)} ROIs")

    # Create all possible pairs of regions
    roi_pairs = list(itertools.combinations(range(len(label_names)), 2))

    logger.info("Extracting ROI time courses...")

    # Extract time courses from each stc in the list
    roi_data_list = []
    for stc in stc_list:
        roi_time_courses = [
            stc.extract_label_time_course(label, src=None, mode="mean", verbose=False)[
                0
            ]
            for label in labels
        ]
        roi_data_list.append(np.array(roi_time_courses))

    # Concatenate all time courses
    roi_data = np.hstack(roi_data_list) if len(roi_data_list) > 1 else roi_data_list[0]
    logger.info(f"ROI data shape after concatenation: {roi_data.shape}")

    # Create epochs for connectivity calculation
    n_times = roi_data.shape[1]
    samples_per_epoch = int(epoch_length * sfreq)
    max_epochs = n_times // samples_per_epoch
    if max_epochs < n_epochs:
        logger.warning(
            f"Requested {n_epochs} epochs, but only {max_epochs} possible. Using {max_epochs}."
        )
        n_epochs = max_epochs

    # Use non-overlapping epochs from the concatenated data
    epoch_starts = np.arange(0, n_epochs) * samples_per_epoch
    epoched_data = np.stack(
        [
            roi_data[:, start : start + samples_per_epoch]
            for start in epoch_starts
            if start + samples_per_epoch <= n_times
        ],
        axis=0,
    )
    logger.info(f"Epoched data shape: {epoched_data.shape}")

    # Function to calculate AEC
    def calculate_aec(data, band_range, sfreq):
        # Filter the data for the specific frequency band
        filtered_data = filter_data(
            data, sfreq=sfreq, l_freq=band_range[0], h_freq=band_range[1], verbose=False
        )

        # Get the amplitude envelope using Hilbert transform
        analytic_signal = hilbert(filtered_data, axis=-1)
        amplitude_envelope = np.abs(analytic_signal)

        # Compute correlation between envelopes
        n_signals = amplitude_envelope.shape[0]
        aec_matrix = np.zeros((n_signals, n_signals))

        for i in range(n_signals):
            for j in range(i + 1, n_signals):  # Only compute upper triangle
                corr = np.corrcoef(amplitude_envelope[i], amplitude_envelope[j])[0, 1]
                aec_matrix[i, j] = corr
                aec_matrix[j, i] = corr  # Mirror to lower triangle for convenience

        return aec_matrix

    connectivity_data = []
    conn_matrices = {}

    # Calculate AEC for each frequency band
    for band_name, band_range in bands.items():
        start_band = time.time()
        logger.info(f"Computing AEC connectivity in {band_name} band...")

        # For each epoch, calculate AEC, then average
        aec_matrices = []
        for epoch_idx in range(epoched_data.shape[0]):
            epoch_data = epoched_data[epoch_idx]
            aec_matrix = calculate_aec(epoch_data, band_range, sfreq)
            aec_matrices.append(aec_matrix)

        # Average across epochs
        con_matrix = np.mean(aec_matrices, axis=0)
        conn_matrices[band_name] = con_matrix

        # Save the full connectivity matrix
        con_df = pd.DataFrame(con_matrix, columns=label_names, index=label_names)
        matrix_filename = os.path.join(
            output_dir, f"{subject_id}_aec_{band_name}_matrix.csv"
        )
        con_df.to_csv(matrix_filename)
        logger.info(f"Saved connectivity matrix to {matrix_filename}")

        # Create a long-format dataframe with all connections
        for i, j in roi_pairs:
            value = con_matrix[i, j]
            connectivity_data.append(
                {
                    "subject": subject_id,
                    "method": "aec",
                    "band": band_name,
                    "roi1": label_names[i],
                    "roi2": label_names[j],
                    "connectivity": value,
                }
            )

        band_time = time.time() - start_band
        logger.info(f"Completed {band_name} band in {band_time:.1f} seconds")

    # Create and save summary dataframe
    conn_df = pd.DataFrame(connectivity_data)
    if not conn_df.empty:
        summary_path = os.path.join(output_dir, f"{subject_id}_aec_connectivity.csv")
        conn_df.to_csv(summary_path, index=False)
        logger.info(f"Saved connectivity summary to {summary_path}")
    else:
        logger.warning("No connectivity data was generated")

    return conn_df, conn_matrices


def calculate_source_pac(
    stc,
    labels=None,
    subjects_dir=None,
    subject="fsaverage",
    n_jobs=4,
    output_dir=None,
    subject_id=None,
    sfreq=None,
):
    """
    Calculate phase-amplitude coupling (PAC) from source-localized data with specific focus
    on ALS-relevant coupling and regions.

    Parameters
    ----------
    stc : instance of SourceEstimate
        The source time course to calculate PAC from
    labels : list of Labels | None
        List of ROI labels to use. If None, will load Desikan-Killiany atlas
    subjects_dir : str | None
        Path to the freesurfer subjects directory. If None, uses the environment variable
    subject : str
        Subject name in the subjects_dir (default: 'fsaverage')
    n_jobs : int
        Number of parallel jobs to use for computation
    output_dir : str | None
        Directory to save output files. If None, saves in current directory
    subject_id : str | None
        Subject identifier for file naming
    sfreq : float | None
        Sampling frequency. If None, will use stc.sfreq

    Returns
    -------
    pac_df : DataFrame
        DataFrame containing PAC values for all ROIs and frequency band pairs
    file_path : str
        Path to the saved summary file
    """

    import pandas as pd

    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = os.getcwd()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "pac"), exist_ok=True)

    if subject_id is None:
        subject_id = "unknown_subject"

    if sfreq is None:
        sfreq = stc.sfreq

    print(f"Calculating ALS-focused phase-amplitude coupling for {subject_id}...")

    # Define frequency bands - narrower beta band definition for better ALS sensitivity
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "lowbeta": (13, 20),  # Lower beta associated with motor control
        "highbeta": (20, 30),  # Higher beta associated with inhibitory control
        "gamma": (30, 45),
    }

    # Define PAC pairs focused on ALS-relevant couplings
    # Priority on beta-gamma coupling which is most relevant for motor dysfunction in ALS
    coupling_pairs = [
        ("lowbeta", "gamma"),  # Primary focus: motor system dysfunction
        ("highbeta", "gamma"),  # Secondary focus: inhibitory control issues in ALS
        ("alpha", "lowbeta"),  # Changed from 'alpha', 'beta'
        ("theta", "lowbeta"),  # Changed from 'theta', 'beta'
    ]
    # Load labels if not provided
    if labels is None:
        print("Loading Desikan-Killiany atlas labels...")
        labels = mne.read_labels_from_annot(
            subject, parc="aparc", subjects_dir=subjects_dir
        )
        labels = [label for label in labels if "unknown" not in label.name]

    # Focus on ALS-specific ROIs (motor network emphasis)
    selected_rois = [
        "precentral-lh",
        "precentral-rh",  # Primary motor cortex (key for ALS)
        "postcentral-lh",
        "postcentral-rh",  # Primary sensory cortex
        "paracentral-lh",
        "paracentral-rh",  # Supplementary motor area
        "caudalmiddlefrontal-lh",
        "caudalmiddlefrontal-rh",  # Premotor cortex
        "superiorfrontal-lh",
        "superiorfrontal-rh",  # Executive function (affected in ~50% of ALS cases)
        "inferiorparietal-lh",
        "inferiorparietal-rh",  # Sensorimotor integration
    ]

    # Filter labels to keep only selected ROIs
    label_names = [label.name for label in labels]
    selected_labels = []
    selected_roi_names = []

    for roi in selected_rois:
        if roi in label_names:
            selected_labels.append(labels[label_names.index(roi)])
            selected_roi_names.append(roi)
        else:
            print(f"Warning: ROI {roi} not found in the available labels")

    # If no ROIs matched, use a subset of all labels
    if not selected_labels:
        print("No selected ROIs found, using a subset of available labels")
        # Take a reasonable subset to avoid excessive computation
        selected_labels = labels[:16]  # First 16 labels
        selected_roi_names = label_names[:16]
    else:
        print(
            f"Using {len(selected_labels)} selected ROIs for ALS-focused PAC analysis"
        )

    # Function to segment data into epochs to speed up computation
    def epoch_data(data, epoch_length=4, n_epochs=40, sfreq=250):
        """Split continuous data into epochs"""
        samples_per_epoch = int(epoch_length * sfreq)
        total_samples = len(data)

        # If we have enough data, take the first n_epochs
        if total_samples >= n_epochs * samples_per_epoch:
            epochs = data[: n_epochs * samples_per_epoch].reshape(
                n_epochs, samples_per_epoch
            )
        else:
            # If not enough data, use as many complete epochs as possible
            max_complete_epochs = total_samples // samples_per_epoch
            print(
                f"Warning: Only {max_complete_epochs} complete epochs available (requested {n_epochs})"
            )
            epochs = data[: max_complete_epochs * samples_per_epoch].reshape(
                max_complete_epochs, samples_per_epoch
            )

        return epochs

    # Function to calculate PAC for a single time series
    def phase_amplitude_coupling(signal, phase_band, amp_band, fs):
        """Calculate phase-amplitude coupling between two frequency bands"""
        # Segment into epochs for faster processing
        epoch_length = 4  # seconds
        n_epochs = (
            40  # enough for reliable PAC estimate while keeping computation manageable
        )

        try:
            signal_epochs = epoch_data(
                signal, epoch_length=epoch_length, n_epochs=n_epochs, sfreq=fs
            )
            n_actual_epochs = signal_epochs.shape[0]

            # Calculate PAC for each epoch and average
            epoch_mis = []

            for epoch_idx in range(n_actual_epochs):
                epoch = signal_epochs[epoch_idx]

                # Filter signal for phase frequency band
                phase_signal = mne.filter.filter_data(
                    epoch, fs, phase_band[0], phase_band[1], method="iir"
                )

                # Filter signal for amplitude frequency band
                amp_signal = mne.filter.filter_data(
                    epoch, fs, amp_band[0], amp_band[1], method="iir"
                )

                # Extract phase and amplitude using Hilbert transform
                phase = np.angle(hilbert(phase_signal))
                amplitude = np.abs(hilbert(amp_signal))

                # Calculate modulation index (MI)
                n_bins = 18
                phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
                mean_amp = np.zeros(n_bins)

                for bin_idx in range(n_bins):
                    bin_mask = np.logical_and(
                        phase >= phase_bins[bin_idx], phase < phase_bins[bin_idx + 1]
                    )
                    if np.any(bin_mask):  # Check if the bin has any data points
                        mean_amp[bin_idx] = np.mean(amplitude[bin_mask])

                # Normalize mean amplitude
                if np.sum(mean_amp) > 0:  # Avoid division by zero
                    mean_amp /= np.sum(mean_amp)

                    # Calculate MI using Kullback-Leibler divergence
                    uniform = np.ones(n_bins) / n_bins
                    # Avoid log(0) by adding a small epsilon
                    epsilon = 1e-10
                    mi = np.sum(
                        mean_amp * np.log((mean_amp + epsilon) / (uniform + epsilon))
                    )
                    epoch_mis.append(mi)
                else:
                    epoch_mis.append(0)

            # Average MI across epochs
            if epoch_mis:
                return np.mean(epoch_mis)
            else:
                return 0

        except Exception as e:
            print(f"Error calculating PAC: {e}")
            return 0

    # Extract time courses for each ROI
    print("Extracting ROI time courses...")
    roi_time_courses = {}

    for i, label in enumerate(selected_labels):
        # Extract time course for this label
        tc = stc.extract_label_time_course(label, src=None, mode="mean", verbose=False)
        roi_time_courses[selected_roi_names[i]] = tc[0]

    # Initialize data storage for PAC values
    pac_data = []

    # Function to process a single ROI and coupling pair
    def process_pac(roi, phase_band_name, amp_band_name):
        signal = roi_time_courses[roi]
        phase_range = bands[phase_band_name]
        amp_range = bands[amp_band_name]

        # Calculate PAC
        mi = phase_amplitude_coupling(signal, phase_range, amp_range, sfreq)

        return {
            "roi": roi,
            "phase_band": phase_band_name,
            "amp_band": amp_band_name,
            "mi": mi,
        }

    # Process all ROIs and coupling pairs in parallel
    print("Calculating PAC for all ROIs and frequency band pairs...")

    # Create task list for parallel processing with priority for motor regions and beta-gamma coupling
    tasks = []

    # Add high-priority tasks first (beta-gamma in motor areas)
    motor_regions = [
        "precentral-lh",
        "precentral-rh",
        "postcentral-lh",
        "postcentral-rh",
    ]
    beta_gamma_pairs = [
        pair for pair in coupling_pairs if "beta" in pair[0] and "gamma" in pair[1]
    ]

    for roi in selected_roi_names:
        # Prioritize motor regions with beta-gamma coupling
        if roi in motor_regions:
            for phase_band, amp_band in beta_gamma_pairs:
                tasks.append((roi, phase_band, amp_band))

    # Then add all other combinations
    for roi in selected_roi_names:
        for phase_band, amp_band in coupling_pairs:
            if not (
                (roi in motor_regions) and ((phase_band, amp_band) in beta_gamma_pairs)
            ):
                tasks.append((roi, phase_band, amp_band))

    # Run tasks in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_pac)(roi, phase_band, amp_band)
        for roi, phase_band, amp_band in tasks
    )

    # Add all results to data storage
    for result in results:
        result["subject"] = subject_id
        pac_data.append(result)

    # Create and save PAC dataframe
    pac_df = pd.DataFrame(pac_data)

    # Save to CSV
    file_path = os.path.join(output_dir, f"{subject_id}_als_pac_summary.csv")
    pac_df.to_csv(file_path, index=False)
    print(f"Saved ALS-focused PAC summary to {file_path}")

    # Also save a more readable pivot table version
    pivot_data = pac_df.pivot_table(
        index=["roi", "phase_band"], columns="amp_band", values="mi"
    ).reset_index()

    pivot_path = os.path.join(output_dir, f"{subject_id}_als_pac_pivot.csv")
    pivot_data.to_csv(pivot_path, index=False)
    print(f"Saved PAC pivot table to {pivot_path}")

    # Generate a summary with special focus on motor-region beta-gamma coupling
    coupling_summary = []

    # Calculate motor-specific summaries for beta-gamma coupling
    motor_regions = [
        "precentral-lh",
        "precentral-rh",
        "postcentral-lh",
        "postcentral-rh",
    ]

    # First, summarize beta-gamma coupling in motor regions (most relevant for ALS)
    motor_beta_gamma = pac_df[
        (pac_df["roi"].isin(motor_regions))
        & (pac_df["phase_band"].isin(["lowbeta", "highbeta"]))
        & (pac_df["amp_band"] == "gamma")
    ]

    if not motor_beta_gamma.empty:
        motor_mean = motor_beta_gamma["mi"].mean()
        motor_max = motor_beta_gamma["mi"].max()
        motor_max_roi = (
            motor_beta_gamma.loc[motor_beta_gamma["mi"].idxmax(), "roi"]
            if len(motor_beta_gamma) > 0
            else "none"
        )
        motor_max_band = (
            motor_beta_gamma.loc[motor_beta_gamma["mi"].idxmax(), "phase_band"]
            if len(motor_beta_gamma) > 0
            else "none"
        )

        coupling_summary.append(
            {
                "subject": subject_id,
                "coupling_name": "motor_beta_gamma",
                "description": "Beta-gamma coupling in motor regions (ALS primary focus)",
                "mean_mi": motor_mean,
                "max_mi": motor_max,
                "max_roi": motor_max_roi,
                "max_phase_band": motor_max_band,
            }
        )

    # Then add summaries for each coupling pair across all regions
    for phase_band, amp_band in coupling_pairs:
        pair_data = pac_df[
            (pac_df["phase_band"] == phase_band) & (pac_df["amp_band"] == amp_band)
        ]

        # Calculate statistics for this coupling pair
        mean_mi = pair_data["mi"].mean()
        max_mi = pair_data["mi"].max()
        max_roi = (
            pair_data.loc[pair_data["mi"].idxmax(), "roi"]
            if len(pair_data) > 0
            else "none"
        )

        coupling_summary.append(
            {
                "subject": subject_id,
                "coupling_name": f"{phase_band}-{amp_band}",
                "description": f"{phase_band}-{amp_band} coupling across all regions",
                "mean_mi": mean_mi,
                "max_mi": max_mi,
                "max_roi": max_roi,
                "max_phase_band": phase_band,
            }
        )

    # Save coupling summary
    summary_df = pd.DataFrame(coupling_summary)
    summary_path = os.path.join(
        output_dir, f"{subject_id}_als_pac_coupling_summary.csv"
    )
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved ALS-focused coupling summary to {summary_path}")

    return pac_df, file_path


def calculate_vertex_level_spectral_power_list(
    stc_list, bands=None, n_jobs=10, output_dir=None, subject_id=None
):
    """
    Calculate spectral power at the vertex level across the entire brain for pre-epoched data,
    with better handling of inactive vertices and PSD visualization for quality control.

    Parameters
    ----------
    stc_list : list of SourceEstimate
        List of source time courses to calculate power from
    bands : dict | None
        Dictionary of frequency bands. If None, uses standard bands
    n_jobs : int
        Number of parallel jobs to use for computation
    output_dir : str | None
        Directory to save output files
    subject_id : str | None
        Subject identifier for file naming

    Returns
    -------
    power_dict : dict
        Dictionary containing power values for each frequency band for each vertex
    file_path : str
        Path to the saved vertex power file
    """

    import numpy as np

    # Setup output directory
    if output_dir is None:
        output_dir = os.getcwd()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "psd_plots"), exist_ok=True)

    if subject_id is None:
        subject_id = "unknown_subject"

    # Define frequency bands if not provided
    if bands is None:
        bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "lowalpha": (8, 10),
            "highalpha": (10, 13),
            "lowbeta": (13, 20),
            "highbeta": (20, 30),
            "gamma": (30, 45),
        }

    logging.info(f"Starting spectral power calculation for {subject_id}")

    # Get data info
    n_vertices = stc_list[0].data.shape[0]
    sfreq = stc_list[0].sfreq

    # First, identify active vertices
    logging.info("Identifying active vertices based on signal variance...")
    vertex_variance = np.zeros(n_vertices)

    # Calculate variance for a subset of vertices (for efficiency)
    sample_indices = np.linspace(0, n_vertices - 1, 1000, dtype=int)
    for vertex_idx in tqdm(sample_indices, desc="Sampling vertex variance"):
        try:
            vertex_data = np.concatenate([stc.data[vertex_idx] for stc in stc_list])
            vertex_variance[vertex_idx] = np.var(vertex_data)
        except Exception as e:
            logging.error(
                f"Error calculating variance for vertex {vertex_idx}: {str(e)}"
            )

    # Determine threshold based on sampled vertices
    non_zero_vars = vertex_variance[sample_indices][vertex_variance[sample_indices] > 0]
    if len(non_zero_vars) > 0:
        var_threshold = np.percentile(
            non_zero_vars, 10
        )  # 10th percentile of non-zero values
    else:
        var_threshold = 1e-12  # Fallback if no non-zero values

    logging.info(f"Variance threshold set to {var_threshold:.3e}")

    # Create a plot of the variance distribution
    plt.figure(figsize=(10, 6))
    plt.hist(np.log10(vertex_variance[sample_indices] + 1e-20), bins=50)
    plt.axvline(
        np.log10(var_threshold),
        color="r",
        linestyle="--",
        label=f"Threshold: {var_threshold:.3e}",
    )
    plt.xlabel("Log10 Vertex Variance")
    plt.ylabel("Count")
    plt.title("Distribution of Vertex Signal Variance")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{subject_id}_vertex_variance_dist.png"))
    plt.close()

    # Function to calculate power for a batch of vertices
    def process_vertex_batch(vertex_indices):
        batch_powers = {band: np.zeros(len(vertex_indices)) for band in bands}

        for i, vertex_idx in enumerate(vertex_indices):
            try:
                # Concatenate data from all epochs for this vertex
                vertex_data = np.concatenate([stc.data[vertex_idx] for stc in stc_list])

                # Calculate variance for this vertex
                v_var = np.var(vertex_data)

                # Skip computation for low-variance vertices
                if v_var < var_threshold:
                    # Set powers to zero or NaN
                    for band in bands:
                        batch_powers[band][i] = 0  # or np.nan
                    continue

                # Detrend the data
                vertex_data = signal.detrend(vertex_data)

                # Apply window function
                vertex_data = vertex_data * np.hamming(len(vertex_data))

                # Calculate PSD using Welch's method
                f, psd = signal.welch(
                    vertex_data,
                    fs=sfreq,
                    window="hann",
                    nperseg=min(
                        len(vertex_data), int(4 * sfreq)
                    ),  # 4-second windows or shorter
                    noverlap=min(len(vertex_data), int(2 * sfreq)),  # 50% overlap
                    nfft=None,
                    detrend=None,  # Already detrended
                )

                # Visualize PSD for a few vertices (evenly spaced)
                if vertex_idx % 2000 == 0:
                    plt.figure(figsize=(10, 6))
                    plt.semilogy(f, psd)
                    plt.xlabel("Frequency (Hz)")
                    plt.ylabel("PSD (V^2/Hz)")
                    plt.title(f"Vertex {vertex_idx} - Variance: {v_var:.3e}")

                    # Add band markers
                    for band_name, (fmin, fmax) in bands.items():
                        plt.axvspan(fmin, fmax, alpha=0.2, label=band_name)

                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            output_dir,
                            "psd_plots",
                            f"{subject_id}_vertex{vertex_idx}_psd.png",
                        )
                    )
                    plt.close()

                # Calculate power in each frequency band
                for band, (fmin, fmax) in bands.items():
                    freq_mask = (f >= fmin) & (f <= fmax)

                    if np.any(freq_mask):
                        # Calculate band power (area under the curve)
                        band_freqs = f[freq_mask]
                        band_psd = psd[freq_mask]

                        # Use trapezoid rule for integration
                        power = np.trapz(band_psd, band_freqs)
                        batch_powers[band][i] = power
                    else:
                        logging.warning(
                            f"No frequencies in band {band} ({fmin}-{fmax} Hz) for vertex {vertex_idx}"
                        )
                        batch_powers[band][i] = 0

            except Exception as e:
                logging.error(f"Error processing vertex {vertex_idx}: {str(e)}")
                # Initialize with zeros
                for band in bands:
                    batch_powers[band][i] = 0

        return batch_powers, vertex_indices

    # Process vertices in batches to manage memory
    batch_size = 1000
    n_batches = int(np.ceil(n_vertices / batch_size))
    all_vertex_batches = [
        range(i * batch_size, min((i + 1) * batch_size, n_vertices))
        for i in range(n_batches)
    ]

    logging.info(f"Processing {n_vertices} vertices in {n_batches} batches...")

    # Run processing in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_vertex_batch)(vertex_batch)
        for vertex_batch in tqdm(all_vertex_batches, desc="Processing batches")
    )

    # Combine results
    power_dict = {band: np.zeros(n_vertices) for band in bands}

    for batch_powers, vertex_indices in results:
        for band in bands:
            power_dict[band][vertex_indices] = batch_powers[band]

    # Save results to disk
    file_path = os.path.join(output_dir, f"{subject_id}_vertex_power.h5")

    with h5py.File(file_path, "w") as f:
        # Store vertex information
        f.attrs["n_vertices"] = n_vertices
        f.attrs["lh_vertices"] = len(stc_list[0].vertices[0])
        f.attrs["rh_vertices"] = len(stc_list[0].vertices[1])

        # Store power values
        for band, power_values in power_dict.items():
            f.create_dataset(
                band, data=power_values, compression="gzip", compression_opts=9
            )

    # Create a CSV file for statistical analysis
    csv_file_path = os.path.join(output_dir, f"{subject_id}_vertex_power.csv")

    # Create a DataFrame with vertex indices and power values for each band
    df_data = {"vertex_idx": np.arange(n_vertices)}

    # Add hemisphere information
    lh_vertices = len(stc_list[0].vertices[0])
    hemispheres = ["left"] * lh_vertices + ["right"] * (n_vertices - lh_vertices)
    df_data["hemisphere"] = hemispheres

    # Add power values for each band
    for band, power_values in power_dict.items():
        df_data[f"{band}_power"] = power_values
        # Add log-transformed values for visualization
        # Add small constant to avoid log(0)
        df_data[f"{band}_log_power"] = np.log10(power_values + 1e-15)

    # Create and save the DataFrame
    vertex_power_df = pd.DataFrame(df_data)
    vertex_power_df.to_csv(csv_file_path, index=False)

    # Create summary plots of all bands
    plt.figure(figsize=(12, 8))
    band_means = {}
    for band in bands:
        # Get non-zero values only for plotting
        non_zero_mask = power_dict[band] > 0
        if np.any(non_zero_mask):
            band_means[band] = np.mean(power_dict[band][non_zero_mask])
        else:
            band_means[band] = 0

    # Sort bands by frequency
    band_order = sorted(
        bands.keys(), key=lambda x: bands[x][0]
    )  # Sort by lower bound of band

    # Create bar plot
    plt.bar(
        range(len(band_order)),
        [band_means[band] for band in band_order],
        tick_label=band_order,
    )
    plt.ylabel("Mean Power (active vertices)")
    plt.title(f"Mean Band Power - {subject_id}")
    plt.savefig(os.path.join(output_dir, f"{subject_id}_band_powers.png"))

    # Also create a log-scale version
    plt.figure(figsize=(12, 8))
    plt.bar(
        range(len(band_order)),
        [np.log10(band_means[band] + 1e-15) for band in band_order],
        tick_label=band_order,
    )
    plt.ylabel("Log10 Mean Power (active vertices)")
    plt.title(f"Log Mean Band Power - {subject_id}")
    plt.savefig(os.path.join(output_dir, f"{subject_id}_log_band_powers.png"))

    logging.info(f"Saved vertex-level spectral power to {file_path}")
    logging.info(f"Saved CSV table for statistical analysis to {csv_file_path}")
    logging.info(
        f"Generated PSD plots for sample vertices in {os.path.join(output_dir, 'psd_plots')}"
    )

    return power_dict, file_path


def calculate_vertex_level_spectral_power(
    stc, bands=None, n_jobs=10, output_dir=None, subject_id=None
):
    """
    Calculate spectral power at the vertex level across the entire brain.

    Parameters
    ----------
    stc : instance of SourceEstimate
        The source time course to calculate power from
    bands : dict | None
        Dictionary of frequency bands. If None, uses standard bands
    n_jobs : int
        Number of parallel jobs to use for computation
    output_dir : str | None
        Directory to save output files
    subject_id : str | None
        Subject identifier for file naming

    Returns
    -------
    power_dict : dict
        Dictionary containing power values for each frequency band
        for each vertex
    file_path : str
        Path to the saved vertex power file
    """

    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    if subject_id is None:
        subject_id = "unknown_subject"

    if bands is None:
        bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "lowbeta": (13, 20),
            "highbeta": (20, 30),
            "gamma": (30, 45),
        }

    # Get data and sampling frequency
    data = stc.data
    sfreq = stc.sfreq
    n_vertices = data.shape[0]

    print(f"Calculating vertex-level spectral power for {subject_id}...")
    print(f"Source data shape: {data.shape}")

    # Parameters for Welch's method
    window_length = int(4 * sfreq)  # 4-second windows
    n_overlap = window_length // 2  # 50% overlap

    # Function to calculate power for a batch of vertices
    def process_vertex_batch(vertex_indices):
        # Initialize a dictionary to store power values for each band
        batch_powers = {band: np.zeros(len(vertex_indices)) for band in bands}

        for i, vertex_idx in enumerate(vertex_indices):
            # Calculate PSD using Welch's method
            f, psd = signal.welch(
                data[vertex_idx],
                fs=sfreq,
                window="hann",
                nperseg=window_length,
                noverlap=n_overlap,
                nfft=None,
                scaling="density",
            )

            # Calculate average power in each frequency band
            for band, (fmin, fmax) in bands.items():
                freq_mask = (f >= fmin) & (f <= fmax)
                if np.any(freq_mask):
                    batch_powers[band][i] = np.mean(psd[freq_mask])
                else:
                    batch_powers[band][i] = 0

        return batch_powers, vertex_indices

    # Process vertices in batches to manage memory
    batch_size = 1000
    n_batches = int(np.ceil(n_vertices / batch_size))
    all_vertex_batches = []

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_vertices)
        all_vertex_batches.append(range(start_idx, end_idx))

    print(f"Processing {n_vertices} vertices in {n_batches} batches...")

    # Run processing in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_vertex_batch)(vertex_batch)
        for vertex_batch in all_vertex_batches
    )

    # Combine results
    power_dict = {band: np.zeros(n_vertices) for band in bands}

    for batch_powers, vertex_indices in results:
        for band in bands:
            power_dict[band][vertex_indices] = batch_powers[band]

    # Save results to disk
    # HDF5 format is good for large arrays and provides compression
    file_path = os.path.join(output_dir, f"{subject_id}_vertex_power.h5")

    with h5py.File(file_path, "w") as f:
        # Store vertex information
        f.attrs["n_vertices"] = n_vertices
        f.attrs["lh_vertices"] = len(stc.vertices[0])
        f.attrs["rh_vertices"] = len(stc.vertices[1])

        # Create a group for each frequency band
        for band, power_values in power_dict.items():
            f.create_dataset(
                band, data=power_values, compression="gzip", compression_opts=9
            )

    print(f"Saved vertex-level spectral power to {file_path}")

    return power_dict, file_path


def apply_spatial_smoothing(
    power_dict, stc, smoothing_steps=5, subject_id=None, output_dir=None
):
    """
    Apply spatial smoothing to vertex-level power data.

    Parameters
    ----------
    power_dict : dict
        Dictionary containing power values for each frequency band
    stc : instance of SourceEstimate
        The source time course object (needed for vertices info)
    smoothing_steps : int
        Number of smoothing steps to apply
    subject_id : str | None
        Subject identifier for file naming
    output_dir : str | None
        Directory to save output files

    Returns
    -------
    smoothed_dict : dict
        Dictionary containing smoothed power values
    file_path : str
        Path to the saved smoothed data file
    """

    import numpy as np

    if output_dir is None:
        output_dir = os.getcwd()

    if subject_id is None:
        subject_id = "unknown_subject"

    print(f"Applying spatial smoothing (steps={smoothing_steps}) to vertex data...")

    # Create a source space for smoothing
    # We need the same source space that was used to create the stc
    src = mne.source_space.SourceSpaces(
        [
            mne.source_space.SourceSpace(
                vertices=stc.vertices[0], hemisphere=0, coord_frame=5
            ),
            mne.source_space.SourceSpace(
                vertices=stc.vertices[1], hemisphere=1, coord_frame=5
            ),
        ]
    )

    smoothed_dict = {}

    # Apply smoothing to each frequency band
    for band, power_values in power_dict.items():
        print(f"Smoothing {band} band data...")

        # Create a temporary SourceEstimate to use MNE's smoothing function
        temp_stc = mne.SourceEstimate(
            power_values[:, np.newaxis],  # Add a time dimension
            vertices=stc.vertices,
            tmin=0,
            tstep=1,
        )

        # Apply spatial smoothing
        smoothed_stc = mne.spatial_src_adjacency(temp_stc, src, n_steps=smoothing_steps)

        # Store the smoothed data
        smoothed_dict[band] = smoothed_stc.data[:, 0]  # Remove the time dimension

    # Save the smoothed data
    file_path = os.path.join(output_dir, f"{subject_id}_smoothed_vertex_power.h5")

    with h5py.File(file_path, "w") as f:
        # Store vertex information
        f.attrs["n_vertices"] = len(smoothed_dict[next(iter(smoothed_dict))])
        f.attrs["lh_vertices"] = len(stc.vertices[0])
        f.attrs["rh_vertices"] = len(stc.vertices[1])
        f.attrs["smoothing_steps"] = smoothing_steps

        # Create a group for each frequency band
        for band, power_values in smoothed_dict.items():
            f.create_dataset(
                band, data=power_values, compression="gzip", compression_opts=9
            )

    print(f"Saved smoothed vertex-level spectral power to {file_path}")

    return smoothed_dict, file_path


def calculate_vertex_psd_for_fooof(
    stc, fmin=1.0, fmax=45.0, n_jobs=10, output_dir=None, subject_id=None
):
    """
    Calculate full power spectral density at the vertex level for FOOOF analysis.

    Parameters
    ----------
    stc : instance of SourceEstimate
        The source time course to calculate power from
    fmin : float
        Minimum frequency of interest
    fmax : float
        Maximum frequency of interest
    n_jobs : int
        Number of parallel jobs to use for computation
    output_dir : str | None
        Directory to save output files
    subject_id : str | None
        Subject identifier for file naming

    Returns
    -------
    stc_psd : instance of SourceEstimate
        Source estimate containing PSD values with frequencies as time points
    file_path : str
        Path to the saved PSD file
    """

    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    if subject_id is None:
        subject_id = "unknown_subject"

    # Get data and sampling frequency
    data = stc.data
    sfreq = stc.sfreq
    n_vertices = data.shape[0]

    print(f"Calculating vertex-level PSD for FOOOF analysis - {subject_id}...")
    print(f"Source data shape: {data.shape}")

    # Parameters for Welch's method
    window_length = int(4 * sfreq)  # 4-second windows
    n_overlap = window_length // 2  # 50% overlap

    # First, calculate the frequency axis (same for all vertices)
    f, _ = signal.welch(
        data[0],
        fs=sfreq,
        window="hann",
        nperseg=window_length,
        noverlap=n_overlap,
        nfft=None,
    )

    # Filter to frequency range of interest
    freq_mask = (f >= fmin) & (f <= fmax)
    freqs = f[freq_mask]
    n_freqs = len(freqs)

    print(f"Calculating PSD for {n_freqs} frequency points from {fmin} to {fmax} Hz")

    # Function to calculate PSD for a batch of vertices
    def process_vertex_batch(vertex_indices):
        batch_psd = np.zeros((len(vertex_indices), n_freqs))

        for i, vertex_idx in enumerate(vertex_indices):
            # Calculate PSD using Welch's method
            _, psd = signal.welch(
                data[vertex_idx],
                fs=sfreq,
                window="hann",
                nperseg=window_length,
                noverlap=n_overlap,
                nfft=None,
                scaling="density",
            )

            # Store PSD for frequencies in our range
            batch_psd[i] = psd[freq_mask]

        return batch_psd

    # Process vertices in batches to manage memory
    batch_size = 4000
    n_batches = int(np.ceil(n_vertices / batch_size))
    all_psds = np.zeros((n_vertices, n_freqs))

    print(f"Processing {n_vertices} vertices in {n_batches} batches...")

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_vertices)
        vertex_batch = range(start_idx, end_idx)

        print(
            f"Processing batch {batch_idx + 1}/{n_batches}, vertices {start_idx}-{end_idx}"
        )

        # Calculate PSD for this batch
        batch_psd = process_vertex_batch(vertex_batch)

        # Store in the full array
        all_psds[start_idx:end_idx] = batch_psd

    # Create a source estimate with the PSD data
    # This uses frequencies as time points for easy manipulation
    stc_psd = mne.SourceEstimate(
        all_psds,
        vertices=stc.vertices,
        tmin=freqs[0],
        tstep=(freqs[-1] - freqs[0]) / (n_freqs - 1),
    )

    # Save the PSD source estimate
    file_path = os.path.join(output_dir, f"{subject_id}_psd-stc.h5")
    stc_psd.save(file_path, overwrite=True)

    print(f"Saved vertex-level PSD to {file_path}")
    print(
        f"PSD shape: {all_psds.shape}, frequency range: {freqs[0]:.1f}-{freqs[-1]:.1f} Hz"
    )

    return stc_psd, file_path


def calculate_fooof_aperiodic(
    stc_psd, subject_id, output_dir, n_jobs=10, aperiodic_mode="knee"
):
    """
    Run FOOOF to model aperiodic parameters for all vertices with robust error handling.

    Parameters
    ----------
    stc_psd : instance of SourceEstimate
        The source estimate containing PSD data
    subject_id : str
        Subject identifier for file naming
    output_dir : str
        Directory to save output files
    n_jobs : int
        Number of parallel jobs to use for computation
    aperiodic_mode : str
        Aperiodic mode for FOOOF ('fixed' or 'knee')

    Returns
    -------
    aperiodic_df : DataFrame
        DataFrame with aperiodic parameters
    file_path : str
        Path to saved file
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Calculating FOOOF aperiodic parameters for {subject_id}...")

    if not FOOOF_AVAILABLE:
        print("FOOOF library not available. Skipping aperiodic parameter analysis.")
        return pd.DataFrame(), None

    # Get data from stc_psd
    psds = stc_psd.data
    freqs = stc_psd.times

    n_vertices = psds.shape[0]
    print(f"Processing FOOOF analysis for {n_vertices} vertices...")

    # FOOOF parameters with fallback options
    fooof_params = {
        "peak_width_limits": [1, 8.0],
        "max_n_peaks": 6,
        "min_peak_height": 0.0,
        "peak_threshold": 2.0,
        "aperiodic_mode": aperiodic_mode,
        "verbose": False,
    }

    fallback_params = {
        "peak_width_limits": [1, 8.0],
        "max_n_peaks": 3,
        "min_peak_height": 0.1,
        "peak_threshold": 2.5,
        "aperiodic_mode": "fixed",  # Fall back to fixed mode which is more stable
        "verbose": False,
    }

    # Function to process a batch of vertices with error handling
    def process_batch(vertices):
        batch_psds = psds[vertices, :]

        # First attempt with primary parameters
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            try:
                fg = FOOOFGroup(**fooof_params)
                fg.fit(freqs, batch_psds)

                # Check if fits were successful
                if np.any(~fg.get_params("aperiodic_params")[:, 0].astype(bool)):
                    # Some fits failed, try fallback parameters
                    raise RuntimeError("Some fits failed with primary parameters")

            except Exception:
                # Try again with fallback parameters
                try:
                    fg = FOOOFGroup(**fallback_params)
                    fg.fit(freqs, batch_psds)
                except Exception:
                    # Create dummy results for completely failed fits
                    results = []
                    for i, vertex_idx in enumerate(vertices):
                        results.append(
                            {
                                "vertex": vertex_idx,
                                "offset": np.nan,
                                "knee": np.nan,
                                "exponent": np.nan,
                                "r_squared": np.nan,
                                "error": np.nan,
                                "status": "FITTING_FAILED",
                            }
                        )
                    return results

        # Extract aperiodic parameters
        aperiodic_params = fg.get_params("aperiodic_params")
        r_squared = fg.get_params("r_squared")
        error = fg.get_params("error")

        # Process results
        results = []
        for i, vertex_idx in enumerate(vertices):
            # Check for valid parameters
            if np.any(np.isnan(aperiodic_params[i])) or np.any(
                np.isinf(aperiodic_params[i])
            ):
                results.append(
                    {
                        "vertex": vertex_idx,
                        "offset": np.nan,
                        "knee": np.nan,
                        "exponent": np.nan,
                        "r_squared": np.nan,
                        "error": np.nan,
                        "status": "NAN_PARAMS",
                    }
                )
                continue

            # Extract parameters based on aperiodic mode
            if aperiodic_mode == "knee":
                offset = aperiodic_params[i, 0]
                knee = aperiodic_params[i, 1]
                exponent = aperiodic_params[i, 2]

                # Additional validation for knee mode
                if knee <= 0 or exponent <= 0:
                    results.append(
                        {
                            "vertex": vertex_idx,
                            "offset": np.nan,
                            "knee": np.nan,
                            "exponent": np.nan,
                            "r_squared": np.nan,
                            "error": np.nan,
                            "status": "INVALID_PARAMS",
                        }
                    )
                    continue
            else:  # fixed mode
                offset = aperiodic_params[i, 0]
                knee = np.nan
                exponent = aperiodic_params[i, 1]

                # Additional validation for fixed mode
                if exponent <= 0:
                    results.append(
                        {
                            "vertex": vertex_idx,
                            "offset": np.nan,
                            "knee": np.nan,
                            "exponent": np.nan,
                            "r_squared": np.nan,
                            "error": np.nan,
                            "status": "INVALID_EXPONENT",
                        }
                    )
                    continue

            # Add valid result
            results.append(
                {
                    "vertex": vertex_idx,
                    "offset": offset,
                    "knee": knee,
                    "exponent": exponent,
                    "r_squared": r_squared[i],
                    "error": error[i],
                    "status": "SUCCESS",
                }
            )

        # Clear memory
        del fg, batch_psds
        gc.collect()

        return results

    # Process in batches
    batch_size = 2000
    n_batches = int(np.ceil(n_vertices / batch_size))
    vertex_batches = []

    for i in range(0, n_vertices, batch_size):
        vertex_batches.append(range(i, min(i + batch_size, n_vertices)))

    print(f"Processing {n_batches} batches with {n_jobs} parallel jobs...")

    # Run in parallel with warning suppression
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        all_results = Parallel(n_jobs=n_jobs)(
            delayed(process_batch)(batch) for batch in vertex_batches
        )

    # Flatten results
    flat_results = [item for sublist in all_results for item in sublist]

    # Create DataFrame
    aperiodic_df = pd.DataFrame(flat_results)

    # Add subject_id
    aperiodic_df.insert(0, "subject", subject_id)

    # Save results
    file_path = os.path.join(output_dir, f"{subject_id}_fooof_aperiodic.parquet")
    aperiodic_df.to_csv(
        os.path.join(output_dir, f"{subject_id}_fooof_aperiodic.csv"), index=False
    )
    aperiodic_df.to_parquet(file_path)

    # Calculate statistics for better reporting
    success_count = (aperiodic_df["status"] == "SUCCESS").sum()
    success_rate = (success_count / len(aperiodic_df)) * 100

    print(f"Saved FOOOF aperiodic parameters to {file_path}")
    print(
        f"Success rate: {success_rate:.1f}% ({success_count}/{len(aperiodic_df)} vertices)"
    )

    # Report average values for successful fits
    successful_fits = aperiodic_df[aperiodic_df["status"] == "SUCCESS"]
    if len(successful_fits) > 0:
        print(f"Average exponent: {successful_fits['exponent'].mean():.3f}")
        if aperiodic_mode == "knee":
            print(f"Average knee: {successful_fits['knee'].mean():.3f}")
        print(f"Average R²: {successful_fits['r_squared'].mean():.3f}")

    return aperiodic_df, file_path


def visualize_fooof_results(
    aperiodic_df,
    stc_psd,
    peaks_df=None,
    subjects_dir=None,
    subject="fsaverage",
    output_dir=None,
    subject_id=None,
    plot_examples=True,
    plot_brain=True,
    use_log=False,
):
    """
    Create a comprehensive visualization of FOOOF analysis results.

    Parameters
    ----------
    aperiodic_df : DataFrame
        DataFrame with aperiodic parameters from run_fooof_aperiodic_fit
    stc_psd : instance of SourceEstimate
        Source estimate containing PSD data
    peaks_df : DataFrame | None
        DataFrame with peak parameters from calculate_vertex_peak_frequencies
    subjects_dir : str | None
        Path to FreeSurfer subjects directory
    subject : str
        Subject name (default: 'fsaverage')
    output_dir : str | None
        Directory to save output files
    subject_id : str | None
        Subject identifier for file naming
    plot_examples : bool
        Whether to plot example fits (default: True)
    plot_brain : bool
        Whether to plot brain visualizations (default: True)
    use_log : bool
        Whether to use log scale for power (default: False)

    Returns
    -------
    fig : matplotlib Figure
        The multi-panel figure
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    if subject_id is None and "subject" in aperiodic_df.columns:
        subject_id = aperiodic_df["subject"].iloc[0]
    elif subject_id is None:
        subject_id = "unknown_subject"

    # Filter to successful fits
    success_df = aperiodic_df[aperiodic_df["status"] == "SUCCESS"].copy()

    # Create figure with multiple panels
    fig = plt.figure(figsize=(18, 12))

    # Create GridSpec for flexible panel arrangement
    gs = gridspec.GridSpec(3, 4, figure=fig)

    # 1. Summary statistics panel
    ax_stats = fig.add_subplot(gs[0, 0])
    ax_stats.axis("off")

    # Calculate summary statistics
    total_vertices = len(aperiodic_df)
    success_count = len(success_df)
    success_rate = (success_count / total_vertices) * 100

    summary_text = [
        f"FOOOF Analysis Summary: {subject_id}",
        f"Total vertices: {total_vertices}",
        f"Successful fits: {success_count} ({success_rate:.1f}%)",
        f"Average exponent: {success_df['exponent'].mean():.3f} ± {success_df['exponent'].std():.3f}",
    ]

    if "knee" in success_df.columns and not all(np.isnan(success_df["knee"])):
        summary_text.append(
            f"Average knee: {success_df['knee'].dropna().mean():.3f} ± {success_df['knee'].dropna().std():.3f}"
        )

    summary_text.append(
        f"Average R²: {success_df['r_squared'].mean():.3f} ± {success_df['r_squared'].std():.3f}"
    )

    # Add peak information if available
    if peaks_df is not None:
        peaks_success = peaks_df[peaks_df["status"] == "SUCCESS"]
        if len(peaks_success) > 0:
            summary_text.append(
                f"Peak detection: {len(peaks_success)} vertices ({len(peaks_success) / total_vertices * 100:.1f}%)"
            )
            summary_text.append(
                f"Average peak freq: {peaks_success['peak_freq'].mean():.2f} ± {peaks_success['peak_freq'].std():.2f} Hz"
            )

    # Display statistics
    ax_stats.text(
        0.05,
        0.95,
        "\n".join(summary_text),
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # 2. Exponent distribution histogram
    ax_exp = fig.add_subplot(gs[0, 1])
    sns.histplot(success_df["exponent"], bins=30, kde=True, ax=ax_exp)
    ax_exp.set_title("Aperiodic Exponent Distribution")
    ax_exp.set_xlabel("Exponent")
    ax_exp.set_ylabel("Count")

    # 3. Knee distribution histogram (if available)
    if "knee" in success_df.columns and not all(np.isnan(success_df["knee"])):
        ax_knee = fig.add_subplot(gs[0, 2])
        knee_values = success_df["knee"].dropna()
        if len(knee_values) > 0:
            sns.histplot(knee_values, bins=30, kde=True, ax=ax_knee)
            ax_knee.set_title("Knee Parameter Distribution")
            ax_knee.set_xlabel("Knee")
            ax_knee.set_ylabel("Count")

    # 4. R² distribution histogram
    ax_r2 = fig.add_subplot(gs[0, 3])
    sns.histplot(success_df["r_squared"], bins=30, kde=True, ax=ax_r2)
    ax_r2.set_title("R² Distribution")
    ax_r2.set_xlabel("R²")
    ax_r2.set_ylabel("Count")

    # 5. Status breakdown pie chart
    ax_status = fig.add_subplot(gs[1, 0])
    status_counts = aperiodic_df["status"].value_counts()
    ax_status.pie(
        status_counts,
        labels=status_counts.index,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
    )
    ax_status.set_title("Fitting Status Distribution")

    # 6. Example fits (if requested)
    if plot_examples and len(success_df) > 0:
        # Get data
        psds = stc_psd.data
        freqs = stc_psd.times

        if not FOOOF_AVAILABLE:
            print("FOOOF library not available. Skipping aperiodic visualization.")
            return

        # Create FOOOF model for visualization
        fm = FOOOF(
            peak_width_limits=[1, 8.0],
            aperiodic_mode="knee" if "knee" in success_df.columns else "fixed",
        )

        # Plot 3 examples: best fit, median fit, and worst fit (among successful)
        r2_sorted = success_df.sort_values("r_squared", ascending=False)
        best_idx = r2_sorted.iloc[0]["vertex"]
        median_idx = r2_sorted.iloc[len(r2_sorted) // 2]["vertex"]
        worst_idx = r2_sorted.iloc[-1]["vertex"]

        def plot_fit_custom(fm, ax, plt_log=use_log):
            """Custom function to plot FOOOF fits on linear or log scale"""
            # Get data from FOOOF model
            ap_fit = fm._ap_fit
            model_fit = fm.fooofed_spectrum_
            freqs = fm.freqs
            power_spectrum = fm.power_spectrum

            # Plot original spectrum
            ax.plot(
                freqs, power_spectrum, "k-", linewidth=2.0, label="Original Spectrum"
            )

            # Plot model fit
            ax.plot(
                freqs, model_fit, "r-", linewidth=2.5, alpha=0.5, label="Full Model Fit"
            )

            # Plot aperiodic fit
            ax.plot(
                freqs, ap_fit, "b--", linewidth=2.5, alpha=0.5, label="Aperiodic Fit"
            )

            # Configure plot
            ax.set_xlim([freqs.min(), freqs.max()])
            if plt_log:
                ax.set_ylabel("log(Power)")
            else:
                ax.set_ylabel("Power")
            ax.set_xlabel("Frequency (Hz)")
            ax.legend(fontsize="small")

            # Return axis
            return ax

        # Best fit example
        ax_best = fig.add_subplot(gs[1, 1])
        fm.fit(freqs, psds[int(best_idx)])
        plot_fit_custom(fm, ax_best)
        ax_best.set_title(f"Best Fit (R²={r2_sorted.iloc[0]['r_squared']:.3f})")

        # Median fit example
        ax_median = fig.add_subplot(gs[1, 2])
        fm.fit(freqs, psds[int(median_idx)])
        plot_fit_custom(fm, ax_median)
        ax_median.set_title(
            f"Median Fit (R²={r2_sorted.iloc[len(r2_sorted) // 2]['r_squared']:.3f})"
        )

        # Worst fit example
        ax_worst = fig.add_subplot(gs[1, 3])
        fm.fit(freqs, psds[int(worst_idx)])
        plot_fit_custom(fm, ax_worst)
        ax_worst.set_title(
            f"Worst Successful Fit (R²={r2_sorted.iloc[-1]['r_squared']:.3f})"
        )

    # 7. Brain maps of parameters (if requested)
    if plot_brain and subjects_dir is not None:
        # Create source estimates for visualization
        vertices = stc_psd.vertices

        # A) Exponent brain map
        exponent_data = np.ones(total_vertices) * np.nan
        for _, row in success_df.iterrows():
            exponent_data[int(row["vertex"])] = row["exponent"]

        exponent_stc = mne.SourceEstimate(
            exponent_data[:, np.newaxis], vertices=vertices, tmin=0, tstep=1
        )

        # Plot exponent brain map
        brain_exp = exponent_stc.plot(
            subject=subject,
            surface="pial",
            hemi="both",
            colormap="viridis",
            clim=dict(
                kind="value",
                lims=[
                    np.nanpercentile(exponent_data, 5),
                    np.nanpercentile(exponent_data, 50),
                    np.nanpercentile(exponent_data, 95),
                ],
            ),
            subjects_dir=subjects_dir,
            title="Aperiodic Exponent",
            background="white",
            size=(800, 600),
        )

        # Save the brain visualization
        brain_exp.save_image(
            os.path.join(output_dir, f"{subject_id}_exponent_brain.png")
        )

        # Add the brain image to the figure
        img = plt.imread(os.path.join(output_dir, f"{subject_id}_exponent_brain.png"))
        ax_brain_exp = fig.add_subplot(gs[2, :2])
        ax_brain_exp.imshow(img)
        ax_brain_exp.set_title("Aperiodic Exponent Brain Map")
        ax_brain_exp.axis("off")

        # B) Peak frequency brain map (if available)
        if peaks_df is not None:
            peaks_success = peaks_df[peaks_df["status"] == "SUCCESS"]
            if len(peaks_success) > 0:
                peak_data = np.ones(total_vertices) * np.nan
                for _, row in peaks_success.iterrows():
                    peak_data[int(row["vertex"])] = row["peak_freq"]

                peak_stc = mne.SourceEstimate(
                    peak_data[:, np.newaxis], vertices=vertices, tmin=0, tstep=1
                )

                # Plot peak frequency brain map
                brain_peak = peak_stc.plot(
                    subject=subject,
                    surface="pial",
                    hemi="both",
                    colormap="plasma",
                    clim=dict(
                        kind="value",
                        lims=[
                            np.nanpercentile(peak_data, 5),
                            np.nanpercentile(peak_data, 50),
                            np.nanpercentile(peak_data, 95),
                        ],
                    ),
                    subjects_dir=subjects_dir,
                    title="Peak Frequency",
                    background="white",
                    size=(800, 600),
                )

                # Save the brain visualization
                brain_peak.save_image(
                    os.path.join(output_dir, f"{subject_id}_peak_freq_brain.png")
                )

                # Add the brain image to the figure
                img = plt.imread(
                    os.path.join(output_dir, f"{subject_id}_peak_freq_brain.png")
                )
                ax_brain_peak = fig.add_subplot(gs[2, 2:])
                ax_brain_peak.imshow(img)
                ax_brain_peak.set_title("Peak Frequency Brain Map")
                ax_brain_peak.axis("off")

    # Adjust layout and save
    plt.tight_layout()
    scale_type = "log" if use_log else "linear"
    fig.savefig(
        os.path.join(output_dir, f"{subject_id}_fooof_summary_{scale_type}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(output_dir, f"{subject_id}_fooof_summary_{scale_type}.pdf"),
        bbox_inches="tight",
    )

    print(
        f"Visualization saved to {os.path.join(output_dir, f'{subject_id}_fooof_summary_{scale_type}.png')}"
    )

    return fig


def calculate_fooof_periodic(
    stc,
    freq_bands=None,
    n_jobs=10,
    output_dir=None,
    subject_id=None,
    aperiodic_mode="knee",
):
    """
    Calculate FOOOF periodic parameters from source-localized data and save results.

    Parameters
    ----------
    stc : instance of SourceEstimate
        The source time course containing spectral data
    freq_bands : dict | None
        Dictionary of frequency bands to analyze, e.g., {'alpha': (8, 13)}
        If None, will use default bands: delta, theta, alpha, beta, gamma
    n_jobs : int
        Number of parallel jobs to use for computation
    output_dir : str | None
        Directory to save output files
    subject_id : str | None
        Subject identifier for file naming
    aperiodic_mode : str
        Aperiodic mode for FOOOF ('fixed' or 'knee')

    Returns
    -------
    periodic_df : DataFrame
        DataFrame containing periodic parameters for each vertex and frequency band
    file_path : str
        Path to the saved data file
    """

    if not FOOOF_AVAILABLE:
        raise ImportError(
            "FOOOF is required for this function. Install with 'pip install fooof'"
        )

    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    if subject_id is None:
        subject_id = "unknown_subject"

    if freq_bands is None:
        freq_bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 45),
        }

    print(f"Calculating FOOOF oscillatory parameters for {subject_id}...")

    # Get data from stc
    if hasattr(stc, "data") and hasattr(stc, "times"):
        # Assuming stc.data contains PSDs and stc.times contains frequencies
        psds = stc.data
        freqs = stc.times
    else:
        raise ValueError(
            "Input stc must have 'data' and 'times' attributes with PSDs and frequencies"
        )

    # Determine full frequency range
    freq_range = (
        min([band[0] for band in freq_bands.values()]),
        max([band[1] for band in freq_bands.values()]),
    )

    # Check if frequencies are within the specified range
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if not np.any(freq_mask):
        raise ValueError(
            f"No frequencies found within the specified range {freq_range}"
        )

    # Trim data to specified frequency range
    freqs_to_fit = freqs[freq_mask]
    psds_to_fit = psds[:, freq_mask]

    n_vertices = psds.shape[0]
    print(
        f"Processing FOOOF analysis for {n_vertices} vertices and {len(freq_bands)} frequency bands..."
    )

    # FOOOF parameters
    fooof_params = {
        "peak_width_limits": [1, 12.0],
        "max_n_peaks": 6,
        "min_peak_height": 0.0,
        "peak_threshold": 2.0,
        "aperiodic_mode": aperiodic_mode,
        "verbose": False,
    }

    # Function to process a batch of vertices
    def process_batch(vertices):
        # Extract data for these vertices
        batch_psds = psds_to_fit[vertices, :]

        # Create FOOOF model and fit
        fg = FOOOFGroup(**fooof_params)
        fg.fit(freqs_to_fit, batch_psds)

        # Extract periodic parameters for each frequency band
        results = []

        for i, vertex_idx in enumerate(vertices):
            for band_name, band_range in freq_bands.items():
                # Get FOOOF model for this vertex
                fm = fg.get_fooof(i)

                # Extract peak in this band
                peak_params = get_band_peak_fm(fm, band_range, select_highest=True)

                if peak_params is not None:
                    cf, pw, bw = peak_params
                else:
                    cf, pw, bw = np.nan, np.nan, np.nan

                results.append(
                    {
                        "vertex": vertex_idx,
                        "band": band_name,
                        "center_frequency": cf,
                        "power": pw,
                        "bandwidth": bw,
                    }
                )

        # Clear memory
        del fg, batch_psds
        gc.collect()

        return results

    # Process in batches to manage memory
    batch_size = 2000  # Adjust based on memory constraints
    vertex_batches = []

    for i in range(0, n_vertices, batch_size):
        vertex_batches.append(range(i, min(i + batch_size, n_vertices)))

    print(f"Processing {len(vertex_batches)} batches with {n_jobs} parallel jobs...")

    # Run in parallel
    all_results = Parallel(n_jobs=n_jobs)(
        delayed(process_batch)(batch) for batch in vertex_batches
    )

    # Flatten results
    flat_results = [item for sublist in all_results for item in sublist]

    # Convert to DataFrame
    periodic_df = pd.DataFrame(flat_results)

    # Add subject_id
    periodic_df.insert(0, "subject", subject_id)

    # Save results
    file_path = os.path.join(output_dir, f"{subject_id}_fooof_periodic.parquet")
    periodic_df.to_csv(
        os.path.join(output_dir, f"{subject_id}_fooof_periodic.csv"), index=False
    )
    periodic_df.to_parquet(file_path)

    print(f"Saved FOOOF periodic parameters to {file_path}")

    return periodic_df, file_path


def calculate_vertex_peak_frequencies(
    stc,
    freq_range=(6, 12),
    alpha_range=(6, 12),
    n_jobs=10,
    output_dir=None,
    subject_id=None,
    smoothing_method="savitzky_golay",
):
    """
    Calculate peak frequencies at the vertex level across the source space.

    Parameters
    ----------
    stc : instance of SourceEstimate
        The source time course containing spectral data
    freq_range : tuple
        Frequency range for analysis (default: 6-12 Hz)
    alpha_range : tuple
        Alpha band range for peak finding (default: 6-12 Hz)
    n_jobs : int
        Number of parallel jobs to use for computation
    output_dir : str | None
        Directory to save output files
    subject_id : str | None
        Subject identifier for file naming
    smoothing_method : str
        Method for spectral smoothing ('savitzky_golay', 'moving_average', 'gaussian', 'median')

    Returns
    -------
    peaks_df : DataFrame
        DataFrame containing peak frequencies and analysis parameters for each vertex
    file_path : str
        Path to the saved data file
    """

    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    if subject_id is None:
        subject_id = "unknown_subject"

    print(f"Calculating vertex-level peak frequencies for {subject_id}...")

    # Get data from stc
    if hasattr(stc, "data") and hasattr(stc, "times"):
        # Assuming stc.data contains PSDs and stc.times contains frequencies
        psds = stc.data
        freqs = stc.times
    else:
        raise ValueError(
            "Input stc must have 'data' and 'times' attributes with PSDs and frequencies"
        )

    # Check if frequencies are within the specified range
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if not np.any(freq_mask):
        raise ValueError(
            f"No frequencies found within the specified range {freq_range}"
        )

    # Define Gaussian function for peak fitting
    def gaussian(x, a, x0, sigma):
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))

    # Define function to model 1/f trend
    def model_1f_trend(frequencies, powers):
        log_freq = np.log10(frequencies)
        log_power = np.log10(powers)
        slope, intercept, _, _, _ = stats.linregress(log_freq, log_power)
        return slope * log_freq + intercept

    # Define smoothing function
    def enhanced_smooth_spectrum(spectrum, method=smoothing_method):
        if method == "moving_average":
            window_size = 3
            return np.convolve(
                spectrum, np.ones(window_size) / window_size, mode="same"
            )

        elif method == "gaussian":
            window_size = 3
            sigma = 1.0
            gaussian_window = np.exp(
                -((np.arange(window_size) - window_size // 2) ** 2) / (2 * sigma**2)
            )
            gaussian_window /= np.sum(gaussian_window)
            return np.convolve(spectrum, gaussian_window, mode="same")

        elif method == "savitzky_golay":
            window_length = 5
            poly_order = 2
            return savgol_filter(spectrum, window_length, poly_order)

        elif method == "median":
            window_size = 3
            return np.array(
                [
                    np.median(
                        spectrum[
                            max(0, i - window_size // 2) : min(
                                len(spectrum), i + window_size // 2 + 1
                            )
                        ]
                    )
                    for i in range(len(spectrum))
                ]
            )

        else:
            return spectrum

    # Define function to find alpha peak using Dickinson method for a single vertex
    def dickinson_method_vertex(powers, vertex_idx):
        log_powers = np.log10(powers)
        log_trend = model_1f_trend(freqs, powers)
        detrended_log_powers = enhanced_smooth_spectrum(log_powers - log_trend)

        # Focus on alpha range
        alpha_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        alpha_freqs = freqs[alpha_mask]
        alpha_powers = detrended_log_powers[alpha_mask]

        if len(alpha_freqs) == 0:
            return {
                "vertex": vertex_idx,
                "peak_freq": np.nan,
                "peak_power": np.nan,
                "r_squared": np.nan,
                "status": "NO_DATA_IN_RANGE",
            }

        # Find peaks in the detrended spectrum
        peaks, _ = find_peaks(alpha_powers, width=1)

        if len(peaks) == 0:
            return {
                "vertex": vertex_idx,
                "peak_freq": np.nan,
                "peak_power": np.nan,
                "r_squared": np.nan,
                "status": "NO_PEAKS_FOUND",
            }

        # Sort peaks by prominence
        peak_prominences = alpha_powers[peaks] - np.min(alpha_powers)
        sorted_peaks = [
            p for _, p in sorted(zip(peak_prominences, peaks), reverse=True)
        ]

        # Try to fit Gaussian to each peak, starting with the most prominent
        for peak_idx in sorted_peaks:
            peak_freq = alpha_freqs[peak_idx]
            if alpha_range[0] <= peak_freq <= alpha_range[1]:
                try:
                    p0 = [alpha_powers[peak_idx], peak_freq, 0.2]
                    popt, pcov = curve_fit(
                        gaussian, alpha_freqs, alpha_powers, p0=p0, maxfev=1000
                    )

                    if alpha_range[0] <= popt[1] <= alpha_range[1]:
                        # Calculate R-squared for the fit
                        fitted_curve = gaussian(alpha_freqs, *popt)
                        ss_tot = np.sum((alpha_powers - np.mean(alpha_powers)) ** 2)
                        ss_res = np.sum((alpha_powers - fitted_curve) ** 2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                        return {
                            "vertex": vertex_idx,
                            "peak_freq": popt[1],
                            "peak_power": popt[0],
                            "peak_width": popt[2],
                            "r_squared": r_squared,
                            "status": "SUCCESS",
                        }
                except Exception:
                    continue

        # If no valid fit found, use the max peak in the alpha range
        alpha_range_mask = (alpha_freqs >= alpha_range[0]) & (
            alpha_freqs <= alpha_range[1]
        )
        if np.any(alpha_range_mask):
            max_idx = np.argmax(alpha_powers[alpha_range_mask])
            max_peak_freq = alpha_freqs[alpha_range_mask][max_idx]
            max_peak_power = alpha_powers[alpha_range_mask][max_idx]

            return {
                "vertex": vertex_idx,
                "peak_freq": max_peak_freq,
                "peak_power": max_peak_power,
                "peak_width": np.nan,
                "r_squared": np.nan,
                "status": "MAX_PEAK_USED",
            }
        else:
            return {
                "vertex": vertex_idx,
                "peak_freq": np.nan,
                "peak_power": np.nan,
                "r_squared": np.nan,
                "status": "NO_VALID_PEAK",
            }

    # Process vertices in batches for memory efficiency
    def process_batch(vertex_indices):
        batch_results = []
        for i in vertex_indices:
            result = dickinson_method_vertex(psds[i], i)
            batch_results.append(result)
        return batch_results

    n_vertices = psds.shape[0]
    print(f"Processing peak frequencies for {n_vertices} vertices...")

    # Create batches of vertices
    batch_size = 2000  # Adjust based on memory constraints
    vertex_batches = []

    for i in range(0, n_vertices, batch_size):
        vertex_batches.append(range(i, min(i + batch_size, n_vertices)))

    # Process batches in parallel
    all_results = Parallel(n_jobs=n_jobs)(
        delayed(process_batch)(batch) for batch in vertex_batches
    )

    # Flatten results
    flat_results = [item for sublist in all_results for item in sublist]

    # Create DataFrame
    peaks_df = pd.DataFrame(flat_results)

    # Add metadata
    peaks_df["subject"] = subject_id
    peaks_df["freq_range"] = f"{freq_range[0]}-{freq_range[1]}"
    peaks_df["alpha_range"] = f"{alpha_range[0]}-{alpha_range[1]}"
    peaks_df["analysis_date"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save results
    file_path = os.path.join(
        output_dir, f"{subject_id}_vertex_peak_frequencies.parquet"
    )
    peaks_df.to_csv(
        os.path.join(output_dir, f"{subject_id}_vertex_peak_frequencies.csv"),
        index=False,
    )
    peaks_df.to_parquet(file_path)

    print(f"Saved vertex-level peak frequencies to {file_path}")

    # Create summary statistics
    success_rate = (peaks_df["status"] == "SUCCESS").mean() * 100
    avg_peak = peaks_df.loc[peaks_df["peak_freq"].notna(), "peak_freq"].mean()

    print(
        f"Analysis complete: {success_rate:.1f}% of vertices with successful Gaussian fits"
    )
    print(f"Average peak frequency: {avg_peak:.2f} Hz")

    return peaks_df, file_path


def visualize_peak_frequencies(
    peaks_df,
    stc_template,
    subjects_dir=None,
    subject="fsaverage",
    output_dir=None,
    colormap="viridis",
    vmin=None,
    vmax=None,
):
    """
    Visualize peak frequencies on a brain surface.

    Parameters
    ----------
    peaks_df : DataFrame
        DataFrame with peak frequency results from calculate_vertex_peak_frequencies
    stc_template : instance of SourceEstimate
        Template source estimate to use for visualization
    subjects_dir : str | None
        Path to FreeSurfer subjects directory
    subject : str
        Subject name (default: 'fsaverage')
    output_dir : str | None
        Directory to save output files
    colormap : str
        Colormap for visualization
    vmin, vmax : float | None
        Minimum and maximum values for color scaling

    Returns
    -------
    brain : instance of Brain
        The visualization object
    """

    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    vertices = peaks_df["vertex"].values
    peak_freqs = peaks_df["peak_freq"].values

    # Set default color limits if not provided
    if vmin is None:
        vmin = np.nanpercentile(peak_freqs, 5)
    if vmax is None:
        vmax = np.nanpercentile(peak_freqs, 95)

    # Create a data array of the right size, initialized with NaN
    data = np.ones_like(stc_template.data[:, 0]) * np.nan

    # Fill in the peak frequency values
    for vertex, freq in zip(vertices, peak_freqs):
        if not np.isnan(freq):
            data[vertex] = freq

    # Create a new SourceEstimate with peak frequency data
    stc_viz = mne.SourceEstimate(
        data[:, np.newaxis], vertices=stc_template.vertices, tmin=0, tstep=1
    )

    # Visualize
    brain = stc_viz.plot(
        subject=subject,
        surface="pial",
        hemi="both",
        colormap=colormap,
        clim=dict(kind="value", lims=[vmin, (vmin + vmax) / 2, vmax]),
        subjects_dir=subjects_dir,
        title="Peak Frequency Distribution",
    )

    # Add a colorbar
    brain.add_annotation("aparc", borders=2, alpha=0.7)

    # Save images
    brain.save_image(os.path.join(output_dir, "peak_frequency_lateral.png"))
    brain.show_view("medial")
    brain.save_image(os.path.join(output_dir, "peak_frequency_medial.png"))

    return brain


def convert_stc_to_eeg(
    stc, subject="fsaverage", subjects_dir=None, output_dir=None, subject_id=None
):
    """
    Convert a source estimate (stc) to EEG SET format with DK atlas regions as channels.

    Parameters
    ----------
    stc : instance of SourceEstimate
        The source time course to convert
    subject : str
        Subject name in FreeSurfer subjects directory (default: 'fsaverage')
    subjects_dir : str | None
        Path to FreeSurfer subjects directory
    output_dir : str | None
        Directory to save output files
    subject_id : str | None
        Subject identifier for file naming

    Returns
    -------
    raw_eeg : instance of mne.io.Raw
        The converted EEG data in MNE Raw format
    eeglab_out_file : str
        Path to the saved EEGLAB .set file
    """

    # Set up paths
    if subjects_dir is None:
        subjects_dir = os.path.dirname(fetch_fsaverage())

    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    if subject_id is None:
        subject_id = "stc_to_eeg"

    print(f"Converting stc to EEG format for {subject_id}...")

    # Load the parcellation labels from DK atlas
    labels = mne.read_labels_from_annot(
        subject, parc="aparc", subjects_dir=subjects_dir
    )
    labels = [label for label in labels if "unknown" not in label.name]

    # Extract time series for each label
    label_ts = mne.extract_label_time_course(
        stc, labels, src=None, mode="mean", verbose=True
    )

    # Get data properties
    n_regions = len(labels)
    sfreq = (
        1.0 / stc.tstep if hasattr(stc, "tstep") else 1000.0
    )  # Default 1000Hz if not available
    ch_names = [label.name for label in labels]

    # Create an array of channel positions - we'll use spherical coordinates
    # based on region centroids
    ch_pos = {}
    for i, label in enumerate(labels):
        # Extract centroid of the label
        if hasattr(label, "pos") and len(label.pos) > 0:
            centroid = np.mean(label.pos, axis=0)
        else:
            # If no positions available, create a point on a unit sphere
            # We'll distribute them evenly by using golden ratio
            phi = (1 + np.sqrt(5)) / 2
            idx = i + 1
            theta = 2 * np.pi * idx / phi**2
            phi = np.arccos(1 - 2 * ((idx % phi**2) / phi**2))
            centroid = (
                np.array(
                    [
                        np.sin(phi) * np.cos(theta),
                        np.sin(phi) * np.sin(theta),
                        np.cos(phi),
                    ]
                )
                * 0.1
            )  # Scaled to approximate head radius

        # Store in dictionary
        ch_pos[label.name] = centroid

    # Create MNE Info object with channel information
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_regions)

    # Update channel positions
    for idx, ch_name in enumerate(ch_names):
        info["chs"][idx]["loc"][:3] = ch_pos[ch_name]

    # Create Raw object with label time courses as data
    raw_eeg = mne.io.RawArray(np.array(label_ts), info, verbose=True)

    # Add annotations if available in the original stc
    if hasattr(stc, "annotations"):
        raw_eeg.set_annotations(stc.annotations)

    # Save to various formats
    eeglab_out_file = os.path.join(output_dir, f"{subject_id}_dk_regions.set")
    raw_eeg.export(eeglab_out_file, fmt="eeglab", overwrite=True)

    print(
        f"Saved EEG SET file with {n_regions} channels (DK regions) to {eeglab_out_file}"
    )

    return raw_eeg, eeglab_out_file


def convert_stc_list_to_eeg(
    stc_list,
    subject="fsaverage",
    subjects_dir=None,
    output_dir=None,
    subject_id=None,
    events=None,
    event_id=None,
):
    """
    Convert a list of source estimates (stc) to EEG SET format with DK atlas regions as channels.

    Parameters
    ----------
    stc_list : list of SourceEstimate
        List of source time courses to convert, representing different trials or segments
    subject : str
        Subject name in FreeSurfer subjects directory (default: 'fsaverage')
    subjects_dir : str | None
        Path to FreeSurfer subjects directory
    output_dir : str | None
        Directory to save output files
    subject_id : str | None
        Subject identifier for file naming
    events : array, shape (n_events, 3) | None
        Events array to use when creating the epochs. If None, will create generic events.
    event_id : dict | None
        Dictionary mapping event types to IDs. If None, will use {1: 'event'}.

    Returns
    -------
    epochs : instance of mne.Epochs
        The converted EEG data in MNE Epochs format
    eeglab_out_file : str
        Path to the saved EEGLAB .set file
    """

    import pandas as pd

    # Set up paths
    if subjects_dir is None:
        subjects_dir = os.path.dirname(fetch_fsaverage())

    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    if subject_id is None:
        subject_id = "stc_to_eeg"

    print(
        f"Converting {len(stc_list)} source estimates to EEG epochs format for {subject_id}..."
    )

    # Check if all stc objects have the same structure
    n_times_list = [stc.data.shape[1] for stc in stc_list]
    if len(set(n_times_list)) > 1:
        raise ValueError(
            f"Source estimates have different time dimensions: {n_times_list}"
        )

    # Load the parcellation labels from DK atlas
    labels = mne.read_labels_from_annot(
        subject, parc="aparc", subjects_dir=subjects_dir
    )
    labels = [label for label in labels if "unknown" not in label.name]

    # Extract time series for each label for each stc
    all_label_ts = []
    for stc in stc_list:
        # Extract label time courses for this stc
        label_ts = mne.extract_label_time_course(
            stc, labels, src=None, mode="mean", verbose=False
        )
        all_label_ts.append(label_ts)

    # Stack to get 3D array (n_epochs, n_regions, n_times)
    label_data = np.array(all_label_ts)

    # Get data properties from the first stc
    n_epochs = len(stc_list)
    n_regions = len(labels)
    sfreq = 1.0 / stc_list[0].tstep
    ch_names = [label.name for label in labels]

    # Create an array of channel positions based on region centroids
    ch_pos = {}
    for i, label in enumerate(labels):
        # Extract centroid of the label
        if hasattr(label, "pos") and len(label.pos) > 0:
            centroid = np.mean(label.pos, axis=0)
        else:
            # If no positions available, create a point on a unit sphere
            phi = (1 + np.sqrt(5)) / 2
            idx = i + 1
            theta = 2 * np.pi * idx / phi**2
            phi = np.arccos(1 - 2 * ((idx % phi**2) / phi**2))
            centroid = (
                np.array(
                    [
                        np.sin(phi) * np.cos(theta),
                        np.sin(phi) * np.sin(theta),
                        np.cos(phi),
                    ]
                )
                * 0.1
            )  # Scaled to approximate head radius

        # Store in dictionary
        ch_pos[label.name] = centroid

    # Create MNE Info object with channel information
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_regions)

    # Update channel positions
    for idx, ch_name in enumerate(ch_names):
        info["chs"][idx]["loc"][:3] = ch_pos[ch_name]

    # Create events array if not provided
    if events is None:
        events = np.array([[i, 0, 1] for i in range(n_epochs)])

    # Create event_id dictionary if not provided
    if event_id is None:
        event_id = {"event": 1}

    # Create MNE Epochs object from the extracted label time courses
    tmin = stc_list[0].tmin
    epochs = mne.EpochsArray(
        label_data, info, events=events, event_id=event_id, tmin=tmin
    )

    # Save to EEGLAB format
    eeglab_out_file = os.path.join(output_dir, f"{subject_id}_dk_regions.set")
    epochs.export(eeglab_out_file, fmt="eeglab")

    print(
        f"Saved EEG SET file with {n_regions} channels (DK regions) to {eeglab_out_file}"
    )

    # Create and save a montage file to help with visualization
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    montage_file = os.path.join(output_dir, f"{subject_id}_dk_montage.fif")
    montage.save(montage_file)

    print(f"Saved montage file to {montage_file}")

    # Export additional metadata to help with interpretation
    region_info = {
        "names": ch_names,
        "hemisphere": ["lh" if "-lh" in name else "rh" for name in ch_names],
        "centroid_x": [ch_pos[name][0] for name in ch_names],
        "centroid_y": [ch_pos[name][1] for name in ch_names],
        "centroid_z": [ch_pos[name][2] for name in ch_names],
    }

    info_file = os.path.join(output_dir, f"{subject_id}_region_info.csv")
    pd.DataFrame(region_info).to_csv(info_file, index=False)

    print(f"Saved region information to {info_file}")

    return epochs, eeglab_out_file
