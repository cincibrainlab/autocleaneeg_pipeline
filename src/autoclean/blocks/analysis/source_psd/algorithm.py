"""Source-level power spectral density calculation algorithms.

This module contains scientifically validated functions for calculating power spectral
density (PSD) from source-localized EEG data with region-of-interest (ROI) averaging.

These functions are EXACT copies from the AutoCleanEEG pipeline source.py module
and must NOT be modified to maintain reproducibility of scientific results.

The PSD calculation uses Welch's method with adaptive windowing and parallel processing
for efficiency. Results are parcellated into anatomical ROIs using the Desikan-Killiany
atlas (68 cortical regions).

References
----------
Welch P (1967). The use of fast Fourier transform for the estimation of power spectra:
A method based on time averaging over short, modified periodograms. IEEE Transactions
on Audio and Electroacoustics, 15(2), 70-73.

Desikan RS, et al. (2006). An automated labeling system for subdividing the human
cerebral cortex on MRI scans into gyral based regions of interest. NeuroImage, 31(3), 968-980.
"""

import os
import time

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.gridspec import GridSpec
from scipy import signal


def calculate_roi_psd(
    data,
    segment_duration=80,
    n_jobs=4,
    output_dir=None,
    subject_id=None,
    generate_plots=True,
):
    """
    Optimized function to calculate PSD from 68-channel ROI EEG data.

    This function is designed for source-localized data that has already been
    converted to 68 Desikan-Killiany ROI channels. It's much faster than vertex-level
    PSD since it processes 68 channels instead of 20,484 vertices.

    Parameters
    ----------
    data : instance of Raw or Epochs
        The ROI EEG data (68 channels). Each channel represents one DK atlas region.
    segment_duration : float or None
        Duration in seconds to process. If None, processes the entire data.
        Default is 80 seconds for optimal balance of accuracy and performance.
    n_jobs : int
        Number of parallel jobs to use for computation (currently not used for 68 channels)
    output_dir : str | None
        Directory to save output files. If None, saves in current directory
    subject_id : str | None
        Subject identifier for file naming
    generate_plots : bool
        Whether to generate diagnostic PSD plots (default: True)

    Returns
    -------
    psd_df : DataFrame
        DataFrame containing ROI PSD values with columns:
        [subject, roi, hemisphere, frequency, psd]
    file_path : str
        Path to the saved parquet file
    """
    import pandas as pd

    # Start timing
    start_time = time.time()

    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    if subject_id is None:
        subject_id = "unknown_subject"

    # Determine data type
    is_raw = isinstance(data, mne.io.BaseRaw)
    is_epochs = isinstance(data, mne.BaseEpochs)

    if not (is_raw or is_epochs):
        raise TypeError(f"Data must be Raw or Epochs, got {type(data)}")

    # Get basic info
    sfreq = data.info["sfreq"]
    n_channels = len(data.ch_names)

    print(f"Processing {n_channels} ROI channels at {sfreq} Hz")

    # Determine available duration
    if is_raw:
        total_duration = data.times[-1] - data.times[0]
        print(f"Total available signal: {total_duration:.1f} seconds")
    else:
        epoch_duration = data.times[-1] - data.times[0]
        total_duration = epoch_duration * len(data)
        print(
            f"Total available signal: {total_duration:.1f} seconds ({len(data)} epochs of {epoch_duration:.1f}s each)"
        )

    # Select data segment if needed
    if segment_duration is not None and segment_duration < total_duration:
        if is_raw:
            # For Raw, crop from the middle
            start_time_crop = (total_duration - segment_duration) / 2
            end_time_crop = start_time_crop + segment_duration
            data_to_use = data.copy().crop(tmin=start_time_crop, tmax=end_time_crop)
            print(f"Using {segment_duration:.1f}s from middle of recording")
        else:
            # For Epochs, select middle epochs
            n_epochs_needed = int(np.ceil(segment_duration / epoch_duration))
            n_epochs_needed = min(n_epochs_needed, len(data))
            start_idx = (len(data) - n_epochs_needed) // 2
            data_to_use = data[start_idx : start_idx + n_epochs_needed]
            actual_duration = len(data_to_use) * epoch_duration
            print(
                f"Using {len(data_to_use)} epochs ({actual_duration:.1f}s) from middle of recording"
            )
    else:
        data_to_use = data
        if is_raw:
            print(f"Using all data ({total_duration:.1f}s)")
        else:
            print(f"Using all {len(data)} epochs ({total_duration:.1f}s)")

    # Define frequency parameters
    fmin = 0.5
    fmax = 45.0

    # Determine optimal window length for Welch's method
    if is_raw:
        # For Raw data, use the full available duration and ensure enough windows for averaging
        available_duration = data_to_use.times[-1] - data_to_use.times[0]
        # Prefer 4-second windows, but ensure at least 8 windows fit in available duration
        window_length = int(min(4 * sfreq, available_duration * sfreq / 8))
    else:
        # For Epochs, MNE's compute_psd() processes each epoch independently
        # We average PSDs across epochs, so we just need the window to fit within one epoch
        # Use the maximum window that fits in a single epoch for best frequency resolution
        epoch_duration = data_to_use.times[-1] - data_to_use.times[0]
        epoch_samples = int(epoch_duration * sfreq)
        # Prefer 4-second windows, but can't exceed epoch length
        window_length = int(min(4 * sfreq, epoch_samples))
    n_overlap = window_length // 2  # 50% overlap

    print(f"Using {window_length / sfreq:.2f}s windows with 50% overlap")

    # Define frequency bands
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

    # Calculate PSD for each channel using MNE's compute_psd
    print("Calculating PSD for each ROI channel...")

    if is_raw:
        # Use MNE's compute_psd for Raw
        psd_data = data_to_use.compute_psd(
            method="welch",
            fmin=fmin,
            fmax=fmax,
            n_fft=window_length,
            n_overlap=n_overlap,
            n_jobs=1,  # Single core sufficient for 68 channels
        )
        freqs = psd_data.freqs
        psd_values = psd_data.get_data()  # Shape: (n_channels, n_freqs)
    else:
        # Use MNE's compute_psd for Epochs (averages across epochs)
        psd_data = data_to_use.compute_psd(
            method="welch",
            fmin=fmin,
            fmax=fmax,
            n_fft=window_length,
            n_overlap=n_overlap,
            n_jobs=1,
        )
        freqs = psd_data.freqs
        psd_values = psd_data.get_data()  # Shape: (n_epochs, n_channels, n_freqs)
        # Average across epochs
        psd_values = np.mean(psd_values, axis=0)  # Shape: (n_channels, n_freqs)

    print(f"PSD calculation complete in {time.time() - start_time:.1f} seconds")

    # Build DataFrame with ROI information
    roi_psds = []
    band_psds = []

    for ch_idx, ch_name in enumerate(data.ch_names):
        # Parse channel name to extract ROI and hemisphere
        # Format from eeg2source: "lh_roi_name" or "rh_roi_name"
        if ch_name.startswith("lh_"):
            hemisphere = "lh"
            roi_name = (
                ch_name[3:] + "-lh"
            )  # Convert "lh_superior_frontal" to "superior_frontal-lh"
        elif ch_name.startswith("rh_"):
            hemisphere = "rh"
            roi_name = ch_name[3:] + "-rh"
        else:
            # Fallback: assume format is already "roiName-hemisphere"
            if "-lh" in ch_name:
                hemisphere = "lh"
                roi_name = ch_name
            elif "-rh" in ch_name:
                hemisphere = "rh"
                roi_name = ch_name
            else:
                print(f"Warning: Cannot parse channel name '{ch_name}', skipping")
                continue

        # Get PSD for this channel
        channel_psd = psd_values[ch_idx]

        # Add frequency-resolved PSD data
        for freq_idx, freq in enumerate(freqs):
            roi_psds.append(
                {
                    "subject": subject_id,
                    "roi": roi_name,
                    "hemisphere": hemisphere,
                    "frequency": freq,
                    "psd": channel_psd[freq_idx],
                }
            )

        # Calculate band powers
        for band_name, (band_min, band_max) in bands.items():
            band_mask = (freqs >= band_min) & (freqs < band_max)
            if np.sum(band_mask) > 0:
                band_power = np.mean(channel_psd[band_mask])
            else:
                band_power = 0

            band_psds.append(
                {
                    "subject": subject_id,
                    "roi": roi_name,
                    "hemisphere": hemisphere,
                    "band": band_name,
                    "band_start_hz": band_min,
                    "band_end_hz": band_max,
                    "power": band_power,
                }
            )

    # Create DataFrames
    psd_df = pd.DataFrame(roi_psds)
    band_df = pd.DataFrame(band_psds)

    # Save to files
    file_path = os.path.join(output_dir, f"{subject_id}_roi_psd.parquet")
    psd_df.to_parquet(file_path)

    csv_path = os.path.join(output_dir, f"{subject_id}_roi_bands.csv")
    band_df.to_csv(csv_path, index=False)

    print(f"Saved ROI PSD to {file_path}")
    print(f"Saved frequency band summary to {csv_path}")

    # Create visualizations if requested
    if generate_plots:
        print("Creating summary visualizations...")

        # Group by band and hemisphere
        band_summary = (
            band_df.groupby(["band", "hemisphere"])["power"].mean().reset_index()
        )
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
        plt.close()

        # Log scale version
        plt.figure(figsize=(12, 8))
        np.log10(pivot_df).plot(kind="bar", ax=plt.gca())
        plt.title(f"Log10 Mean Band Power by Hemisphere - {subject_id}")
        plt.ylabel("Log10 Power")
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{subject_id}_hemisphere_bands_log.png"))
        plt.close()

    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.1f} seconds")
    print(
        f"ROI-optimized PSD: {n_channels} channels (vs 20,484 vertices in vertex-level mode)"
    )

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
