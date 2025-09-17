"""Vertex-level spectral utilities."""
from __future__ import annotations

import logging
import os
from typing import Dict, Sequence

import h5py
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mne import SourceEstimate
from tqdm import tqdm

from ._utils import coerce_stc_to_single, ensure_stc_list
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

    stc_list, _ = ensure_stc_list(stc_list)

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
    stc: SourceEstimate | Sequence[SourceEstimate],
    bands=None,
    n_jobs=10,
    output_dir=None,
    subject_id=None,
):
    """
    Calculate spectral power at the vertex level across the entire brain.

    Parameters
    ----------
    stc : SourceEstimate | Sequence[SourceEstimate]
        The source time course(s) to calculate power from. When a sequence is
        supplied, the specialised list implementation is used automatically.
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

    stc_list, multiple = ensure_stc_list(stc)
    if multiple:
        return calculate_vertex_level_spectral_power_list(
            stc_list,
            bands=bands,
            n_jobs=n_jobs,
            output_dir=output_dir,
            subject_id=subject_id,
        )

    stc = stc_list[0]

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
    stc: SourceEstimate | Sequence[SourceEstimate],
    fmin=1.0,
    fmax=45.0,
    n_jobs=10,
    output_dir=None,
    subject_id=None,
):
    """
    Calculate full power spectral density at the vertex level for FOOOF analysis.

    Parameters
    ----------
    stc : SourceEstimate | Sequence[SourceEstimate]
        The source time course(s) to calculate power from. Sequences are
        concatenated to preserve temporal ordering.
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

    stc, was_sequence = coerce_stc_to_single(stc)
    if was_sequence:
        print(
            "Concatenated multiple source estimates before computing vertex-level PSD."
        )

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

__all__ = [
    'calculate_vertex_level_spectral_power_list',
    'calculate_vertex_level_spectral_power',
    'apply_spatial_smoothing',
    'calculate_vertex_psd_for_fooof',
]
