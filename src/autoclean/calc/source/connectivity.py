"""Connectivity and network metric utilities."""
from __future__ import annotations

import itertools
import logging
import os
import tempfile
import time
import traceback
from typing import Sequence

import mne
import numpy as np
import pandas as pd
from mne import SourceEstimate
from mne.filter import filter_data
from mne_connectivity import spectral_connectivity_time
from scipy.signal import hilbert

from ._compat import (
    NETWORK_ANALYSIS_AVAILABLE,
    charpath,
    clustering_coef_wu,
    efficiency_wei,
    louvain_communities,
    modularity,
    nx,
)
from ._utils import ensure_stc_list

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
    max_duration: float | None = 80,
):
    """
    Calculate connectivity metrics between brain regions from source-localized data.

    Parameters
    ----------
    stc : SourceEstimate | Sequence[SourceEstimate]
        The source time course(s) to calculate connectivity from. Sequences
        are routed to :func:`calculate_source_connectivity_list` automatically.
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
    max_duration : float | None
        Optional cap (in seconds) applied when ``stc`` represents multiple
        epochs. ``None`` disables down-selection.

    Returns
    -------
    conn_df : DataFrame
        DataFrame containing connectivity values
    summary_path : str
        Path to the saved summary file

    """
    stc_list, multiple = ensure_stc_list(stc)
    if multiple:
        return calculate_source_connectivity_list(
            stc_list,
            labels=labels,
            subjects_dir=subjects_dir,
            subject=subject,
            n_jobs=n_jobs,
            output_dir=output_dir,
            subject_id=subject_id,
            sfreq=sfreq,
            epoch_length=epoch_length,
            n_epochs=n_epochs,
            max_duration=max_duration,
        )

    stc = stc_list[0]
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

    stc_list, _ = ensure_stc_list(stc_list)

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

    stc_list, _ = ensure_stc_list(stc_list)

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

__all__ = [
    'calculate_source_connectivity',
    'test_connectivity_function_list',
    'calculate_source_connectivity_list',
    'test_connectivity_function',
    'calculate_aec_connectivity',
]
