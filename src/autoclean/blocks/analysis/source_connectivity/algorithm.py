"""Source-level functional connectivity and graph theory algorithms.

This module contains scientifically validated functions for calculating functional
connectivity between brain regions from source-localized EEG data and computing graph
theory metrics to characterize brain network properties.

These functions are EXACT copies from the AutoCleanEEG pipeline source.py module
and must NOT be modified to maintain reproducibility of scientific results.

The connectivity calculation uses multiple methods (WPLI, PLV, coherence, PLI, AEC)
across frequency bands (delta through gamma) and includes comprehensive graph theory
analysis (clustering, efficiency, modularity, small-worldness).

References
----------
Vinck M, et al. (2011). The pairwise phase consistency: a bias-free measure of
rhythmic neuronal synchronization. NeuroImage, 51(1), 112-122.

Rubinov M & Sporns O (2010). Complex network measures of brain connectivity: Uses
and interpretations. NeuroImage, 52(3), 1059-1069.
"""

import itertools
import logging
import os

import mne
import numpy as np
from mne.filter import filter_data
from mne_connectivity import spectral_connectivity_time
from scipy.signal import hilbert

# Optional imports with availability flags
try:
    import networkx as nx

    NETWORK_ANALYSIS_AVAILABLE = True
except ImportError:
    NETWORK_ANALYSIS_AVAILABLE = False

try:
    from bctpy import charpath, clustering_coef_wu, efficiency_wei

    BCTPY_AVAILABLE = True
except ImportError:
    BCTPY_AVAILABLE = False


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
