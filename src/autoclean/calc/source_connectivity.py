"""Source-level functional connectivity and graph theory algorithms.

This module contains scientifically validated functions for calculating functional
connectivity between brain regions from source-localized EEG data and computing graph
theory metrics to characterize brain network properties.

This module is the single implementation of connectivity for both the sensor
and source paths. ``blocks/analysis/source_connectivity/algorithm.py`` delegates
to it. Estimates are band-specific (across-epoch multitaper with 2 Hz
bandwidth for spectral methods; continuous band-pass plus Hilbert for AEC),
so source-path values differ from older releases that used wide time-resolved
multitaper smoothing and per-epoch FIR filtering. Do not compare values
across those versions.

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
from typing import Mapping, Sequence

import mne
import numpy as np
import pandas as pd
from mne_connectivity import spectral_connectivity_epochs
from scipy.signal import butter, hilbert, sosfiltfilt

# Optional imports with availability flags
try:
    import networkx as nx

    NETWORK_ANALYSIS_AVAILABLE = True
except ImportError:
    NETWORK_ANALYSIS_AVAILABLE = False

try:
    from bct import (
        charpath,
        clustering_coef_wu,
        distance_wei,
        efficiency_wei,
        weight_conversion,
    )

    BCTPY_AVAILABLE = True
except ImportError:
    BCTPY_AVAILABLE = False


def _characteristic_path_length(weights: np.ndarray) -> float:
    """Mean shortest-path length with connection length = 1 / weight.

    ``weights`` must be non-negative; zero entries are non-edges. Unreachable
    node pairs are excluded from the mean.
    """
    if not np.any(np.asarray(weights) > 0):
        return float("nan")
    distances, _ = distance_wei(weight_conversion(weights, "lengths"))
    return float(charpath(distances, include_infinite=False)[0])


SPECTRAL_METHODS = ("wpli", "wpli2_debiased", "plv", "coh", "pli")
SUPPORTED_METHODS = SPECTRAL_METHODS + ("aec",)


# Fraction of an epoch discarded at each outer edge of a filtered span before
# AEC is computed (filter/Hilbert edge transients).
AEC_EDGE_FRACTION = 0.1


def _calculate_aec(spans, band_range, sfreq, edge_samples):
    """Amplitude envelope correlation matrix (n_ch, n_ch), zero diagonal.

    Each span is a contiguous (n_ch, n_samples) array. It is band-passed with a
    zero-phase Butterworth filter and Hilbert-transformed as a whole, then
    ``edge_samples`` are dropped at both outer ends (edge transients). The
    envelopes of all spans are pooled and correlated once. Because spans are
    filtered whole, epoch boundaries inside a span add no edge artifacts.
    """
    sos = butter(4, band_range, btype="bandpass", fs=sfreq, output="sos")
    kept = []
    for span in spans:
        env = np.abs(hilbert(sosfiltfilt(sos, span, axis=-1), axis=-1))
        if edge_samples > 0:
            env = env[:, edge_samples:-edge_samples]
        if env.shape[-1] > 1:
            kept.append(env)
    if not kept:
        raise ValueError("No data left for AEC after discarding filter edges")
    envelope = np.concatenate(kept, axis=-1)
    aec_matrix = np.corrcoef(envelope)
    np.fill_diagonal(aec_matrix, 0.0)
    return aec_matrix


# Multitaper full bandwidth (Hz) for across-epoch spectral estimates. About
# +/- 1 Hz of smoothing keeps estimates specific to the requested band; the
# estimate is pooled across epochs, so unconnected pairs have a low noise floor.
# Bands narrower than 2 * MT_BANDWIDTH Hz are rejected: the smoothing would be
# as wide as the band.
MT_BANDWIDTH = 2.0


def _validate_connectivity_inputs(epoched, names, sfreq, bands, methods, lowpass=None):
    if epoched.ndim != 3:
        raise ValueError(
            f"Epoched data must be (n_epochs, n_channels, n_times), got {epoched.shape}"
        )
    unknown = [m for m in methods if m not in SUPPORTED_METHODS]
    if unknown:
        raise ValueError(
            f"Unknown connectivity method(s) {unknown}; supported: {list(SUPPORTED_METHODS)}"
        )
    if epoched.shape[1] < 2:
        raise ValueError("Connectivity needs at least 2 channels")
    if len(names) != epoched.shape[1]:
        raise ValueError("Number of channel names does not match data")
    if epoched.shape[0] < 2:
        raise ValueError(
            f"Connectivity needs at least 2 epochs, got {epoched.shape[0]}"
        )
    if not np.all(np.isfinite(epoched)):
        raise ValueError("Data contains NaN or infinite values")
    nyquist = sfreq / 2.0
    for band, (low, high) in bands.items():
        if low >= high:
            raise ValueError(f"Band {band}: low ({low}) must be below high ({high})")
        if high - low < MT_BANDWIDTH:
            raise ValueError(
                f"Band {band}: width {high - low} Hz is below the {MT_BANDWIDTH} Hz "
                "spectral resolution; use a wider band"
            )
        if high >= nyquist:
            raise ValueError(
                f"Band {band}: high edge {high} Hz is at or above Nyquist "
                f"({nyquist} Hz). Raise or disable resample_step."
            )
        if lowpass is not None and lowpass < nyquist and high > lowpass:
            raise ValueError(
                f"Band {band}: high edge {high} Hz exceeds the data low-pass "
                f"({float(lowpass):g} Hz); raise the filtering h_freq above "
                f"{high} Hz."
            )


def _connectivity_from_epoched_array(
    epoched, names, sfreq, bands, methods, n_jobs=1, aec_spans=None, lowpass=None
):
    """Shared connectivity core.

    Parameters
    ----------
    epoched : ndarray, shape (n_epochs, n_channels, n_times)
    names : sequence of str
        Channel / ROI names, in row order of ``epoched``.
    sfreq : float
    bands : mapping of name -> (low, high) in Hz
    methods : sequence of str
        Any of wpli, plv, coh, pli, aec.
    aec_spans : list of ndarray (n_channels, n_samples) | None
        Contiguous data spans that contain the epochs. AEC filters each span
        whole. If None, each epoch is its own span (edges are then discarded).
    lowpass : float | None
        Data low-pass (``info["lowpass"]``); bands above it are rejected.

    Returns
    -------
    DataFrame with columns channel_a, channel_b, method, band, band_low,
    band_high, value, n_epochs, epoch_length_s. One row per unordered pair
    (a before b in ``names`` order) per method per band.
    """
    epoched = np.asarray(epoched, dtype=float)
    names = list(names)
    methods = list(methods)
    _validate_connectivity_inputs(epoched, names, sfreq, bands, methods, lowpass)

    n_ep, n_ch, n_times = epoched.shape
    epoch_length_s = n_times / sfreq
    pairs = list(itertools.combinations(range(n_ch), 2))
    spans = list(aec_spans) if aec_spans is not None else list(epoched)
    edge_samples = int(round(AEC_EDGE_FRACTION * n_times))
    rows = []
    for method in methods:
        for band, (low, high) in bands.items():
            try:
                if method == "aec":
                    con_matrix = _calculate_aec(spans, (low, high), sfreq, edge_samples)
                else:
                    con = spectral_connectivity_epochs(
                        epoched,
                        names=names,
                        method=method,
                        sfreq=sfreq,
                        mode="multitaper",
                        fmin=low,
                        fmax=high,
                        faverage=True,
                        mt_bandwidth=MT_BANDWIDTH,
                        n_jobs=n_jobs,
                        verbose=False,
                    )
                    con_matrix = con.get_data(output="dense").squeeze()
                if con_matrix.shape != (n_ch, n_ch):
                    raise ValueError(
                        f"Unexpected connectivity matrix shape {con_matrix.shape}, "
                        f"expected {(n_ch, n_ch)}"
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Connectivity method {method} failed for band {band}: {e}"
                ) from e
            for i, j in pairs:
                # spectral_connectivity_epochs fills the lower triangle; AEC is symmetric
                rows.append(
                    {
                        "channel_a": names[i],
                        "channel_b": names[j],
                        "method": method,
                        "band": band,
                        "band_low": float(low),
                        "band_high": float(high),
                        "value": float(con_matrix[j, i]),
                        "n_epochs": n_ep,
                        "epoch_length_s": epoch_length_s,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "channel_a",
            "channel_b",
            "method",
            "band",
            "band_low",
            "band_high",
            "value",
            "n_epochs",
            "epoch_length_s",
        ],
    )


def compute_sensor_connectivity(
    data,
    bands: Mapping[str, Sequence[float]],
    methods: Sequence[str] = ("wpli", "wpli2_debiased", "plv", "coh", "pli", "aec"),
    picks=None,
    epoch_length: float = 2.0,
    n_epochs: int | None = None,
) -> pd.DataFrame:
    """Channel-level connectivity for Raw or Epochs data.

    Raw data is cut into consecutive non-overlapping epochs of ``epoch_length``
    seconds with ``mne.make_fixed_length_epochs(reject_by_annotation=True)``,
    so epochs never overlap BAD_ annotated spans (``n_epochs=None`` keeps all;
    otherwise the first N). Epochs are used as-is (``epoch_length`` is taken
    from the data; ``n_epochs`` keeps the first N). ``picks`` selects channels
    by name or type (default: data channels). Channels in ``info["bads"]`` are
    always excluded, even if named in ``picks``.

    Spectral methods are estimated across epochs (multitaper, ~+/-1 Hz
    smoothing); ``wpli2_debiased`` has a near-zero floor for unconnected pairs.

    AEC is computed from band-passed, Hilbert-transformed contiguous spans
    (Raw) or per epoch with 10% edges discarded (Epochs input).

    Returns a long DataFrame: channel_a, channel_b, method, band, band_low,
    band_high, value, n_epochs (epochs actually used), epoch_length_s.
    """
    picked = data.copy().pick(picks if picks is not None else "data", exclude="bads")
    # MNE keeps explicitly named bad channels; drop them so bads never enter
    bads = [ch for ch in picked.ch_names if ch in picked.info["bads"]]
    if bads:
        picked.drop_channels(bads)
    names = list(picked.ch_names)
    sfreq = float(picked.info["sfreq"])
    lowpass = picked.info["lowpass"]
    aec_spans = None

    if isinstance(picked, mne.BaseEpochs):
        epoched = picked.get_data(copy=True)
        if n_epochs is not None:
            epoched = epoched[:n_epochs]
    else:
        samples = int(round(epoch_length * sfreq))
        if samples < 1:
            raise ValueError("epoch_length is shorter than one sample")
        epochs = mne.make_fixed_length_epochs(
            picked,
            duration=samples / sfreq,
            reject_by_annotation=True,
            preload=True,
            verbose=False,
        )
        epoched = epochs.get_data(copy=True)
        starts = epochs.events[:, 0] - picked.first_samp
        if n_epochs is not None:
            epoched = epoched[:n_epochs]
            starts = starts[:n_epochs]
        if epoched.shape[0] < 2:
            raise ValueError(
                f"Connectivity needs at least 2 epochs, only {epoched.shape[0]} "
                f"clean {epoch_length}-second epochs available"
            )
        # Group epochs that are back to back into contiguous spans for AEC
        raw_data = picked.get_data()
        aec_spans = []
        run_start = prev = starts[0]
        for st in starts[1:]:
            if st != prev + samples:
                aec_spans.append(raw_data[:, run_start : prev + samples])
                run_start = st
            prev = st
        aec_spans.append(raw_data[:, run_start : prev + samples])
    # MNE's multitaper needs 5 cycles of the band's low edge per epoch for a
    # reliable estimate. Checked for the sensor path only: the source path keeps
    # its default delta band and lets MNE warn instead.
    epoch_len_s = epoched.shape[-1] / sfreq
    for band, (low, _high) in bands.items():
        if low < 5.0 / epoch_len_s:
            raise ValueError(
                f"Band {band}: low edge {low} Hz has fewer than 5 cycles in a "
                f"{epoch_len_s:g} s epoch; use epoch_length >= {5.0 / low:g} s "
                "(or a higher low edge)."
            )
    return _connectivity_from_epoched_array(
        epoched, names, sfreq, bands, methods, aec_spans=aec_spans, lowpass=lowpass
    )


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

    # For AEC we'll need to handle it separately since it's not a spectral method
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
    selected_rois = [label.name for label in selected_labels]
    if not selected_labels:
        logger.warning("No selected ROIs found, using all available labels")
        selected_labels = labels
        selected_rois = label_names
    logger.info(f"Using {len(selected_labels)} selected ROIs: {selected_rois}")

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

    if n_epochs < 2:
        raise ValueError(
            f"Source connectivity needs at least 2 epochs, but only {n_epochs} "
            f"{epoch_length}-second epoch(s) fit in {n_times / sfreq:.1f} s of data. "
            "Use a shorter epoch_length or more data."
        )
    epoched_data_span = roi_data[:, : n_epochs * samples_per_epoch]
    epoched_data = np.stack(
        [
            roi_data[:, k * samples_per_epoch : (k + 1) * samples_per_epoch]
            for k in range(n_epochs)
        ],
        axis=0,
    )
    logger.info(f"Epoched data shape: {epoched_data.shape}")

    logger.info("Calculating connectivity metrics...")
    core_df = _connectivity_from_epoched_array(
        epoched_data,
        selected_rois,
        sfreq,
        bands,
        list(conn_methods) + (["aec"] if include_aec else []),
        n_jobs=n_jobs,
        aec_spans=[epoched_data_span],
    )

    connectivity_data = []
    for (method, band_name), grp in core_df.groupby(["method", "band"], sort=False):
        n_roi = len(selected_rois)
        con_matrix = np.zeros((n_roi, n_roi))
        index = {name: k for k, name in enumerate(selected_rois)}
        for r in grp.itertuples():
            ia, ib = index[r.channel_a], index[r.channel_b]
            con_matrix[ib, ia] = r.value
            if method == "aec":
                con_matrix[ia, ib] = r.value
        con_df = pd.DataFrame(con_matrix, columns=selected_rois, index=selected_rois)
        matrix_filename = os.path.join(
            output_dir, f"{subject_id}_{method}_{band_name}_matrix.csv"
        )
        con_df.to_csv(matrix_filename)
        logger.info(f"Saved connectivity matrix to {matrix_filename}")
        for r in grp.itertuples():
            connectivity_data.append(
                {
                    "subject": subject_id,
                    "method": method,
                    "band": band_name,
                    "roi1": r.channel_a,
                    "roi2": r.channel_b,
                    "connectivity": r.value,
                }
            )

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
    graph_methods = list(conn_methods) + (["aec"] if include_aec else [])
    if not BCTPY_AVAILABLE:
        logger.warning("bctpy (module 'bct') is not installed. Skipping graph metrics.")
        graph_methods = []
    elif not NETWORK_ANALYSIS_AVAILABLE:
        logger.warning("networkx is not installed. Skipping graph metrics.")
        graph_methods = []
    else:
        from networkx.algorithms.community import louvain_communities, modularity

    for method in graph_methods:
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
            np.fill_diagonal(adj_matrix, 0.0)

            try:
                clustering = np.mean(clustering_coef_wu(adj_matrix))
                global_efficiency = efficiency_wei(adj_matrix)
                char_path_length = _characteristic_path_length(adj_matrix)
                communities = louvain_communities(G, resolution=1.0, seed=0)
                modularity_score = modularity(G, communities, weight="weight")
                strength = np.mean(np.sum(adj_matrix, axis=1))

                # Additional graph metrics
                # Assortativity measures how similar connected nodes are
                assortativity = nx.degree_assortativity_coefficient(G, weight="weight")

                # Small-worldness
                # (requires computing random networks for comparison - simplified version)
                if np.isnan(char_path_length):
                    small_worldness = float("nan")
                else:
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
