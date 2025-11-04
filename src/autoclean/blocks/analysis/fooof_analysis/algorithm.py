"""specparam analysis algorithms for source-localized EEG data.

This module provides vertex-level spectral parameterization using the specparam
(Fitting Oscillations & One Over F) algorithm.

Functions
---------
calculate_vertex_psd_for_fooof : Prepare vertex-level PSD for specparam analysis
calculate_fooof_aperiodic : Extract aperiodic (1/f) parameters
calculate_fooof_periodic : Extract periodic (oscillatory) parameters
visualize_fooof_results : Create comprehensive specparam visualizations

References
----------
Donoghue T, et al. (2020). Parameterizing neural power spectra into periodic and
aperiodic components. Nature Neuroscience, 23(12), 1655-1665.
"""

from __future__ import annotations

import gc
import os
import warnings

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import signal

# SpecParam dependency for spectral parameterization
try:
    from specparam import SpectralGroupModel
    from specparam.analysis import get_band_peak

    SPECPARAM_AVAILABLE = True
except ImportError:
    SPECPARAM_AVAILABLE = False


def calculate_vertex_psd_for_fooof(
    stc, fmin=1.0, fmax=45.0, n_jobs=10, output_dir=None, subject_id=None
):
    """
    Calculate full power spectral density at the vertex level for specparam analysis.

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

    print(f"Calculating vertex-level PSD for specparam analysis - {subject_id}...")
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
    Run specparam to model aperiodic parameters for all vertices with robust error handling.

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
        Aperiodic mode for specparam ('fixed' or 'knee')

    Returns
    -------
    aperiodic_df : DataFrame
        DataFrame with aperiodic parameters
    file_path : str
        Path to saved file
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Calculating specparam aperiodic parameters for {subject_id}...")

    if not SPECPARAM_AVAILABLE:
        print("specparam library not available. Skipping aperiodic parameter analysis.")
        return pd.DataFrame(), None

    # Get data from stc_psd
    psds = stc_psd.data
    freqs = stc_psd.times

    n_vertices = psds.shape[0]
    print(f"Processing specparam analysis for {n_vertices} vertices...")

    # specparam parameters with fallback options
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
                fg = SpectralGroupModel(**fooof_params)
                fg.fit(freqs, batch_psds)

                # Check if fits were successful
                if np.any(~fg.get_params("aperiodic_params")[:, 0].astype(bool)):
                    # Some fits failed, try fallback parameters
                    raise RuntimeError("Some fits failed with primary parameters")

            except Exception:
                # Try again with fallback parameters
                try:
                    fg = SpectralGroupModel(**fallback_params)
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

    print(f"Saved specparam aperiodic parameters to {file_path}")
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


def calculate_fooof_periodic(
    stc,
    freq_bands=None,
    n_jobs=10,
    output_dir=None,
    subject_id=None,
    aperiodic_mode="knee",
):
    """
    Calculate specparam periodic parameters from source-localized data and save results.

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
        Aperiodic mode for specparam ('fixed' or 'knee')

    Returns
    -------
    periodic_df : DataFrame
        DataFrame containing periodic parameters for each vertex and frequency band
    file_path : str
        Path to the saved data file
    """

    if not SPECPARAM_AVAILABLE:
        raise ImportError(
            "specparam is required for this function. Install with 'pip install specparam'"
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

    print(f"Calculating specparam oscillatory parameters for {subject_id}...")

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
        f"Processing specparam analysis for {n_vertices} vertices and {len(freq_bands)} frequency bands..."
    )

    # specparam parameters
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

        # Create specparam model and fit
        fg = SpectralGroupModel(**fooof_params)
        fg.fit(freqs_to_fit, batch_psds)

        # Extract periodic parameters for each frequency band
        results = []

        for i, vertex_idx in enumerate(vertices):
            for band_name, band_range in freq_bands.items():
                # Get specparam model for this vertex
                fm = fg.get_model(i)

                # Extract peak in this band
                peak_params = get_band_peak(fm, band_range, select_highest=True)

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

    print(f"Saved specparam periodic parameters to {file_path}")

    return periodic_df, file_path
