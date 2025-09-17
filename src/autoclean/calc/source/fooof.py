"""FOOOF-based spectral decomposition utilities."""
from __future__ import annotations

import gc
import os
import warnings
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.gridspec import GridSpec
from mne import SourceEstimate
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

from ._compat import FOOOF, FOOOFGroup, FOOOF_AVAILABLE
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
    gs = GridSpec(3, 4, figure=fig)

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

__all__ = [
    'calculate_fooof_aperiodic',
    'visualize_fooof_results',
    'calculate_fooof_periodic',
    'calculate_vertex_peak_frequencies',
    'visualize_peak_frequencies',
]
