"""Phase-amplitude coupling helpers."""
from __future__ import annotations

import os
from typing import Sequence

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mne import SourceEstimate
from mne.filter import filter_data
from scipy.signal import hilbert

from ._utils import coerce_stc_to_single

def calculate_source_pac(
    stc: SourceEstimate | Sequence[SourceEstimate],
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
    stc : SourceEstimate | Sequence[SourceEstimate]
        The source time course(s) to calculate PAC from. Sequences are
        concatenated along the time axis to preserve epoch ordering.
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

    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = os.getcwd()

    os.makedirs(output_dir, exist_ok=True)

    stc, was_sequence = coerce_stc_to_single(stc)
    if was_sequence:
        print(
            "Concatenated multiple source estimates into a single time series for PAC analysis."
        )
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

__all__ = [
    'calculate_source_pac',
]
