"""Source localization algorithms for EEG data.

This module contains scientifically validated functions for estimating cortical
sources from sensor-space EEG data using minimum norm estimation (MNE) and converting
source estimates to standardized EEG format for BIDS derivatives.

These functions are EXACT copies from the AutoCleanEEG pipeline source.py module
and must NOT be modified to maintain reproducibility of scientific results.

References
----------
Hämäläinen MS & Ilmoniemi RJ (1994). Interpreting magnetic fields of the brain:
minimum norm estimates. Medical & Biological Engineering & Computing, 32(1), 35-42.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.datasets import fetch_fsaverage

from autoclean.io.export import save_stc_to_file


def estimate_source_function_raw(raw: mne.io.Raw, config: dict = None, save_stc: bool = False):
    """
    Perform source localization on continuous resting-state EEG data using an identity matrix
    for noise covariance, keeping it as raw data.

    Args:
        raw: MNE Raw object
        config: Configuration dictionary
        save_stc: If True, save vertex-level STC file (default: False)
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

    if config is not None and save_stc:
        save_stc_to_file(stc, config, stage="post_source_localization")

    matplotlib.use("Agg")
    return stc


def estimate_source_function_epochs(epochs: mne.Epochs, config: dict = None, save_stc: bool = False):
    """
    Perform source localization on epoched EEG data using an identity matrix
    for noise covariance.

    Args:
        epochs: MNE Epochs object
        config: Configuration dictionary
        save_stc: If True, save vertex-level STC files for first 3 epochs (default: False)
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

    if config is not None and save_stc:
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