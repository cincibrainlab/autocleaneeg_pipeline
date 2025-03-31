from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import mne
import numpy as np
import pandas as pd
from autoreject import AutoReject
from matplotlib import pyplot as plt
from mne.preprocessing import bads

from autoclean.io.export import save_epochs_to_set
from autoclean.utils.database import manage_database
from autoclean.utils.logging import message

# Force non-interactive matplotlib backend
import matplotlib
matplotlib.use('Agg')  # Ensure non-interactive backend for all plots

__all__ = [
    "step_create_regular_epochs",
    "step_create_eventid_epochs",
    "step_prepare_epochs_for_ica",
    "step_gfp_clean_epochs",
    "step_apply_autoreject",
]


def step_create_regular_epochs(
    cleaned_raw: mne.io.Raw, pipeline: Any, autoclean_dict: Dict[str, Any]
) -> mne.Epochs:
    message("header", "step_create_regular_epochs")
    # Check if epoching is enabled
    task = autoclean_dict["task"]
    epoch_settings = autoclean_dict["tasks"][task]["settings"]["epoch_settings"]
    if not epoch_settings["enabled"]:
        return None

    tmin = epoch_settings["value"]["tmin"]
    tmax = epoch_settings["value"]["tmax"]
    baseline = (
        tuple(epoch_settings["remove_baseline"]["window"])
        if epoch_settings["remove_baseline"]["enabled"]
        else None
    )
    volt_threshold = (
        tuple(epoch_settings["threshold_rejection"]["volt_threshold"])
        if epoch_settings["threshold_rejection"]["enabled"]
        else None
    )
    if isinstance(volt_threshold, (int, float)):
        volt_threshold = {"eeg": volt_threshold}
        
    if isinstance(volt_threshold, dict):
        volt_threshold["eeg"] = float(volt_threshold["eeg"])
    
    # Create initial epochs with reject_by_annotation=False to handle annotations manually
    events = mne.make_fixed_length_events(
        cleaned_raw, duration=tmax - tmin, overlap=0, start=abs(tmin)
    )
    epochs = mne.Epochs(
        cleaned_raw,
        events,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject=volt_threshold,
        preload=True,
        reject_by_annotation=False,
    )

    # Initialize metadata DataFrame
    epochs.metadata = pd.DataFrame(index=range(len(epochs)))

    # Find epochs that overlap with BAD_REF_AF annotations
    bad_ref_epochs = []
    for ann in cleaned_raw.annotations:
        if ann["description"] == "BAD_REF_AF":
            # Find epochs that overlap with this annotation
            ann_start = ann["onset"]
            ann_end = ann["onset"] + ann["duration"]

            # Check each epoch
            for idx, event in enumerate(epochs.events):
                epoch_start = event[0] / epochs.info["sfreq"]  # Convert to seconds
                epoch_end = epoch_start + (tmax - tmin)

                # Check for overlap
                if (epoch_start <= ann_end) and (epoch_end >= ann_start):
                    bad_ref_epochs.append(idx)

    # Remove duplicates and sort
    bad_ref_epochs = sorted(list(set(bad_ref_epochs)))

    # Mark bad reference epochs in metadata
    epochs.metadata["BAD_REF_AF"] = [
        idx in bad_ref_epochs for idx in range(len(epochs))
    ]
    message("info", f"Marked {len(bad_ref_epochs)} unique epochs as BAD_REF_AF")

    # Detect Muscle Beta Focus
    bad_muscle_epochs = _detect_muscle_beta_focus_robust(
        epochs.copy(), pipeline, autoclean_dict, freq_band=(20, 100), scale_factor=2.0
    )

    # Remove duplicates and sort
    bad_muscle_epochs = sorted(list(set(bad_muscle_epochs)))

    # Add muscle artifact information to metadata
    epochs.metadata["BAD_MOVEMENT"] = [
        idx in bad_muscle_epochs for idx in range(len(epochs))
    ]
    message("info", f"Marked {len(bad_muscle_epochs)} unique epochs as BAD_MOVEMENT")

    # Add annotations for visualization
    for idx in bad_muscle_epochs:
        onset = epochs.events[idx, 0] / epochs.info["sfreq"]
        duration = tmax - tmin
        description = "BAD_MOVEMENT"
        epochs.annotations.append(onset, duration, description)

    # Save epochs with bad epochs marked but not dropped
    save_epochs_to_set(epochs, autoclean_dict, "post_epochs")

    # Create a copy for dropping
    epochs_clean = epochs.copy()

    # Combine all bad epochs and remove duplicates
    all_bad_epochs = sorted(list(set(bad_ref_epochs + bad_muscle_epochs)))

    # Drop all bad epochs at once
    if all_bad_epochs:
        epochs_clean.drop(all_bad_epochs)
        message("info", f"Dropped {len(all_bad_epochs)} unique bad epochs")

    # Drop remaining bad epochs
    epochs_clean.drop_bad()

    # Analyze drop log to tally different annotation types
    drop_log = epochs_clean.drop_log
    total_epochs = len(drop_log)
    good_epochs = sum(1 for log in drop_log if len(log) == 0)

    # Dynamically collect all unique annotation types
    annotation_types = {}
    for log in drop_log:
        if len(log) > 0:  # If epoch was dropped
            for annotation in log:
                # Convert numpy string to regular string if needed
                annotation = str(annotation)
                annotation_types[annotation] = annotation_types.get(annotation, 0) + 1

    message("info", "\nEpoch Drop Log Summary:")
    message("info", f"Total epochs: {total_epochs}")
    message("info", f"Good epochs: {good_epochs}")
    for annotation, count in annotation_types.items():
        message("info", f"Epochs with {annotation}: {count}")

    # Add good and total to the annotation_types dictionary
    annotation_types["KEEP"] = good_epochs
    annotation_types["TOTAL"] = total_epochs

    metadata = {
        "step_create_regular_epochs": {
            "creationDateTime": datetime.now().isoformat(),
            "duration": tmax - tmin,
            "reject_by_annotation": False,  # We handled annotations manually
            "initial_epoch_count": len(epochs),
            "final_epoch_count": len(epochs_clean),
            "single_epoch_duration": epochs.times[-1] - epochs.times[0],
            "single_epoch_samples": epochs.times.shape[0],
            "durationSec": (epochs.times[-1] - epochs.times[0]) * len(epochs_clean),
            "numberSamples": epochs.times.shape[0] * len(epochs_clean),
            "channelCount": len(epochs.ch_names),
            "annotation_types": annotation_types,
            "unique_bad_ref_epochs": len(bad_ref_epochs),
            "unique_bad_muscle_epochs": len(bad_muscle_epochs),
            "total_unique_bad_epochs": len(all_bad_epochs),
            "marked_epochs_file": "post_epochs",
            "cleaned_epochs_file": "post_drop_bads",
            "tmin": tmin,
            "tmax": tmax,
        }
    }

    manage_database(
        operation="update",
        update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
    )

    return epochs_clean


def step_create_eventid_epochs(
    cleaned_raw: mne.io.Raw, pipeline: Any, autoclean_dict: Dict[str, Any]
) -> Optional[mne.Epochs]:
    message("header", "step_create_eventid_epochs")
    task = autoclean_dict["task"]

    # Get epoch settings
    epoch_settings = autoclean_dict["tasks"][task]["settings"]["epoch_settings"]
    if not epoch_settings["enabled"]:
        return None

    # Get event_id settings from within epoch_settings
    # If event_id is None, it means it's disabled
    event_types = epoch_settings.get("event_id")
    if event_types is None:
        message("warning", "Event ID is not specified in epoch_settings (set to null)")
        return None

    tmin = epoch_settings["value"]["tmin"]
    tmax = epoch_settings["value"]["tmax"]
    message("info", f"Using tmin: {tmin} and tmax: {tmax}")
    baseline = (
        tuple(epoch_settings["remove_baseline"]["window"])
        if epoch_settings["remove_baseline"]["enabled"]
        else None
    )
    message("info", f"Using baseline: {baseline}")

    # Process voltage threshold
    volt_threshold = None
    if epoch_settings["threshold_rejection"]["enabled"]:
        threshold_config = epoch_settings["threshold_rejection"]["volt_threshold"]
        if isinstance(threshold_config, (int, float)):
            volt_threshold = {"eeg": float(threshold_config)}
        elif isinstance(threshold_config, dict):
            volt_threshold = {k: float(v) for k, v in threshold_config.items()}
        message("info", f"Using voltage threshold: {volt_threshold}")

    # Create regexp pattern for all event types
    target_event_type = list(event_types.keys())[0]
    reg_exp = f".*{target_event_type}.*"
    message("info", f"Looking for events matching pattern: {reg_exp}")

    # Get events using regexp
    events, event_id = mne.events_from_annotations(cleaned_raw, regexp=reg_exp)

    if len(events) == 0:
        message("warning", "No matching events found")
        return None

    message("info", f"Found {len(events)} events matching the patterns:")
    for event_name, event_num in event_id.items():
        message("info", f"  {event_name}: {event_num}")

    # Create epochs centered around events
    epochs = mne.Epochs(
        cleaned_raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject=volt_threshold,
        reject_by_annotation=True,
        preload=True,
    )

    # Remove existing annotations
    epochs.set_annotations(None)

    # Verify epoch events match original events
    message("info", "First 5 epoch events:")
    for i in range(min(5, len(epochs.events))):
        sample = epochs.events[i, 0]
        time = sample / epochs.info["sfreq"]
        message(
            "info",
            f"  Event {i}: sample={sample}, time={time:.3f}s, id={epochs.events[i, 2]}",
        )

    message("info", f"Created {len(epochs)} epochs")
    message(
        "info", f"Each epoch spans from {tmin}s to {tmax}s relative to the event at t=0"
    )

    # Add metadata about the epoching
    metadata = {
        "step_create_eventid_epochs": {
            "creationDateTime": datetime.now().isoformat(),
            "event_types": list(event_types),
            "number_of_events": len(events),
            "number_of_epochs": len(epochs),
            "epoch_duration": epochs.times[-1] - epochs.times[0],
            "samples_per_epoch": len(epochs.times),
            "total_duration": (epochs.times[-1] - epochs.times[0]) * len(epochs),
            "total_samples": len(epochs.times) * len(epochs),
            "channel_count": len(epochs.ch_names),
            "event_counts": {
                name: sum(events[:, 2] == num) for name, num in event_id.items()
            },
            "tmin": tmin,
            "tmax": tmax,
        }
    }

    manage_database(
        operation="update",
        update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
    )

    return epochs


def step_prepare_epochs_for_ica(
    epochs: mne.Epochs, pipeline: Any, autoclean_dict: Dict[str, Any]
) -> mne.Epochs:
    """
    Drops epochs that were marked bad based on a global outlier detection.
    This implementation for the preliminary epoch rejection was based on the
    Python implementation of the FASTER algorithm from Marijn van Vliet
    https://gist.github.com/wmvanvliet/d883c3fe1402c7ced6fc
    Parameters
    ----------
    epochs

    Returns
    -------
    Epochs instance
    """
    message("header", "step_prepare_epochs_for_ica")

    def _deviation(data: np.ndarray) -> np.ndarray:
        """
        Computes the deviation from mean for each channel.
        """
        channels_mean = np.mean(data, axis=2)
        return channels_mean - np.mean(channels_mean, axis=0)

    metrics = {
        "amplitude": lambda x: np.mean(np.ptp(x, axis=2), axis=1),
        "deviation": lambda x: np.mean(_deviation(x), axis=1),
        "variance": lambda x: np.mean(np.var(x, axis=2), axis=1),
    }

    epochs_data = epochs.get_data()

    bad_epochs = []
    bad_epochs_by_metric = {}
    for metric in metrics:
        scores = metrics[metric](epochs_data)
        outliers = bads._find_outliers(scores, threshold=3.0)
        message("info", f"Bad epochs by {metric}: {outliers}")
        bad_epochs.extend(outliers)
        bad_epochs_by_metric[metric] = list(outliers)

    # Convert numpy int64 values to regular integers for JSON serialization
    bad_epochs_by_metric_dict = {
        metric: [int(epoch) for epoch in epochs]
        for metric, epochs in bad_epochs_by_metric.items()
    }

    bad_epochs = list(set(bad_epochs))
    epochs_faster = epochs.copy().drop(bad_epochs, reason="BAD_EPOCHS")

    metadata = {
        "step_prepare_epochs_for_ica": {
            "creationDateTime": datetime.now().isoformat(),
            "initial_epochs": len(epochs),
            "final_epochs": len(epochs_faster),
            "rejected_epochs": len(bad_epochs),
            "rejection_percent": round((len(bad_epochs) / len(epochs)) * 100, 2),
            "bad_epochs_by_metric": bad_epochs_by_metric_dict,
            "total_bad_epochs": bad_epochs,
            "epoch_duration": epochs.times[-1] - epochs.times[0],
            "samples_per_epoch": epochs.times.shape[0],
            "total_duration_sec": (epochs.times[-1] - epochs.times[0])
            * len(epochs_faster),
            "total_samples": epochs.times.shape[0] * len(epochs_faster),
            "channel_count": len(epochs.ch_names),
        }
    }

    manage_database(
        operation="update",
        update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
    )

    return epochs_faster


def step_gfp_clean_epochs(
    epochs: mne.Epochs,
    pipeline: Any,
    autoclean_dict: Dict[str, Any],
    gfp_threshold: float = 3.0,
    number_of_epochs: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> mne.Epochs:
    """
    Clean an MNE Epochs object by removing outlier epochs based on GFP.
    Only calculates GFP on scalp electrodes (excluding those defined in channel_region_map).

    Args:
        epochs (mne.Epochs): The input epoched EEG data.
        gfp_threshold (float, optional): Z-score threshold for GFP-based outlier detection.
                                         Epochs with GFP z-scores above this value are removed.
                                         Defaults to 3.0.
        number_of_epochs (int, optional): If specified, randomly selects this number of epochs from the cleaned data.
                                           If None, retains all cleaned epochs. Defaults to None.
        random_seed (int, optional): Seed for random number generator to ensure reproducibility when selecting epochs.
                                     Defaults to None.

    Returns:
        Tuple[mne.Epochs, Dict[str, any]]: A tuple containing the cleaned Epochs object and a dictionary of statistics.
    """
    message("header", "step_gfp_clean_epochs")

    import random

    # Force preload to avoid RuntimeError
    if not epochs.preload:
        epochs.load_data()

    epochs_clean = epochs.copy()

    # Define non-scalp electrodes to exclude
    channel_region_map = {
        "E17": "OTHER",
        "E38": "OTHER",
        "E43": "OTHER",
        "E44": "OTHER",
        "E48": "OTHER",
        "E49": "OTHER",
        "E56": "OTHER",
        "E73": "OTHER",
        "E81": "OTHER",
        "E88": "OTHER",
        "E94": "OTHER",
        "E107": "OTHER",
        "E113": "OTHER",
        "E114": "OTHER",
        "E119": "OTHER",
        "E120": "OTHER",
        "E121": "OTHER",
        "E125": "OTHER",
        "E126": "OTHER",
        "E127": "OTHER",
        "E128": "OTHER",
    }

    # Get scalp electrode indices (all channels except those in channel_region_map)
    non_scalp_channels = list(channel_region_map.keys())
    all_channels = epochs_clean.ch_names
    scalp_channels = [ch for ch in all_channels if ch not in non_scalp_channels]
    scalp_indices = [epochs_clean.ch_names.index(ch) for ch in scalp_channels]

    # Step 2: Calculate Global Field Power (GFP) only for scalp electrodes
    message(
        "info",
        "Calculating Global Field Power (GFP) for each epoch using only scalp electrodes.",
    )
    gfp = np.sqrt(
        np.mean(epochs_clean.get_data()[:, scalp_indices, :] ** 2, axis=(1, 2))
    )  # Shape: (n_epochs,)

    # Step 3: Epoch Statistics
    epoch_stats = pd.DataFrame(
        {
            "epoch": np.arange(len(gfp)),
            "gfp": gfp,
            "mean_amplitude": epochs_clean.get_data()[:, scalp_indices, :].mean(
                axis=(1, 2)
            ),
            "max_amplitude": epochs_clean.get_data()[:, scalp_indices, :].max(
                axis=(1, 2)
            ),
            "min_amplitude": epochs_clean.get_data()[:, scalp_indices, :].min(
                axis=(1, 2)
            ),
            "std_amplitude": epochs_clean.get_data()[:, scalp_indices, :].std(
                axis=(1, 2)
            ),
        }
    )
    # Step 4: Remove Outlier Epochs based on GFP
    message("info", "Removing outlier epochs based on GFP z-scores.")
    gfp_mean = epoch_stats["gfp"].mean()
    gfp_std = epoch_stats["gfp"].std()
    z_scores = np.abs((epoch_stats["gfp"] - gfp_mean) / gfp_std)
    good_epochs_mask = z_scores < gfp_threshold
    removed_by_gfp = np.sum(~good_epochs_mask)
    epochs_final = epochs_clean[good_epochs_mask]
    epoch_stats_final = epoch_stats[good_epochs_mask]
    message("info", f"Outlier epochs removed based on GFP: {removed_by_gfp}")

    # Step 5: Handle epoch selection with warning if needed
    requested_epochs_exceeded = False
    if number_of_epochs is not None:
        if len(epochs_final) < number_of_epochs:
            warning_msg = f"Requested number_of_epochs={number_of_epochs} exceeds the available cleaned epochs={len(epochs_final)}. Using all available epochs."
            message("warning", warning_msg)
            requested_epochs_exceeded = True
            number_of_epochs = len(epochs_final)

        if random_seed is not None:
            random.seed(random_seed)
        selected_indices = random.sample(range(len(epochs_final)), number_of_epochs)
        epochs_final = epochs_final[selected_indices]
        epoch_stats_final = epoch_stats_final.iloc[selected_indices]
        message("info", f"Selected {number_of_epochs} epochs from the cleaned data.")

    # Analyze drop log to tally different annotation types
    drop_log = epochs.drop_log
    total_epochs = len(drop_log)
    good_epochs = sum(1 for log in drop_log if len(log) == 0)

    # Dynamically collect all unique annotation types
    annotation_types = {}
    for log in drop_log:
        if len(log) > 0:  # If epoch was dropped
            for annotation in log:
                # Convert numpy string to regular string if needed
                annotation = str(annotation)
                annotation_types[annotation] = annotation_types.get(annotation, 0) + 1

    # Add good and total to the annotation_types dictionary
    annotation_types["KEEP"] = good_epochs
    annotation_types["TOTAL"] = total_epochs
    # Create GFP barplot
    plt.figure(figsize=(12, 4))

    # Plot all epochs in red first (marking removed epochs)
    plt.bar(epoch_stats.index, epoch_stats["gfp"], width=0.8, color="red", alpha=0.3)

    # Then overlay kept epochs in blue
    plt.bar(epoch_stats_final.index, epoch_stats_final["gfp"], width=0.8, color="blue")

    plt.xlabel("Epoch Number")
    plt.ylabel("Global Field Power (GFP)")
    plt.title("GFP Values by Epoch (Red = Removed, Blue = Kept)")

    # Save plot using BIDS derivative name
    # Create output path for the PDF report
    derivatives_path = pipeline.get_derivative_path(autoclean_dict["bids_path"])

    plot_fname = derivatives_path.copy().update(
        suffix="gfp", extension="png", check=False
    )
    plt.savefig(Path(plot_fname), dpi=150, bbox_inches="tight")
    plt.close()

    # Create GFP heatmap with larger figure size and improved readability
    plt.figure(figsize=(30, 18))

    # Calculate number of rows and columns for grid layout
    n_epochs = len(epoch_stats)
    n_cols = 8  # Reduced number of columns for larger cells
    n_rows = int(np.ceil(n_epochs / n_cols))

    # Create a grid of values
    grid = np.full((n_rows, n_cols), np.nan)
    for i, (idx, gfp) in enumerate(epoch_stats["gfp"].items()):
        row = i // n_cols
        col = i % n_cols
        grid[row, col] = gfp

    # Create heatmap with larger spacing between cells
    im = plt.imshow(grid, cmap="RdYlBu_r", aspect="auto")
    plt.colorbar(im, label="GFP Value (×10⁻⁶)", fraction=0.02, pad=0.04)

    # Add text annotations with increased font size and spacing
    for i, (idx, gfp) in enumerate(epoch_stats["gfp"].items()):
        row = i // n_cols
        col = i % n_cols
        kept = idx in epoch_stats_final.index
        color = "black" if kept else "red"
        plt.text(
            col,
            row,
            f"ID: {idx}\nGFP: {gfp:.1e}",
            ha="center",
            va="center",
            color=color,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, pad=0.8),
        )

    # Improve title and labels with larger font sizes
    plt.title("GFP Heatmap by Epoch (Red = Removed, Black = Kept)", fontsize=14, pad=20)
    plt.xlabel("Column", fontsize=12, labelpad=10)
    plt.ylabel("Row", fontsize=12, labelpad=10)

    # Adjust layout to prevent text overlap
    plt.tight_layout()

    # Save heatmap plot with higher DPI for better quality
    plot_fname = derivatives_path.copy().update(
        suffix="gfp-heatmap", extension="png", check=False
    )
    plt.savefig(Path(plot_fname), dpi=300, bbox_inches="tight")
    plt.close()

    metadata = {
        "step_gfp_clean_epochs": {
            "creationDateTime": datetime.now().isoformat(),
            "initial_epochs": len(epochs),
            "final_epochs": len(epochs_final),
            "removed_by_gfp": removed_by_gfp,
            "mean_amplitude": float(epoch_stats_final["mean_amplitude"].mean()),
            "max_amplitude": float(epoch_stats_final["max_amplitude"].max()),
            "min_amplitude": float(epoch_stats_final["min_amplitude"].min()),
            "std_amplitude": float(epoch_stats_final["std_amplitude"].mean()),
            "mean_gfp": float(epoch_stats_final["gfp"].mean()),
            "gfp_threshold": float(gfp_threshold),
            "removed_total": removed_by_gfp,
            "annotation_types": annotation_types,
            "epoch_duration": epochs.times[-1] - epochs.times[0],
            "samples_per_epoch": epochs.times.shape[0],
            "total_duration_sec": (epochs.times[-1] - epochs.times[0])
            * len(epochs_final),
            "total_samples": epochs.times.shape[0] * len(epochs_final),
            "channel_count": len(epochs.ch_names),
            "scalp_channels_used": scalp_channels,
            "requested_epochs_exceeded": requested_epochs_exceeded,
        }
    }

    manage_database(
        operation="update",
        update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
    )

    message("info", "Epoch GFP cleaning process completed.")

    return epochs_final


def step_apply_autoreject(
    epochs: mne.Epochs, pipeline: Any, autoclean_dict: Dict[str, Any]
) -> mne.Epochs:
    """
    Apply AutoReject to clean epochs.

    Args:
        epochs (mne.Epochs): The input epoched EEG data
        pipeline: The pipeline object containing configuration and metadata
        autoclean_dict (dict): Dictionary containing pipeline configuration and metadata

    Returns:
        mne.Epochs: The cleaned epochs after AutoReject
    """
    message("header", "Applying AutoReject for artifact rejection.")
    ar = AutoReject()
    epochs_clean = ar.fit_transform(epochs)
    rejected_epochs = len(epochs) - len(epochs_clean)
    message(
        "info", f"Artifacts rejected: {rejected_epochs} epochs removed by AutoReject."
    )

    metadata = {
        "step_apply_autoreject": {
            "creationDateTime": datetime.now().isoformat(),
            "initial_epochs": len(epochs),
            "final_epochs": len(epochs_clean),
            "rejected_epochs": rejected_epochs,
            "rejection_percent": round((rejected_epochs / len(epochs)) * 100, 2),
            "epoch_duration": epochs.times[-1] - epochs.times[0],
            "samples_per_epoch": epochs.times.shape[0],
            "total_duration_sec": (epochs.times[-1] - epochs.times[0])
            * len(epochs_clean),
            "total_samples": epochs.times.shape[0] * len(epochs_clean),
            "channel_count": len(epochs.ch_names),
        }
    }

    manage_database(
        operation="update",
        update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
    )

    return epochs_clean


def _detect_muscle_beta_focus_robust(
    epochs, pipeline, autoclean_dict, freq_band=(20, 30), scale_factor=3.0
):
    """
    Detect muscle artifacts using a robust measure (median + MAD * scale_factor)
    focusing only on electrodes labeled as 'OTHER'.
    This reduces forced removal of epochs in very clean data.
    """

    # Ensure data is loaded
    epochs.load_data()

    # Filter in beta band
    epochs_beta = epochs.copy().filter(
        l_freq=freq_band[0], h_freq=freq_band[1], verbose=False
    )

    # Get channel names
    ch_names = epochs_beta.ch_names

    # Build channel_region_map from the provided channel data
    # Make sure all "OTHER" electrodes are listed here
    channel_region_map = {
        "E17": "OTHER",
        "E38": "OTHER",
        "E43": "OTHER",
        "E44": "OTHER",
        "E48": "OTHER",
        "E49": "OTHER",
        "E56": "OTHER",
        "E73": "OTHER",
        "E81": "OTHER",
        "E88": "OTHER",
        "E94": "OTHER",
        "E107": "OTHER",
        "E113": "OTHER",
        "E114": "OTHER",
        "E119": "OTHER",
        "E120": "OTHER",
        "E121": "OTHER",
        "E125": "OTHER",
        "E126": "OTHER",
        "E127": "OTHER",
        "E128": "OTHER",
    }

    # Select only OTHER channels
    selected_ch_indices = [
        i for i, ch in enumerate(ch_names) if channel_region_map.get(ch, "") == "OTHER"
    ]

    # If no OTHER channels are found, return empty
    if not selected_ch_indices:
        return np.array([], dtype=int)

    # Extract data from OTHER channels only
    data = epochs_beta.get_data()[
        :, selected_ch_indices, :
    ]  # shape: (n_epochs, n_sel_channels, n_times)

    # Compute peak-to-peak amplitude per epoch and selected channels
    p2p = data.max(axis=2) - data.min(axis=2)

    # Compute maximum peak-to-peak amplitude across the selected channels
    max_p2p = p2p.max(axis=1)

    # Compute median and MAD
    med = np.median(max_p2p)
    mad = np.median(np.abs(max_p2p - med))

    # Robust threshold
    threshold = med + scale_factor * mad

    # Identify bad epochs
    bad_epochs = np.where(max_p2p > threshold)[0].tolist()

    metadata = {
        "muscle_beta_focus_robust": {
            "creationDateTime": datetime.now().isoformat(),
            "muscle_beta_focus_robust": True,
            "freq_band": freq_band,
            "scale_factor": scale_factor,
            "bad_epochs": bad_epochs,
        }
    }

    manage_database(
        operation="update",
        update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
    )

    return bad_epochs
