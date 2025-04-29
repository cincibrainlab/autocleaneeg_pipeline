"""Continuous preprocessing steps."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import mne
import numpy as np
import pylossless as ll
import yaml
from matplotlib import pyplot as plt
from mne_bids import read_raw_bids
from pyprep.find_noisy_channels import NoisyChannels

from autoclean.io.export import save_raw_to_set
from autoclean.utils.bids import step_convert_to_bids
from autoclean.utils.database import manage_database
from autoclean.utils.logging import message

__all__ = [
    "step_pre_pipeline_processing",
    "step_create_bids_path",
    "step_clean_bad_channels",
    "step_run_pylossless",
    "step_run_ll_rejection_policy",
]


def step_pre_pipeline_processing(
    raw: mne.io.Raw, autoclean_dict: Dict[str, Any]
) -> mne.io.Raw:
    message("header", "\nPre-pipeline Processing Steps")

    task = autoclean_dict["task"]

    # Get enabled/disabled status for each step
    apply_resample_toggle = autoclean_dict["tasks"][task]["settings"]["resample_step"][
        "enabled"
    ]
    apply_drop_outerlayer_toggle = autoclean_dict["tasks"][task]["settings"][
        "drop_outerlayer"
    ]["enabled"]
    apply_eog_toggle = autoclean_dict["tasks"][task]["settings"]["eog_step"]["enabled"]
    apply_average_reference_toggle = autoclean_dict["tasks"][task]["settings"][
        "reference_step"
    ]["enabled"]
    apply_trim_toggle = autoclean_dict["tasks"][task]["settings"]["trim_step"][
        "enabled"
    ]
    apply_crop_toggle = autoclean_dict["tasks"][task]["settings"]["crop_step"][
        "enabled"
    ]

    # Print status of each step
    message(
        "info",
        f"{'✓' if apply_resample_toggle else '✗'} Resample: {apply_resample_toggle}",
    )
    message(
        "info",
        f"{'✓' if apply_drop_outerlayer_toggle else '✗'} Drop Outer Layer: {apply_drop_outerlayer_toggle}",
    )
    message(
        "info", f"{'✓' if apply_eog_toggle else '✗'} EOG Assignment: {apply_eog_toggle}"
    )
    message(
        "info",
        f"{'✓' if apply_average_reference_toggle else '✗'} Average Reference: {apply_average_reference_toggle}",
    )
    message(
        "info",
        f"{'✓' if apply_trim_toggle else '✗'} Edge Trimming: {apply_trim_toggle}",
    )
    message(
        "info",
        f"{'✓' if apply_crop_toggle else '✗'} Duration Cropping: {apply_crop_toggle}",
    )

    # Initialize metadata
    metadata = {
        "pre_pipeline_processing": {
            "creationDateTime": datetime.now().isoformat(),
            "ResampleHz": None,
            "TrimSec": None,
            "CropDurationSec": None,
            "AverageReference": apply_average_reference_toggle,
            "EOGChannels": None,
            "OuterLayerChannels": None,
        }
    }

    # Resample
    if apply_resample_toggle:
        message("header", "Resampling data...")
        target_sfreq = autoclean_dict["tasks"][task]["settings"]["resample_step"][
            "value"
        ]
        raw = raw.resample(target_sfreq)
        message("info", f"  - Data resampled to {target_sfreq} Hz")
        metadata["pre_pipeline_processing"]["ResampleHz"] = target_sfreq
        save_raw_to_set(raw, autoclean_dict, "post_resample")

    # Drop Outer Layer
    if apply_drop_outerlayer_toggle:
        message("header", "Dropping Outer Layer Channels...")
        outer_layer_channels = autoclean_dict["tasks"][task]["settings"][
            "drop_outerlayer"
        ]["value"]
        raw = raw.drop_channels(outer_layer_channels)
        message("info", f"  - Outer Layer Channels dropped: {outer_layer_channels}")
        metadata["pre_pipeline_processing"]["OuterLayerChannels"] = outer_layer_channels
        save_raw_to_set(raw, autoclean_dict, "post_outerlayer")

    # EOG Assignment
    if apply_eog_toggle:
        message("header", "Setting EOG channels...")
        eog_channels = autoclean_dict["tasks"][task]["settings"]["eog_step"]["value"]
        eog_channels = [f"E{ch}" for ch in sorted(eog_channels)]
        raw.set_channel_types({ch: "eog" for ch in raw.ch_names if ch in eog_channels})
        message("info", "  - EOG channels assigned")
        metadata["pre_pipeline_processing"]["EOGChannels"] = eog_channels

    # # Average Reference
    # if apply_average_reference_toggle:
    #     message("header", "Applying average reference...")
    #     ref_type = autoclean_dict["tasks"][task]["settings"]["reference_step"]["value"]
    #     if ref_type == "average":
    #         raw = raw.set_eeg_reference(ref_type, projection=False)
    #     else:
    #         raw = raw.set_eeg_reference(ref_type)
    #     message("info", "  - Average reference applied")
    #     save_raw_to_set(raw, autoclean_dict, "post_reference")

    # Trim Edges
    if apply_trim_toggle:
        message("header", "Trimming data edges...")
        trim = autoclean_dict["tasks"][task]["settings"]["trim_step"]["value"]
        start_time = raw.times[0]
        end_time = raw.times[-1]
        raw.crop(tmin=start_time + trim, tmax=end_time - trim)
        message("info", f"  - Data trimmed by {trim}s from each end")
        metadata["pre_pipeline_processing"]["TrimSec"] = trim
        save_raw_to_set(raw, autoclean_dict, "post_trim")

    # Crop Duration
    if apply_crop_toggle:
        message("header", "Cropping data duration...")
        start_time = autoclean_dict["tasks"][task]["settings"]["crop_step"]["value"][
            "start"
        ]
        end_time = autoclean_dict["tasks"][task]["settings"]["crop_step"]["value"][
            "end"
        ]
        if end_time is None:
            end_time = raw.times[-1]  # Use full duration if end is null
        raw.crop(tmin=start_time, tmax=end_time)
        target_crop_duration = raw.times[-1] - raw.times[0]
        message("info", f"  - Data cropped to {target_crop_duration:.1f}s")
        metadata["pre_pipeline_processing"]["CropDurationSec"] = target_crop_duration
        metadata["pre_pipeline_processing"]["CropStartSec"] = start_time
        metadata["pre_pipeline_processing"]["CropEndSec"] = end_time
        save_raw_to_set(raw, autoclean_dict, "post_crop")

    metadata["pre_pipeline_processing"]["channelCount"] = len(raw.ch_names)
    metadata["pre_pipeline_processing"]["durationSec"] = (
        int(raw.n_times) / raw.info["sfreq"]
    )
    metadata["pre_pipeline_processing"]["numberSamples"] = int(raw.n_times)

    run_id = autoclean_dict["run_id"]
    manage_database(
        operation="update", update_record={"run_id": run_id, "metadata": metadata}
    )

    # self._verify_annotations(self.raw, "post_prepipeline")
    save_raw_to_set(raw, autoclean_dict, "post_prepipeline")

    return raw


def step_create_bids_path(
    raw: mne.io.Raw, autoclean_dict: Dict[str, Any]
) -> Tuple[mne.io.Raw, Dict[str, Any]]:
    """Create BIDS-compliant paths."""
    message("header", "step_create_bids_path")
    unprocessed_file = autoclean_dict["unprocessed_file"]
    task = autoclean_dict["task"]
    mne_task = autoclean_dict["tasks"][task]["mne_task"]
    bids_dir = autoclean_dict["bids_dir"]
    eeg_system = autoclean_dict["eeg_system"]
    config_file = autoclean_dict["config_file"]

    try:
        line_freq = autoclean_dict["filtering"]["notch_filter_args"]["freqs"][0]
    except:
        line_freq = 60.0

    try:
        bids_path = step_convert_to_bids(
            raw,
            output_dir=str(bids_dir),
            task=mne_task,
            participant_id=None,
            line_freq=line_freq,
            overwrite=True,
            study_name=unprocessed_file.stem,
            autoclean_dict=autoclean_dict,
        )

        autoclean_dict["bids_path"] = bids_path
        autoclean_dict["bids_basename"] = bids_path.basename
        metadata = {
            "step_convert_to_bids": {
                "creationDateTime": datetime.now().isoformat(),
                "bids_subject": bids_path.subject,
                "bids_task": bids_path.task,
                "bids_run": bids_path.run,
                "bids_session": bids_path.session,
                "bids_dir": str(bids_dir),
                "bids_datatype": bids_path.datatype,
                "bids_suffix": bids_path.suffix,
                "bids_extension": bids_path.extension,
                "bids_root": str(bids_path.root),
                "eegSystem": eeg_system,
                "configFile": str(config_file),
                "line_freq": line_freq,
            }
        }

        manage_database(
            operation="update",
            update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
        )

        manage_database(
            operation="update_status",
            update_record={
                "run_id": autoclean_dict["run_id"],
                "status": "bids_path_created",
            },
        )

        return raw, autoclean_dict

    except Exception as e:
        message("error", f"Error converting raw to bids: {e}")
        raise e


def step_clean_bad_channels(
    raw: mne.io.Raw, autoclean_dict: Dict[str, Any]
) -> mne.io.Raw:
    """Clean bad channels."""
    message("header", "step_clean_bad_channels")
    # Setup options
    options = {
        "random_state": 1337,
        "ransac": True,
        "channel_wise": False,
        "max_chunk_size": None,
        "corr_thresh": 0.75,
    }

    # check if "eog" is in channel type dictionary
    if (
        "eog" in raw.get_channel_types()
        and not autoclean_dict["tasks"][autoclean_dict["task"]]["settings"]["eog_step"][
            "enabled"
        ]
    ):
        eog_picks = mne.pick_types(raw.info, eog=True)
        eog_ch_names = [raw.ch_names[idx] for idx in eog_picks]
        raw.set_channel_types({ch: "eeg" for ch in eog_ch_names})

    # Run noisy channels detection
    cleaned_raw = NoisyChannels(raw, random_state=options["random_state"])
    cleaned_raw.find_bad_by_SNR()
    cleaned_raw.find_bad_by_correlation(
        correlation_secs=1.0, correlation_threshold=0.2, frac_bad=0.01
    )  # frac_bad refers to "bad" correlation window over whole recording
    cleaned_raw.find_bad_by_deviation(deviation_threshold=4.0)

    cleaned_raw.find_bad_by_ransac(
        n_samples=100,
        sample_prop=0.5,
        corr_thresh=0.5,
        frac_bad=0.25,
        corr_window_secs=4.0,
        channel_wise=False,
        max_chunk_size=None,
    )
    bad_channels = cleaned_raw.get_bads(as_dict=True)
    raw.info["bads"].extend([str(ch) for ch in bad_channels["bad_by_ransac"]])
    raw.info["bads"].extend([str(ch) for ch in bad_channels["bad_by_deviation"]])
    raw.info["bads"].extend([str(ch) for ch in bad_channels["bad_by_correlation"]])
    raw.info["bads"].extend([str(ch) for ch in bad_channels["bad_by_SNR"]])

    print(raw.info["bads"])

    # Record metadata with options
    metadata = {
        "step_clean_bad_channels": {
            "creationDateTime": datetime.now().isoformat(),
            "method": "NoisyChannels",
            "options": options,
            "bads": raw.info["bads"],
            "channelCount": len(raw.ch_names),
            "durationSec": int(raw.n_times) / raw.info["sfreq"],
            "numberSamples": int(raw.n_times),
        }
    }

    manage_database(
        operation="update",
        update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
    )

    return raw


def step_run_pylossless(autoclean_dict: Dict[str, Any]) -> Tuple[Any, mne.io.Raw]:
    """Run PyLossless pipeline."""
    message("header", "step_run_pylossless")
    task = autoclean_dict["task"]
    bids_path = autoclean_dict["bids_path"]
    config_path = autoclean_dict["tasks"][task]["lossless_config"]
    derivative_name = "pylossless"

    # Use the already processed raw data if available, otherwise load from disk
    if "_current_raw" in autoclean_dict:
        raw = autoclean_dict["_current_raw"]
    else:
        raw = read_raw_bids(bids_path, verbose="ERROR", extra_params={"preload": True})

    try:
        pipeline = ll.LosslessPipeline(config_path)
        raw = pipeline.run_with_raw(raw)

        derivatives_path = pipeline.get_derivative_path(bids_path, derivative_name)
        pipeline.save(derivatives_path, overwrite=True, format="BrainVision")

    except Exception as e:
        message("error", f"Failed to run pylossless: {str(e)}")
        raise e

    try:
        pylossless_config = yaml.safe_load(open(config_path))

        metadata = {
            "step_run_pylossless": {
                "creationDateTime": datetime.now().isoformat(),
                "derivativeName": derivative_name,
                "configFile": str(config_path),
                "pylossless_config": pylossless_config,
                "channelCount": len(pipeline.raw.ch_names),
                "durationSec": int(pipeline.raw.n_times) / pipeline.raw.info["sfreq"],
                "numberSamples": int(pipeline.raw.n_times),
            }
        }
        manage_database(
            operation="update",
            update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
        )

    except Exception as e:
        message("error", f"Failed to load pylossless config: {str(e)}")
        raise e

    return pipeline, raw


def step_get_pylossless_pipeline(
    autoclean_dict: Dict[str, Any],
) -> Tuple[Any, mne.io.Raw]:
    """Run PyLossless pipeline."""
    message("header", "step_run_pylossless")
    task = autoclean_dict["task"]
    bids_path = autoclean_dict["bids_path"]
    config_path = autoclean_dict["tasks"][task]["lossless_config"]
    derivative_name = "pylossless"

    try:
        raw = read_raw_bids(bids_path, verbose="ERROR", extra_params={"preload": True})

        pipeline = ll.LosslessPipeline(config_path)

        pipeline.raw = raw

        derivatives_path = pipeline.get_derivative_path(bids_path, derivative_name)

    except Exception as e:
        message("error", f"Failed to run pylossless: {str(e)}")
        raise e

    try:
        pylossless_config = yaml.safe_load(open(config_path))

        metadata = {
            "step_get_pylossless_pipeline": {
                "creationDateTime": datetime.now().isoformat(),
                "derivativeName": derivative_name,
                "derivativePath": derivatives_path,
                "configFile": str(config_path),
                "pylossless_config": pylossless_config,
            }
        }
        manage_database(
            operation="update",
            update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
        )

    except Exception as e:
        message("error", f"Failed to load pylossless config: {str(e)}")
        raise e

    return pipeline


def step_run_ll_rejection_policy(
    pipeline: Any, autoclean_dict: Dict[str, Any]
) -> Tuple[Any, mne.io.Raw]:
    """Apply PyLossless rejection policy."""
    message("header", "step_run_ll_rejection_policy")
    rejection_policy = _get_rejection_policy(autoclean_dict)

    # Convert all channel flags to numpy arrays to avoid 'tolist' AttributeError
    for key in pipeline.flags["ch"]:
        if isinstance(pipeline.flags["ch"][key], list):
            pipeline.flags["ch"][key] = np.array(pipeline.flags["ch"][key])

    cleaned_raw = rejection_policy.apply(pipeline)
    cleaned_raw = _extended_BAD_LL_noisy_ICs_annotations(
        cleaned_raw, pipeline, autoclean_dict, extra_duration=1
    )

    ica = pipeline.ica2
    rejected_ics = ica.exclude
    n_components = int(ica.n_components_)
    # Calculate total duration of BAD annotations
    total_bad_duration = 0
    bad_annotation_count = 0
    distinct_annotation_types = set()

    if cleaned_raw.annotations:
        for annotation in cleaned_raw.annotations:
            if annotation["description"].startswith("BAD"):
                total_bad_duration += annotation["duration"]
                bad_annotation_count += 1
                distinct_annotation_types.add(annotation["description"])

    plot_bad_channels_with_topography(
        raw_original=pipeline.raw,
        raw_cleaned=cleaned_raw,
        pipeline=pipeline,
        autoclean_dict=autoclean_dict,
        zoom_duration=30,  # Duration for the zoomed-in plot
        zoom_start=0,  # Start time for the zoomed-in plot
    )
    # Validate inputs
    if cleaned_raw is None or not hasattr(cleaned_raw, "ch_names"):
        raise ValueError("cleaned_raw must be a valid MNE Raw object")
    if not isinstance(rejection_policy, (str, dict)):
        raise ValueError("rejection_policy must be a string or dictionary")
    if not isinstance(autoclean_dict, dict) or "run_id" not in autoclean_dict:
        raise ValueError("autoclean_dict must be a dictionary containing 'run_id'")

    # Safely calculate duration and sampling rate
    try:
        duration_sec = int(cleaned_raw.n_times) / cleaned_raw.info["sfreq"]
        total_time = cleaned_raw.times[-1] if len(cleaned_raw.times) > 0 else 0
    except (AttributeError, KeyError, IndexError, ZeroDivisionError) as e:
        raise RuntimeError(f"Error calculating duration/timing info: {str(e)}")

    # Safely calculate percentage with bounds checking
    try:
        percent_recording = (
            round((total_bad_duration / total_time) * 100, 2) if total_time > 0 else 0
        )
    except ZeroDivisionError:
        percent_recording = 0

    metadata = {
        "step_run_ll_rejection_policy": {
            "creationDateTime": datetime.now().isoformat(),
            "rejection_policy": rejection_policy,
            "channelCount": len(cleaned_raw.ch_names),
            "durationSec": duration_sec,
            "numberSamples": int(cleaned_raw.n_times),
            "bad_annotations_pending": {
                "number_of_annotations": int(bad_annotation_count),  # Ensure integer
                "total_duration_seconds": float(total_bad_duration),  # Ensure float
                "total_duration_minutes": round(float(total_bad_duration) / 60, 2),
                "percent_of_recording": percent_recording,
                "distinct_annotation_types": sorted(
                    list(distinct_annotation_types)
                ),  # Sort for consistency
            },
            "ica_components": rejected_ics,
            "n_components": n_components,
        }
    }

    try:
        manage_database(
            operation="update",
            update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
        )
    except Exception as e:
        raise RuntimeError(f"Failed to update database: {str(e)}")

    ##remove annotations before saving
    # cleaned_raw.set_annotations(None)
    save_raw_to_set(cleaned_raw, autoclean_dict, "post_rejection_policy")

    return pipeline, cleaned_raw


def _get_rejection_policy(autoclean_dict: dict) -> dict:
    task = autoclean_dict["task"]
    # Create a new rejection policy for cleaning channels and removing ICs
    rejection_policy = ll.RejectionPolicy()

    # Set parameters for channel rejection
    rejection_policy["ch_flags_to_reject"] = autoclean_dict["tasks"][task][
        "rejection_policy"
    ]["ch_flags_to_reject"]
    rejection_policy["ch_cleaning_mode"] = autoclean_dict["tasks"][task][
        "rejection_policy"
    ]["ch_cleaning_mode"]
    rejection_policy["interpolate_bads_kwargs"] = {
        "method": autoclean_dict["tasks"][task]["rejection_policy"][
            "interpolate_bads_kwargs"
        ]["method"]
    }

    # Set parameters for IC rejection
    rejection_policy["ic_flags_to_reject"] = autoclean_dict["tasks"][task][
        "rejection_policy"
    ]["ic_flags_to_reject"]
    rejection_policy["ic_rejection_threshold"] = autoclean_dict["tasks"][task][
        "rejection_policy"
    ]["ic_rejection_threshold"]
    rejection_policy["remove_flagged_ics"] = autoclean_dict["tasks"][task][
        "rejection_policy"
    ]["remove_flagged_ics"]

    # Add metadata about rejection policy
    metadata = {
        "rejection_policy": {
            "creationDateTime": datetime.now().isoformat(),
            "task": task,
            "ch_flags_to_reject": rejection_policy["ch_flags_to_reject"],
            "ch_cleaning_mode": rejection_policy["ch_cleaning_mode"],
            "interpolate_method": rejection_policy["interpolate_bads_kwargs"]["method"],
            "ic_flags_to_reject": rejection_policy["ic_flags_to_reject"],
            "ic_rejection_threshold": rejection_policy["ic_rejection_threshold"],
            "remove_flagged_ics": rejection_policy["remove_flagged_ics"],
        }
    }

    manage_database(
        operation="update",
        update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
    )

    # Log rejection policy details using messaging function
    message("info", "Rejection Policy Settings:")
    message(
        "info", f"Channel flags to reject: {rejection_policy['ch_flags_to_reject']}"
    )
    message("info", f"Channel cleaning mode: {rejection_policy['ch_cleaning_mode']}")
    message(
        "info",
        f"Interpolation method: {rejection_policy['interpolate_bads_kwargs']['method']}",
    )
    message("info", f"IC flags to reject: {rejection_policy['ic_flags_to_reject']}")
    message(
        "info", f"IC rejection threshold: {rejection_policy['ic_rejection_threshold']}"
    )
    message("info", f"Remove flagged ICs: {rejection_policy['remove_flagged_ics']}")

    return rejection_policy


def _extended_BAD_LL_noisy_ICs_annotations(
    raw, pipeline, autoclean_dict, extra_duration=1
):
    from collections import OrderedDict

    # Extend each annotation by 1 second on each side
    for ann in raw.annotations:
        new_annotation = OrderedDict(
            [
                ("onset", np.float64(ann["onset"])),
                ("duration", np.float64(ann["duration"])),
                ("description", np.str_(ann["description"])),
                ("orig_time", ann.get("orig_time", None)),
            ]
        )

        # print(
        #     f"'{new_annotation['description']}' goes from {new_annotation['onset']} to {new_annotation['onset'] + new_annotation['duration']}"
        # )

    from collections import OrderedDict

    updated_annotations = []
    for annotation in raw.annotations:
        if annotation["description"] == "BAD_LL_noisy_ICs":
            start = annotation["onset"]  # Extend start by 1 second
            duration = (
                annotation["duration"] + extra_duration
            )  # Extend duration by extra_duration
            new_ann = mne.Annotations(
                onset=start, duration=duration, description=annotation["description"]
            )
            updated_annotations.append(new_ann)
        else:
            updated_annotations.append(annotation)  # Keep other annotations unchanged

    # Create new annotation structure from updated_annotations list
    combined_onset = []
    combined_duration = []
    combined_description = []
    combined_orig_time = None

    # Extract data from each annotation
    for ann in updated_annotations:
        if isinstance(ann, mne.Annotations):
            # Handle single annotation objects
            combined_onset.extend(ann.onset)
            combined_duration.extend(ann.duration)
            combined_description.extend(ann.description)
            if combined_orig_time is None and hasattr(ann, "orig_time"):
                combined_orig_time = ann.orig_time
        else:
            # Handle individual annotation entries
            combined_onset.append(ann["onset"])
            combined_duration.append(ann["duration"])
            combined_description.append(ann["description"])
            if combined_orig_time is None and "orig_time" in ann:
                combined_orig_time = ann["orig_time"]

    # Create new consolidated Annotations object
    new_annotations = mne.Annotations(
        onset=np.array(combined_onset),
        duration=np.array(combined_duration),
        description=np.array(combined_description),
        orig_time=combined_orig_time,
    )

    raw.set_annotations(new_annotations)

    # Extract indices and info for BAD_LL_noisy_ICs after extension
    bad_indices = np.where(raw.annotations.description == "BAD_LL_noisy_ICs")[0]
    n_segments = len(bad_indices)

    if n_segments > 0:
        # Create figure with subplots for each segment
        fig, axes = plt.subplots(
            n_segments, 1, figsize=(15, 4 * n_segments), sharex=True
        )
        if n_segments == 1:
            axes = [axes]  # Ensure axes is iterable

        sfreq = raw.info["sfreq"]
        for idx, i_ann in enumerate(bad_indices):
            onset = raw.annotations.onset[i_ann]
            duration = raw.annotations.duration[i_ann]

            # Calculate start and end times with padding
            start_time = max(onset - 5, raw.times[0])
            end_time = min(onset + duration + 5, raw.times[-1])

            # Convert times to sample indices
            start_sample = raw.time_as_index(start_time)[0]
            end_sample = raw.time_as_index(end_time)[0]

            # Get data and corresponding times
            data, times = raw.get_data(
                start=start_sample, stop=end_sample, return_times=True
            )

            # Plot the data
            axes[idx].plot(times, data.T, "k", linewidth=0.5, alpha=0.5)

            # Highlight the annotation region
            axes[idx].axvspan(
                onset,
                onset + duration,
                color="red",
                alpha=0.2,
                label="BAD_LL_noisy_ICs",
            )

            # Add vertical lines at annotation boundaries
            axes[idx].axvline(onset, color="red", linestyle="--", alpha=0.5)
            axes[idx].axvline(onset + duration, color="red", linestyle="--", alpha=0.5)

            axes[idx].set_title(
                f"Segment {idx + 1}: {onset:.1f}s - {(onset + duration):.1f}s",
                fontsize=10,
            )
            axes[idx].set_ylabel("Amplitude")
            axes[idx].legend(loc="upper right")

        axes[-1].set_xlabel("Time (s)")

        plt.tight_layout()

        # Save figure
        derivatives_path = pipeline.get_derivative_path(autoclean_dict["bids_path"])
        target_figure = str(
            derivatives_path.copy().update(
                suffix="bad_ll_noisy_segments", extension=".pdf", datatype="eeg"
            )
        )

        fig.savefig(target_figure, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Add catalog of BAD_LL_noisy_ICs to metadata
    bad_ll_noisy_ics_count = (
        sum(
            1
            for ann in updated_annotations
            if isinstance(ann, mne.Annotations)
            and ann.description == "BAD_LL_noisy_ICs"
        )
        if updated_annotations
        else 0
    )

    # Initialize metadata with required fields
    metadata = {
        "extended_BAD_LL_noisy_ICs_annotations": {
            "creationDateTime": datetime.now().isoformat(),
            "extended_BAD_LL_noisy_ICs_annotations": True,
            "extra_duration": extra_duration,
            "bad_LL_noisy_ICs_count": bad_ll_noisy_ics_count,  # Count of BAD_LL_noisy_ICs
        }
    }

    # Only add figure info if it was generated
    if "target_figure" in locals():
        metadata["extended_BAD_LL_noisy_ICs_annotations"][
            "extended_BAD_LL_noisy_ICs_annotations_figure"
        ] = Path(target_figure).name

    manage_database(
        operation="update",
        update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
    )

    return raw


def plot_bad_channels_with_topography(
    raw_original, raw_cleaned, pipeline, autoclean_dict, zoom_duration=30, zoom_start=0
):
    """
    Plot bad channels with a topographical map and time series overlays for both full duration and a zoomed-in window.

    Parameters:
    -----------
    raw_original : mne.io.Raw
        Original raw EEG data before cleaning.
    raw_cleaned : mne.io.Raw
        Cleaned raw EEG data after interpolation of bad channels.
    pipeline : pylossless.Pipeline
        Pipeline object containing flags and raw data.
    autoclean_dict : dict
        Autoclean dictionary containing metadata.
    zoom_duration : float, optional
        Duration in seconds for the zoomed-in time series plot. Default is 30 seconds.
    zoom_start : float, optional
        Start time in seconds for the zoomed-in window. Default is 0 seconds.
    """
    import matplotlib.pyplot as plt
    import mne
    import numpy as np
    from matplotlib.gridspec import GridSpec

    # ----------------------------
    # 1. Collect Bad Channels
    # ----------------------------
    bad_channels_info = {}

    # Mapping from channel to reason(s)
    for reason, channels in pipeline.flags.get("ch", {}).items():
        for ch in channels:
            if ch in bad_channels_info:
                if reason not in bad_channels_info[ch]:
                    bad_channels_info[ch].append(reason)
            else:
                bad_channels_info[ch] = [reason]

    bad_channels = list(bad_channels_info.keys())

    if not bad_channels:
        print("No bad channels were identified.")
        return

    # Debugging: Print bad channels
    print(f"Identified Bad Channels: {bad_channels}")

    # ----------------------------
    # 2. Identify Good Channels
    # ----------------------------
    all_channels = raw_original.ch_names
    good_channels = [ch for ch in all_channels if ch not in bad_channels]

    # Debugging: Print good channels count
    print(f"Number of Good Channels: {len(good_channels)}")

    # ----------------------------
    # 3. Extract Data for Bad Channels
    # ----------------------------
    picks_bad_original = mne.pick_channels(raw_original.ch_names, bad_channels)
    picks_bad_cleaned = mne.pick_channels(raw_cleaned.ch_names, bad_channels)

    if len(picks_bad_original) == 0:
        print("No bad channels found in original data.")
        return

    if len(picks_bad_cleaned) == 0:
        print("No bad channels found in cleaned data.")
        return

    data_original, times = raw_original.get_data(
        picks=picks_bad_original, return_times=True
    )
    data_cleaned = raw_cleaned.get_data(picks=picks_bad_cleaned)

    channel_labels = [raw_original.ch_names[i] for i in picks_bad_original]
    n_channels = len(channel_labels)

    # Debugging: Print number of bad channels being plotted
    print(f"Number of Bad Channels to Plot: {n_channels}")

    # ----------------------------
    # 4. Downsample Data if Necessary
    # ----------------------------
    sfreq = raw_original.info["sfreq"]
    desired_sfreq = 100  # Target sampling rate
    downsample_factor = int(sfreq // desired_sfreq)
    if downsample_factor > 1:
        data_original = data_original[:, ::downsample_factor]
        data_cleaned = data_cleaned[:, ::downsample_factor]
        times = times[::downsample_factor]
        print(
            f"Data downsampled by a factor of {downsample_factor} to {desired_sfreq} Hz."
        )

    # ----------------------------
    # 5. Normalize and Scale Data
    # ----------------------------
    data_original_normalized = np.zeros_like(data_original)
    data_cleaned_normalized = np.zeros_like(data_cleaned)
    # Dynamic spacing based on number of bad channels
    spacing = 10 + (n_channels * 2)  # Adjusted spacing

    for idx in range(n_channels):
        channel_data_original = data_original[idx]
        channel_data_cleaned = data_cleaned[idx]
        # Remove DC offset
        channel_data_original -= np.mean(channel_data_original)
        channel_data_cleaned -= np.mean(channel_data_cleaned)
        # Normalize by standard deviation
        std_orig = np.std(channel_data_original)
        std_clean = np.std(channel_data_cleaned)
        if std_orig == 0:
            std_orig = 1  # Prevent division by zero
        if std_clean == 0:
            std_clean = 1
        data_original_normalized[idx] = channel_data_original / std_orig
        data_cleaned_normalized[idx] = channel_data_cleaned / std_clean

    # Scaling factor for better visibility
    scaling_factor = 5  # Increased scaling factor
    data_original_scaled = data_original_normalized * scaling_factor
    data_cleaned_scaled = data_cleaned_normalized * scaling_factor

    # Calculate offsets
    offsets = np.arange(n_channels) * spacing

    # ----------------------------
    # 6. Define Zoom Window
    # ----------------------------
    zoom_end = zoom_start + zoom_duration
    if zoom_end > times[-1]:
        zoom_end = times[-1]
        zoom_start = max(zoom_end - zoom_duration, times[0])

    # ----------------------------
    # 7. Create Figure with GridSpec
    # ----------------------------
    fig_height = 10 + (n_channels * 0.3)
    fig = plt.figure(constrained_layout=True, figsize=(20, fig_height))
    gs = GridSpec(3, 2, figure=fig)

    # ----------------------------
    # 8. Topography Subplot
    # ----------------------------
    ax_topo = fig.add_subplot(gs[0, :])

    # Plot sensors with ch_groups for good and bad channels
    ch_groups = [
        [int(raw_original.ch_names.index(ch)) for ch in good_channels],
        [int(raw_original.ch_names.index(ch)) for ch in bad_channels],
    ]
    colors = "RdYlBu_r"

    # Plot again for the main figure subplot
    mne.viz.plot_sensors(
        raw_original.info,
        kind="topomap",
        ch_type="eeg",
        title="Sensor Topography: Good vs Bad Channels",
        show_names=True,
        ch_groups=ch_groups,
        pointsize=75,
        linewidth=0,
        cmap=colors,
        show=False,
        axes=ax_topo,
    )

    ax_topo.legend(["Good Channels", "Bad Channels"], loc="upper right", fontsize=12)
    ax_topo.set_title("Topography of Good and Bad Channels", fontsize=16)

    # ----------------------------
    # 9. Full Duration Time Series Subplot
    # ----------------------------
    ax_full = fig.add_subplot(gs[1, 0])
    for idx in range(n_channels):
        # Plot original data
        ax_full.plot(
            times,
            data_original_scaled[idx] + offsets[idx],
            color="red",
            linewidth=1,
            linestyle="-",
        )
        # Plot cleaned data
        ax_full.plot(
            times,
            data_cleaned_scaled[idx] + offsets[idx],
            color="black",
            linewidth=1,
            linestyle="-",
        )

    ax_full.set_xlabel("Time (seconds)", fontsize=14)
    ax_full.set_ylabel("Bad Channels", fontsize=14)
    ax_full.set_title(
        "Bad Channels: Original vs Interpolated (Full Duration)", fontsize=16
    )
    ax_full.set_xlim(times[0], times[-1])
    ax_full.set_ylim(-spacing, offsets[-1] + spacing)
    ax_full.set_yticks([])  # Hide y-ticks
    ax_full.invert_yaxis()

    # Add legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="red", lw=2, linestyle="-", label="Original Data"),
        Line2D([0], [0], color="black", lw=2, linestyle="-", label="Interpolated Data"),
    ]
    ax_full.legend(handles=legend_elements, loc="upper right", fontsize=12)

    # ----------------------------
    # 10. Zoomed-In Time Series Subplot
    # ----------------------------
    ax_zoom = fig.add_subplot(gs[1, 1])
    for idx in range(n_channels):
        # Plot original data
        ax_zoom.plot(
            times,
            data_original_scaled[idx] + offsets[idx],
            color="red",
            linewidth=1,
            linestyle="-",
        )
        # Plot cleaned data
        ax_zoom.plot(
            times,
            data_cleaned_scaled[idx] + offsets[idx],
            color="black",
            linewidth=1,
            linestyle="-",
        )

    ax_zoom.set_xlabel("Time (seconds)", fontsize=14)
    ax_zoom.set_title(
        f"Bad Channels: Original vs Interpolated (Zoom: {zoom_start}-{zoom_end} s)",
        fontsize=16,
    )
    ax_zoom.set_xlim(zoom_start, zoom_end)
    ax_zoom.set_ylim(-spacing, offsets[-1] + spacing)
    ax_zoom.set_yticks([])  # Hide y-ticks
    ax_zoom.invert_yaxis()

    # Add legend
    ax_zoom.legend(handles=legend_elements, loc="upper right", fontsize=12)

    # ----------------------------
    # 11. Add Channel Labels
    # ----------------------------
    for idx, ch in enumerate(channel_labels):
        label = f"{ch}\n({', '.join(bad_channels_info[ch])})"
        ax_full.text(
            times[0] - (0.05 * (times[-1] - times[0])),
            offsets[idx],
            label,
            horizontalalignment="right",
            fontsize=10,
            verticalalignment="center",
        )

    # ----------------------------
    # 12. Finalize and Save the Figure
    # ----------------------------
    plt.tight_layout()

    # Get output path for bad channels figure
    bids_path = autoclean_dict.get("bids_path", "")
    if bids_path:
        derivatives_path = pipeline.get_derivative_path(bids_path)
    else:
        derivatives_path = "."

    # Assuming pipeline.get_derivative_path returns a Path-like object with a copy method
    # and update method as per the initial code
    try:
        target_figure = str(
            derivatives_path.copy().update(
                suffix="step_bad_channels_with_map", extension=".png", datatype="eeg"
            )
        )
    except AttributeError:
        # Fallback if copy or update is not implemented
        target_figure = os.path.join(
            derivatives_path, "bad_channels_with_topography.png"
        )

    # Save the figure
    fig.savefig(target_figure, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Bad channels with topography plot saved to {target_figure}")

    metadata = {
        "artifact_reports": {
            "creationDateTime": datetime.now().isoformat(),
            "plot_bad_channels_with_topography": Path(target_figure).name,
        }
    }

    manage_database(
        operation="update",
        update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
    )

    return fig


def extend_annotations(
    raw, pipeline, autoclean_dict, extra_duration=0.5, label="BAD_LL_noisy_ICs"
):
    from collections import OrderedDict

    # Extend each annotation by 1 second on each side
    for ann in raw.annotations:
        new_annotation = OrderedDict(
            [
                ("onset", np.float64(ann["onset"])),
                ("duration", np.float64(ann["duration"])),
                ("description", np.str_(ann["description"])),
                ("orig_time", ann.get("orig_time", None)),
            ]
        )

        # print(
        #     f"'{new_annotation['description']}' goes from {new_annotation['onset']} to {new_annotation['onset'] + new_annotation['duration']}"
        # )

    from collections import OrderedDict

    updated_annotations = []
    for annotation in raw.annotations:
        if annotation["description"] == label:
            start = annotation["onset"]  # Extend start by 1 second
            duration = (
                annotation["duration"] + extra_duration
            )  # Extend duration by extra_duration
            new_ann = mne.Annotations(
                onset=start, duration=duration, description=annotation["description"]
            )
            updated_annotations.append(new_ann)
        else:
            updated_annotations.append(annotation)  # Keep other annotations unchanged

    # Create new annotation structure from updated_annotations list
    combined_onset = []
    combined_duration = []
    combined_description = []
    combined_orig_time = None

    # Extract data from each annotation
    for ann in updated_annotations:
        if isinstance(ann, mne.Annotations):
            # Handle single annotation objects
            combined_onset.extend(ann.onset)
            combined_duration.extend(ann.duration)
            combined_description.extend(ann.description)
            if combined_orig_time is None and hasattr(ann, "orig_time"):
                combined_orig_time = ann.orig_time
        else:
            # Handle individual annotation entries
            combined_onset.append(ann["onset"])
            combined_duration.append(ann["duration"])
            combined_description.append(ann["description"])
            if combined_orig_time is None and "orig_time" in ann:
                combined_orig_time = ann["orig_time"]

    # Create new consolidated Annotations object
    new_annotations = mne.Annotations(
        onset=np.array(combined_onset),
        duration=np.array(combined_duration),
        description=np.array(combined_description),
        orig_time=combined_orig_time,
    )

    raw.set_annotations(new_annotations)

    # Extract indices and info for annotations
    bad_indices = np.where(raw.annotations.description == label)[0]
    n_segments = len(bad_indices)

    if n_segments > 0:
        # Create figure with subplots for each segment
        fig, axes = plt.subplots(
            n_segments, 1, figsize=(15, 4 * n_segments), sharex=True
        )
        if n_segments == 1:
            axes = [axes]  # Ensure axes is iterable

        sfreq = raw.info["sfreq"]
        for idx, i_ann in enumerate(bad_indices):
            onset = raw.annotations.onset[i_ann]
            duration = raw.annotations.duration[i_ann]

            # Calculate start and end times with padding
            start_time = max(onset - 5, raw.times[0])
            end_time = min(onset + duration + 5, raw.times[-1])

            # Convert times to sample indices
            start_sample = raw.time_as_index(start_time)[0]
            end_sample = raw.time_as_index(end_time)[0]

            # Get data and corresponding times
            data, times = raw.get_data(
                start=start_sample, stop=end_sample, return_times=True
            )

            # Plot the data
            axes[idx].plot(times, data.T, "k", linewidth=0.5, alpha=0.5)

            # Highlight the annotation region
            axes[idx].axvspan(
                onset,
                onset + duration,
                color="red",
                alpha=0.2,
                label=label,
            )

            # Add vertical lines at annotation boundaries
            axes[idx].axvline(onset, color="red", linestyle="--", alpha=0.5)
            axes[idx].axvline(onset + duration, color="red", linestyle="--", alpha=0.5)

            axes[idx].set_title(
                f"Segment {idx + 1}: {onset:.1f}s - {(onset + duration):.1f}s",
                fontsize=10,
            )
            axes[idx].set_ylabel("Amplitude")
            axes[idx].legend(loc="upper right")

        axes[-1].set_xlabel("Time (s)")

        plt.tight_layout()

        # Save figure
        derivatives_path = pipeline.get_derivative_path(autoclean_dict["bids_path"])
        target_figure = str(
            derivatives_path.copy().update(
                suffix=label, extension=".pdf", datatype="eeg"
            )
        )

        fig.savefig(target_figure, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Add catalog of annotations to metadata
    annotations_count = (
        sum(
            1
            for ann in updated_annotations
            if isinstance(ann, mne.Annotations) and ann.description == label
        )
        if updated_annotations
        else 0
    )

    # Initialize metadata with required fields
    metadata = {
        "extended_annotations": {
            "creationDateTime": datetime.now().isoformat(),
            "extended_annotations": True,
            "extra_duration": extra_duration,
            "annotations_count": annotations_count,  # Count of annotations
        }
    }

    # Only add figure info if it was generated
    if "target_figure" in locals():
        metadata["extended_annotations"]["extended_annotations_figure"] = Path(
            target_figure
        ).name

    manage_database(
        operation="update",
        update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
    )

    return raw
