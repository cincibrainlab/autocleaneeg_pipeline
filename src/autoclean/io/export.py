"""Export functions for autoclean pipeline."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import mne
import numpy as np
import scipy.io as sio
from eeglabio.epochs import export_set

from autoclean.utils.database import manage_database_conditionally
from autoclean.utils.logging import message

__all__ = [
    "save_stc_to_file",
    "save_raw_to_set",
    "save_epochs_to_set",
    "copy_final_files",
    "_get_stage_number",
]


def save_stc_to_file(
    stc: mne.SourceEstimate,
    autoclean_dict: Dict[str, Any],
    stage: str = "post_source_localization",
    output_path: Optional[Path] = None,
) -> Path:
    """Save source estimate (STC) data to file.

    This function saves an MNE SourceEstimate object at a specified processing stage,
    consistent with the pipeline's directory structure and configuration.

    Parameters
    ----------
        stc : mne.SourceEstimate
            SourceEstimate object to save
        autoclean_dict : dict
            Configuration dictionary
        stage : str
            Processing stage identifier (default: "post_source_localization")
        output_path : Optional[Path]
            Optional custom output path. If None, uses config

    Returns
    -------
        Path: Path
            Path to the saved file (stage path)

    """
    # Generate suffix from stage name
    suffix = f"_{stage.replace('post_', '')}"
    basename = Path(autoclean_dict["unprocessed_file"]).stem
    stage_num = _get_stage_number(stage, autoclean_dict)

    # Determine output path - BIDS-compliant structure
    if output_path is None:
        output_path = autoclean_dict["stage_dir"]
    subfolder = output_path / f"{stage_num}{suffix}"
    subfolder.mkdir(parents=True, exist_ok=True)
    stage_path = subfolder / f"{basename}{suffix}-stc.h5"

    # Only save to stage directory - final files will be copied separately
    paths = [stage_path]

    # Save the STC to all specified paths
    for path in paths:
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            stc.save(fname=path, ftype="h5", overwrite=True, verbose=False)
            message("success", f"✓ Saved {stage} STC file to: {path}")
        except Exception as e:
            error_msg = f"Failed to save {stage} STC file to {path}: {str(e)}"
            message("error", error_msg)
            raise RuntimeError(error_msg) from e

    # Create metadata for database logging
    metadata = {
        "save_stc_to_file": {
            "creationDateTime": datetime.now().isoformat(),
            "stage": stage,
            "stage_number": stage_num,
            "outputPaths": [str(p) for p in paths],
            "suffix": suffix,
            "basename": basename,
            "format": "h5",
            "n_vertices": stc.data.shape[0],
            "n_times": stc.data.shape[1],
            "tmin": stc.tmin,
            "tstep": stc.tstep,
        }
    }

    # Update database
    run_id = autoclean_dict["run_id"]
    manage_database_conditionally(
        operation="update", update_record={"run_id": run_id, "metadata": metadata}
    )
    manage_database_conditionally(
        operation="update_status",
        update_record={"run_id": run_id, "status": f"{stage} completed"},
    )

    return paths[0]  # Return stage path for consistency


def save_raw_to_set(
    raw: mne.io.Raw,
    autoclean_dict: Dict[str, Any],
    stage: str = "post_import",
    output_path: Optional[Path] = None,
    flagged: bool = False,
) -> Path:
    """Save continuous EEG data to file.

    This function saves raw EEG data at various processing stages.

    Parameters
    ----------
        raw : mne.io.Raw
            Raw EEG data to save
        autoclean_dict : dict
            Configuration dictionary
        stage : str
            Processing stage identifier (e.g., "post_import")
        output_path : Optional[Path]
            Optional custom output path. If None, uses config
        flagged : bool
            If True, appends FLAGGED_ to the stage file name

    Returns
    -------
        Path: Path
            Path to the saved file (stage path)

    """

    # Generate suffix from stage name
    suffix = f"_{stage.replace('post_', '')}"
    basename = Path(autoclean_dict["unprocessed_file"]).stem
    stage_num = _get_stage_number(stage, autoclean_dict)

    flagged = flagged and autoclean_dict.get("move_flagged_files", True)

    # Save to BIDS-compliant intermediate directory structure
    if flagged:
        output_path = autoclean_dict["stage_dir"]
        subfolder = output_path / f"FLAGGED_{stage_num}{suffix}"
    elif output_path is None:
        output_path = autoclean_dict["stage_dir"]
        subfolder = output_path / f"{stage_num}{suffix}"
    else:
        subfolder = output_path
    subfolder.mkdir(parents=True, exist_ok=True)
    stage_path = subfolder / f"{basename}{suffix}_raw.set"

    # Only save to stage directory - final files will be copied separately
    paths = [stage_path]

    # Save to all paths
    raw.info["description"] = autoclean_dict["run_id"]
    for path in paths:
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            raw.export(path, fmt="eeglab", overwrite=True)
            message("success", f"✓ Saved {stage} file to: {path}")
        except Exception as e:
            error_msg = f"Failed to save {stage} file to {path}: {str(e)}"
            message("error", error_msg)
            # For dynamic stages, provide more helpful error information
            if stage not in autoclean_dict["stage_files"]:
                message(
                    "info",
                    f"Note: Stage '{stage}' was auto-generated. Check directory permissions and disk space.",
                )
            raise RuntimeError(error_msg) from e

    metadata = {
        "save_raw_to_set": {
            "creationDateTime": datetime.now().isoformat(),
            "stage": stage,
            "stage_number": stage_num,
            "outputPath": str(stage_path),
            "suffix": suffix,
            "basename": basename,
            "format": "eeglab",
            "n_channels": len(raw.ch_names),
            "actual_sfreq": raw.info["sfreq"],
            "actual_duration": raw.times[-1] - raw.times[0],
        }
    }

    run_id = autoclean_dict["run_id"]
    manage_database_conditionally(
        operation="update", update_record={"run_id": run_id, "metadata": metadata}
    )
    manage_database_conditionally(
        operation="update_status",
        update_record={"run_id": run_id, "status": f"{stage} completed"},
    )

    return paths[0]  # Return stage path for consistency


def save_epochs_to_set(
    epochs: mne.Epochs,
    autoclean_dict: Dict[str, Any],
    stage: str = "post_clean_epochs",
    output_path: Optional[Path] = None,
    flagged: bool = False,
) -> Path:
    """Save epoched EEG data to EEGLAB .set format with metadata preservation.

    Parameters
    ----------
        epochs : mne.Epochs
            The epoched EEG data to save
        autoclean_dict : Dict[str, Any]
            Pipeline configuration containing stage settings, paths, and run identifier
        stage : str, default="post_clean_epochs"
            Processing stage identifier used for file naming and organization
        output_path : Optional[Path], default=None
            Custom output directory; if None, uses stage_dir from config
        flagged : bool, default=False
            If True, appends FLAGGED_ to the stage file name

    Returns
    -------
        Path: Path
            Path to the saved file (stage path)

    """

    # Generate suffix from stage name
    suffix = f"_{stage.replace('post_', '')}"
    basename = Path(autoclean_dict["unprocessed_file"]).stem
    stage_num = _get_stage_number(stage, autoclean_dict)

    flagged = flagged and autoclean_dict.get("move_flagged_files", True)

    # Determine output directory based on flagged status - BIDS-compliant structure
    if flagged:
        output_path = autoclean_dict["stage_dir"]
        subfolder = output_path / f"FLAGGED_{stage_num}{suffix}"
    elif output_path is None:
        output_path = autoclean_dict["stage_dir"]
        subfolder = output_path / f"{stage_num}{suffix}"
    else:
        subfolder = output_path
    subfolder.mkdir(parents=True, exist_ok=True)
    stage_path = subfolder / f"{basename}{suffix}_epo.set"

    # Only save to stage directory - final files will be copied separately
    paths = [stage_path]

    # Handle epoch metadata for event preservation
    if epochs.metadata is None:
        message("warning", "No additional event metadata found for epochs")
        events_in_epochs = None
        event_id_rebuilt = None
    else:
        try:
            # Check for metadata-events alignment
            if len(epochs.metadata) != len(epochs.events):
                message(
                    "warning",
                    "Mismatch in metadata vs events: "
                    f"{len(epochs.metadata)} vs {len(epochs.events)} — truncating to align.",
                )

            # Extract events from metadata if available
            if (
                "additional_events"
                in epochs.metadata.columns
                # and not epochs.metadata["additional_events"].empty # This check might be too simple if NaNs are present
            ):
                # Calculate timing parameters for event reconstruction
                sfreq = epochs.info["sfreq"]
                # offset = int(round(-epochs.tmin * sfreq))  # Samples from epoch start to time 0. Will recalculate sample pos directly.
                n_samples = len(epochs.times)  # Total samples per epoch

                # Build event dictionary from all unique event labels
                all_labels = set()
                # Iterate over potentially NaN-containing 'additional_events' Series safely
                for additional_event_list_for_epoch in epochs.metadata[
                    "additional_events"
                ].dropna():
                    if isinstance(additional_event_list_for_epoch, list):
                        for event_tuple in additional_event_list_for_epoch:
                            # Ensure the event_tuple is a tuple/list and has at least one element (the label)
                            if (
                                isinstance(event_tuple, (list, tuple))
                                and len(event_tuple) >= 1
                            ):
                                label = event_tuple[0]  # Get the label
                                all_labels.add(str(label))  # Ensure label is a string
                            else:
                                message(
                                    "debug",
                                    f"Skipping malformed event tuple: {event_tuple} in additional_events.",
                                )
                    # else: it's not a list after dropna(), so it was NaN or another non-list type.

                # Preserve original event codes instead of creating sequential ones
                event_id_rebuilt = {}
                if hasattr(epochs, "event_id") and epochs.event_id:
                    # Use original event codes from epochs object
                    for label in all_labels:
                        # Find matching event code from original event_id
                        for orig_label, orig_code in epochs.event_id.items():
                            if str(orig_label) == str(label):
                                event_id_rebuilt[label] = orig_code
                                break
                        else:
                            # Fallback: if no match found, use sequential numbering
                            event_id_rebuilt[label] = len(event_id_rebuilt) + 1
                else:
                    # Fallback: if no original event_id available, use sequential numbering
                    event_id_rebuilt = {
                        label: idx + 1
                        for idx, label in enumerate(sorted(list(all_labels)))
                    }
                # Reconstruct events array with global sample positions
                events_in_epochs = []
                used_samples = set()  # Track used samples to prevent collisions

                # Iterate through metadata rows, but ensure we don't go beyond the actual number of events
                # This directly addresses the "82 vs 77" mismatch.
                for i, meta_row_tuple in enumerate(
                    epochs.metadata.head(len(epochs.events)).itertuples(
                        index=False, name="Row"
                    )
                ):
                    current_additional_events_for_epoch = getattr(
                        meta_row_tuple, "additional_events", None
                    )
                    if isinstance(current_additional_events_for_epoch, list):
                        for label, rel_time in current_additional_events_for_epoch:
                            try:
                                # rel_time is time from epoch's t=0 (trigger).
                                # epochs.tmin is the start of the epoch data window relative to t=0.
                                # Sample index for rel_time within the epoch's data array (0 to n_samples-1) is:
                                event_sample_within_epoch_data = int(
                                    round((float(rel_time) - epochs.tmin) * sfreq)
                                )

                                # Check bounds: sample must be within the current epoch's data segment [0, n_samples-1]
                                if not (
                                    0 <= event_sample_within_epoch_data < n_samples
                                ):
                                    message(
                                        "warning",
                                        f"Epoch {i}, event '{label}': rel_time {rel_time}s -> sample {event_sample_within_epoch_data} is outside epoch data window [0, {n_samples-1}]. Skipping.",
                                    )
                                    continue

                                # global_sample is for concatenated data, as expected by eeglabio.export_set
                                # It's the start of the i-th epoch's data block + the sample within that block.
                                global_sample = (
                                    i * n_samples
                                ) + event_sample_within_epoch_data

                                # Prevent sample collisions by incrementing if needed
                                while global_sample in used_samples:
                                    global_sample += 1
                                used_samples.add(global_sample)

                                str_label = str(
                                    label
                                )  # Ensure label is string for dict lookup
                                if str_label not in event_id_rebuilt:
                                    message(
                                        "warning",
                                        f"Label '{str_label}' (from epoch {i}, rel_time {rel_time}) not found in rebuilt event_id. Available: {list(event_id_rebuilt.keys())}. Skipping this event.",
                                    )
                                    continue
                                code = event_id_rebuilt[str_label]
                                events_in_epochs.append(
                                    [global_sample, 0, code]
                                )  # Assuming duration 0 for point events
                            except ValueError:
                                message(
                                    "warning",
                                    f"Epoch {i}, event '{label}': Could not convert rel_time '{rel_time}' to float. Skipping this event.",
                                )
                                continue
                            except (
                                Exception
                            ) as e_inner:  # pylint: disable=broad-exception-caught
                                message(
                                    "error",
                                    f"Unexpected error processing event '{label}' in epoch {i} (rel_time: {rel_time}): {e_inner}",
                                )
                                continue
                    # else: current_additional_events_for_epoch is not a list (e.g., NaN), so skip for this epoch.

                if events_in_epochs:  # Only convert to numpy array if list is not empty
                    events_in_epochs = np.array(events_in_epochs, dtype=int)
                else:  # If list is empty, set to None or an empty array as eeglabio expects
                    events_in_epochs = None  # Or np.empty((0,3), dtype=int) depending on eeglabio's preference for empty

            else:
                message(
                    "warning",
                    "No 'additional_events' column found in epochs.metadata or it is empty.",
                )
                events_in_epochs = None
        except Exception as e:  # pylint: disable=broad-exception-caught
            message("error", f"Failed to rebuild events_in_epochs: {str(e)}")

    # Save to all target paths
    epochs.info["description"] = autoclean_dict["run_id"]
    epochs.apply_proj()  # Apply projectors before saving
    for path in paths:
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # Use specialized export for preserving complex event structures
            if events_in_epochs is not None and len(events_in_epochs) > 0:

                export_set(
                    fname=str(path),
                    data=epochs.get_data(),
                    sfreq=epochs.info["sfreq"],
                    events=events_in_epochs,
                    tmin=epochs.tmin,
                    tmax=epochs.tmax,
                    ch_names=epochs.ch_names,
                    event_id=event_id_rebuilt,
                    precision="single",
                )
            else:
                # Use MNE's built-in exporter for simple cases
                epochs.export(path, fmt="eeglab", overwrite=True)
            # Add run_id to EEGLAB's etc field for tracking
            # pylint: disable=invalid-name
            EEG = sio.loadmat(path)
            for k in ["__header__", "__version__", "__globals__"]:
                EEG.pop(k, None)
            EEG["etc"] = {}
            EEG["etc"]["run_id"] = autoclean_dict["run_id"]
            sio.savemat(path, EEG, do_compression=False)
            message("success", f"✓ Saved {stage} file to: {path}")
        except Exception as e:
            error_msg = f"Failed to save {stage} file to {path}: {str(e)}"
            message("error", error_msg)
            raise RuntimeError(error_msg) from e

    # Record save operation in database
    metadata = {
        "save_epochs_to_set": {
            "creationDateTime": datetime.now().isoformat(),
            "stage": stage,
            "stage_number": stage_num,
            "outputPaths": [str(p) for p in paths],
            "suffix": suffix,
            "basename": basename,
            "format": "eeglab",
            "n_epochs": len(epochs),
            "n_channels": len(epochs.ch_names),
            "actual_sfreq": epochs.info["sfreq"],
            "actual_duration": len(epochs) * (epochs.tmax - epochs.tmin),
            "tmin": epochs.tmin,
            "tmax": epochs.tmax,
        }
    }

    # Update database with save metadata and status
    run_id = autoclean_dict["run_id"]
    manage_database_conditionally(
        operation="update", update_record={"run_id": run_id, "metadata": metadata}
    )
    manage_database_conditionally(
        operation="update_status",
        update_record={"run_id": run_id, "status": f"{stage} completed"},
    )

    return paths[0]  # Return stage path for consistency


def save_ica_to_fif(ica, autoclean_dict, pre_ica_raw):
    """Save ICA results to FIF files.

    This function saves ICA results to FIF files in the derivatives directory.

    Parameters
    ----------
        ica : mne.preprocessing.ICA
            ICA object
        autoclean_dict : dict
            Autoclean dictionary
        pre_ica_raw : mne.io.Raw
            Raw data before ICA
    """
    try:
        derivatives_dir = Path(autoclean_dict["derivatives_dir"])
        basename = Path(autoclean_dict["unprocessed_file"]).stem
        ica_root = Path(
            autoclean_dict.get("ica_dir") or derivatives_dir.parent / "ica_fif"
        )
        ica_root.mkdir(parents=True, exist_ok=True)
    except Exception as e:  # pylint: disable=broad-exception-caught
        message("error", f"Failed to save ICA to FIF files: {str(e)}")
        return

    components = []
    ica_path: Optional[Path] = None

    if ica is not None:
        ica_path = ica_root / f"{basename}-ica.fif"
        ica.save(ica_path, overwrite=True)
        components.append(ch for ch in ica.exclude)

    # Use standard pipeline saving for pre_ica data
    pre_ica_path = save_raw_to_set(pre_ica_raw, autoclean_dict, stage="pre_ica")
    metadata = {
        "save_ica_to_fif": {
            "creationDateTime": datetime.now().isoformat(),
            "components": components,
            "ica_directory": str(ica_root),
            "ica_path": ica_path.name if ica_path else None,
            "pre_ica_path": str(pre_ica_path),
        }
    }
    run_id = autoclean_dict["run_id"]

    manage_database_conditionally(
        operation="update", update_record={"run_id": run_id, "metadata": metadata}
    )


# Keep the existing save functions with minor updates to ensure backward compatibility
def _get_stage_number(stage: str, autoclean_dict: Dict[str, Any]) -> str:
    """Get two-digit number based on export counter.

    Increments and tracks export count to assign sequential stage numbers.

    Args:
        stage: Name of the stage to get number for
        autoclean_dict: Configuration dictionary

    Returns:
        Two-digit string representation of stage number
    """
    # Initialize export counter if not present
    if "_export_counter" not in autoclean_dict:
        autoclean_dict["_export_counter"] = 0

    # Increment counter for this export
    autoclean_dict["_export_counter"] += 1

    return f"{autoclean_dict['_export_counter']:02d}"


def copy_final_files(autoclean_dict: Dict[str, Any]) -> None:
    """Copy final files from post_comp stage and processing log to the final_files directory.

    This function finds all files in the highest numbered post_comp stage directory
    and copies them to the dedicated final_files directory for easy access. It also
    copies the most recent processing log for user convenience while keeping the
    original log in the derivatives/logs directory.

    Parameters
    ----------
    autoclean_dict : Dict[str, Any]
        Configuration dictionary containing directory paths and run information.
    """

    stage_dir = Path(autoclean_dict["stage_dir"])
    final_files_dir = Path(autoclean_dict["final_files_dir"])
    logs_dir = Path(autoclean_dict["logs_dir"])
    basename = Path(autoclean_dict["unprocessed_file"]).stem

    # Find the post_comp stage directory (highest numbered directory with 'comp' in name)
    post_comp_dirs = [
        d
        for d in stage_dir.iterdir()
        if d.is_dir() and "comp" in d.name.lower() and not d.name.startswith("FLAGGED")
    ]

    if not post_comp_dirs:
        message(
            "warning", "No post_comp stage directory found - no final files to copy"
        )
        return

    # Sort by stage number to get the latest one
    post_comp_dirs.sort(key=lambda x: x.name)
    latest_post_comp = post_comp_dirs[-1]

    message(
        "info",
        f"Copying final files from {latest_post_comp.name} to final_files directory",
    )

    # Copy all files from the post_comp directory to final_files
    final_files_dir.mkdir(parents=True, exist_ok=True)
    files_copied = 0

    for file_path in latest_post_comp.iterdir():
        if file_path.is_file():
            dest_path = final_files_dir / file_path.name
            try:
                shutil.copy2(file_path, dest_path)
                files_copied += 1
                message("debug", f"Copied {file_path.name} to final_files")
            except Exception as e:
                message("error", f"Failed to copy {file_path.name}: {str(e)}")

    # Copy the most recent processing log to final_files for user convenience
    if logs_dir.exists():
        log_files = [
            f
            for f in logs_dir.iterdir()
            if f.is_file() and f.name.startswith("autoclean_") and f.suffix == ".log"
        ]

        if log_files:
            # Sort by modification time and get the most recent
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            log_dest = final_files_dir / f"{basename}_processing.log"

            try:
                shutil.copy2(latest_log, log_dest)
                files_copied += 1
                message("debug", "Copied processing log to final_files")
            except Exception as e:
                message("error", f"Failed to copy processing log: {str(e)}")
        else:
            message("warning", "No processing log files found to copy")
    else:
        message("warning", f"Logs directory not found: {logs_dir}")

    if files_copied > 0:
        message(
            "success",
            f"Copied {files_copied} final files (including processing log) to {final_files_dir}",
        )

        # Update metadata
        metadata = {
            "copy_final_files": {
                "creationDateTime": datetime.now().isoformat(),
                "source_stage": latest_post_comp.name,
                "files_copied": files_copied,
                "final_files_dir": str(final_files_dir),
                "includes_processing_log": True,
            }
        }

        run_id = autoclean_dict["run_id"]
        manage_database_conditionally(
            operation="update", update_record={"run_id": run_id, "metadata": metadata}
        )
    else:
        message("warning", "No files found in post_comp stage directory")
