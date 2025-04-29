# src/autoclean/utils/bids.py
import json
import sys
import threading
import traceback
from contextlib import contextmanager  # Imported for dummy lock
from pathlib import Path
from threading import Lock as ThreadingLock
from typing import Optional

import pandas as pd
from mne.io.constants import FIFF
from mne_bids import BIDSPath, update_sidecar_json, write_raw_bids

from ..utils.logging import message


def step_convert_to_bids(
    raw,
    output_dir,
    task="rest",
    participant_id=None,
    line_freq=60.0,
    overwrite=False,
    events=None,
    event_id=None,
    study_name="EEG Study",
    autoclean_dict: Optional[dict] = None,
):
    """
    Converts a single EEG data file into BIDS format with default/dummy metadata.
    Handles concurrent access to participants.tsv using a threading.Lock passed
    via autoclean_dict. Ensures specific column order and dtype=object for the TSV.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw data to convert to BIDS.
    output_dir : str
        The directory where the BIDS dataset will be created.
    task : str
        The task name for BIDS.
    participant_id : str
        The participant ID (if None, generated from filename).
    line_freq : float
        The power line frequency.
    overwrite : bool
        Whether to overwrite existing BIDS files.
    events : mne.events_data
        The events array.
    event_id : dict
        The event_id dictionary.
    study_name : str
        The name of the study for dataset_description.json.
    autoclean_dict : dict
        The run configuration, MUST include 'participants_tsv_lock' (a threading.Lock) for concurrent safety.

    Returns
    -------
    bids_path : BIDSPath
        The BIDS path of the converted file.
    """
    import hashlib

    file_path = raw.filenames[0]
    file_name = Path(file_path).name

    # Retrieve Lock from autoclean_dict if available for thread-safe TSV access.
    lock = None
    lock_valid = False
    if autoclean_dict and "participants_tsv_lock" in autoclean_dict:
        retrieved_lock = autoclean_dict["participants_tsv_lock"]
        # Validate the lock object based on expected methods and type name ('lock').
        if (
            hasattr(retrieved_lock, "acquire")
            and hasattr(retrieved_lock, "release")
            and retrieved_lock.__class__.__name__ == "lock"
        ):
            lock = retrieved_lock
            lock_valid = True
            message(
                "debug",
                "Successfully validated threading.Lock object from autoclean_dict.",
            )
        else:
            message(
                "warning",
                f"participants_tsv_lock found in autoclean_dict but is not a valid threading.Lock object (type: {type(retrieved_lock).__name__}, value: {retrieved_lock!r}). Proceeding without lock.",
            )

    if not lock_valid:
        message(
            "warning",
            "participants_tsv_lock not found or invalid. Concurrent writes to participants.tsv may be unsafe.",
        )

        # Use a dummy context manager if no valid lock is found to allow execution.
        @contextmanager
        def dummy_lock():
            yield

        lock_context = dummy_lock()
    else:
        lock_context = lock  # Use the actual lock context.

    bids_root = Path(output_dir)
    bids_root.mkdir(parents=True, exist_ok=True)

    # Define participants file path and the desired column order.
    participants_file = bids_root / "participants.tsv"
    desired_column_order = [
        "participant_id",
        "file_name",
        "bids_path",
        "age",
        "sex",
        "group",
        "hand",
        "weight",
        "height",
        "eegid",
        "file_hash",
    ]

    # Determine participant ID (generate if not provided).
    if participant_id is None:
        participant_id = step_sanitize_id(file_path)
    subject_id = str(participant_id)

    # Set default metadata values.
    session = None
    run = None
    age = "n/a"
    sex = "n/a"
    group = "n/a"

    # Create BIDSPath object.
    bids_path = BIDSPath(
        subject=subject_id,
        session=session,
        task=task,
        run=run,
        datatype="eeg",
        root=bids_root,
        suffix="eeg",
    )

    fif_file = Path(file_path)

    # Calculate file hash.
    try:
        file_hash = hashlib.sha256(fif_file.read_bytes()).hexdigest()
    except Exception as e:
        message("error", f"Failed to read {fif_file} for hashing: {e}")
        raise

    # Prepare MNE Raw object metadata for BIDS conversion.
    raw.info["subject_info"] = {"id": int(subject_id)}
    raw.info["line_freq"] = line_freq
    for ch in raw.info["chs"]:
        ch["unit"] = FIFF.FIFF_UNIT_V

    # Prepare arguments for mne_bids.write_raw_bids.
    bids_kwargs = {
        "raw": raw,
        "bids_path": bids_path,
        "overwrite": overwrite,
        "verbose": False,
        "format": "BrainVision",
        "events": events,
        "event_id": event_id,
        "allow_preload": True,
    }

    # Create derivatives directory structure (outside the lock).
    derivatives_dir = (
        bids_root / "derivatives" / "pylossless" / f"sub-{subject_id}" / "eeg"
    )
    derivatives_dir.mkdir(parents=True, exist_ok=True)
    message("info", f"Created derivatives directory structure at {derivatives_dir}")

    # --- Critical Section: Accessing shared BIDS files ---
    # Use the lock (real or dummy) to protect file access.
    message("debug", f"Acquiring participants.tsv lock for {file_name}...")
    with lock_context:
        message("debug", f"Acquired participants.tsv lock for {file_name}.")

        # Ensure participants.tsv exists with correct headers and dtype=object
        # *before* calling mne_bids, which might interact with it.
        try:
            if not participants_file.exists():
                message(
                    "info",
                    f"Creating participants.tsv with headers at {participants_file}",
                )
                header_df = pd.DataFrame(columns=desired_column_order, dtype=object)
                header_df.to_csv(
                    participants_file, sep="	", index=False, na_rep="n/a"
                )
        except Exception as header_err:
            message("error", f"Failed to create participants.tsv header: {header_err}")
            raise

        # Call mne_bids to write the core BIDS data.
        try:
            write_raw_bids(**bids_kwargs)
            message("success", f"Converted {fif_file.name} to BIDS format.")
            # Update sidecar JSON with additional info.
            entries = {"Manufacturer": "Unknown", "PowerLineFrequency": line_freq}
            sidecar_path = bids_path.copy().update(extension=".json")
            update_sidecar_json(bids_path=sidecar_path, entries=entries)
        except Exception as e:
            message("error", f"Failed to write BIDS for {fif_file.name}: {e}")
            print(f"Detailed error: {str(e)}")
            traceback.print_exc()
            raise

        # --- Update participants.tsv with custom/calculated metadata ---
        try:
            # Read the potentially modified participants.tsv, enforcing object dtype.
            try:
                dtype_mapping = {col: object for col in desired_column_order}
                # Read assuming all desired columns should exist; add missing ones later.
                # na_filter=False prevents 'NA' strings from becoming NaN if object dtype is used.
                participants_df = pd.read_csv(
                    participants_file, sep="	", dtype=dtype_mapping, na_filter=False
                )

                # Validate and fix columns after reading.
                missing_cols = [
                    col
                    for col in desired_column_order
                    if col not in participants_df.columns
                ]
                if missing_cols:
                    message(
                        "warning",
                        f"participants.tsv is missing columns: {missing_cols}. Adding them with 'n/a'.",
                    )
                    for col in missing_cols:
                        participants_df[col] = "n/a"
                    participants_df = participants_df.astype(
                        {col: object for col in missing_cols}
                    )

                # Handle cases where the file might be corrupted or unexpectedly empty.
                if participants_df.empty and participants_file.stat().st_size > 0:
                    message(
                        "warning",
                        "participants.tsv exists but pandas read an empty DataFrame. Recreating.",
                    )
                    participants_df = pd.DataFrame(
                        columns=desired_column_order, dtype=object
                    )
                elif (
                    not participants_df.empty
                    and "participant_id" not in participants_df.columns
                ):
                    message(
                        "warning",
                        "participants.tsv is missing 'participant_id'. Recreating.",
                    )
                    participants_df = pd.DataFrame(
                        columns=desired_column_order, dtype=object
                    )

            except pd.errors.EmptyDataError:
                # Handle case where mne_bids might have left the file empty.
                message(
                    "warning",
                    f"participants.tsv is empty after MNE-BIDS write. Starting with headers.",
                )
                participants_df = pd.DataFrame(
                    columns=desired_column_order, dtype=object
                )
            except Exception as pd_read_err:
                message(
                    "error",
                    f"Error reading participants.tsv after MNE-BIDS write: {pd_read_err}. Attempting overwrite.",
                )
                participants_df = pd.DataFrame(
                    columns=desired_column_order, dtype=object
                )

            # Prepare the entry for the current participant.
            new_entry = {
                "participant_id": f"sub-{subject_id}",
                "file_name": file_name,
                "bids_path": str(bids_path.match()[0]),
                "age": age,
                "sex": sex,
                "group": group,
                # Add standard optional BIDS columns with 'n/a' if not provided elsewhere.
                "hand": "n/a",
                "weight": "n/a",
                "height": "n/a",
                "eegid": fif_file.stem,
                "file_hash": file_hash,
            }

            # Update existing row or append new row.
            participant_col_id = f"sub-{subject_id}"
            if participant_col_id not in participants_df["participant_id"].values:
                # Append new row using pd.concat for better type handling.
                new_row_df = pd.DataFrame([new_entry]).astype(dtype=object)
                participants_df = pd.concat(
                    [participants_df, new_row_df], ignore_index=True
                )
                message(
                    "debug",
                    f"Appended new entry for {participant_col_id} to participants.tsv.",
                )
            else:
                # Update existing row.
                message(
                    "debug",
                    f"Participant {participant_col_id} already exists. Updating row.",
                )
                idx = participants_df.index[
                    participants_df["participant_id"] == participant_col_id
                ].tolist()
                if idx:
                    row_index = idx[0]
                    for key, value in new_entry.items():
                        if key in participants_df.columns:
                            # Ensure value assignment respects object dtype.
                            participants_df.loc[row_index, key] = (
                                str(value) if value is not None else "n/a"
                            )
                        else:
                            message(
                                "warning",
                                f"Column '{key}' not found in participants.tsv during update for {participant_col_id}.",
                            )
                else:
                    # Fallback if index search fails.
                    message(
                        "warning",
                        f"Could not find index for existing participant {participant_col_id}. Appending instead.",
                    )
                    new_row_df = pd.DataFrame([new_entry]).astype(dtype=object)
                    participants_df = pd.concat(
                        [participants_df, new_row_df], ignore_index=True
                    )

            # Ensure no duplicate participant IDs remain.
            participants_df.drop_duplicates(
                subset="participant_id", keep="last", inplace=True
            )

            # Ensure final DataFrame columns match desired order, preserving extras.
            # Note: This assumes desired_column_order contains all keys from new_entry that should be primary columns.
            final_columns = desired_column_order + [
                col
                for col in participants_df.columns
                if col not in desired_column_order
            ]
            participants_df = participants_df[final_columns]

            # Write the updated DataFrame back to TSV.
            participants_df.to_csv(
                participants_file, sep="	", index=False, na_rep="n/a"
            )
            message("debug", f"Updated participants.tsv for {file_name}")

            # Create metadata JSON files if they don't exist.
            dataset_description_file = bids_root / "dataset_description.json"
            if not dataset_description_file.exists():
                step_create_dataset_desc(bids_root, study_name=study_name)

            participants_json_file = bids_root / "participants.json"
            if not participants_json_file.exists():
                step_create_participants_json(bids_root)

        except Exception as update_err:
            message(
                "error",
                f"Failed during participants.tsv update or associated file creation: {update_err}",
            )
            traceback.print_exc()
            raise

    # Lock is automatically released when exiting the 'with' block.
    message("debug", f"Released participants.tsv lock for {file_name}.")

    return bids_path


def step_sanitize_id(filename):
    """
    Generates a reproducible numeric participant ID from a filename using MD5 hashing.

    Parameters
    ----------
    filename : str
        The filename to generate a participant ID from.

    """
    import hashlib

    def filename_to_number(filename, max_value=1000000):
        # Generate MD5 hash of the filename.
        hash_object = hashlib.md5(filename.encode())
        # Convert first 8 bytes of hash to an integer.
        hash_int = int.from_bytes(hash_object.digest()[:8], "big")
        # Scale to the desired range using modulo.
        return hash_int % max_value

    basename = Path(filename).stem
    participant_id = filename_to_number(basename)
    message("info", f"Generated participant ID for {basename}: {participant_id}")

    return participant_id


def step_create_dataset_desc(output_path, study_name):
    """
    Creates BIDS dataset_description.json file.

    Parameters
    ----------
    output_path : str
        The path to the output directory.
    study_name : str
        The name of the study.
    """
    dataset_description = {
        "Name": study_name,
        "BIDSVersion": "1.6.0",  # Specify BIDS version used.
        "DatasetType": "raw",
    }
    filepath = output_path / "dataset_description.json"
    try:
        with open(filepath, "w") as f:
            json.dump(dataset_description, f, indent=4)
        message("success", f"Created {filepath.name}")
    except Exception as e:
        message("error", f"Failed to create {filepath.name}: {e}")


def step_create_participants_json(output_path):
    """
    Creates BIDS participants.json sidecar file describing participants.tsv columns.

    Parameters
    ----------
    output_path : str
        The path to the output directory.
    """
    # Describes columns in participants.tsv, including standard and custom ones.
    participants_json = {
        "participant_id": {"Description": "Unique participant identifier"},
        "file_name": {"Description": "Original source filename"},
        "bids_path": {"Description": "Relative path to the primary BIDS data file"},
        "age": {"Description": "Age of the participant", "Units": "years"},
        "sex": {
            "Description": "Biological sex of the participant",
            "Levels": {
                "M": "Male",
                "F": "Female",
                "O": "Other",
                "n/a": "Not available",
            },
        },
        "group": {"Description": "Participant group membership", "Levels": {}},
        "hand": {
            "Description": "Dominant hand of the participant",
            "Levels": {
                "L": "Left",
                "R": "Right",
                "A": "Ambidextrous",
                "n/a": "Not available",
            },
        },
        "weight": {"Description": "Weight of the participant", "Units": "kg"},
        "height": {"Description": "Height of the participant", "Units": "m"},
        "eegid": {"Description": "Original participant identifier/source file stem"},
        "file_hash": {"Description": "SHA256 hash of the original source file"},
    }
    filepath = output_path / "participants.json"
    try:
        with open(filepath, "w") as f:
            json.dump(participants_json, f, indent=4)
        message("success", f"Created {filepath.name}")
    except Exception as e:
        message("error", f"Failed to create {filepath.name}: {e}")
