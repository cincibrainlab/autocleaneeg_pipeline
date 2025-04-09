# src/autoclean/utils/bids.py
import json
import sys
from pathlib import Path
import threading
from threading import Lock as ThreadingLock
from typing import Optional

import pandas as pd
from mne.io.constants import FIFF
from mne_bids import BIDSPath, update_sidecar_json, write_raw_bids
import traceback

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

    Parameters:
    - file_path (str or Path): Path to the EEG data file.
    - output_dir (str or Path): Directory where the BIDS dataset will be created.
    - task (str, optional): Task name. Defaults to 'resting'.
    - participant_id (str, optional): Participant ID. Defaults to sanitized basename of the file.
    - line_freq (float, optional): Power line frequency in Hz. Defaults to 60.0.
    - overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
    - study_name (str, optional): Name of the study. Defaults to "EEG Study".
    - autoclean_dict (dict, optional): Dictionary containing run configuration, including the lock.

    Dependent Functions:
    - step_sanitize_id(): Sanitizes and formats participant ID from filename
    - step_create_dataset_desc(): Creates BIDS dataset description JSON file
    - step_create_participants_json(): Creates participants.json metadata file
    - update_sidecar_json(): Updates sidecar JSON files with additional metadata
    """
    import hashlib

    file_path = raw.filenames[0]
    file_name = Path(file_path).name

    # Retrieve Lock if available
    lock = None
    lock_valid = False # Flag to track if we found a valid lock
    if autoclean_dict and 'participants_tsv_lock' in autoclean_dict:
        retrieved_lock = autoclean_dict['participants_tsv_lock']
        # Check based on attributes/name instead of isinstance
        if hasattr(retrieved_lock, 'acquire') and hasattr(retrieved_lock, 'release') and retrieved_lock.__class__.__name__ == 'lock':
            lock = retrieved_lock
            lock_valid = True
            message("debug", "Successfully validated threading.Lock object from autoclean_dict.")
        else:
            # Log what we actually got if it's not a valid lock
            message("warning", f"participants_tsv_lock found in autoclean_dict but is not a valid threading.Lock object (type: {type(retrieved_lock).__name__}, value: {retrieved_lock!r}). Proceeding without lock.")
            # lock remains None
    
    if not lock_valid: # Use the boolean flag for clarity
        message("warning", "participants_tsv_lock not found or invalid in autoclean_dict. Concurrent writes to participants.tsv may be unsafe.")
        from contextlib import contextmanager
        @contextmanager
        def dummy_lock():
            yield
        lock_context = dummy_lock()
    else:
        lock_context = lock

    bids_root = Path(output_dir)
    bids_root.mkdir(parents=True, exist_ok=True)

    # Define participants file path and DESIRED column order
    participants_file = bids_root / "participants.tsv"
    # Use the order from the user's new_entry example
    desired_column_order = [
        "participant_id", "file_name", "bids_path", "age", 
        "sex", "group", "eegid", "file_hash"
    ]
    expected_cols = set(desired_column_order) # Keep the set for checks if needed

    # Determine participant ID
    if participant_id is None:
        participant_id = step_sanitize_id(file_path)
    # subject_id = participant_id.zfill(5)
    subject_id = str(participant_id)

    # Default metadata
    session = None
    run = None
    age = "n/a"
    sex = "n/a"
    group = "n/a"

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

    # Read the raw data
    try:
        file_hash = hashlib.sha256(fif_file.read_bytes()).hexdigest()
        file_name = fif_file.name
    except Exception as e:
        message("error", f"Failed to read {fif_file}: {e}")
        raise

    # Prepare additional metadata
    raw.info["subject_info"] = {"id": int(subject_id)}

    raw.info["line_freq"] = line_freq

    # Prepare unit for BIDS
    for ch in raw.info["chs"]:
        ch["unit"] = FIFF.FIFF_UNIT_V  # Assuming units are in Volts

    # Additional BIDS parameters
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

    # Create derivatives directory structure
    derivatives_dir = bids_root / "derivatives" / "pylossless" / f"sub-{subject_id}" / "eeg"
    derivatives_dir.mkdir(parents=True, exist_ok=True)
    message("info", f"Created derivatives directory structure at {derivatives_dir}")

    # Acquire lock (if available) before critical section
    message("debug", f"Acquiring participants.tsv lock for {file_name}...")
    with lock_context:
        message("debug", f"Acquired participants.tsv lock for {file_name}.")

        # --- Ensure participants.tsv exists with headers before MNE-BIDS --- 
        try:
            if not participants_file.exists():
                message("info", f"Creating participants.tsv with headers at {participants_file}")
                # Create DataFrame with desired column order and dtype=object
                header_df = pd.DataFrame(columns=desired_column_order, dtype=object)
                header_df.to_csv(participants_file, sep="\t", index=False, na_rep="n/a")
        except Exception as header_err:
            message("error", f"Failed to create participants.tsv header: {header_err}")
            # Decide if we should proceed or raise - raising is safer
            raise

        # Write BIDS data (MNE-BIDS might modify participants.tsv)
        try:
            write_raw_bids(**bids_kwargs)
            message("success", f"Converted {fif_file.name} to BIDS format.")
            entries = {"Manufacturer": "Unknown", "PowerLineFrequency": line_freq}
            sidecar_path = bids_path.copy().update(extension=".json")
            update_sidecar_json(bids_path=sidecar_path, entries=entries)
        except Exception as e:
            message("error", f"Failed to write BIDS for {fif_file.name}: {e}")
            print(f"Detailed error: {str(e)}")
            traceback.print_exc()
            raise

        # Update participants.tsv
        try:
            # Use try-except for robustness when reading potentially incomplete files
            try:
                # Specify dtype=object when reading to avoid type mismatch issues
                dtype_mapping = {col: object for col in desired_column_order} # Use desired order here too
                participants_df = pd.read_csv(participants_file, sep="\t", dtype=dtype_mapping, na_filter=False)
                
                # --- Column Validation/Fixing --- 
                # Check if all desired columns exist, add missing ones if necessary
                missing_cols = [col for col in desired_column_order if col not in participants_df.columns]
                if missing_cols:
                    message("warning", f"participants.tsv is missing columns: {missing_cols}. Adding them with 'n/a'.")
                    for col in missing_cols:
                        participants_df[col] = "n/a" # Add missing column with default value
                    # Ensure object dtype for newly added columns
                    participants_df = participants_df.astype({col: object for col in missing_cols})
                
                # Check if the file was effectively empty or corrupted
                if participants_df.empty and participants_file.stat().st_size > 0:
                    message("warning", "participants.tsv exists but pandas read an empty DataFrame. Check file content.")
                    # Recreate with desired order
                    participants_df = pd.DataFrame(columns=desired_column_order, dtype=object)
                elif not participants_df.empty and "participant_id" not in participants_df.columns:
                    message("warning", "participants.tsv is missing 'participant_id' column after MNE-BIDS write. Recreating headers.")
                    # Recreate with desired order
                    participants_df = pd.DataFrame(columns=desired_column_order, dtype=object)

            except pd.errors.EmptyDataError:
                message("warning", f"participants.tsv is empty after MNE-BIDS write. Starting with headers.")
                # Ensure object dtype and desired order
                participants_df = pd.DataFrame(columns=desired_column_order, dtype=object)
            except Exception as pd_read_err:
                message("error", f"Error reading participants.tsv after MNE-BIDS write: {pd_read_err}. Attempting to overwrite.")
                # Ensure object dtype and desired order when overwriting
                participants_df = pd.DataFrame(columns=desired_column_order, dtype=object)

            new_entry = {
                "participant_id": f"sub-{subject_id}",
                "file_name": file_name, 
                "bids_path": str(bids_path.match()[0]), 
                "age": age, 
                "sex": sex, 
                "group": group, 
                "eegid": fif_file.stem,
                "file_hash": file_hash,
            }

            # --- Row Update/Append --- 
            participant_col_id = f"sub-{subject_id}"
            if participant_col_id not in participants_df["participant_id"].values:
                # Append new row if participant doesn't exist
                participants_df = pd.concat([participants_df, pd.DataFrame([new_entry])], ignore_index=True)
                message("debug", f"Appended new entry for {participant_col_id} to participants.tsv.")
            else:
                # Update existing row if participant already exists
                message("debug", f"Participant {participant_col_id} already exists in participants.tsv. Updating row.")
                # Find the index of the row to update
                idx = participants_df.index[participants_df["participant_id"] == participant_col_id].tolist()
                if idx:
                    # Update the row at the found index (use the first index if multiple found)
                    for key, value in new_entry.items():
                        if key in participants_df.columns: # Ensure column exists before assigning
                            participants_df.loc[idx[0], key] = value
                        else:
                             message("warning", f"Column '{key}' not found in participants.tsv during update for {participant_col_id}.")
                else:
                    # Should theoretically not happen if the ID was found in .values, but good to handle
                    message("warning", f"Could not find index for existing participant {participant_col_id} to update. Appending instead.")
                    # Fallback to appending if index search fails unexpectedly
                    participants_df = pd.concat([participants_df, pd.DataFrame([new_entry])], ignore_index=True)

            # Drop duplicates just in case, keeping the last entry
            participants_df.drop_duplicates(subset="participant_id", keep="last", inplace=True)
            
            # --- Final Reordering and Write --- 
            # Ensure columns are in the desired order before writing
            # Include any extra columns that might have been added (e.g., by mne-bids)
            final_columns = desired_column_order + [col for col in participants_df.columns if col not in desired_column_order]
            participants_df = participants_df[final_columns]
            
            participants_df.to_csv(participants_file, sep="\t", index=False, na_rep="n/a")
            message("debug", f"Updated participants.tsv for {file_name}")

            # Create dataset_description.json if it doesn't exist (safe under lock)
            dataset_description_file = bids_root / "dataset_description.json"
            if not dataset_description_file.exists():
                step_create_dataset_desc(bids_root, study_name=study_name)

            # Create participants.json if it doesn't exist (safe under lock)
            participants_json_file = bids_root / "participants.json"
            if not participants_json_file.exists():
                step_create_participants_json(bids_root)

        except Exception as update_err:
            message("error", f"Failed during participants.tsv update or associated file creation: {update_err}")
            traceback.print_exc()
            raise
    
    message("debug", f"Released participants.tsv lock for {file_name}.")

    return bids_path


def step_sanitize_id(filename):
    """
    Sanitizes the participant ID extracted from the filename to comply with BIDS conventions.

    Parameters:
    - filename (str): The filename to sanitize.

    Returns:
    - str: A sanitized participant ID.
    """
    import hashlib

    def filename_to_number(filename, max_value=1000000):
        # Create a hash of the filename
        hash_object = hashlib.md5(filename.encode())
        # Get the first 8 bytes of the hash as an integer
        hash_int = int.from_bytes(hash_object.digest()[:8], "big")
        # Use modulo to get a number within the desired range
        return hash_int % max_value

    basename = Path(filename).stem
    participant_id = filename_to_number(basename)
    message("info", f"Unique Number for {basename}: {participant_id}")

    return participant_id


def step_create_dataset_desc(output_path, study_name):
    """
    Creates BIDS dataset description JSON file.

    Parameters:
    - output_path (Path): Directory where the file will be created
    - study_name (str): Name of the study
    """
    dataset_description = {
        "Name": study_name,
        "BIDSVersion": "1.6.0",
        "DatasetType": "raw",
    }
    with open(output_path / "dataset_description.json", "w") as f:
        json.dump(dataset_description, f, indent=4)
    message("success", "Created dataset_description.json")


def step_create_participants_json(output_path):
    """
    Creates participants.json metadata file.

    Parameters:
    - output_path (Path): Directory where the file will be created
    """
    participants_json = {
        "participant_id": {"Description": "Unique participant identifier"},
        "bids_path": {"Description": "Path to the BIDS file"},
        "file_hash": {"Description": "Hash of the original file"},
        "file_name": {"Description": "Name of the original file"},
        "eegid": {"Description": "Original participant identifier"},
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
        "group": {"Description": "Participant group", "Levels": {}},
    }
    with open(output_path / "participants.json", "w") as f:
        json.dump(participants_json, f, indent=4)
    message("success", "Created participants.json")
