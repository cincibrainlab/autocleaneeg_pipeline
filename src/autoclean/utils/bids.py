# src/autoclean/utils/bids.py
import json
import sys
from pathlib import Path
import threading
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
    if autoclean_dict and 'participants_tsv_lock' in autoclean_dict:
        lock = autoclean_dict['participants_tsv_lock']
        if not isinstance(lock, threading.Lock):
            message("warning", "participants_tsv_lock found in autoclean_dict but is not a threading.Lock. Proceeding without lock.")
            lock = None
    
    if lock is None:
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

    # Sanitize and set participant ID
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
        # Write BIDS data
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
        participants_file = bids_root / "participants.tsv"
        try:
            if not participants_file.exists():
                participants_df = pd.DataFrame(
                    columns=["participant_id", "age", "sex", "group", "bids_path", "eegid", "file_name", "file_hash"]
                )
            else:
                try:
                    participants_df = pd.read_csv(participants_file, sep="\t")
                    expected_cols = {"participant_id", "age", "sex", "group", "bids_path", "eegid", "file_name", "file_hash"}
                    if not expected_cols.issubset(participants_df.columns):
                        message("warning", f"participants.tsv missing expected columns. Will recreate with headers.")
                        participants_df = pd.DataFrame(columns=list(expected_cols))
                except pd.errors.EmptyDataError:
                    message("warning", f"participants.tsv is empty. Creating a new one.")
                    participants_df = pd.DataFrame(columns=["participant_id", "age", "sex", "group", "bids_path", "eegid", "file_name", "file_hash"])
                except Exception as pd_read_err:
                    message("error", f"Error reading participants.tsv: {pd_read_err}. Attempting to overwrite.")
                    participants_df = pd.DataFrame(columns=["participant_id", "age", "sex", "group", "bids_path", "eegid", "file_name", "file_hash"])

            new_entry = {
                "participant_id": f"sub-{subject_id}",
                "bids_path": str(bids_path.match()[0]),
                "age": "n/a",
                "sex": "n/a",
                "group": "n/a",
                "eegid": fif_file.stem,
                "file_name": file_name,
                "file_hash": file_hash,
            }

            if f"sub-{subject_id}" not in participants_df["participant_id"].values:
                participants_df = pd.concat([participants_df, pd.DataFrame([new_entry])], ignore_index=True)
            else:
                message("info", f"Participant sub-{subject_id} already exists in participants.tsv. Skipping append.")

            participants_df.drop_duplicates(subset="participant_id", keep="last", inplace=True)
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
