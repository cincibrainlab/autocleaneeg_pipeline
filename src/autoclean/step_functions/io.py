#src/autoclean/step_functions/io.py
"""Input/Output functions for EEG data.

This module provides functions for loading raw EEG data and saving processed results.
It handles:
- Raw data import from various formats
- Saving intermediate processing results
- Exporting final processed data

The functions handle proper data validation and error checking to ensure
data integrity throughout the processing pipeline.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import mne
import numpy as np
import pandas as pd
import scipy.io as sio

from autoclean.utils.database import manage_database, get_run_record
from autoclean.utils.logging import message



def step_import(autoclean_dict: dict, preload: bool = True) -> mne.io.Raw:
    """Import raw EEG data from file.
    
    This function handles loading raw EEG data from various file formats.
    It supports both continuous data (e.g., resting state) and event-related
    data (e.g., chirp, ASSR). The function:
    1. Loads the data file based on format
    2. Sets up appropriate montage
    3. Handles events/triggers if present
    4. Validates data integrity
    
    Args:
        autoclean_dict: Configuration dictionary containing:
            - unprocessed_file: Path to raw EEG data file
            - eeg_system: Name of the EEG system/montage
            - task: Task name for processing type
            - tasks: Task-specific settings including event_id
        preload: Whether to load data into memory (default: True)
            
    Returns:
        mne.io.Raw: Loaded and validated raw EEG data
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If file format is invalid or data is corrupted
        RuntimeError: If loading or validation fails
    """
    unprocessed_file = autoclean_dict["unprocessed_file"]
    eeg_system = autoclean_dict["eeg_system"]
    task = autoclean_dict.get("task", None)
    file_ext = unprocessed_file.suffix.lower()
    extract_eeglab_events = False

    events = None
    event_dict = None

    message("info", f"Importing {unprocessed_file} with system {eeg_system}")

    try:
        # Determine how to read the file
        if file_ext == ".set":
            try:
                raw = mne.io.read_raw_eeglab(
                    input_fname=unprocessed_file, preload=preload, verbose=True
                )
                extract_eeglab_events = True
            except Exception as e:
                raise RuntimeError(f"Failed to read .set file: {str(e)}")
        elif file_ext == ".raw":
            try:
                raw = mne.io.read_raw_egi(
                    input_fname=unprocessed_file, preload=preload, events_as_annotations=True
                )
            except Exception as e:
                message("critical", f"Failed to read .raw file")
                raise RuntimeError(f"{str(e)}")
        else:
            message("error", f"Unsupported file type: {file_ext}")
            raise ValueError(f"Unsupported file type: {file_ext}")

        # Montage and channel setup
        if eeg_system == "GSN-HydroCel-129":
            try:
                montage = mne.channels.make_standard_montage(eeg_system)
                montage.ch_names[128] = "E129"
                raw.set_montage(montage, match_case=False)
                raw.pick("eeg")
            except Exception as e:
                raise RuntimeError(f"Failed to set up GSN-HydroCel-129 montage: {str(e)}")
            
        elif eeg_system == "GSN-HydroCel-124" and file_ext == ".set":
            try:
                ecg_channels = ["E125","E126","E127","E128"]
                ecg_mapping = {ch: 'ecg' for ch in ecg_channels}
                raw.set_channel_types(ecg_mapping)
                raw.drop_channels(ecg_channels)
                raw.pick("eeg")
            except Exception as e:
                raise RuntimeError(f"Failed to configure GSN-HydroCel-124 channels: {str(e)}")
            
        elif eeg_system == "MEA30" and file_ext == ".set":
            try:
                raw.pick("eeg")
            except Exception as e:
                raise RuntimeError(f"Failed to pick EEG channels for MEA30: {str(e)}")
            
        else:
            raise ValueError(f"Unsupported system or file type: {eeg_system}, {file_ext}")

        if extract_eeglab_events:
            try:
                events_df = _get_matlab_annotations_table(autoclean_dict)
                message("success", "Events table loaded successfully:")
                message("info", f"  - Number of events: {len(events_df)}")
                message("info", f"  - Number of variables: {len(events_df.columns)}")
                message("info", f"  - Variables: {', '.join(events_df.columns)}")
            except Exception as e:
                message("warning", f"Could not load events table: {str(e)}")
                events_df = None

            try:
                subset_events_df = events_df[['Task', 'type', 'onset', 'Condition']]
                new_annotations = mne.Annotations(
                    onset=subset_events_df['onset'].values,
                    duration=np.zeros(len(subset_events_df)),  # Point events
                    description=[
                        f"{row['Task']}/{row['type']}/{row['Condition']}" 
                        for _, row in subset_events_df.iterrows()
                    ]
                )
                raw.set_annotations(new_annotations)
            except Exception as e:
                message("warning", f"Could not set annotations: {str(e)}")

        else:
            events_df = None

        if task == 'hbcd_mmn':
            # Adds missing EEGLAB Metadata
            subset_events_df = events_df[['Task', 'type', 'onset', 'Condition']]
            new_annotations = mne.Annotations(
                onset=subset_events_df['onset'].values,
                duration=np.zeros(len(subset_events_df)),  # Point events
                description=[
                    f"{row['Task']}/{row['type']}/{row['Condition']}" 
                    for _, row in subset_events_df.iterrows()
                ]
            )
            raw.set_annotations(new_annotations)
            # target_event_type = list(autoclean_dict["tasks"][task]["settings"]["event_id"]["value"].keys())[0]
            # reg_exp = f'.*{target_event_type}.*'
            # events, event_id = mne.events_from_annotations(raw, regexp=reg_exp)
            # event_dict = {str(k): int(v) for k, v in event_id.items()}
        else:
            events, event_dict, events_df = None, None, None
            if task and autoclean_dict["tasks"][task]["settings"]["event_id"]["enabled"]:
                target_event_id = autoclean_dict["tasks"][task]["settings"]["event_id"]["value"]
                events, event_id = _update_events(raw, target_event_id)
                rev_target_event_id = dict(map(reversed, target_event_id.items()))
                raw.set_annotations(None)
                raw.set_annotations(mne.annotations_from_events(
                    events, raw.info["sfreq"], event_desc=rev_target_event_id
                ))
        
        # def parse_description(description):
        #     parts = description.split('/')
        #     return parts[:2] + ['n/a'] if len(parts) > 3 else parts
        # # Convert annotations descriptions to DataFrame
        # if raw.annotations is not None and task == 'hbcd_mmn':
        #     parsed_data = [parse_description(desc) for desc in raw.annotations.description]
        #     parsed_df = pd.DataFrame(parsed_data, columns=['Task', 'type', 'Condition'])
        
        # # Get unique event types
        #     unique_types = parsed_df['type'].unique()
        #     message("info", f"Unique event types: {unique_types}")
            #event_dict = create_event_id_dictionary(raw)

        # Prepare metadata
        metadata = {
            "step_import": {
                "import_function": "step_import",
                "creationDateTime": datetime.now().isoformat(),
                "unprocessedFile": str(unprocessed_file.name),
                "eegSystem": eeg_system,
                "sampleRate": raw.info["sfreq"],
                "channelCount": len(raw.ch_names),
                "durationSec": int(raw.n_times) / raw.info["sfreq"],
                "numberSamples": int(raw.n_times),
                "hasEvents": events is not None,
            }
        }
        
        # Add event information to metadata if present
        if events is not None:
            metadata["step_import"].update({
                "event_dict": event_dict,
                "event_count": len(events),
                "unique_event_types": list(set(events[:, 2]))
            })
            
        # if events_df is not None:
        #     metadata["step_import"]["additional_event_info"] = {
        #         "variables": list(events_df.columns),
        #         "event_count": len(events_df)
        #     }

        # Update database
        manage_database(
            operation="update",
            update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
        )

        message("success", "✓ Raw EEG data imported successfully")
        return raw

    except Exception as e:
        message("error", f"Failed to import raw EEG data: {str(e)}")
        raise

def _update_events(raw, event_id_code):
    events, event_id = mne.events_from_annotations(raw, event_id=event_id_code)
    return events, event_id

def _get_matlab_annotations_table(autoclean_dict: dict) -> pd.DataFrame:
    unprocessed_file = autoclean_dict["unprocessed_file"]
    eeglab_data = sio.loadmat(unprocessed_file, squeeze_me=True, struct_as_record=False)
    full_events = eeglab_data['EEG'].event

    event_list = []
    for event in full_events:
        event_dict = {}
        for field_name in event._fieldnames:
            event_dict[field_name] = getattr(event, field_name)
        event_list.append(event_dict)

    events_df = pd.DataFrame(event_list)

    return events_df

def _get_stage_number(stage: str, autoclean_dict: Dict[str, Any]) -> str:
    """Get two-digit number based on enabled stages order."""
    enabled_stages = [s for s, cfg in autoclean_dict["stage_files"].items() if cfg["enabled"]]
    return f"{enabled_stages.index(stage) + 1:02d}"

def save_raw_to_set(
    raw: mne.io.Raw,
    autoclean_dict: Dict[str, Any],
    stage: str = "post_import",
    output_path: Optional[Path] = None
) -> None:
    """Save raw EEG data to file.
    
    This function saves raw EEG data at various processing stages.
    It handles:
    1. Creating appropriate filenames
    2. Ensuring output directories exist
    3. Saving data in the correct format
    4. Validating saved files
    
    Args:
        raw: Raw EEG data to save
        config: Configuration dictionary containing:
            - stage_files (dict): Configuration for save stages
            - unprocessed_file (str): Original input file path
        stage: Processing stage identifier (e.g., "post_import")
        output_path: Optional custom output path. If None, uses config
            
    Raises:
        ValueError: If stage is not configured
        RuntimeError: If saving fails
        
    Example:
        >>> # Save data after preprocessing
        >>> save_raw_to_set(raw, config, "post_prepipeline")
    """
    if not autoclean_dict["stage_files"][stage]["enabled"]:
        message("info", f"Saving disabled for stage: {stage}")
        return None
        
    suffix = autoclean_dict["stage_files"][stage]["suffix"]
    basename = Path(autoclean_dict["unprocessed_file"]).stem
    stage_num = _get_stage_number(stage, autoclean_dict)
    
    # Save to stage directory
    if output_path is None:
        output_path = autoclean_dict["stage_dir"]
    subfolder = output_path / f"{stage_num}{suffix}"
    subfolder.mkdir(exist_ok=True)
    stage_path = subfolder / f"{basename}{suffix}_raw.set"
    
    # Save to both locations for post_comp
    paths = [stage_path]
    if stage == "post_comp":
        clean_path = autoclean_dict["clean_dir"] / f"{basename}{suffix}.set"
        autoclean_dict["clean_dir"].mkdir(exist_ok=True)
        paths.append(clean_path)

    # Save to all paths
    raw.info["description"] = autoclean_dict["run_id"]
    for path in paths:
        raw.export(path, fmt="eeglab", overwrite=True)
        message("success", f"✓ Saved {stage} file to: {path}")

    metadata = {
        "save_raw_to_set": {
            "creationDateTime": datetime.now().isoformat(),
            "stage": stage,
            "stage_number": stage_num,
            "outputPath": str(stage_path),
            "suffix": suffix,
            "basename": basename,
            "format": "eeglab",
        }
    }

    run_id = autoclean_dict["run_id"]
    manage_database(operation="update", update_record={"run_id": run_id, "metadata": metadata})
    manage_database(
        operation="update_status",
        update_record={"run_id": run_id, "status": f"{stage} completed"},
    )

    return paths[0]  # Return stage path for consistency


def save_epochs_to_set(
    epochs: mne.Epochs,
    autoclean_dict: Dict[str, Any],
    stage: str = "post_clean_epochs",
    output_path: Optional[Path] = None
) -> None:
    """Save epoched EEG data to file.
    
    This function saves epoched EEG data, typically after processing
    is complete. It handles:
    1. Creating appropriate filenames
    2. Ensuring output directories exist
    3. Saving data in the correct format
    4. Validating saved files
    
    Args:
        epochs: Epoched EEG data to save
        config: Configuration dictionary containing:
            - stage_files (dict): Configuration for save stages
            - unprocessed_file (str): Original input file path
        stage: Processing stage identifier (default: "post_clean_epochs")
        output_path: Optional custom output path. If None, uses config
            
    Raises:
        ValueError: If stage is not configured
        RuntimeError: If saving fails
        
    Example:
        >>> # Save final cleaned epochs
        >>> save_epochs_to_set(epochs, autoclean_dict)
    """
    if not autoclean_dict["stage_files"][stage]["enabled"]:
        return None

    suffix = autoclean_dict["stage_files"][stage]["suffix"]
    basename = Path(autoclean_dict["unprocessed_file"]).stem
    stage_num = _get_stage_number(stage, autoclean_dict)
    
    # Save to stage directory
    if output_path is None:
        output_path = autoclean_dict["stage_dir"]
    subfolder = output_path / f"{stage_num}{suffix}"
    subfolder.mkdir(exist_ok=True)
    stage_path = subfolder / f"{basename}{suffix}_epo.set"
    
    # Save to both locations for post_comp
    paths = [stage_path]
    if stage == "post_comp":
        clean_path = autoclean_dict["clean_dir"] / f"{basename}{suffix}.set"
        autoclean_dict["clean_dir"].mkdir(exist_ok=True)
        paths.append(clean_path)

    # Save to all paths
    epochs.info["description"] = autoclean_dict["run_id"]
    epochs.apply_proj()
    for path in paths:
        epochs.export(path, fmt="eeglab", overwrite=True)
        # Add run_id to each file
        EEG = sio.loadmat(path)
        EEG["etc"] = {}
        EEG["etc"]["run_id"] = autoclean_dict["run_id"]
        sio.savemat(path, EEG, do_compression=False)
        message("success", f"✓ Saved {stage} file to: {path}")

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
            "tmin": epochs.tmin,
            "tmax": epochs.tmax,
        }
    }

    run_id = autoclean_dict["run_id"]
    manage_database(operation="update", update_record={"run_id": run_id, "metadata": metadata})
    manage_database(
        operation="update_status",
        update_record={"run_id": run_id, "status": f"{stage} completed"},
    )

    return paths[0]  # Return stage path for consistency

def _create_event_id_dictionary(raw):
    """
    Create a dictionary mapping unique event types to sequential IDs by parsing raw.annotations descriptions.
    
    Parameters:
        raw (mne.io.Raw): MNE Raw object containing annotations with descriptions in format 'type/Task/Condition'
        
    Returns:
        dict: Mapping of unique event types to sequential integers (starting from 1)
    """
    def parse_description(description):
        parts = description.split('/')
        return parts[:2] + ['n/a'] if len(parts) > 3 else parts
    
    # Parse descriptions to get event types
    parsed_data = [parse_description(desc) for desc in raw.annotations.description]
    parsed_df = pd.DataFrame(parsed_data, columns=['Task', 'type', 'Condition'])
    # Get unique types and sort for consistency
    unique_types = sorted(set(parsed_df.type))
    # Create sequential mapping
    event_id_dict = {event_type: idx + 1 for idx, event_type in enumerate(unique_types)}
    return event_id_dict