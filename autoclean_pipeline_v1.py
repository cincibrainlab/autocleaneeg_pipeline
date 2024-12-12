# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mne",
#     "rich", 
#     "numpy",
#     "python-dotenv",
#     "openneuro-py",
#     "pyyaml",
#     "schema",
#     "mne-bids",
#     "pandas",
#     "pathlib",
#     "pybv",
#     "torch",
#     "pyprep",
#     "eeglabio",
#     "autoreject",
#     "python-ulid",
#     "pylossless @ /Users/ernie/Documents/GitHub/EegServer/pylossless",
#     "textual",
#     "textual-dev",
#     "asyncio",
#     "mplcairo"
# ]
# ///

"""
autoclean: Automated EEG Processing Pipeline

A streamlined pipeline for automated EEG data processing and cleaning.
Handles BIDS conversion, preprocessing, artifact rejection, and quality control.
"""

from pathlib import Path
from typing import Union
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
import os
import sys
import logging
from dotenv import load_dotenv
from rich.table import Table
from rich.logging import RichHandler

import mne_bids as mb
import autoreject
import mne
import pylossless as ll

import yaml
from schema import Schema, Optional, Or

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

import json

from datetime import datetime
import datetime

from mne_bids import BIDSPath, write_raw_bids, update_sidecar_json
from mne.io.constants import FIFF
import pandas as pd
from pathlib import Path




# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autoclean.log'),
        #logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('autoclean')

load_dotenv()

# sys.tracebacklimit = 0

console = Console()


def step_convert_to_bids(
    raw,
    output_dir,
    task="rest", 
    participant_id=None,
    line_freq=60.0,
    overwrite=False,
    events=None,
    event_id=None,
    study_name="EEG Study"):
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

    Dependent Functions:
    - step_sanitize_id(): Sanitizes and formats participant ID from filename
    - step_create_dataset_desc(): Creates BIDS dataset description JSON file
    - step_create_participants_json(): Creates participants.json metadata file
    - update_sidecar_json(): Updates sidecar JSON files with additional metadata
    """
    import hashlib


    file_path = raw.filenames[0]

    console = Console()
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
        console.print(f"[red]Failed to read {fif_file}: {e}[/red]")
        sys.exit(1)

    # Prepare additional metadata
    raw.info["subject_info"] = {"id": int(subject_id), "age": None, "sex": sex}

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
        "allow_preload": True
    }

    # Write BIDS data
    try:
        write_raw_bids(**bids_kwargs)
        console.print(f"[green]Converted {fif_file.name} to BIDS format.[/green]")
        entries = {"Manufacturer": "Unknown", "PowerLineFrequency": line_freq}
        sidecar_path = bids_path.copy().update(extension=".json")
        update_sidecar_json(bids_path=sidecar_path, entries=entries)
    except Exception as e:
        console.print(f"[red]Failed to write BIDS for {fif_file.name}: {e}[/red]")
        sys.exit(1)

    # Update participants.tsv
    participants_file = bids_root / "participants.tsv"
    if not participants_file.exists():
        participants_df = pd.DataFrame(
            columns=["participant_id", "age", "sex", "group"]
        )
    else:
        participants_df = pd.read_csv(participants_file, sep="\t")

    new_entry = {
        "participant_id": f"sub-{subject_id}",
        "bids_path": bids_path,
        "age": age,
        "sex": sex,
        "group": group,
        "eegid": fif_file.stem,
        "file_name": file_name,
        "file_hash": file_hash,
    }

    participants_df = participants_df._append(new_entry, ignore_index=True)
    participants_df.drop_duplicates(subset="participant_id", keep="last", inplace=True)
    participants_df.to_csv(participants_file, sep="\t", index=False, na_rep="n/a")

    # Create dataset_description.json if it doesn't exist
    dataset_description_file = bids_root / "dataset_description.json"
    if not dataset_description_file.exists():
        step_create_dataset_desc(bids_root, study_name=study_name)

    # Create participants.json if it doesn't exist
    participants_json_file = bids_root / "participants.json"
    if not participants_json_file.exists():
        step_create_participants_json(bids_root)

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
    console.print(f"Unique Number for {basename}: {participant_id}")

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
    console.print("[green]Created dataset_description.json[/green]")

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
    console.print("[green]Created participants.json[/green]")


def step_resample_data(raw, resample_freq):
    """Resample data using frequency from config."""
    return raw.resample(resample_freq)

def step_mark_eog_channels(raw, eog_channels):
    """Set EOG channels based on config."""
    eog_channels = [
        f"E{ch}" for ch in sorted(eog_channels)
    ]
    raw.set_channel_types({ch: "eog" for ch in raw.ch_names if ch in eog_channels})
    return raw

def step_crop_data(raw, crop_start, crop_end):
    """Crop data based on config settings."""
    if crop_end is None:
        tmax = raw.times[-1]  # Use the maximum time available
    return raw.load_data().crop(tmin=crop_start, tmax=crop_end)

def step_set_reference(raw, ref_type):
    """Set EEG reference based on config."""
    return raw.step_set_reference(ref_type, projection=True)

def step_log_start(unprocessed_file: Union[str, Path], eeg_system: str, task: str, autoclean_config_file: Union[str, Path]) -> None:
    """Log and display pipeline startup information."""
    logger.info("Starting autoclean pipeline")
    logger.debug(f"Input parameters: file={unprocessed_file}, system={eeg_system}, task={task}, config={autoclean_config_file}")
    
    console.print(Panel("[bold blue]autoclean: Processing EEG Data[/bold blue]"))
    console.print(f"[cyan]Input File:[/cyan] {unprocessed_file}")
    console.print(f"[cyan]EEG System:[/cyan] {eeg_system}")
    console.print(f"[cyan]Task:[/cyan] {task}")
    console.print(f"[cyan]AutocleanConfig File:[/cyan] {autoclean_config_file}")

def log_pipeline_progress(step: int) -> None:
    """Log pipeline processing progress."""
    logger.debug(f"Processing step {step+1}/3")

def log_pipeline_completion() -> None:
    """Log and display pipeline completion."""
    logger.info("Pipeline processing completed successfully")
    console.print(Panel("[bold green]✅ autoclean Processing Complete![/bold green]"))
    console.print(f"[cyan]Log location:[/cyan] {os.path.abspath('autoclean.log')}\n")

def step_import_raw(autoclean_dict: dict, preload: bool = True) -> mne.io.Raw:
    """Import and configure raw EEG data.
    
    Args:
        autoclean_dict: Dictionary containing pipeline configuration including:
                       - unprocessed_file: Path to raw EEG data file
                       - eeg_system: Name of the EEG system montage
        preload: If True, data will be loaded into memory at initialization.
                If False, data will not be loaded until explicitly called.
        
    Returns:
        mne.io.Raw: Imported and configured raw EEG data with appropriate montage set
        
    Raises:
        ValueError: If the specified EEG system is not supported
        FileNotFoundError: If the input file does not exist
        RuntimeError: If there is an error importing or configuring the data
    """
    unprocessed_file = autoclean_dict["unprocessed_file"]
    eeg_system = autoclean_dict["eeg_system"]
    
    logger.info(f"Importing raw EEG data from {unprocessed_file} using {eeg_system} system")
    console.print("[cyan]Importing raw EEG data...[/cyan]")
    
    try:
        # Import based on EEG system type
        if eeg_system == "GSN-HydroCel-129":
            raw = mne.io.read_raw_egi(input_fname=unprocessed_file, preload=preload, events_as_annotations=False)
            montage = mne.channels.make_standard_montage(eeg_system)
            montage.ch_names[128] = "E129"
            raw.set_montage(montage, match_case=False)
            raw.pick('eeg')
        else:
            raise ValueError(f"Unsupported EEG system: {eeg_system}")
            
        logger.info("Raw EEG data imported successfully")
        console.print("[green]✓ Raw EEG data imported successfully[/green]")

        metadata = {
            "step_import_raw": {
                "creationDateTime": datetime.now().isoformat(),
                "sampleRate": raw.info["sfreq"],
                "channelCount": len(raw.ch_names),
                "durationSec": int(raw.n_times) / raw.info["sfreq"],
                "numberSamples": int(raw.n_times)
            }
        }

        # Save metadata
        step_handle_metadata(autoclean_dict, metadata, mode='save')

        return raw
        
    except Exception as e:
        logger.error(f"Failed to import raw EEG data: {str(e)}")
        console.print("[red]Error importing raw EEG data[/red]")
        raise

def step_load_config(config_file: Union[str, Path]) -> dict:
    """Load and validate the autoclean configuration file.
    
    Args:
        config_file: Path to YAML configuration file
        
    Returns:
        dict: Task-specific configuration settings
    """
    
    logger.info(f"Loading configuration from {config_file}")
    console.print("[cyan]Loading configuration...[/cyan]")

    # Define configuration schema matching autoclean_config.yaml structure
    config_schema = Schema({
        'tasks': {
            str: {
                'mne_task': str,
                'description': str,
                'lossless_config': str,
                'settings': {
                    'resample_step': {
                        'enabled': bool,
                        'value': int
                    },
                    'eog_step': {
                        'enabled': bool,
                        'value': list
                    },
                    'trim_step': {
                        'enabled': bool,
                        'value': int
                    },
                    'crop_step': {
                        'enabled': bool,
                        'value': {
                            'start': int,
                            'end': Or(float, None)
                        }
                    },
                    'reference_step': {
                        'enabled': bool,
                        'value': str
                    },
                    'filter_step': {
                        'enabled': bool,
                        'value': {
                            'l_freq': Or(float, None),
                            'h_freq': Or(float, None)
                        }
                    },
                    'montage': {
                        'enabled': bool,
                        'value': str
                    }
                },
                'rejection_policy': {
                    'ch_flags_to_reject': list,
                    'ch_cleaning_mode': str,
                    'interpolate_bads_kwargs': {
                        'method': str
                    },
                    'ic_flags_to_reject': list,
                    'ic_rejection_threshold': float,
                    'remove_flagged_ics': bool
                }
            }
        },
        'stage_files': {
            str: {
                'enabled': bool,
                'suffix': str
            }
        }
    })

    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)
            
        # Validate against schema
        autoclean_dict = config_schema.validate(config)
    
        # Log loaded configuration
        logger.info("Configuration loaded successfully")
        console.print("[green]✓ Configuration loaded successfully[/green]")
        logger.debug(f"Task configurations: {autoclean_dict}")
        
        return autoclean_dict
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        console.print("[red]Error loading configuration file[/red]")
        raise

def step_list_tasks(autoclean_dict: dict) -> list[str]:
    """Get available tasks from autoclean dictionary."""
    return list(autoclean_dict['tasks'].keys())

def step_select_task_config(task: str, task_configs: dict) -> dict:
    """Select and return the configuration for a given task."""
    return task_configs[task]
    
def step_handle_metadata(autoclean_dict, metadata=None, mode='load'):
    """
    Handle metadata operations for loading, saving, and updating JSON metadata files.
    
    This function manages a JSON metadata file that tracks processing steps and parameters
    throughout the pipeline. It can initialize new metadata, load existing metadata, 
    and update metadata with new information.

    Args:
        autoclean_dict (dict): Dictionary containing pipeline configuration and paths
        metadata (dict, optional): New metadata to add/update. Defaults to None.
        mode (str): Operation mode - either 'load' to read existing metadata or 
                   'save' to write/update metadata. Defaults to 'load'.
    
    Returns:
        dict: The loaded or updated metadata dictionary

    The metadata file is stored in the metadata directory specified in autoclean_dict,
    with filename format: {unprocessed_filename_stem}_autoclean_metadata.json

    Special handling is included for 'save_raw_to_set' entries, which are stored as a list
    to track multiple saves of the same type.
    """
    metadata_dir = autoclean_dict["metadata_dir"]
    unprocessed_file = Path(autoclean_dict["unprocessed_file"]) 
    json_file = metadata_dir / f"{unprocessed_file.stem}_autoclean_metadata.json"
    autoclean_dict["metadata_file"] = json_file

    logger.debug(f"Handling metadata in {mode} mode for file: {json_file}")

    if mode == 'load':
        try:
            logger.info(f"Loading metadata from {json_file}")
            with open(json_file, "r") as f:
                metadata = json.load(f)
            # Add unprocessed key if it doesn't exist
            if "step_handle_metadata" not in metadata:
                logger.debug("Adding step_handle_metadata key to metadata")
                metadata["step_handle_metadata"] = {
                    "creationDateTime": datetime.now().isoformat(),
                    "unprocessedFile": str(autoclean_dict["unprocessed_file"].name),
                    "unprocessedPath": str(autoclean_dict["unprocessed_file"].parent),
                    "task": autoclean_dict["task"],
                    "eegSystem": autoclean_dict["eeg_system"], 
                    "configFile": str(autoclean_dict["config_file"])
                }
                # Save updated metadata
                logger.debug("Saving updated metadata with step_handle_metadata")
                with open(json_file, "w") as f:
                    json.dump(metadata, f, indent=4)
            return metadata
        except FileNotFoundError:
            logger.info(f"No existing metadata found at {json_file}. Creating new metadata file")
            # Initialize new metadata on first run
            metadata = {
                "step_handle_metadata": {
                    "creationDateTime": datetime.now().isoformat(),
                    "unprocessedFile": str(autoclean_dict["unprocessed_file"].name),
                    "unprocessedPath": str(autoclean_dict["unprocessed_file"].parent),
                    "task": autoclean_dict["task"],
                    "eegSystem": autoclean_dict["eeg_system"],
                    "configFile": str(autoclean_dict["config_file"])
                }
            }
            # Save initial metadata
            with open(json_file, "w") as f:
                json.dump(metadata, f, indent=4)
            logger.info("Successfully created new metadata file")
            return metadata
            
    elif mode == 'save':
        logger.info(f"Saving metadata to {json_file}")
        # Load existing metadata if available
        existing_metadata = {}
        if json_file.exists():
            logger.debug("Loading existing metadata before update")
            with open(json_file, "r") as f:
                existing_metadata = json.load(f)
                
            # Add unprocessed key if it doesn't exist
            if "step_handle_metadata" not in existing_metadata:
                logger.debug("Adding step_handle_metadata key to existing metadata")
                existing_metadata["step_handle_metadata"] = {
                    "creationDateTime": datetime.now().isoformat(),
                    "unprocessedFile": str(autoclean_dict["unprocessed_file"].name),
                    "unprocessedPath": str(autoclean_dict["unprocessed_file"].parent),
                    "task": autoclean_dict["task"],
                    "eegSystem": autoclean_dict["eeg_system"],
                    "configFile": str(autoclean_dict["config_file"])
                }
        
        # Update with new metadata if provided
        if metadata is not None:
            logger.debug("Updating metadata with new information")
            # If save_raw_to_set exists, handle as list of dicts
            if "save_raw_to_set" in metadata:
                logger.debug("Handling save_raw_to_set metadata")
                if "save_raw_to_set" not in existing_metadata:
                    existing_metadata["save_raw_to_set"] = [metadata["save_raw_to_set"]]
                else:
                    if not isinstance(existing_metadata["save_raw_to_set"], list):
                        existing_metadata["save_raw_to_set"] = [existing_metadata["save_raw_to_set"]]
                    existing_metadata["save_raw_to_set"].append(metadata["save_raw_to_set"])
                del metadata["save_raw_to_set"]
            
            # Update remaining metadata
            existing_metadata.update(metadata)

        # Save updated metadata
        try:
            with open(json_file, "w") as f:
                json.dump(existing_metadata, f, indent=4)
            logger.info("Successfully saved metadata to JSON file")
            console.print("[green]Successfully saved metadata JSON file[/green]")
        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")
            raise
            
        return existing_metadata

def step_get_rejection_policy(autoclean_dict: dict) -> dict:

    task = autoclean_dict["task"]
    # Create a new rejection policy for cleaning channels and removing ICs
    rejection_policy = ll.RejectionPolicy()

    # Set parameters for channel rejection
    rejection_policy["ch_flags_to_reject"] = autoclean_dict['tasks'][task]['rejection_policy']['ch_flags_to_reject']
    rejection_policy["ch_cleaning_mode"] = autoclean_dict['tasks'][task]['rejection_policy']['ch_cleaning_mode']
    rejection_policy["interpolate_bads_kwargs"] = {"method": autoclean_dict['tasks'][task]['rejection_policy']['interpolate_bads_kwargs']['method']}

    # Set parameters for IC rejection
    rejection_policy["ic_flags_to_reject"] = autoclean_dict['tasks'][task]['rejection_policy']['ic_flags_to_reject']
    rejection_policy["ic_rejection_threshold"] = autoclean_dict['tasks'][task]['rejection_policy']['ic_rejection_threshold']
    rejection_policy["remove_flagged_ics"] = autoclean_dict['tasks'][task]['rejection_policy']['remove_flagged_ics']

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
            "remove_flagged_ics": rejection_policy["remove_flagged_ics"]
        }
    }
    step_handle_metadata(autoclean_dict, metadata, mode='save')

    # Log rejection policy details
    console.print("[bold blue]Rejection Policy Settings:[/bold blue]")
    console.print(f"Channel flags to reject: {rejection_policy['ch_flags_to_reject']}")
    console.print(f"Channel cleaning mode: {rejection_policy['ch_cleaning_mode']}")
    console.print(f"Interpolation method: {rejection_policy['interpolate_bads_kwargs']['method']}")
    console.print(f"IC flags to reject: {rejection_policy['ic_flags_to_reject']}")
    console.print(f"IC rejection threshold: {rejection_policy['ic_rejection_threshold']}")
    console.print(f"Remove flagged ICs: {rejection_policy['remove_flagged_ics']}")

    return rejection_policy


def step_plot_ica_full(pipeline, autoclean_dict):
    """
    Plot ICA components over the full duration with their labels and probabilities.
    
    Parameters:
    -----------
    pipeline : pylossless.Pipeline
        PyLossless pipeline object containing raw data and ICA
    autoclean_dict : dict
        Autoclean dictionary containing metadata
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Get raw and ICA from pipeline
    raw = pipeline.raw
    ica = pipeline.ica2
    ic_labels = pipeline.flags['ic']

    # Get ICA activations and create time vector
    ica_sources = ica.get_sources(raw)
    ica_data = ica_sources.get_data()
    sfreq = raw.info['sfreq']
    times = raw.times
    n_components, n_samples = ica_data.shape

    # Normalize each component individually for better visibility
    for idx in range(n_components):
        component = ica_data[idx]
        # Scale to have a consistent peak-to-peak amplitude
        ptp = np.ptp(component)
        if ptp == 0:
            scaling_factor = 2.5  # Avoid division by zero
        else:
            scaling_factor = 2.5 / ptp
        ica_data[idx] = component * scaling_factor

    # Determine appropriate spacing
    spacing = 2  # Fixed spacing between components

    # Calculate figure size proportional to duration
    total_duration = times[-1] - times[0]
    width_per_second = 0.1  # Increased from 0.02 to 0.1 for wider view
    fig_width = total_duration * width_per_second
    max_fig_width = 200  # Doubled from 100 to allow wider figures
    fig_width = min(fig_width, max_fig_width)
    fig_height = max(6, n_components * 0.5)  # Ensure a minimum height

    # Create plot with wider figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create a colormap for the components
    cmap = plt.cm.get_cmap('tab20', n_components)
    line_colors = [cmap(i) for i in range(n_components)]

    # Plot components in original order
    for idx in range(n_components):
        offset = idx * spacing
        ax.plot(times, ica_data[idx] + offset, color=line_colors[idx], linewidth=0.5)

    # Set y-ticks and labels
    yticks = [idx * spacing for idx in range(n_components)]
    yticklabels = []
    for idx in range(n_components):
        label_text = f"IC{idx + 1}: {ic_labels['ic_type'][idx]} ({ic_labels['confidence'][idx]:.2f})"
        yticklabels.append(label_text)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8)

    # Customize axes
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_title('ICA Component Activations (Full Duration)', fontsize=14)
    ax.set_xlim(times[0], times[-1])

    # Adjust y-axis limits
    ax.set_ylim(-spacing, (n_components - 1) * spacing + spacing)

    # Remove y-axis label as we have custom labels
    ax.set_ylabel('')

    # Invert y-axis to have the first component at the top
    ax.invert_yaxis()

    # Color the labels red or black based on component type
    artifact_types = ['eog', 'muscle', 'ecg', 'other']
    for ticklabel, idx in zip(ax.get_yticklabels(), range(n_components)):
        ic_type = ic_labels['ic_type'][idx]
        if ic_type in artifact_types:
            ticklabel.set_color('red')
        else:
            ticklabel.set_color('black')

    # Adjust layout
    plt.tight_layout()

    # Get output path for ICA components figure
    derivatives_path = pipeline.get_derivative_path(autoclean_dict["bids_path"])
    target_figure = str(derivatives_path.copy().update(
        suffix='ica_components_full_duration',
        extension='.png',
        datatype='eeg'
    ))

    # Save figure with higher DPI for better resolution of wider plot
    fig.savefig(target_figure, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return fig

def step_plot_overlay(raw_original, raw_cleaned, pipeline, autoclean_dict, suffix=''):
    """
    Plot raw data channels over the full duration, overlaying the original and cleaned data.
    Original data is plotted in red, cleaned data in black.

    Parameters:
    -----------
    raw_original : mne.io.Raw
        Original raw EEG data before cleaning.
    raw_cleaned : mne.io.Raw
        Cleaned raw EEG data after preprocessing.
    pipeline : pylossless.Pipeline
        Pipeline object (can be None if not used).
    autoclean_dict : dict
        Autoclean dictionary containing metadata.
    suffix : str
        Suffix for the filename.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Ensure that the original and cleaned data have the same channels and times
    if raw_original.ch_names != raw_cleaned.ch_names:
        raise ValueError("Channel names in raw_original and raw_cleaned do not match.")
    if raw_original.times.shape != raw_cleaned.times.shape:
        raise ValueError("Time vectors in raw_original and raw_cleaned do not match.")

    # Get raw data
    channel_labels = raw_original.ch_names
    n_channels = len(channel_labels)
    sfreq = raw_original.info['sfreq']
    times = raw_original.times
    n_samples = len(times)
    data_original = raw_original.get_data()
    data_cleaned = raw_cleaned.get_data()

    # Increase downsample factor to reduce file size
    desired_sfreq = 100  # Reduced sampling rate to 100 Hz
    downsample_factor = int(sfreq // desired_sfreq)
    if downsample_factor > 1:
        data_original = data_original[:, ::downsample_factor]
        data_cleaned = data_cleaned[:, ::downsample_factor]
        times = times[::downsample_factor]

    # Normalize each channel individually for better visibility
    data_original_normalized = np.zeros_like(data_original)
    data_cleaned_normalized = np.zeros_like(data_cleaned)
    spacing = 10  # Fixed spacing between channels
    for idx in range(n_channels):
        # Original data
        channel_data_original = data_original[idx]
        channel_data_original = channel_data_original - np.mean(channel_data_original)  # Remove DC offset
        std = np.std(channel_data_original)
        if std == 0:
            std = 1  # Avoid division by zero
        data_original_normalized[idx] = channel_data_original / std  # Normalize to unit variance

        # Cleaned data
        channel_data_cleaned = data_cleaned[idx]
        channel_data_cleaned = channel_data_cleaned - np.mean(channel_data_cleaned)  # Remove DC offset
        # Use same std for normalization to ensure both signals are on the same scale
        data_cleaned_normalized[idx] = channel_data_cleaned / std

    # Multiply by a scaling factor to control amplitude
    scaling_factor = 2  # Adjust this factor as needed for visibility
    data_original_scaled = data_original_normalized * scaling_factor
    data_cleaned_scaled = data_cleaned_normalized * scaling_factor

    # Calculate offsets for plotting
    offsets = np.arange(n_channels) * spacing

    # Create plot
    total_duration = times[-1] - times[0]
    width_per_second = 0.1  # Adjust this factor as needed
    fig_width = min(total_duration * width_per_second, 50)
    fig_height = max(6, n_channels * 0.25)  # Adjusted for better spacing

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Plot channels
    for idx in range(n_channels):
        ch_name = channel_labels[idx]
        offset = offsets[idx]

        # Plot original data in red
        ax.plot(times, data_original_scaled[idx] + offset, color='red', linewidth=0.5, linestyle='-')

        # Plot cleaned data in black
        ax.plot(times, data_cleaned_scaled[idx] + offset, color='black', linewidth=0.5, linestyle='-')

    # Set y-ticks and labels
    ax.set_yticks(offsets)
    ax.set_yticklabels(channel_labels, fontsize=8)

    # Customize axes
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_title('Raw Data Channels: Original vs Cleaned (Full Duration)', fontsize=14)
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-spacing, offsets[-1] + spacing)
    ax.set_ylabel('')
    ax.invert_yaxis()

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=0.5, linestyle='-', label='Original Data'),
        Line2D([0], [0], color='black', lw=0.5, linestyle='-', label='Cleaned Data')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()

    # Create Artifact Report
    derivatives_path  = pipeline.get_derivative_path(autoclean_dict["bids_path"])

    # Independent Components
    target_figure = str(derivatives_path.copy().update(
        suffix='data_trace_overlay',
        extension='.png',
        datatype='eeg'
    ))

    # Save as PNG with high DPI for quality
    fig.savefig(target_figure, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Raw channels overlay full duration plot saved to {target_figure}")


    return fig
def step_plot_psd_overlay(raw_original, raw_cleaned, pipeline, autoclean_dict, suffix=''):
    """
    Generate and save PSD overlays for the original and cleaned data.

    Parameters:
    -----------
    raw_original : mne.io.Raw
        Original raw EEG data before cleaning.
    raw_cleaned : mne.io.Raw
        Cleaned raw EEG data after preprocessing.
    pipeline : pylossless.Pipeline
        Pipeline object (can be None if not used).
    autoclean_dict : dict
        Autoclean dictionary containing metadata.
    suffix : str
        Suffix for the filename.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import mne

    # Select all EEG channels
    picks = mne.pick_types(raw_original.info, eeg=True)
    if len(picks) == 0:
        raise ValueError("No EEG channels found in raw data.")

    # Parameters for PSD
    fmin = 0
    fmax = 100
    n_fft = int(raw_original.info['sfreq'] * 2)  # Window length of 2 seconds

    # Compute PSD for original data
    psd_original = raw_original.compute_psd(
        method='welch',
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        picks=picks,
        average='mean',
        verbose=False
    )
    freqs = psd_original.freqs
    psd_original_data = psd_original.get_data()

    # Compute PSD for cleaned data
    psd_cleaned = raw_cleaned.compute_psd(
        method='welch',
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        picks=picks,
        average='mean',
        verbose=False
    )
    psd_cleaned_data = psd_cleaned.get_data()

    # Average PSD across channels
    psd_original_mean = np.mean(psd_original_data, axis=0)
    psd_cleaned_mean = np.mean(psd_cleaned_data, axis=0)

    # Convert power to dB
    psd_original_db = 10 * np.log10(psd_original_mean)
    psd_cleaned_db = 10 * np.log10(psd_cleaned_mean)

    # Create figure for PSD
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, psd_original_db, color='red', label='Original')
    plt.plot(freqs, psd_cleaned_db, color='black', label='Cleaned')

    # Add vertical lines for power bands
    power_bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 100)
    }

    for band_name, (f_start, f_end) in power_bands.items():
        plt.axvline(f_start, color='grey', linestyle='--', linewidth=1)
        plt.axvline(f_end, color='grey', linestyle='--', linewidth=1)
        # Fill the band area
        plt.fill_betweenx(plt.ylim(), f_start, f_end, color='grey', alpha=0.1)
        # Annotate band names
        plt.text((f_start + f_end) / 2, plt.ylim()[1] - 5, band_name,
                 horizontalalignment='center', verticalalignment='top', fontsize=9, color='grey')

    plt.xlim(fmin, fmax)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB)')
    plt.title('Average Power Spectral Density (0–100 Hz)')
    plt.legend()
    plt.tight_layout()

    # Get output path for bad channels figure
    derivatives_path = pipeline.get_derivative_path(autoclean_dict["bids_path"])
    target_figure = str(derivatives_path.copy().update(
        suffix='psd_overlay',
        extension='.png',
        datatype='eeg'
    ))


    plt.savefig(target_figure, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"PSD overlay saved to {target_figure}")


def step_plot_band_topos(raw, pipeline, autoclean_dict, bands=None, metadata=None):
    """
    Generate and save a single high-resolution topographical map image 
    for multiple EEG frequency bands arranged horizontally. Also creates a 
    JSON sidecar with metadata for database ingestion.

    Parameters:
    -----------
    raw : mne.io.Raw
        The raw EEG data.
    output_dir : str
        Directory where the topoplot image and JSON sidecar will be saved.
    bands : list of tuple, optional
        List of frequency bands to plot. Each tuple should contain 
        (band_name, lower_freq, upper_freq).
    metadata : dict, optional
        Additional metadata to include in the JSON sidecar.

    Returns:
    --------
    image_path : str
        Path to the saved topoplot image.
    sidecar_path : str
        Path to the saved JSON sidecar.
    """

        # Create Artifact Report
    derivatives_path            = pipeline.get_derivative_path(autoclean_dict["bids_path"])

    # Independent Components
    target_figure = str(derivatives_path.copy().update(
        suffix='topoplot',
        extension='.png',
        datatype='eeg'
    ))

    # Define default frequency bands if none provided
    if bands is None:
        bands = [
            ("Delta", 1, 4),
            ("Theta", 4, 8),
            ("Alpha", 8, 12),
            ("Beta", 12, 30),
            ("Gamma1", 30, 60),
            ("Gamma2", 60, 80)
        ]
    
    # Ensure output directory exists
    # os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dictionary to store band powers for metadata
    band_powers_metadata = {}
    
    # Compute PSD using compute_psd
    spectrum = raw.compute_psd(method='welch', fmin=1, fmax=80, picks='eeg')
    
    # Compute band power for each frequency band
    band_powers = []
    for band_name, l_freq, h_freq in bands:
        # Get band power using the spectrum object
        band_power = spectrum.get_data(fmin=l_freq, fmax=h_freq).mean(axis=-1)
        band_powers.append(band_power)
        
        # Store in metadata
        band_powers_metadata[band_name] = {
            "frequency_band": f"{l_freq}-{h_freq} Hz",
            "band_power_mean": float(np.mean(band_power)),
            "band_power_std": float(np.std(band_power))
        }
    # Create a figure with subplots arranged horizontally
    num_bands = len(bands)
    fig, axes = plt.subplots(1, num_bands, figsize=(5*num_bands, 6))  # Increased height
    
    # If only one band, axes is not a list
    if num_bands == 1:
        axes = [axes]
    
    # Add a title to the entire figure with the filename
    fig.suptitle(os.path.basename(raw.filenames[0]), fontsize=16)
    
    for ax, (band, power) in zip(axes, zip(bands, band_powers)):
        band_name, l_freq, h_freq = band
        # Plot topomap
        mne.viz.plot_topomap(
            power, raw.info, axes=ax, show=False, contours=0, cmap='jet'
        )
        ax.set_title(f"{band_name}\n({l_freq}-{h_freq} Hz)", fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to accommodate suptitle
    
# Generate unique filename
    image_path = target_figure
    
    # Save the figure in high resolution
    fig.savefig(image_path, dpi=300)
    plt.close(fig)
    
    return image_path

def step_run_pylossless(autoclean_dict):

    task                    = autoclean_dict["task"]
    bids_path               = autoclean_dict["bids_path"]
    config_path             = autoclean_dict["tasks"][task]["lossless_config"] 
    derivative_name         = "pylossless"
    raw = mb.read_raw_bids(
        bids_path, verbose="ERROR", extra_params={"preload": True}
    )

    
    try:
        pipeline = ll.LosslessPipeline(config_path)
        pipeline.run_with_raw(raw)
        
        derivatives_path = pipeline.get_derivative_path(
            bids_path, derivative_name
        )
        pipeline.save(
            derivatives_path, overwrite=True, format="BrainVision")

    except Exception as e:
        console.print(f"[red]Error: Failed to run pylossless: {str(e)}[/red]")
        raise e

    try:
        pylossless_config = yaml.safe_load(open(config_path))
        metadata = {
            "step_run_pylossless": {
                "creationDateTime": datetime.now().isoformat(),
                "derivativeName": derivative_name,
                "configFile": str(config_path),
                "pylossless_config": pylossless_config
            }
        }
        step_handle_metadata(autoclean_dict, metadata, mode='save')
    except Exception as e:
        console.print(f"[red]Error: Failed to load pylossless config: {str(e)}[/red]")
        raise e

    return pipeline



import numpy as np
from scipy import signal

def step_detect_bad_eog(raw, decimate_factor=10, window_size=100, threshold_std=5):
    # Decimate the data first
    data = raw.get_data(picks=['eog'])
    decimated_data = signal.decimate(data, decimate_factor, axis=1)
    
    # Adjust window size for decimated data
    adj_window = window_size // decimate_factor
    
    # Use stride tricks on decimated data
    strides = np.lib.stride_tricks.sliding_window_view(decimated_data, 
                                                     window_shape=adj_window, 
                                                     axis=-1)
    
    # Vectorized operations on smaller data
    rolling_std = np.std(strides, axis=-1)
    global_std = np.std(rolling_std, axis=1, keepdims=True)
    global_mean = np.mean(rolling_std, axis=1, keepdims=True)
    outlier_mask = np.any(rolling_std > global_mean + threshold_std * global_std, axis=1)
    
    bad_channels = [raw.ch_names[i] for i in np.where(outlier_mask)[0]]
    return bad_channels


def step_clean_bad_channels(raw):
    
    from time import perf_counter

    import mne
    import numpy as np
    from scipy import signal as signal

    from pyprep.find_noisy_channels import NoisyChannels


    # bad_channels = step_detect_bad_eog(raw)

    # Temporarily switch EOG channels to EEG type
    eog_picks = mne.pick_types(raw.info, eog=True, exclude=[])
    eog_ch_names = [raw.ch_names[idx] for idx in eog_picks]
    raw.set_channel_types({ch: 'eeg' for ch in eog_ch_names})

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])

    cleaned_raw = NoisyChannels(raw, random_state=1337)
    cleaned_raw.find_all_bads(ransac=True, channel_wise=False, max_chunk_size=None)


    # Set EOG channels to EEG temporarily
    #raw.set_channel_types({ch: 'eog' for ch in eog_ch_names})

    threshold=3.0

    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    # start_time = perf_counter()
    # cleaned_raw.find_bad_by_ransac(channel_wise=True)
    # cleaned_raw.find_bad_by_SNR()
    # cleaned_raw.find_bad_by_correlation()
    # cleaned_raw.find_bad_by_deviation(deviation_threshold=5.0)
    # cleaned_raw.find_bad_by_hfnoise(HF_zscore_threshold=5.0)
    # cleaned_raw.find_bad_by_nan_flat()
    # cleaned_raw.get_bads()
    # print("--- %s seconds ---" % (perf_counter() - start_time))
    # breakpoint()

    print(raw.info["bads"])
    raw.info["bads"].extend(cleaned_raw.get_bads())

    return raw

import mne
import numpy as np
import pandas as pd
from autoreject import AutoReject
import random
from typing import Tuple, Dict, Optional
import logging

def clean_epochs(
    epochs: mne.Epochs,
    gfp_threshold: float = 3.0,
    number_of_epochs: Optional[int] = None,
    apply_autoreject: bool = False,
    random_seed: Optional[int] = None
) -> Tuple[mne.Epochs, Dict[str, any]]:
    """
    Clean an MNE Epochs object by applying artifact rejection and removing outlier epochs based on GFP.
    
    Args:
        epochs (mne.Epochs): The input epoched EEG data.
        gfp_threshold (float, optional): Z-score threshold for GFP-based outlier detection. 
                                         Epochs with GFP z-scores above this value are removed.
                                         Defaults to 3.0.
        number_of_epochs (int, optional): If specified, randomly selects this number of epochs from the cleaned data.
                                           If None, retains all cleaned epochs. Defaults to None.
        apply_autoreject (bool, optional): Whether to apply AutoReject for artifact correction. Defaults to True.
        random_seed (int, optional): Seed for random number generator to ensure reproducibility when selecting epochs.
                                     Defaults to None.
    
    Returns:
        Tuple[mne.Epochs, Dict[str, any]]: A tuple containing the cleaned Epochs object and a dictionary of statistics.
    
    Raises:
        ValueError: If after cleaning, the number of epochs is less than `number_of_epochs` when specified.
    """
    logger = logging.getLogger('clean_epochs')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # Prevent adding multiple handlers in interactive environments
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    logger.info("Starting epoch cleaning process.")
    
    # Force preload to avoid RuntimeError
    if not epochs.preload:
        epochs.load_data()
    
    # Step 1: Artifact Rejection using AutoReject
    if apply_autoreject:
        logger.info("Applying AutoReject for artifact rejection.")
        ar = AutoReject()
        epochs_clean = ar.fit_transform(epochs)
        logger.info(f"Artifacts rejected: {len(epochs) - len(epochs_clean)} epochs removed by AutoReject.")
    else:
        epochs_clean = epochs.copy()
        logger.info("AutoReject not applied. Proceeding without artifact rejection.")
    
    # Step 2: Calculate Global Field Power (GFP)
    logger.info("Calculating Global Field Power (GFP) for each epoch.")
    gfp = np.sqrt(np.mean(epochs_clean.get_data() ** 2, axis=(1, 2)))  # Shape: (n_epochs,)
    
    # Step 3: Epoch Statistics
    epoch_stats = pd.DataFrame({
        'epoch': np.arange(len(gfp)),
        'gfp': gfp,
        'mean_amplitude': epochs_clean.get_data().mean(axis=(1, 2)),
        'max_amplitude': epochs_clean.get_data().max(axis=(1, 2)),
        'min_amplitude': epochs_clean.get_data().min(axis=(1, 2)),
        'std_amplitude': epochs_clean.get_data().std(axis=(1, 2))
    })
    
    # Step 4: Remove Outlier Epochs based on GFP
    logger.info("Removing outlier epochs based on GFP z-scores.")
    gfp_mean = epoch_stats['gfp'].mean()
    gfp_std = epoch_stats['gfp'].std()
    z_scores = np.abs((epoch_stats['gfp'] - gfp_mean) / gfp_std)
    good_epochs_mask = z_scores < gfp_threshold
    removed_by_gfp = np.sum(~good_epochs_mask)
    epochs_final = epochs_clean[good_epochs_mask]
    epoch_stats_final = epoch_stats[good_epochs_mask]
    logger.info(f"Outlier epochs removed based on GFP: {removed_by_gfp}")
    
    # Step 5: Randomly Select a Specified Number of Epochs
    if number_of_epochs is not None:
        if len(epochs_final) < number_of_epochs:
            error_msg = (f"Requested number_of_epochs={number_of_epochs} exceeds the available cleaned epochs={len(epochs_final)}.")
            logger.error(error_msg)
            raise ValueError(error_msg)
        if random_seed is not None:
            random.seed(random_seed)
        selected_indices = random.sample(range(len(epochs_final)), number_of_epochs)
        epochs_final = epochs_final[selected_indices]
        epoch_stats_final = epoch_stats_final.iloc[selected_indices]
        logger.info(f"Randomly selected {number_of_epochs} epochs from the cleaned data.")
    
    # Compile Statistics
    stats = {
        'initial_epochs': len(epochs),
        'after_autoreject': len(epochs_clean),
        'removed_by_autoreject': len(epochs) - len(epochs_clean),
        'removed_by_gfp': removed_by_gfp,
        'final_epochs': len(epochs_final),
        'mean_amplitude': epoch_stats_final['mean_amplitude'].mean(),
        'max_amplitude': epoch_stats_final['max_amplitude'].max(),
        'min_amplitude': epoch_stats_final['min_amplitude'].min(),
        'std_amplitude': epoch_stats_final['std_amplitude'].mean(),
        'mean_gfp': epoch_stats_final['gfp'].mean(),
        'gfp_threshold': gfp_threshold,
        'removed_total': (len(epochs) - len(epochs_clean)) + removed_by_gfp
    }
    
    logger.info("Epoch cleaning process completed.")
    logger.info(f"Final number of epochs: {stats['final_epochs']}")
    
    return epochs_final, stats

import logging
import random
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
import mne
from autoreject import AutoReject

def clean_epochs_ranked(
    epochs: mne.Epochs,
    gfp_threshold: float = 3.0,
    number_of_epochs: Optional[int] = None,
    apply_autoreject: bool = False,
    random_seed: Optional[int] = None
) -> Tuple[mne.Epochs, Dict[str, any]]:
    """
    Clean an MNE Epochs object by applying artifact rejection and selecting the top N cleanest epochs based on GFP.

    Args:
        epochs (mne.Epochs): The input epoched EEG data.
        gfp_threshold (float, optional): Z-score threshold for GFP-based outlier detection.
                                         Epochs with GFP z-scores above this value are removed.
                                         Defaults to 3.0.
        number_of_epochs (int, optional): If specified, selects the top N epochs with the lowest GFP.
                                         If None, retains all cleaned epochs. Defaults to None.
        apply_autoreject (bool, optional): Whether to apply AutoReject for artifact correction. Defaults to False.
        random_seed (int, optional): Seed for random number generator to ensure reproducibility when selecting epochs.
                                     Defaults to None.

    Returns:
        Tuple[mne.Epochs, Dict[str, any]]: A tuple containing the cleaned Epochs object and a dictionary of statistics.

    Raises:
        ValueError: If after cleaning, the number of epochs is less than `number_of_epochs` when specified.
    """
    logger = logging.getLogger('clean_epochs_ranked')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # Prevent adding multiple handlers in interactive environments
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    logger.info("Starting epoch cleaning process.")
    
    # Ensure data is preloaded to allow manipulation
    if not epochs.preload:
        epochs.load_data()
    
    # Step 1: Artifact Rejection using AutoReject
    if apply_autoreject:
        logger.info("Applying AutoReject for artifact rejection.")
        ar = AutoReject()
        epochs_clean = ar.fit_transform(epochs)
        logger.info(f"Artifacts rejected: {len(epochs) - len(epochs_clean)} epochs removed by AutoReject.")
    else:
        epochs_clean = epochs.copy()
        logger.info("AutoReject not applied. Proceeding without artifact rejection.")
    
    # Step 2: Calculate Global Field Power (GFP)
    logger.info("Calculating Global Field Power (GFP) for each epoch.")
    gfp = np.sqrt(np.mean(epochs_clean.get_data() ** 2, axis=(1, 2)))  # Shape: (n_epochs,)
    
    # Step 3: Epoch Statistics
    epoch_stats = pd.DataFrame({
        'gfp': gfp,
        'mean_amplitude': epochs_clean.get_data().mean(axis=(1, 2)),
        'max_amplitude': epochs_clean.get_data().max(axis=(1, 2)),
        'min_amplitude': epochs_clean.get_data().min(axis=(1, 2)),
        'std_amplitude': epochs_clean.get_data().std(axis=(1, 2))
    }, index=np.arange(len(gfp)))
    
    # Step 4: Remove Outlier Epochs based on GFP
    logger.info("Removing outlier epochs based on GFP z-scores.")
    gfp_mean = epoch_stats['gfp'].mean()
    gfp_std = epoch_stats['gfp'].std()
    z_scores = np.abs((epoch_stats['gfp'] - gfp_mean) / gfp_std)
    good_epochs_mask = z_scores < gfp_threshold
    removed_by_gfp = np.sum(~good_epochs_mask)
    epochs_cleaned = epochs_clean[good_epochs_mask]
    epoch_stats_cleaned = epoch_stats[good_epochs_mask].reset_index(drop=True)
    logger.info(f"Outlier epochs removed based on GFP: {removed_by_gfp}")
    
    # Step 5: Rank Epochs by GFP (ascending order)
    logger.info("Ranking epochs based on GFP.")
    # Sort the epochs by GFP in ascending order (cleanest first)
    sorted_epoch_stats = epoch_stats_cleaned.sort_values(by='gfp', ascending=True).reset_index(drop=True)
    
    # Step 6: Select Top N Cleanest Epochs
    if number_of_epochs is not None:
        if len(sorted_epoch_stats) < number_of_epochs:
            error_msg = (f"Requested number_of_epochs={number_of_epochs} exceeds the available cleaned epochs={len(sorted_epoch_stats)}.")
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Select the top N indices
        selected_indices = sorted_epoch_stats.index[:number_of_epochs].tolist()
        # Select the corresponding epochs
        selected_epochs = epochs_cleaned[selected_indices]
        # Extract the corresponding statistics
        epoch_stats_final = sorted_epoch_stats.iloc[:number_of_epochs]
        logger.info(f"Selected top {number_of_epochs} cleanest epochs based on GFP.")
    else:
        selected_epochs = epochs_cleaned
        epoch_stats_final = sorted_epoch_stats
        logger.info("No specific number of epochs requested. Retaining all cleaned epochs.")
    
    # Compile Statistics
    stats = {
        'initial_epochs': len(epochs),
        'after_autoreject': len(epochs_clean),
        'removed_by_autoreject': len(epochs) - len(epochs_clean),
        'removed_by_gfp': removed_by_gfp,
        'final_epochs': len(selected_epochs),
        'mean_amplitude': epoch_stats_final['mean_amplitude'].mean(),
        'max_amplitude': epoch_stats_final['max_amplitude'].max(),
        'min_amplitude': epoch_stats_final['min_amplitude'].min(),
        'std_amplitude': epoch_stats_final['std_amplitude'].mean(),
        'mean_gfp': epoch_stats_final['gfp'].mean(),
        'gfp_threshold': gfp_threshold,
        'removed_total': (len(epochs) - len(epochs_clean)) + removed_by_gfp
    }
    
    logger.info("Epoch cleaning process completed.")
    logger.info(f"Final number of epochs: {stats['final_epochs']}")
    
    return selected_epochs, stats



def pre_pipeline_processing(raw, autoclean_dict):
    console.print("\n[bold]Pre-pipeline Processing Steps[/bold]")

    task                           = autoclean_dict["task"]
    
    # Get enabled/disabled status for each step
    apply_resample_toggle                   = autoclean_dict["tasks"][task]["settings"]["resample_step"]["enabled"]
    apply_eog_toggle                        = autoclean_dict["tasks"][task]["settings"]["eog_step"]["enabled"]
    apply_average_reference_toggle          = autoclean_dict["tasks"][task]["settings"]["reference_step"]["enabled"]
    apply_trim_toggle                       = autoclean_dict["tasks"][task]["settings"]["trim_step"]["enabled"]
    apply_crop_toggle                       = autoclean_dict["tasks"][task]["settings"]["crop_step"]["enabled"]    
    apply_filter_toggle                     = autoclean_dict["tasks"][task]["settings"]["filter_step"]["enabled"]

    # Print status of each step
    console.print(f"{'✓' if apply_resample_toggle else '✗'} Resample: [{'green' if apply_resample_toggle else 'red'}]{apply_resample_toggle}[/]")
    console.print(f"{'✓' if apply_eog_toggle else '✗'} EOG Assignment: [{'green' if apply_eog_toggle else 'red'}]{apply_eog_toggle}[/]")
    console.print(f"{'✓' if apply_average_reference_toggle else '✗'} Average Reference: [{'green' if apply_average_reference_toggle else 'red'}]{apply_average_reference_toggle}[/]")
    console.print(f"{'✓' if apply_trim_toggle else '✗'} Edge Trimming: [{'green' if apply_trim_toggle else 'red'}]{apply_trim_toggle}[/]")
    console.print(f"{'✓' if apply_crop_toggle else '✗'} Duration Cropping: [{'green' if apply_crop_toggle else 'red'}]{apply_crop_toggle}[/]")
    console.print(f"{'✓' if apply_filter_toggle else '✗'} Filtering: [{'green' if apply_filter_toggle else 'red'}]{apply_filter_toggle}[/]\n")

    # Initialize metadata
    metadata = {
        "pre_pipeline_processing": {
            "creationDateTime": datetime.now().isoformat(),
            "ResampleHz": None,
            "TrimSec": None,
            "LowPassHz1": None, 
            "HighPassHz1": None,
            "CropDurationSec": None,
            "AverageReference": apply_average_reference_toggle,
            "EOGChannels": None
        }
    }

    # Resample
    if apply_resample_toggle:
        console.print("[cyan]Resampling data...[/cyan]")
        target_sfreq = autoclean_dict["tasks"][task]["settings"]["resample_step"]["value"]
        raw = step_resample_data(raw, target_sfreq)
        console.print(f"[green]✓ Data resampled to {target_sfreq} Hz[/green]")
        metadata["pre_pipeline_processing"]["ResampleHz"] = target_sfreq
        save_raw_to_set(raw, autoclean_dict, 'post_resample')
    
    # EOG Assignment
    if apply_eog_toggle:
        console.print("[cyan]Setting EOG channels...[/cyan]")
        eog_channels = autoclean_dict["tasks"][task]["settings"]["eog_step"]["value"]
        raw = step_mark_eog_channels(raw, eog_channels)
        console.print("[green]✓ EOG channels assigned[/green]")
        metadata["pre_pipeline_processing"]["EOGChannels"] = eog_channels
    
    # Average Reference
    if apply_average_reference_toggle:
        console.print("[cyan]Applying average reference...[/cyan]")
        ref_type = autoclean_dict["tasks"][task]["settings"]["reference_step"]["value"]
        raw = step_set_reference(raw, ref_type)
        console.print("[green]✓ Average reference applied[/green]")
        save_raw_to_set(raw, autoclean_dict, 'post_reference')
    
    # Trim Edges
    if apply_trim_toggle:
        console.print("[cyan]Trimming data edges...[/cyan]")
        trim = autoclean_dict["tasks"][task]["settings"]["trim_step"]["value"]
        start_time = raw.times[0]
        end_time = raw.times[-1]
        raw.crop(tmin=start_time + trim, tmax=end_time - trim)   
        console.print(f"[green]✓ Data trimmed by {trim}s from each end[/green]")
        metadata["pre_pipeline_processing"]["TrimSec"] = trim
        save_raw_to_set(raw, autoclean_dict, 'post_trim')
    # Crop Duration
    if apply_crop_toggle:
        console.print("[cyan]Cropping data duration...[/cyan]")
        start_time = autoclean_dict["tasks"][task]["settings"]["crop_step"]["value"]['start']
        end_time = autoclean_dict["tasks"][task]["settings"]["crop_step"]["value"]['end']
        if end_time is None:
            end_time = raw.times[-1]  # Use full duration if end is null
        raw.crop(tmin=start_time, tmax=end_time)
        target_crop_duration = raw.times[-1] - raw.times[0]
        console.print(f"[green]✓ Data cropped to {target_crop_duration:.1f}s[/green]")
        metadata["pre_pipeline_processing"]["CropDurationSec"] = target_crop_duration
        metadata["pre_pipeline_processing"]["CropStartSec"] = start_time
        metadata["pre_pipeline_processing"]["CropEndSec"] = end_time
        save_raw_to_set(raw, autoclean_dict, 'post_crop')

    # Pre-Filter    
    if apply_filter_toggle:
        console.print("[cyan]Applying frequency filters...[/cyan]")
        target_lfreq = float(autoclean_dict["tasks"][task]["settings"]["filter_step"]["value"]["l_freq"])
        target_hfreq = float(autoclean_dict["tasks"][task]["settings"]["filter_step"]["value"]["h_freq"])
        raw.filter(l_freq=target_lfreq, h_freq=target_hfreq)
        console.print(f"[green]✓ Applied bandpass filter: {target_lfreq}-{target_hfreq} Hz[/green]")
        metadata["pre_pipeline_processing"]["LowPassHz1"] = target_lfreq
        metadata["pre_pipeline_processing"]["HighPassHz1"] = target_hfreq
        save_raw_to_set(raw, autoclean_dict, 'post_filter')
    
    # Save metadata
    step_handle_metadata(autoclean_dict, metadata, mode='save')
    return raw

def create_bids_path(raw, autoclean_dict):

    unprocessed_file        = autoclean_dict["unprocessed_file"]
    task                    = autoclean_dict["task"]
    mne_task                = autoclean_dict["tasks"][task]["mne_task"]
    bids_dir                = autoclean_dict["bids_dir"]
    eeg_system              = autoclean_dict["eeg_system"]
    config_file             = autoclean_dict["config_file"]

    try:
        bids_path = step_convert_to_bids(
            raw,
            output_dir=str(bids_dir),
            task=mne_task,
            participant_id=None,
            line_freq=60.0,
            overwrite=True,
            study_name=unprocessed_file.stem
        )

        autoclean_dict["bids_path"] = bids_path
        autoclean_dict["bids_basename"] = bids_path.basename

        metadata = {
            "step_convert_to_bids": {
                "creationDateTime": datetime.now().isoformat(),
                "bids_output_dir": str(bids_dir),
                "bids_path": str(bids_path),
                "bids_basename": bids_path.basename,
                "study_name": unprocessed_file.stem,
                "task": mne_task,
                "participant_id": None,
                "line_freq": 60.0,
                "eegSystem": eeg_system,
                "configFile": str(config_file)
            }
        }

        step_handle_metadata(autoclean_dict, metadata, mode='save')

        return raw, autoclean_dict
    
    except Exception as e:
        console.print(f"[red]Error converting raw to bids: {e}[/red]")
        raise e 

def save_raw_to_set(raw, autoclean_dict, stage="post_import", output_path=None):
    """Save raw EEG data to SET format with descriptive filename.
    
    Args:
        raw: MNE Raw object containing EEG data
        output_path: Path object specifying output directory
        autoclean_dict: Dictionary containing configuration and paths
        stage: Processing stage to get suffix from stage_files config
        
    Returns:
        Path: Path to the saved SET file
    """
    #Only save if enabled for this stage in stage_files config
    if not autoclean_dict['stage_files'][stage]['enabled']:
        return None
        
    # Get suffix from stage_files config
    suffix = autoclean_dict['stage_files'][stage]['suffix']

    # Create subfolder using suffix name
    if output_path is None:
        output_path = autoclean_dict["stage_dir"]

    subfolder = output_path / suffix
    subfolder.mkdir(exist_ok=True)

    basename = Path(autoclean_dict["unprocessed_file"]).stem
    set_path = subfolder / f"{basename}{suffix}_raw.set"
    
    raw.export(set_path, fmt='eeglab', overwrite=True)
    console.print(f"[green]Saved stage file for {stage} to: {basename}[/green]")

    metadata = {
        "save_raw_to_set": {
            "creationDateTime": datetime.now().isoformat(),
            "stage": stage,
            "outputPath": str(set_path),
            "suffix": suffix,
            "basename": basename,
            "format": "eeglab"
        }
    }
    step_handle_metadata(autoclean_dict, metadata, mode='save')

    return set_path

def save_epochs_to_set(epochs, autoclean_dict, stage="post_import", output_path=None):
    """Save epoched EEG data to SET format with descriptive filename.
    
    Args:
        epochs: MNE Epochs object containing EEG data
        output_path: Path object specifying output directory
        autoclean_dict: Dictionary containing configuration and paths
        stage: Processing stage to get suffix from stage_files config
        
    Returns:
        Path: Path to the saved SET file
    """
    #Only save if enabled for this stage in stage_files config
    if not autoclean_dict['stage_files'][stage]['enabled']:
        return None
        
    # Get suffix from stage_files config
    suffix = autoclean_dict['stage_files'][stage]['suffix']

    # Create subfolder using suffix name
    if output_path is None:
        output_path = autoclean_dict["stage_dir"]

    subfolder = output_path / suffix
    subfolder.mkdir(exist_ok=True)

    basename = Path(autoclean_dict["unprocessed_file"]).stem
    set_path = subfolder / f"{basename}{suffix}_epo.set"
    
    epochs.export(set_path, fmt='eeglab', overwrite=True)
    console.print(f"[green]Saved stage file for {stage} to: {basename}[/green]")

    metadata = {
        "save_raw_to_set": {
            "creationDateTime": datetime.now().isoformat(),
            "stage": stage,
            "outputPath": str(set_path),
            "suffix": suffix,
            "basename": basename,
            "format": "eeglab",
            "n_epochs": len(epochs),
            "tmin": epochs.tmin,
            "tmax": epochs.tmax
        }
    }
    step_handle_metadata(autoclean_dict, metadata, mode='save')

    return set_path

def plot_bad_channels_full_duration(raw_original, raw_cleaned, pipeline, autoclean_dict):
    """
    Plot only the bad channels over the full duration, overlaying the original
    and interpolated data. Original data is plotted in red, interpolated data in black.

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
    suffix : str
        Suffix for the filename.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Collect bad channels and their reasons
    bad_channels_info = {}

    # Mapping from channel to reason(s)
    for reason, channels in pipeline.flags['ch'].items():
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

    # Get data for the bad channels from both original and cleaned data
    picks_original = mne.pick_channels(raw_original.ch_names, bad_channels)
    picks_cleaned = mne.pick_channels(raw_cleaned.ch_names, bad_channels)

    data_original, times = raw_original.get_data(picks=picks_original, return_times=True)
    data_cleaned = raw_cleaned.get_data(picks=picks_cleaned)

    channel_labels = [raw_original.ch_names[i] for i in picks_original]
    n_channels = len(channel_labels)

    # Increase downsample factor to reduce file size
    sfreq = raw_original.info['sfreq']
    desired_sfreq = 100  # Reduced sampling rate to 100 Hz
    downsample_factor = int(sfreq // desired_sfreq)
    if downsample_factor > 1:
        data_original = data_original[:, ::downsample_factor]
        data_cleaned = data_cleaned[:, ::downsample_factor]
        times = times[::downsample_factor]

    # Normalize each channel individually for better visibility
    data_original_normalized = np.zeros_like(data_original)
    data_cleaned_normalized = np.zeros_like(data_cleaned)
    spacing = 10  # Fixed spacing between channels
    for idx in range(n_channels):
        channel_data_original = data_original[idx]
        channel_data_cleaned = data_cleaned[idx]
        # Remove DC offset
        channel_data_original = channel_data_original - np.mean(channel_data_original)
        channel_data_cleaned = channel_data_cleaned - np.mean(channel_data_cleaned)
        # Use standard deviation of original data for normalization
        std = np.std(channel_data_original)
        if std == 0:
            std = 1  # Avoid division by zero
        data_original_normalized[idx] = channel_data_original / std
        data_cleaned_normalized[idx] = channel_data_cleaned / std

    # Multiply by a scaling factor to control amplitude
    scaling_factor = 2  # Adjust this factor as needed for visibility
    data_original_scaled = data_original_normalized * scaling_factor
    data_cleaned_scaled = data_cleaned_normalized * scaling_factor

    # Calculate offsets for plotting
    offsets = np.arange(n_channels) * spacing

    # Create plot
    total_duration = times[-1] - times[0]
    width_per_second = 0.1  # Adjusted for better scaling
    fig_width = min(total_duration * width_per_second, 50)
    fig_height = max(6, n_channels * 0.5)  # Adjusted for better spacing

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Plot bad channels
    for idx in range(n_channels):
        ch_name = channel_labels[idx]
        offset = offsets[idx]
        reasons = bad_channels_info.get(ch_name, [])
        label = f"{ch_name} ({', '.join(reasons)})"

        # Plot the original data in red
        ax.plot(times, data_original_scaled[idx] + offset, color='red', linewidth=0.5, linestyle='-')

        # Plot the cleaned (interpolated) data in black
        ax.plot(times, data_cleaned_scaled[idx] + offset, color='black', linewidth=0.5, linestyle='-')

        # Add channel label
        ax.text(times[0] - (0.01 * total_duration), offset, label, horizontalalignment='right', fontsize=8)

    # Customize axes
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_title('Bad Channels: Original vs Interpolated (Full Duration)', fontsize=14)
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-spacing, offsets[-1] + spacing)
    ax.set_yticks([])  # Hide y-ticks as we have labels next to each channel
    ax.invert_yaxis()

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=0.5, linestyle='-', label='Original Data'),
        Line2D([0], [0], color='black', lw=0.5, linestyle='-', label='Interpolated Data')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # Enhance the layout
    plt.tight_layout()

    # Get output path for bad channels figure
    derivatives_path = pipeline.get_derivative_path(autoclean_dict["bids_path"])
    target_figure = str(derivatives_path.copy().update(
        suffix='bad_channels',
        extension='.png',
        datatype='eeg'
    ))

    # Save as PNG with high DPI for quality
    fig.savefig(target_figure, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Bad channels full duration plot saved to {target_figure}")

    return fig

import os
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, coherence
from datetime import datetime
import tempfile

def eeg_htpEegAssessPipelineHAPPE(EEG1, EEG2, outputdir=None, resample_rate=500,
                                  group_labels=None, table_only=False,
                                  save_output=True, outputfile=None):
    """
    Description: Adaptation of HAPPE pipeline with visual quality assurance.
    Compares two EEG datasets and provides quality metrics and visualizations.
    
    Parameters:
    - EEG1: MNE Raw or Epochs object for the first EEG dataset.
    - EEG2: MNE Raw or Epochs object for the second EEG dataset.
    - outputdir: Directory to save output files. Defaults to system temp directory.
    - resample_rate: Rate to resample the EEG data for analysis. Defaults to 500 Hz.
    - group_labels: Labels for the two EEG datasets. Defaults to ['EEG1', 'EEG2'].
    - table_only: If True, only generate the summary table without visualizations.
    - save_output: If True, save output files such as images and tables.
    - outputfile: Name of the output file to save the summary table.
    
    Returns:
    - summary_table: A pandas DataFrame containing summary metrics.
    """
    # Set default output directory
    if outputdir is None:
        outputdir = tempfile.gettempdir()
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)
        
    # Set default group labels
    if group_labels is None:
        group_labels = ['EEG1', 'EEG2']
    
    # Create timestamp and function stamp
    timestamp = datetime.now().strftime('%y%m%d%H%M%S')
    functionstamp = 'eeg_htpEegAssessPipelineHAPPE'
    def msglog(x):
        print(f'{functionstamp}: {x}')
        
    # Ensure common sampling rate and resample if necessary
    srate1 = EEG1.info['sfreq']
    srate2 = EEG2.info['sfreq']
    if srate1 != srate2:
        raise ValueError('EEGs have different sampling rates. Use resampling to match rates.')
    else:
        srate = srate1

    if srate >= resample_rate:
        EEG1 = EEG1.copy().resample(resample_rate)
        EEG2 = EEG2.copy().resample(resample_rate)
        srate = resample_rate

    # Function to get continuous data from Raw or Epochs objects
    def get_data_continuous(EEG):
        if isinstance(EEG, mne.io.BaseRaw):
            data = EEG.get_data()  # shape (n_channels, n_times)
        elif isinstance(EEG, mne.BaseEpochs):
            data = EEG.get_data()  # shape (n_epochs, n_channels, n_times)
            n_epochs, n_channels, n_times = data.shape
            data = data.transpose(1,0,2).reshape(n_channels, -1)  # reshape to (n_channels, n_epochs * n_times)
        else:
            raise ValueError('EEG must be Raw or Epochs object')
        return data

    # Extract data from EEG1 and EEG2
    data1 = get_data_continuous(EEG1)
    data2 = get_data_continuous(EEG2)

    # Check that data1 and data2 have the same shape
    if data1.shape != data2.shape:
        raise ValueError('EEG data structures have different dimensions.')

    # Compute correlation coefficients per channel
    cr = np.array([np.corrcoef(data1[ch, :], data2[ch, :])[0,1] for ch in range(data1.shape[0])])

    # Compute coherence per channel and frequency
    fvec = np.arange(0, 81)
    cf = np.zeros((data1.shape[0], len(fvec)))
    for ch in range(data1.shape[0]):
        f_coh, Cxy = coherence(data1[ch,:], data2[ch,:], fs=srate, nperseg=1000, noverlap=0)
        # Interpolate Cxy to match fvec
        Cxy_interp = np.interp(fvec, f_coh, Cxy)
        cf[ch,:] = Cxy_interp

    # Compute other QA measures
    order = 2  # EEG2 - EEG1
    if order == 1:
        orderMod = data1
    else:
        orderMod = data2

    squared_differences = (data1 - data2)**2
    epsilon = 1e-10  # Small constant to prevent division by zero
    DEN = np.sum(squared_differences, axis=1) + epsilon
    MSE = np.mean(squared_differences, axis=1) + epsilon

    NUM = np.sum(orderMod**2, axis=1)

    SNR = np.mean(20 * np.log10(np.sqrt(NUM) / np.sqrt(DEN)))
    PeakSNR = np.mean(20 * np.log10(np.max(orderMod, axis=1) / np.sqrt(MSE)))
    RMSE = np.mean(np.sqrt(np.mean((data1 - data2)**2, axis=1)))
    MAE = np.mean(np.mean(np.abs(data1 - data2), axis=1))

    # Compute power spectra
    fvec_pwr = np.arange(0, 80.25, 0.25)
    pwr1_ch = []
    pwr2_ch = []
    for ch in range(data1.shape[0]):
        f_pwr1, Pxx1 = welch(data1[ch,:], fs=srate, nperseg=int(srate*2), noverlap=0)
        f_pwr2, Pxx2 = welch(data2[ch,:], fs=srate, nperseg=int(srate*2), noverlap=0)
        # Interpolate to match fvec_pwr
        Pxx1_interp = np.interp(fvec_pwr, f_pwr1, Pxx1)
        Pxx2_interp = np.interp(fvec_pwr, f_pwr2, Pxx2)
        pwr1_ch.append(Pxx1_interp)
        pwr2_ch.append(Pxx2_interp)
    pwr1_ch = np.array(pwr1_ch)  # shape (n_channels, len(fvec_pwr))
    pwr2_ch = np.array(pwr2_ch)
    pwr1 = np.mean(pwr1_ch, axis=0)  # Average over channels
    pwr2 = np.mean(pwr2_ch, axis=0)
    pwrdiff = 100 * (pwr2 / pwr1)
    # Eliminate notch differences for plot scaling (55-65 Hz)
    notch_indices = np.where((fvec_pwr > 55) & (fvec_pwr < 65))
    pwrdiff[notch_indices] = 100

    # Prepare amplitude correlation structure
    ampStruct = {
        'label': f'CorrCoef\nRange: {np.min(cr):.2f}-{np.max(cr):.2f}',
        'meancf': cr,
        'bandname': 'Amplitude'
    }

    # Define frequency bands
    BandDefs = [
        ('Delta', 2, 3.5),
        ('Theta', 3.5, 7.5),
        ('Alpha1', 7.5, 10.5),
        ('Alpha2', 10.5, 12.5),
        ('Beta', 13, 30),
        ('Gamma1', 30, 55),
        ('Gamma2', 65, 80),
        # 'Epsilon' band is beyond our frequency vector range
    ]

    # Compute mean coherence in each band
    coefStruct = [ampStruct]
    for band in BandDefs:
        bandname, f1, f2 = band
        if f2 <= fvec[-1]:
            band_indices = np.where((fvec >= f1) & (fvec <= f2))[0]
            cftmp = np.mean(cf[:, band_indices], axis=1)
            rangetmp = f'Range: {np.min(cftmp):.2f}-{np.max(cftmp):.2f}'
            labeltmp = f'{bandname} ({f1}-{f2} Hz)\n{rangetmp}'
            coefStruct.append({
                'meancf': cftmp,
                'label': labeltmp,
                'bandname': bandname
            })

    # Visualization
    if not table_only:
        n_plots = len(coefStruct)
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()

        # Adjust the figure layout to make space for colorbar and header
        fig.subplots_adjust(left=0.05, right=0.85, top=0.85, bottom=0.05, hspace=0.4, wspace=0.3)

        # Plot topomaps without 'vmin' and 'vmax'
        for idx, coef in enumerate(coefStruct):
            ax = axes[idx]
            im, cn = mne.viz.plot_topomap(
                coef['meancf'],
                EEG1.info,
                axes=ax,
                show=False,
                cmap='bone',
                sensors=True,
                names=None
            )
            ax.set_title(coef['label'], fontsize=12)
        
        # Hide unused axes if any
        for idx in range(len(coefStruct), len(axes)):
            axes[idx].axis('off')

        # Add colorbar
        cbar_ax = fig.add_axes([0.86, 0.3, 0.02, 0.4])  # Adjust position to avoid overlapping
        sm = plt.cm.ScalarMappable(cmap='bone')
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label='Corr. Coef.')

        # Histogram of correlation coefficients
        allCoefs = np.hstack([coef['meancf'] for coef in coefStruct])
        ax_hist = axes[8]
        ax_hist.hist([coef['meancf'] for coef in coefStruct], bins=10, label=[coef['bandname'] for coef in coefStruct])
        ax_hist.set_xlabel('Correlation Coefficient', fontsize=12)
        ax_hist.set_ylabel('Count', fontsize=12)
        ax_hist.set_title('Histogram of Amplitude/Frequency Correlation Coefficients', fontsize=14)
        ax_hist.legend(fontsize=10)

        # Check if axes[9] is available
        if len(axes) > 9:
            axes[9].axis('off')  # Hide unused axis

        # Spectrogram comparison
        ax_spec = axes[10]
        ax_spec.plot(fvec_pwr, pwr1, 'k', label=group_labels[0])
        ax_spec.plot(fvec_pwr, pwr2, 'r', label=group_labels[1])
        ax_spec.set_yscale('log')
        ax_spec.set_xlabel('Frequency (Hz)')
        ax_spec.set_ylabel('PSD')
        ax_spec.set_title('Spectrogram Comparison')
        ax_spec.legend()

        # Plot percentage difference
        ax_spec_right = ax_spec.twinx()
        ax_spec_right.plot(fvec_pwr, pwrdiff, ':b', label='EEG2/EEG1 (%)')
        ax_spec_right.set_ylabel('EEG2/EEG1 (%)', color='b')
        ax_spec_right.tick_params(axis='y', labelcolor='b')

        # Difference histogram
        ax_diff = axes[11]
        ax_diff.hist(pwrdiff, bins=25)
        ax_diff.set_xlabel('EEG2/EEG1 (%)')
        ax_diff.set_ylabel('Count of EEG1-EEG2 Differences')
        ax_diff.set_title('Percent EEG2/EEG1 PSD\n(omitted 55-65 Hz)')

        # Add header strip with white text on black background
        fig.subplots_adjust(top=0.85)  # Adjust top to make space for header
        header_text = f"{functionstamp}: Channel Cross-Correlation (Run Date: {timestamp})\nEEG1 = {group_labels[0]}   EEG2 = {group_labels[1]}"
        fig.suptitle(header_text, fontsize=14, color='white', backgroundcolor='black')

        # Save the figure if required
        image_filename = os.path.join(outputdir, f'{functionstamp}_{timestamp}_{group_labels[0]}_{group_labels[1]}.png')
        if save_output:
            plt.savefig(image_filename, bbox_inches='tight')
        else:
            image_filename = 'Not saved'
        plt.close(fig)
    else:
        image_filename = 'Table only mode, no image generated.'

    # Identify channels/frequencies with low correlation
    trouble_channels = []
    for coef in coefStruct:
        low_corr_indices = np.where(coef['meancf'] < 0.95)[0]
        for idx in low_corr_indices:
            trouble_channels.append({
                'setname': EEG1.info.get('filename', 'EEG1'),
                'bandname': coef['bandname'],
                'channel': EEG1.ch_names[idx],
                'value': coef['meancf'][idx],
                'label': f"{coef['bandname']} (Chan: {EEG1.ch_names[idx]})"
            })
    trouble_channel_table = pd.DataFrame(trouble_channels)

    # Prepare summary measures
    summary_data = {
        'timestamp': timestamp,
        'EEG1': group_labels[0],
        'EEG2': group_labels[1],
        'ImageFilename': image_filename,
        'SNR': SNR,
        'PeakSNR': PeakSNR,
        'RMSE': RMSE,
        'MAE': MAE
    }
    for coef in coefStruct:
        summary_data[coef['bandname']] = np.mean(coef['meancf'])
    summary_table = pd.DataFrame([summary_data])

    # Display summary
    print(f"{functionstamp}: Quality Assurance Summary")
    print(summary_table)
    print(f"{functionstamp}: Channels below quality threshold (0.95)")
    print(trouble_channel_table)
    print(f"{functionstamp}: Visualization: {image_filename}")

    # Save summary table if required
    if save_output:
        if outputfile is None:
            outputfile = f'{functionstamp}_{timestamp}.csv'
        csvfile = os.path.join(outputdir, outputfile)
        if os.path.isfile(csvfile):
            summary_table.to_csv(csvfile, mode='a', header=False, index=False)
        else:
            summary_table.to_csv(csvfile, index=False)

    return summary_table


import matplotlib
#matplotlib.use('macosx')  # Set non-interactive backend for file output
import matplotlib.style as mplstyle
matplotlib.use('agg')  # Fastest non-interactive backend
mplstyle.use('fast')  # Enable fast style for better performance

import matplotlib
# Switch to mplcairo's PDF backend for faster, higher-quality PDF rendering
matplotlib.use('module://mplcairo.base')

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import mne

def plot_epochs_to_pdf(epochs: mne.Epochs, pipeline, autoclean_dict: dict, epochs_per_page: int = 6, spacing: float = 50, bad_epochs: list = None) -> None:
    data_array = epochs.get_data() * 1e6
    times = epochs.times
    n_epochs = len(epochs)
    n_channels = len(epochs.ch_names)
    offsets = np.arange(n_channels) * spacing
    pages = int(np.ceil(n_epochs / epochs_per_page))

    fig, axes = plt.subplots(2, 3, figsize=(15, 11))
    axes = axes.flatten()


    # Create output path for the PDF report
    derivatives_path = pipeline.get_derivative_path(autoclean_dict["bids_path"])
    output_file = str(
        derivatives_path.copy().update(suffix='epoch_plots', extension='.pdf')
    )


    with PdfPages(output_file) as pdf:
        for p in range(pages):
            start_idx = p * epochs_per_page
            end_idx = min((p + 1) * epochs_per_page, n_epochs)

            for ax in axes:
                ax.cla()
                ax.set_visible(True)

            page_data = data_array[start_idx:end_idx]
            for i, ep_data in enumerate(page_data):
                ax = axes[i]
                epoch_idx = start_idx + i
                color = 'red' if bad_epochs is not None and epoch_idx in bad_epochs else 'black'
                ax.plot(times[:, None], (ep_data.T + offsets), color=color, linewidth=0.5)
                if i % 3 == 0:
                    ax.set_yticks(offsets)
                    ax.set_yticklabels([f'{ch}' for ch in epochs.ch_names], fontsize=6)
                else:
                    ax.set_yticklabels([])
                ax.set_title(f'Epoch {epoch_idx + 1}', fontsize=10, color=color)
                ax.set_xlim(times[0], times[-1])
                ax.set_ylim(-spacing, offsets[-1] + np.ptp(ep_data) + spacing)
                ax.invert_yaxis()
                ax.grid(True, alpha=0.2)
                if i >= 3:
                    ax.set_xlabel('Time (s)', fontsize=8)

            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)

            plt.subplots_adjust(hspace=0.4, wspace=0.2)
            pdf.savefig(fig, dpi=150)

    print(f"Epoch plots saved to {output_file}")


def detect_muscle_beta_focus_robust(epochs, freq_band=(20, 30), scale_factor=3.0):
    """
    Detect muscle artifacts using a robust measure (median + MAD * scale_factor) 
    focusing only on electrodes labeled as 'OTHER'.
    This reduces forced removal of epochs in very clean data.
    """

    # Ensure data is loaded
    epochs.load_data()
    
    # Filter in beta band
    epochs_beta = epochs.copy().filter(l_freq=freq_band[0], h_freq=freq_band[1], verbose=False)

    # Get channel names
    ch_names = epochs_beta.ch_names
    
    # Build channel_region_map from the provided channel data
    # Make sure all "OTHER" electrodes are listed here
    channel_region_map = {
        "E17":"OTHER", "E38":"OTHER", "E43":"OTHER", "E44":"OTHER", "E48":"OTHER", "E49":"OTHER",
        "E56":"OTHER", "E73":"OTHER", "E81":"OTHER", "E88":"OTHER", "E94":"OTHER", "E107":"OTHER",
        "E113":"OTHER", "E114":"OTHER", "E119":"OTHER", "E120":"OTHER", "E121":"OTHER", "E125":"OTHER",
        "E126":"OTHER", "E127":"OTHER", "E128":"OTHER"
    }
    
    # Select only OTHER channels
    selected_ch_indices = [i for i, ch in enumerate(ch_names) if channel_region_map.get(ch, "") == "OTHER"]
    
    # If no OTHER channels are found, return empty
    if not selected_ch_indices:
        return np.array([], dtype=int)
    
    # Extract data from OTHER channels only
    data = epochs_beta.get_data()[:, selected_ch_indices, :]  # shape: (n_epochs, n_sel_channels, n_times)

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
    bad_epochs = np.where(max_p2p > threshold)[0]

    return bad_epochs

def detect_muscle_beta_focus(epochs, freq_band=(20, 30), percentile_threshold=75):
    """
    Detect muscle artifacts using beta band power from electrodes labeled as 'OTHER'.
    This assumes you have a channel_region_map that associates each channel name with its region.
    """

    # Ensure data is loaded
    epochs.load_data()
    
    # Filter in beta band (in-place filtering if allowed)
    epochs_beta = epochs.copy().filter(l_freq=freq_band[0], h_freq=freq_band[1])

    # Get channel names
    ch_names = epochs_beta.ch_names
    
    # Build channel_region_map from the provided channel data
    channel_region_map = {
        "E17":"OTHER", "E38":"OTHER", "E43":"OTHER", "E44":"OTHER", "E48":"OTHER", "E49":"OTHER",
        "E56":"OTHER", "E73":"OTHER", "E81":"OTHER", "E88":"OTHER", "E94":"OTHER", "E107":"OTHER",
        "E113":"OTHER", "E114":"OTHER", "E119":"OTHER", "E120":"OTHER", "E121":"OTHER", "E125":"OTHER",
        "E126":"OTHER", "E127":"OTHER", "E128":"OTHER"
    }
    
    # Select only OTHER channels
    selected_ch_indices = [i for i, ch in enumerate(ch_names) if channel_region_map.get(ch, "") == "OTHER"]
    
    # If no OTHER channels are found, return empty
    if not selected_ch_indices:
        return np.array([], dtype=int)
    
    # Extract data from OTHER channels only
    data = epochs_beta.get_data()[:, selected_ch_indices, :]  # shape: (n_epochs, n_sel_channels, n_times)

    # Compute peak-to-peak amplitude per epoch and selected channels
    p2p = data.max(axis=2) - data.min(axis=2)

    # Compute maximum peak-to-peak amplitude across the selected channels
    max_p2p = p2p.max(axis=1)

    # Determine threshold
    threshold = np.percentile(max_p2p, percentile_threshold)

    # Identify bad epochs and get additional info
    bad_epochs = np.where(max_p2p > threshold)[0]
    
    # Create metadata dictionary for each bad epoch
    metadata = []
    for bad_idx in bad_epochs:
        # Get the channel with maximum p2p for this epoch
        max_ch_idx = np.argmax(p2p[bad_idx])
        max_ch_name = [ch_names[i] for i in selected_ch_indices][max_ch_idx]
        
        # Calculate how much it exceeds threshold
        excess = float(max_p2p[bad_idx] - threshold)
        percent_above = float((max_p2p[bad_idx] / threshold - 1) * 100)
        
        epoch_info = {
            "epoch_index": int(bad_idx),
            "max_p2p_amplitude": float(max_p2p[bad_idx]),
            "threshold": float(threshold),
            "excess_amount": excess,
            "percent_above_threshold": percent_above,
            "worst_channel": max_ch_name
        }
        metadata.append(epoch_info)
        
        # Still print info for debugging/logging
        print(json.dumps(epoch_info, indent=2))

    return bad_epochs, metadata


def clean_artifacts_continuous(pipeline, autoclean_dict):

    bids_path = autoclean_dict["bids_path"]
    
    # Apply rejection policy
    rejection_policy = step_get_rejection_policy(autoclean_dict)
    cleaned_raw = rejection_policy.apply(pipeline)

    # Save cleaned raw data
    derivatives_path            = pipeline.get_derivative_path(bids_path)
    derivatives_path.suffix     = "eeg"
    pipeline.save(derivatives_path, overwrite=True, format="BrainVision")

    # Save cleaned raw data to set
    save_raw_to_set(cleaned_raw, autoclean_dict, 'post_rejection_policy')

    # # Plot bad channels separately
    plot_bad_channels_full_duration(pipeline.raw, cleaned_raw, pipeline, autoclean_dict)

    # # Plot topoplot for bands
    step_plot_band_topos(cleaned_raw, pipeline,autoclean_dict)

    # # Generate ICA reports
    generate_ica_reports(pipeline, cleaned_raw, autoclean_dict, duration=60)

    # # Plot ICA components full duration
    step_plot_ica_full(pipeline, autoclean_dict)

    # # Call the function to plot and save the overlay
    step_plot_overlay(
        pipeline.raw,
        cleaned_raw,
        pipeline,
        autoclean_dict,
        suffix='overlay'
    )

    step_plot_psd_overlay(pipeline.raw, cleaned_raw, pipeline, autoclean_dict, suffix='spectrogram')

    #breakpoint()
    eeg_htpEegAssessPipelineHAPPE(EEG1=pipeline.raw, EEG2=cleaned_raw)

    epochs = mne.make_fixed_length_epochs(cleaned_raw, duration=2)

    # epochs = epochs[:30]  # Select first 30 epochs
    # bad_epochs = detect_muscle_beta_focus(epochs, freq_band=(20, 30), percentile_threshold=50)
    bad_epochs = detect_muscle_beta_focus_robust(epochs, freq_band=(20, 30), scale_factor=3.0)
    print(f"Detected {len(bad_epochs)} bad epochs")
    
    # Plot epochs to PDF
    import time
    start_time = time.time()
    plot_epochs_to_pdf(epochs, pipeline=pipeline, autoclean_dict=autoclean_dict, bad_epochs=bad_epochs)
    end_time = time.time()
    print(f"Plotting epochs took {end_time - start_time:.2f} seconds")

    # Remove bad epochs detected from muscle artifacts
    if len(bad_epochs) > 0:
        print(f"Removing {len(bad_epochs)} epochs with muscle artifacts...")
        epochs.drop(bad_epochs, reason='muscle')
        print(f"Remaining epochs: {len(epochs)}")
    
    # Store cleaned epochs for later use
    cleaned_epochs = epochs



    #cleaned_epochs, stats = clean_epochs(epochs, number_of_epochs=80, gfp_threshold=3)
    #cleaned_epochs.load_data()
    save_epochs_to_set(cleaned_epochs, autoclean_dict, 'post_clean_epochs')

    report_artifact_rejection(pipeline, cleaned_raw, autoclean_dict)

    return pipeline, autoclean_dict
  
def plot_ica_components(pipeline, cleaned_raw, autoclean_dict, duration=60, components='all'):
    """
    Plots ICA components with labels and saves reports.

    Parameters:
    -----------
    pipeline : pylossless.Pipeline
        Pipeline object containing raw data and ICA.
    autoclean_dict : dict
        Autoclean dictionary containing metadata.
    duration : int
        Duration in seconds to plot.
    components : str
        'all' to plot all components, 'rejected' to plot only rejected components.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.gridspec import GridSpec

    # Get raw and ICA from pipeline
    raw = pipeline.raw
    ica = pipeline.ica2
    ic_labels = pipeline.flags['ic']

    # Determine components to plot
    if components == 'all':
        component_indices = range(ica.n_components_)
        report_name = 'ica_components_all'
    elif components == 'rejected':
        component_indices = ica.exclude
        report_name = 'ica_components_rejected'
        if not component_indices:
            print("No components were rejected. Skipping rejected components report.")
            return
    else:
        raise ValueError("components parameter must be 'all' or 'rejected'.")

    # Get ICA activations
    ica_sources = ica.get_sources(raw)
    ica_data = ica_sources.get_data()

    # Limit data to specified duration
    sfreq = raw.info['sfreq']
    n_samples = int(duration * sfreq)
    times = raw.times[:n_samples]

    # Create output path for the PDF report
    derivatives_path = pipeline.get_derivative_path(autoclean_dict["bids_path"])
    pdf_path = str(
        derivatives_path.copy().update(suffix=report_name, extension='.pdf')
    )

    # Remove existing file
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    with PdfPages(pdf_path) as pdf:
        # First page: Component topographies overview
        fig_topo = ica.plot_components(picks=component_indices, show=False)
        if isinstance(fig_topo, list):
            for f in fig_topo:
                pdf.savefig(f)
                plt.close(f)
        else:
            pdf.savefig(fig_topo)
            plt.close(fig_topo)

        # If rejected components, add overlay plot
        if components == 'rejected':
            fig_overlay = plt.figure()
            end_time = min(30., pipeline.raw.times[-1])
            fig_overlay = pipeline.ica2.plot_overlay(pipeline.raw, start=0, stop=end_time, exclude=component_indices, show=False)
            fig_overlay.set_size_inches(15, 10)  # Set size after creating figure

            pdf.savefig(fig_overlay)
            plt.close(fig_overlay)

        # For each component, create detailed plots
        for idx in component_indices:
            fig = plt.figure(constrained_layout=True, figsize=(12, 8))
            gs = GridSpec(nrows=3, ncols=3, figure=fig)

            # Axes for ica.plot_properties
            ax1 = fig.add_subplot(gs[0, 0])  # Data
            ax2 = fig.add_subplot(gs[0, 1])  # Epochs image
            ax3 = fig.add_subplot(gs[0, 2])  # ERP/ERF
            ax4 = fig.add_subplot(gs[1, 0])  # Spectrum
            ax5 = fig.add_subplot(gs[1, 1])  # Topomap
            ax_props = [ax1, ax2, ax3, ax4, ax5]

            # Plot properties
            ica.plot_properties(
                raw,
                picks=[idx],
                axes=ax_props,
                dB=True,
                plot_std=True,
                log_scale=False,
                reject='auto',
                show=False
            )

            # Add time series plot
            ax_timeseries = fig.add_subplot(gs[2, :])  # Last row, all columns
            ax_timeseries.plot(times, ica_data[idx, :n_samples], linewidth=0.5)
            ax_timeseries.set_xlabel('Time (seconds)')
            ax_timeseries.set_ylabel('Amplitude')
            ax_timeseries.set_title(f'Component {idx + 1} Time Course ({duration}s)')

            # Add labels
            comp_info = ic_labels.iloc[idx]
            label_text = (
                f"Component {idx + 1}\n"
                f"Type: {comp_info['ic_type']}\n"
                f"Confidence: {comp_info['confidence']:.2f}"
            )

            fig.suptitle(
                label_text,
                fontsize=14,
                fontweight='bold',
                color='red' if comp_info['ic_type'] in ['eog', 'muscle', 'ecg', 'other'] else 'black'
            )

            # Save the figure
            pdf.savefig(fig)
            plt.close(fig)

        print(f"Report saved to {pdf_path}")

def generate_ica_reports(pipeline, cleaned_raw, autoclean_dict, duration=60):
    """
    Generates two reports:
    1. All ICA components.
    2. Only the rejected ICA components.

    Parameters:
    -----------
    pipeline : pylossless.Pipeline
        The pipeline object containing the ICA and raw data.
    autoclean_dict : dict
        Dictionary containing configuration and paths.
    duration : int
        Duration in seconds for plotting time series data.
    """
    # Generate report for all components
    plot_ica_components(pipeline, cleaned_raw, autoclean_dict, duration=duration, components='all')

    # Generate report for rejected components
    plot_ica_components(pipeline, cleaned_raw, autoclean_dict, duration=duration, components='rejected')


def report_artifact_rejection(pipeline, cleaned_raw, autoclean_dict):
    """Generate a report for artifact rejection, plotting only the removed ICA components with improved layout."""
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.gridspec import GridSpec

    # Get the list of removed ICA components
    removed_ics = pipeline.ica2.exclude
    cleaned_raw.info['temp'] = {}
    cleaned_raw.info['temp']['removed_ics'] = removed_ics

    # Create the path for saving the artifact report
    derivatives_path = pipeline.get_derivative_path(autoclean_dict["bids_path"])
    ic_flags_path = str(
        derivatives_path.copy().update(suffix='ic_flags', extension='.pdf')
    )

    # Remove file if it exists
    if os.path.exists(ic_flags_path):
        os.remove(ic_flags_path)

    # Prepare to save figures to PDF
    with PdfPages(ic_flags_path) as pdf:
        # For each removed component, create a figure with improved layout
        for idx in removed_ics:
            # Create figure with constrained layout
            fig = plt.figure(constrained_layout=True, figsize=(12, 8))
            gs = GridSpec(nrows=3, ncols=3, figure=fig)

            # Axes for ica.plot_properties (5 axes)
            ax1 = fig.add_subplot(gs[0, 0])  # Data
            ax2 = fig.add_subplot(gs[0, 1])  # Epochs image
            ax3 = fig.add_subplot(gs[0, 2])  # ERP/ERF
            ax4 = fig.add_subplot(gs[1, 0])  # Spectrum
            ax5 = fig.add_subplot(gs[1, 1])  # Topomap
            ax_props = [ax1, ax2, ax3, ax4, ax5]

            # Plot the properties into the axes
            pipeline.ica2.plot_properties(
                cleaned_raw,
                picks=[idx],
                axes=ax_props,
                dB=True,
                plot_std=True,
                log_scale=False,
                reject='auto',
                show=False
            )

            # Add time series plot
            ax_timeseries = fig.add_subplot(gs[2, :])  # Last row, all columns
            # Get the ICA activations (sources)
            ica_sources = pipeline.ica2.get_sources(cleaned_raw)
            ica_data = ica_sources.get_data()
            times = cleaned_raw.times
            # Plot the time series for the component
            ax_timeseries.plot(times, ica_data[idx], linewidth=0.5)
            ax_timeseries.set_xlabel('Time (seconds)')
            ax_timeseries.set_ylabel('Amplitude')
            ax_timeseries.set_title(f'Component {idx} Time Course')

            # Add labels
            # Assuming you have component labels or types in pipeline.flags['ic']
            comp_info = pipeline.flags['ic'].loc[idx]
            label_text = (
                f"Type: {comp_info['ic_type']}\n"
                f"Confidence: {comp_info['confidence']:.3f}"
            )

            fig.suptitle(
                label_text,
                fontsize=14,
                fontweight='bold',
                color='red' if comp_info['ic_type'] in ['muscle', 'eog'] else 'black'
            )

            # Save the figure
            pdf.savefig(fig)
            plt.close(fig)

        # Plot overlay of raw and cleaned data for the removed components
        fig_overlay = pipeline.ica2.plot_overlay(
            pipeline.raw,
            exclude=removed_ics,
            show=False
        )
        pdf.savefig(fig_overlay)
        plt.close(fig_overlay)

        # Plot the bad channels
        # Collect bad channels
        noisy_channels = pipeline.flags['ch']['noisy']
        bridged_channels = pipeline.flags['ch']['bridged']
        rank_deficient_channels = pipeline.flags['ch']['rank']
        uncorrelated_channels = pipeline.flags['ch']['uncorrelated']

        # Combine all bad channels into a single list
        bad_channels = []
        bad_channels.extend(noisy_channels)
        bad_channels.extend(bridged_channels)
        bad_channels.extend(rank_deficient_channels)
        bad_channels.extend(uncorrelated_channels)
        # Remove duplicates while preserving order
        bad_channels = list(dict.fromkeys(bad_channels))
        bad_channels = [str(channel) for channel in bad_channels]

        if bad_channels:
            # Get data for the bad channels
            data, times = cleaned_raw.get_data(picks=bad_channels, return_times=True)

            # Create a figure with appropriate layout
            fig_bad_ch = plt.figure(constrained_layout=True, figsize=(12, 6))
            ax = fig_bad_ch.add_subplot(111)

            # Plot each bad channel with offset
            for idx, channel in enumerate(bad_channels):
                offset = idx * np.ptp(data) * 0.1  # Offset for clarity
                ax.plot(times, data[idx] + offset, label=channel)

            # Customize the plot
            ax.set_title("Bad Channels")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.legend(loc="upper right")

            # Save the figure
            pdf.savefig(fig_bad_ch)
            plt.close(fig_bad_ch)

    

def process_resting_eyesopen(autoclean_dict: dict) -> None:
    """Process resting state eyes-open data."""

    # Initialize metadata tracking
    step_handle_metadata(autoclean_dict, mode='save')

    # Import and save raw EEG data
    raw = step_import_raw(autoclean_dict)
    save_raw_to_set(raw, autoclean_dict, 'post_import')

    # Run preprocessing pipeline and save intermediate result
    raw = pre_pipeline_processing(raw, autoclean_dict)
    save_raw_to_set(raw, autoclean_dict, 'post_prepipeline')

    # Create BIDS-compliant paths and filenames
    raw, autoclean_dict = create_bids_path(raw, autoclean_dict)

    raw = step_clean_bad_channels(raw)

    # Run PyLossless pipeline and save result
    pipeline = step_run_pylossless(autoclean_dict)
    save_raw_to_set(raw, autoclean_dict, 'post_pylossless')

    # Artifact Rejection
    pipeline, autoclean_dict = clean_artifacts_continuous(pipeline, autoclean_dict)

    console.print("[green]✓ Completed[/green]")

def process_chirp_default(raw: mne.io.Raw) -> None:
    """Process chirp default data."""
    pass

def process_assr_default(raw: mne.io.Raw) -> None:
    """Process assr default data."""
    pass

def extract_task_dict(full_dict, task_name):
    task_dict = {
        'task_config': full_dict['tasks'][task_name],
        'stage_files': full_dict['stage_files']
    }
    return task_dict
def entrypoint(
    unprocessed_file: Union[str, Path], 
    task: str
) -> None:
    """
    Main entry point for the autoclean pipeline.
    
    Args:
        unprocessed_file: Path to raw EEG data file
        task: Task/experiment name
        
    This function handles initialization of the pipeline and manages multiprocessing resources.
    """
    try:
        # Set MNE to use only one job to avoid conflicts with multiprocessing
        mne.set_config('MNE_USE_CUDA', 'false')  # Disable CUDA to avoid GPU conflicts
        # mne.utils.set_config('MNE_NUM_JOBS', 1)  # Set to single job

        # Validate environment variables
        autoclean_dir, autoclean_config_file = validate_environment_variables()

        # Validate autoclean configuration  
        autoclean_dict = validate_autoclean_config(autoclean_config_file)

        # Validate task and EEG system
        task = validate_task(task, step_list_tasks(autoclean_dict))
        eeg_system = validate_eeg_system(autoclean_dict['tasks'][task]['settings']['montage']['value'])

        # Validate input file
        validate_input_file(unprocessed_file)

        # Log pipeline start    
        step_log_start(unprocessed_file, eeg_system, task, autoclean_config_file)
        
        # Prepare directories
        autoclean_dir, bids_dir, metadata_dir, clean_dir, stage_dir, debug_dir = step_prepare_directories(task)

        autoclean_dict = {
            'task': task,
            'eeg_system': eeg_system, 
            'config_file': autoclean_config_file,
            'tasks': {
                task: autoclean_dict['tasks'][task]
            },
            'stage_files': autoclean_dict['stage_files'],
            'unprocessed_file': unprocessed_file,
            'autoclean_dir': autoclean_dir,
            'bids_dir': bids_dir,
            'metadata_dir': metadata_dir,
            'clean_dir': clean_dir,
            'debug_dir': debug_dir,
            'stage_dir': stage_dir
        }


        # Branch to task-specific and eeg system-specific processing
        if task == "rest_eyesopen":
            process_resting_eyesopen(autoclean_dict)
        
        elif task == "chirp_default":
            if eeg_system == "EGI_129":
                pass
            else:
                raise ValueError(f"Unsupported EEG system for task: {eeg_system}")

        elif task == "assr_default":
            if eeg_system == "EGI_129":
                pass
            else:
                raise ValueError(f"Unsupported EEG system for task: {eeg_system}")

        else:
            raise ValueError(f"Unsupported task: {task}")

        log_pipeline_completion()

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise
    finally:
        # Reset MNE config to defaults
        console.print("[bold red]Resetting MNE config to defaults[/bold red]")
        #mne.utils.set_config('MNE_NUM_JOBS', None)
        #mne.set_config('MNE_USE_CUDA', None)

def step_prepare_directories(task: str) -> tuple[Path, Path, Path, Path, Path]:
    """Create and return required autoclean pipeline directories."""
    logger.info(f"Setting up pipeline directories for task: {task}")
    console.print("[bold blue]Setting up autoclean pipeline directories...[/bold blue]")
    
    autoclean_dir = Path(os.getenv("AUTOCLEAN_DIR"))
    dirs = {
        "bids": autoclean_dir / task / "bids",
        "metadata": autoclean_dir / task / "metadata", 
        "clean": autoclean_dir / task / "postcomps",
        "debug": autoclean_dir / task / "debug",
        "stage": autoclean_dir / task / "stage"
    }
    
    for name, dir in track(dirs.items(), description="Creating directories"):
        logger.debug(f"Creating directory: {dir}")
        dir.mkdir(parents=True, exist_ok=True)
    
    # Create and display directory table
    table_data = [[name, str(path)] for name, path in dirs.items()]
    table_data.insert(0, ["root", str(autoclean_dir)])
    
    console.print("\n[bold]autoclean Pipeline Directories:[/bold]")
    table = Table(title="Directory Structure", show_header=True, header_style="bold cyan", show_lines=True)
    table.add_column("Type", style="bold green")
    table.add_column("Path", style="white")
    for row in table_data:
        table.add_row(*row)
    console.print(Panel.fit(table))
    
    logger.info("Directory setup completed")
    console.print("[bold green]✓ Directory setup complete![/bold green]")
        
    return autoclean_dir, dirs["bids"], dirs["metadata"], dirs["clean"], dirs["stage"], dirs["debug"]

def validate_environment_variables() -> tuple[str, str]:
    """Check required environment variables are set and return their values.
    
    Returns:
        tuple[str, str]: AUTOCLEAN_DIR and AUTOCLEAN_CONFIG paths
    
    Raises:
        ValueError: If required environment variables are not set
    """
    autoclean_dir = os.getenv("AUTOCLEAN_DIR")
    if not autoclean_dir:
        logger.error("AUTOCLEAN_DIR environment variable is not set")
        raise ValueError("AUTOCLEAN_DIR environment variable (pipeline output root directory) is not set.")

    autoclean_config = os.getenv("AUTOCLEAN_CONFIG") 
    if not autoclean_config:
        logger.error("AUTOCLEAN_CONFIG environment variable is not set")
        raise ValueError("AUTOCLEAN_CONFIG environment variable (path to autoclean configuration file) is not set.")
        
    return autoclean_dir, autoclean_config

def validate_input_file(unprocessed_file: Union[str, Path]) -> None:
    """Check if the input file exists and is a valid EEG file."""
    logger.info(f"Validating input file: {unprocessed_file}")
    console.print(f"[cyan]Validating input file:[/cyan] {unprocessed_file}")

    file_path = Path(unprocessed_file)
    if not file_path.exists():
        error_msg = f"Input file does not exist: {unprocessed_file}"
        logger.error(error_msg)
        console.print(f"[red]✗ {error_msg}[/red]")
        raise FileNotFoundError(error_msg)

    logger.info(f"Input file validation successful: {unprocessed_file}")
    console.print(f"[green]✓ Input file exists and is accessible[/green]")

def validate_task(task: str, available_tasks: list[str]) -> str:
    if task in available_tasks:
        console.print(f"[green]✓ Valid task: {task}[/green]")
        return task
    else:
        console.print(f"[red]✗ Invalid task: {task}[/red]")
        console.print("[yellow]Allowed values are:[/yellow]")
        for t in available_tasks:
            console.print(f"  • [cyan]{t}[/cyan]")
        raise ValueError(f'Invalid task: {task}. Allowed values are: {available_tasks}')

def validate_eeg_system(eeg_system: str) -> str:
    """Validate that the EEG system montage is supported by MNE.
    
    Args:
        eeg_system: Name of the EEG system montage
        
    Returns:
        str: Validated EEG system montage name
        
    Raises:
        ValueError: If EEG system montage is not supported
    """
    VALID_MONTAGES = {
        # Standard system montages
        'standard_1005': '10-05 system (343+3 locations)',
        'standard_1020': '10-20 system (94+3 locations)', 
        'standard_alphabetic': 'LETTER-NUMBER combinations (65+3 locations)',
        'standard_postfixed': '10-20 system with postfixes (100+3 locations)',
        'standard_prefixed': '10-20 system with prefixes (74+3 locations)',
        'standard_primed': '10-20 system with prime marks (100+3 locations)',
        
        # BioSemi montages
        'biosemi16': 'BioSemi 16 electrodes (16+3 locations)',
        'biosemi32': 'BioSemi 32 electrodes (32+3 locations)', 
        'biosemi64': 'BioSemi 64 electrodes (64+3 locations)',
        'biosemi128': 'BioSemi 128 electrodes (128+3 locations)',
        'biosemi160': 'BioSemi 160 electrodes (160+3 locations)',
        'biosemi256': 'BioSemi 256 electrodes (256+3 locations)',
        
        # EasyCap montages
        'easycap-M1': 'EasyCap 10-05 names (74 locations)',
        'easycap-M10': 'EasyCap numbered (61 locations)',
        
        # EGI/GSN montages
        'EGI_256': 'Geodesic Sensor Net (256 locations)',
        'GSN-HydroCel-32': 'HydroCel GSN with Cz (33+3 locations)',
        'GSN-HydroCel-64_1.0': 'HydroCel GSN (64+3 locations)', 
        'GSN-HydroCel-65_1.0': 'HydroCel GSN with Cz (65+3 locations)',
        'GSN-HydroCel-128': 'HydroCel GSN (128+3 locations)',
        'GSN-HydroCel-129': 'HydroCel GSN with Cz (129+3 locations)',
        'GSN-HydroCel-256': 'HydroCel GSN (256 locations)',
        'GSN-HydroCel-257': 'HydroCel GSN with Cz (257+3 locations)',
        
        # MGH montages
        'mgh60': 'MGH 60-channel cap (60+3 locations)',
        'mgh70': 'MGH 70-channel BrainVision (70+3 locations)',
        
        # fNIRS montages
        'artinis-octamon': 'Artinis OctaMon fNIRS (8 sources, 2 detectors)',
        'artinis-brite23': 'Artinis Brite23 fNIRS (11 sources, 7 detectors)'
    }

    if eeg_system in VALID_MONTAGES:
        console.print(f"[green]✓ Valid EEG system: {eeg_system}[/green]")
        console.print(f"[cyan]Description: {VALID_MONTAGES[eeg_system]}[/cyan]")
        return eeg_system
    else:
        console.print(f"[red]✗ Invalid EEG system: {eeg_system}[/red]")
        console.print("[yellow]Supported montages are:[/yellow]")
        for system, desc in VALID_MONTAGES.items():
            console.print(f"  • [cyan]{system}[/cyan]: {desc}")
        raise ValueError(f'Invalid EEG system: {eeg_system}. Must be one of the supported MNE montages.')

def validate_autoclean_config(config_file: Union[str, Path]) -> dict:
    """Validate the autoclean configuration file.
    
    Args:
        config_file: Path to YAML configuration file
        
    Returns:
        tuple containing:
            - list[str]: Available task names
            - list[str]: Available EEG system names 
            - dict[str, str]: Mapping of tasks to their lossless config files
            
    Raises:
        FileNotFoundError: If config file does not exist
        ValueError: If config file is not a valid YAML file
    """
    config_path = Path(config_file)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    try:
        autoclean_dict = step_load_config(config_path)
                
    except Exception as e:
        logger.error(f"Invalid YAML configuration file: {str(e)}")
        raise ValueError(f"Invalid YAML configuration file: {str(e)}")

    return autoclean_dict

def main() -> None:
    logger.info("Initializing autoclean pipeline")
    console.print("[bold]Initializing autoclean Pipeline[/bold]")
    
    # Initialize required variables before calling entrypoint
    unprocessed_file = Path("/Users/ernie/Documents/GitHub/spg_analysis_redo/dataset_raw/0170_rest.raw")  
    task = "rest_eyesopen"

    try:
        entrypoint(unprocessed_file, task)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
