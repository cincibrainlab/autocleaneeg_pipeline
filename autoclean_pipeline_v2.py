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
#     "mplcairo",
#     "unqlite"
# ]
# ///

"""
autoclean: Automated EEG Processing Pipeline

This pipeline handles automated EEG data processing:
- BIDS conversion
- Preprocessing
- Artifact rejection
- Quality control

Environment Variables:
    AUTOCLEAN_DIR (root directory for output)
    AUTOCLEAN_CONFIG (configuration YAML filename)

Color Scheme:
    HEADER: cyan
    SUCCESS: green + ✓
    ERROR: red + ✗
    WARNING: yellow + ⚠
    INFO: white (general information)
    VALUES: dim cyan (paths, parameter values)
"""

import logging
import os
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Union

from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
from schema import Schema, Or
from dotenv import load_dotenv
from unqlite import UnQLite
from ulid import ULID

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import mne
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids, update_sidecar_json
from mne.io.constants import FIFF
from mne.preprocessing import bads

from pyprep.find_noisy_channels import NoisyChannels
import pylossless as ll
from autoreject import AutoReject

logger = logging.getLogger('autoclean')
console = Console()

load_dotenv()

# Single global database connection
db = UnQLite('autoclean.db')

# Configure logging to only write to file, not console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('autoclean.log')],
    force=True
)

def message(msg_type: str, text: str) -> None:
    """
    Unified messaging function for console and logger.
    """
    msg_map = {
        "header":  {"style": "cyan", "symbol": "",    "log_level": logging.INFO},
        "success": {"style": "green", "symbol": "✓ ", "log_level": logging.INFO},
        "error":   {"style": "red",   "symbol": "✗ ", "log_level": logging.ERROR},
        "warning": {"style": "yellow","symbol": "⚠ ", "log_level": logging.WARNING},
        "info":    {"style": "white", "symbol": "",    "log_level": logging.INFO},
        "values":  {"style": "dim cyan", "symbol": "", "log_level": logging.DEBUG},
    }

    if msg_type not in msg_map:
        msg_type = "info"

    style = msg_map[msg_type]["style"]
    symbol = msg_map[msg_type]["symbol"]
    log_level = msg_map[msg_type]["log_level"]

    console.print(f"[{style}]{symbol}{text}[/{style}]")

    # Strip known tags for logging
    clean_text = text
    for ctag in ["[dim cyan]", "[/dim cyan]", "[white]", "[/white]", "[cyan]", "[/cyan]", "[green]", "[/green]", "[red]", "[/red]", "[yellow]", "[/yellow]"]:
        clean_text = clean_text.replace(ctag, "")

    if log_level == logging.INFO:
        logger.info(clean_text)
    elif log_level == logging.ERROR:
        logger.error(clean_text)
    elif log_level == logging.WARNING:
        logger.warning(clean_text)
    elif log_level == logging.DEBUG:
        logger.debug(clean_text)
    else:
        logger.info(clean_text)

def validate_environment_variables(run_id: str) -> tuple[str, str]:
    message("header", "Validating environment variables")
    
    status = {
        "validate_environment_variables": {
            "timestamp": datetime.now().isoformat(),
            "status": "started",
            "variables": {}
        }
    }

    autoclean_dir = os.getenv("AUTOCLEAN_DIR")
    if not autoclean_dir:
        error_msg = "AUTOCLEAN_DIR is not set."
        message("error", error_msg)
        message("warning", "Set AUTOCLEAN_DIR in your .env file.")
        status["validate_environment_variables"]["status"] = "failed"
        status["validate_environment_variables"]["error"] = error_msg
        manage_database(operation='update', update_record={
            'run_id': run_id,
            'metadata': status
        })
        raise ValueError(error_msg)
    else:
        message("success", f"AUTOCLEAN_DIR: [dim cyan]{autoclean_dir}[/dim cyan]")
        status["validate_environment_variables"]["variables"]["AUTOCLEAN_DIR"] = autoclean_dir

    autoclean_config = os.getenv("AUTOCLEAN_CONFIG") 
    if not autoclean_config:
        error_msg = "AUTOCLEAN_CONFIG is not set."
        message("error", error_msg)
        message("warning", "Set AUTOCLEAN_CONFIG in your .env file.")
        status["validate_environment_variables"]["status"] = "failed"
        status["validate_environment_variables"]["error"] = error_msg
        manage_database(operation='update', update_record={
            'run_id': run_id,
            'metadata': status
        })
        raise ValueError(error_msg)
    else:
        message("success", f"AUTOCLEAN_CONFIG: [dim cyan]{autoclean_config}[/dim cyan]")
        status["validate_environment_variables"]["variables"]["AUTOCLEAN_CONFIG"] = autoclean_config

    status["validate_environment_variables"]["status"] = "completed"
    manage_database(operation='update', update_record={
        'run_id': run_id,
        'metadata': status
    })
    
    message("success", "Environment variables validated")
    return autoclean_dir, autoclean_config

def validate_autoclean_config(config_file: Union[str, Path]) -> dict:
    message("header", f"Validating configuration: [dim cyan]{config_file}[/dim cyan]")
    
    config_path = Path(config_file)
    if not config_path.exists():
        error_msg = f"Configuration file not found: {config_path}"
        message("error", error_msg)
        raise FileNotFoundError(error_msg)
        
    try:
        autoclean_dict = step_load_config(config_path)
        message("success", "Configuration validated")
        return autoclean_dict
    except Exception as e:
        error_msg = f"Invalid configuration file: {e}"
        message("error", error_msg)
        raise ValueError(error_msg)

def validate_task(task: str, available_tasks: list[str]) -> str:
    if task in available_tasks:
        message("success", f"Task validated: {task}")
        return task
    else:
        error_msg = f"Invalid task: {task}. Allowed: {', '.join(available_tasks)}"
        message("error", error_msg)
        raise ValueError(error_msg)

def validate_eeg_system(eeg_system: str) -> str:
    VALID_MONTAGES = {
        'standard_1005': '10-05 system',
        'standard_1020': '10-20 system', 
        'standard_alphabetic': 'Letter-number combos',
        'standard_postfixed': '10-20 with postfixes',
        'standard_prefixed': '10-20 with prefixes',
        'standard_primed': '10-20 with primes',
        'biosemi16': 'BioSemi 16',
        'biosemi32': 'BioSemi 32', 
        'biosemi64': 'BioSemi 64',
        'biosemi128': 'BioSemi 128',
        'biosemi160': 'BioSemi 160',
        'biosemi256': 'BioSemi 256',
        'easycap-M1': 'EasyCap M1 (10-05)',
        'easycap-M10': 'EasyCap M10',
        'EGI_256': 'EGI 256',
        'GSN-HydroCel-32': 'HydroCel GSN 32',
        'GSN-HydroCel-64_1.0': 'HydroCel GSN 64', 
        'GSN-HydroCel-65_1.0': 'HydroCel GSN 65',
        'GSN-HydroCel-128': 'HydroCel GSN 128',
        'GSN-HydroCel-129': 'HydroCel GSN 129',
        'GSN-HydroCel-256': 'HydroCel GSN 256',
        'GSN-HydroCel-257': 'HydroCel GSN 257',
        'mgh60': 'MGH 60-channel',
        'mgh70': 'MGH 70-channel',
        'artinis-octamon': 'Artinis OctaMon fNIRS',
        'artinis-brite23': 'Artinis Brite23 fNIRS'
    }

    if eeg_system in VALID_MONTAGES:
        message("success", f"EEG system validated: {eeg_system}")
        return eeg_system
    else:
        error_msg = f"Invalid EEG system: {eeg_system}. Supported: {', '.join(VALID_MONTAGES.keys())}"
        message("error", error_msg)
        raise ValueError(error_msg)

def validate_input_file(unprocessed_file: Union[str, Path]) -> None:
    message("info", f"Validating input file: {unprocessed_file}")

    file_path = Path(unprocessed_file)
    if not file_path.exists():
        error_msg = f"Input file not found: {unprocessed_file}"
        message("error", error_msg)
        raise FileNotFoundError(error_msg)

    message("success", "Input file validated")

def step_load_config(config_file: Union[str, Path]) -> dict:
    message("info", f"Loading config: {config_file}")

    config_schema = Schema({
        'tasks': {
            str: {
                'mne_task': str,
                'description': str,
                'lossless_config': str,
                'settings': {
                    'resample_step': {'enabled': bool, 'value': int},
                    'eog_step': {'enabled': bool, 'value': list},
                    'trim_step': {'enabled': bool, 'value': int},
                    'crop_step': {'enabled': bool, 'value': {'start': int, 'end': Or(float, None)}},
                    'reference_step': {'enabled': bool, 'value': str},
                    'filter_step': {'enabled': bool, 'value': {'l_freq': Or(float, None), 'h_freq': Or(float, None)}},
                    'montage': {'enabled': bool, 'value': str}
                },
                'rejection_policy': {
                    'ch_flags_to_reject': list,
                    'ch_cleaning_mode': str,
                    'interpolate_bads_kwargs': {'method': str},
                    'ic_flags_to_reject': list,
                    'ic_rejection_threshold': float,
                    'remove_flagged_ics': bool
                }
            }
        },
        'stage_files': {
            str: {'enabled': bool, 'suffix': str}
        }
    })

    with open(config_file) as f:
        config = yaml.safe_load(f)
    autoclean_dict = config_schema.validate(config)
    return autoclean_dict

def step_list_tasks(autoclean_dict: dict) -> list[str]:
    return list(autoclean_dict['tasks'].keys())

def step_log_start(unprocessed_file: Union[str, Path], eeg_system: str, task: str, autoclean_config_file: Union[str, Path]) -> dict:
    console.print(
        Panel(
            f"[cyan]Autoclean: Processing EEG Data[/cyan]\n"
            f"[white]File:[/white] [cyan]{unprocessed_file}[/cyan]\n"
            f"[white]EEG System:[/white] [cyan]{eeg_system}[/cyan]\n"
            f"[white]Task:[/white] [cyan]{task}[/cyan]\n"
            f"[white]Config:[/white] [cyan]{autoclean_config_file}[/cyan]",
            title="Pipeline Start"
        )
    )
    return

def step_prepare_directories(task: str, run_id: str) -> tuple[Path, Path, Path, Path, Path]:
    message("header", f"Setting up directories for task: {task}")

    autoclean_dir = Path(os.getenv("AUTOCLEAN_DIR"))
    dirs = {
        "bids": autoclean_dir / task / "bids",
        "metadata": autoclean_dir / task / "metadata", 
        "clean": autoclean_dir / task / "postcomps",
        "debug": autoclean_dir / task / "debug",
        "stage": autoclean_dir / task / "stage"
    }
    
    for _, dir_ in track(dirs.items(), description="Creating directories"):
        dir_.mkdir(parents=True, exist_ok=True)
    
    table_data = [["root", str(autoclean_dir)]] + [[name, str(path)] for name, path in dirs.items()]
    table = Table(title="Directory Structure", show_header=True, header_style="cyan", show_lines=True)
    table.add_column("Type", style="green")
    table.add_column("Path", style="dim cyan")
    for row in table_data:
        table.add_row(*row)
    console.print(Panel.fit(table))

    # Update directories in database
    manage_database(operation='update', update_record={
        'run_id': run_id,
        'metadata': {'step_prepare_directories': {key: str(path) for key, path in dirs.items()}}
    })

    message("success", "Directories ready")
    return autoclean_dir, dirs["bids"], dirs["metadata"], dirs["clean"], dirs["stage"], dirs["debug"]

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
    # Only save if enabled for this stage in stage_files config
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

    raw.info['description'] = autoclean_dict['run_id'] 
    
    raw.export(set_path, fmt='eeglab', overwrite=True)
    message("success", f"Saved stage file for {stage} to: {basename}")

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

    run_id = autoclean_dict['run_id']
    manage_database(operation='update', update_record={
        'run_id': run_id,
        'metadata': metadata
    })

    # Update the main database record status to reflect the stage completed
    manage_database(operation='update_status', update_record={
        'run_id': run_id,
        'status': f'{stage} completed'
    })

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
    # Only save if enabled for this stage in stage_files config
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

    # Add run_id for future database reference
    epochs.info['description'] = autoclean_dict['run_id']  
    epochs.apply_proj()
    epochs.export(set_path, fmt='eeglab', overwrite=True)

    import scipy.io as sio
    EEG = sio.loadmat(set_path)
    EEG['etc'] = {}
    EEG['etc']['run_id'] = autoclean_dict['run_id']
    sio.savemat(set_path, EEG, do_compression=False)

    # epochs2 = mne.read_epochs_eeglab(set_path)

    message("success", f"Saved stage file for {stage} to: {basename}")


    metadata = {
        "save_epochs_to_set": {
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

    run_id = autoclean_dict['run_id']
    manage_database(operation='update', update_record={
        'run_id': run_id,
        'metadata': metadata
    })

    # Update the main database record status to reflect the stage completed
    manage_database(operation='update_status', update_record={
        'run_id': run_id,
        'status': f'{stage} completed'
    })

    return set_path

def save_edited_epochs_to_set(epochs, autoclean_dict, stage="post_edit", output_path=None):
    pass

def manage_database(operation: str = 'connect', run_record: dict = None, update_record: dict = None) -> None:
    try:
        if operation == 'create_collection':
            collection = db.collection('pipeline_runs')
            if not collection.exists():
                collection.create()
                message("success", "Created 'pipeline_runs' collection")

        elif operation == 'drop_collection':
            collection = db.collection('pipeline_runs')
            if collection.exists():
                collection.drop()
                message("warning", "'pipeline_runs' collection dropped")

        elif operation == 'get_collection':
            collection = db.collection('pipeline_runs')
            if collection.exists():
                return collection
            else:
                error_msg = "No 'pipeline_runs' collection found."
                message("error", error_msg)
                raise ValueError(error_msg)

        elif operation == 'store':
            collection = manage_database(operation='get_collection')
            return collection.store(run_record, return_id=True)

        elif operation == 'update_status':
            collection = manage_database(operation='get_collection')
            if update_record and 'run_id' in update_record:
                run_id = update_record['run_id']
                
                # Debug prints
                message("info", f"Attempting to update status for run_id: {run_id}")

                # Find matching record
                existing_record = collection.filter(lambda x: x['run_id'] == run_id)
                
                if existing_record:
                    record_id = existing_record[0]['__id']
                    message("success", f"Found record with ID: {record_id}")
                    current_record = collection.fetch(record_id)
                    
                    # Append timestamped status
                    timestamp = datetime.now().isoformat()
                    current_record['status'] = f"{update_record['status']} at {timestamp}"
                    
                    collection.update(record_id=record_id, record=current_record)
                    message("success", f"Status updated for run_id: {run_id}")
                else:
                    error_msg = f"No record found for run_id: {run_id}"
                    message("error", error_msg)
                    message("info", "Available run_ids: " + 
                           ", ".join([str(r.get('run_id')) for r in collection.all()]))
                    raise ValueError(error_msg)

        elif operation == 'update':
            collection = manage_database(operation='get_collection')
            if update_record and 'run_id' in update_record:
                run_id = update_record['run_id']
                existing_record = collection.filter(lambda x: x['run_id'] == run_id)
                if existing_record:
                    doc = existing_record[0]
                    record_id = doc['__id']
                    # Remove __id since we'll use it directly in update
                    doc.pop('__id', None)

                    # Handle metadata merging
                    if 'metadata' in update_record:
                        # Ensure the doc has a metadata dictionary
                        if 'metadata' not in doc:
                            doc['metadata'] = {}
                        elif not isinstance(doc['metadata'], dict):
                            # Convert to dict if it's not already
                            doc['metadata'] = {}

                        # Update metadata
                        for key, value in update_record['metadata'].items():
                            if key not in doc['metadata']:
                                doc['metadata'][key] = value
                            else:
                                current = doc['metadata'][key]
                                if not isinstance(current, list):
                                    current = [current]
                                if isinstance(value, list):
                                    current.extend(value)
                                else:
                                    current.append(value)
                                doc['metadata'][key] = current

                        # Remove metadata from update_record before general update
                        non_metadata_update = {k: v for k, v in update_record.items() if k != 'metadata'}
                        doc.update(non_metadata_update)
                    else:
                        # No metadata, just update the doc normally
                        doc.update(update_record)

                    # Write merged doc back to the database
                    collection.update(record_id=record_id, record=doc)
                    message("info", "Record updated with appended metadata")
                else:
                    error_msg = f"No record found for run_id: {update_record['run_id']}"
                    message("error", error_msg)
                    raise ValueError(error_msg)

        elif operation == 'get_record':
            collection = manage_database(operation='get_collection')
            if run_record and 'run_id' in run_record:
                run_id = run_record['run_id']
                record = collection.filter(lambda x: x['run_id'] == run_id)
                if record:
                    return record[0]
                else:
                    error_msg = f"No record found for run_id: {run_record['run_id']}"
                    message("error", error_msg)
                    raise ValueError(error_msg)

        elif operation == 'print_record':
            collection = manage_database(operation='get_collection')
            if run_record and 'run_id' in run_record:
                run_id = run_record['run_id']
                record = collection.filter(lambda x: x['run_id'] == run_id)
                if record:
                    from rich.console import Console
                    from rich.pretty import pprint

                    console = Console()
                    console.print("\n[bold blue]Database Record:[/bold blue]")
                    pprint(record[0], expand_all=True)
                else:
                    error_msg = f"No record found for run_id: {run_record['run_id']}"
                    message("error", error_msg)
                    raise ValueError(error_msg)

    except Exception as e:
        message("error", f"Database operation '{operation}' failed: {e}")
        raise

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
    
    message("info", f"Importing raw EEG data from {unprocessed_file} using {eeg_system} system")
    message("header", "Importing raw EEG data...")
    
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
            
        message("info", "Raw EEG data imported successfully")
        message("success", "Raw EEG data imported successfully")

        metadata = {
            "step_import_raw": {
                "creationDateTime": datetime.now().isoformat(),
                "unprocessedFile": str(autoclean_dict["unprocessed_file"].name),
                "eegSystem": autoclean_dict["eeg_system"],
                "sampleRate": raw.info["sfreq"],
                "channelCount": len(raw.ch_names),
                "durationSec": int(raw.n_times) / raw.info["sfreq"],
                "numberSamples": int(raw.n_times)
            }
        }

        run_id = autoclean_dict['run_id']
        manage_database(operation='update', update_record={
            'run_id': run_id,
            'metadata': metadata
        })
        return raw
        
    except Exception as e:
        message("error", f"Failed to import raw EEG data: {str(e)}")
        raise

def pre_pipeline_processing(raw, autoclean_dict):
    message("header", "\nPre-pipeline Processing Steps")

    task = autoclean_dict["task"]
    
    # Get enabled/disabled status for each step
    apply_resample_toggle = autoclean_dict["tasks"][task]["settings"]["resample_step"]["enabled"]
    apply_eog_toggle = autoclean_dict["tasks"][task]["settings"]["eog_step"]["enabled"]
    apply_average_reference_toggle = autoclean_dict["tasks"][task]["settings"]["reference_step"]["enabled"]
    apply_trim_toggle = autoclean_dict["tasks"][task]["settings"]["trim_step"]["enabled"]
    apply_crop_toggle = autoclean_dict["tasks"][task]["settings"]["crop_step"]["enabled"]    
    apply_filter_toggle = autoclean_dict["tasks"][task]["settings"]["filter_step"]["enabled"]

    # Print status of each step
    message("info", f"{'✓' if apply_resample_toggle else '✗'} Resample: {apply_resample_toggle}")
    message("info", f"{'✓' if apply_eog_toggle else '✗'} EOG Assignment: {apply_eog_toggle}")
    message("info", f"{'✓' if apply_average_reference_toggle else '✗'} Average Reference: {apply_average_reference_toggle}")
    message("info", f"{'✓' if apply_trim_toggle else '✗'} Edge Trimming: {apply_trim_toggle}")
    message("info", f"{'✓' if apply_crop_toggle else '✗'} Duration Cropping: {apply_crop_toggle}")
    message("info", f"{'✓' if apply_filter_toggle else '✗'} Filtering: {apply_filter_toggle}\n")

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
        message("header", "Resampling data...")
        target_sfreq = autoclean_dict["tasks"][task]["settings"]["resample_step"]["value"]
        raw = step_resample_data(raw, target_sfreq)
        message("success", f"Data resampled to {target_sfreq} Hz")
        metadata["pre_pipeline_processing"]["ResampleHz"] = target_sfreq
        save_raw_to_set(raw, autoclean_dict, 'post_resample')
    
    # EOG Assignment
    if apply_eog_toggle:
        message("header", "Setting EOG channels...")
        eog_channels = autoclean_dict["tasks"][task]["settings"]["eog_step"]["value"]
        raw = step_mark_eog_channels(raw, eog_channels)
        message("success", "EOG channels assigned")
        metadata["pre_pipeline_processing"]["EOGChannels"] = eog_channels
    
    # Average Reference
    if apply_average_reference_toggle:
        message("header", "Applying average reference...")
        ref_type = autoclean_dict["tasks"][task]["settings"]["reference_step"]["value"]
        raw = step_set_reference(raw, ref_type)
        message("success", "Average reference applied")
        save_raw_to_set(raw, autoclean_dict, 'post_reference')
    
    # Trim Edges
    if apply_trim_toggle:
        message("header", "Trimming data edges...")
        trim = autoclean_dict["tasks"][task]["settings"]["trim_step"]["value"]
        start_time = raw.times[0]
        end_time = raw.times[-1]
        raw.crop(tmin=start_time + trim, tmax=end_time - trim)   
        message("success", f"Data trimmed by {trim}s from each end")
        metadata["pre_pipeline_processing"]["TrimSec"] = trim
        save_raw_to_set(raw, autoclean_dict, 'post_trim')

    # Crop Duration
    if apply_crop_toggle:
        message("header", "Cropping data duration...")
        start_time = autoclean_dict["tasks"][task]["settings"]["crop_step"]["value"]['start']
        end_time = autoclean_dict["tasks"][task]["settings"]["crop_step"]["value"]['end']
        if end_time is None:
            end_time = raw.times[-1]  # Use full duration if end is null
        raw.crop(tmin=start_time, tmax=end_time)
        target_crop_duration = raw.times[-1] - raw.times[0]
        message("success", f"Data cropped to {target_crop_duration:.1f}s")
        metadata["pre_pipeline_processing"]["CropDurationSec"] = target_crop_duration
        metadata["pre_pipeline_processing"]["CropStartSec"] = start_time
        metadata["pre_pipeline_processing"]["CropEndSec"] = end_time
        save_raw_to_set(raw, autoclean_dict, 'post_crop')

    # Pre-Filter    
    if apply_filter_toggle:
        message("header", "Applying frequency filters...")
        target_lfreq = float(autoclean_dict["tasks"][task]["settings"]["filter_step"]["value"]["l_freq"])
        target_hfreq = float(autoclean_dict["tasks"][task]["settings"]["filter_step"]["value"]["h_freq"])
        raw.filter(l_freq=target_lfreq, h_freq=target_hfreq)
        message("success", f"Applied bandpass filter: {target_lfreq}-{target_hfreq} Hz")
        metadata["pre_pipeline_processing"]["LowPassHz1"] = target_lfreq
        metadata["pre_pipeline_processing"]["HighPassHz1"] = target_hfreq
        save_raw_to_set(raw, autoclean_dict, 'post_filter')
    
    metadata["pre_pipeline_processing"]["channelCount"] = len(raw.ch_names)
    metadata["pre_pipeline_processing"]["durationSec"] = int(raw.n_times) / raw.info["sfreq"]
    metadata["pre_pipeline_processing"]["numberSamples"] = int(raw.n_times)

    run_id = autoclean_dict['run_id']
    manage_database(operation='update', update_record={
        'run_id': run_id,
        'metadata': metadata
    })

    return raw

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
                "bids_subject": bids_path.subject,
                "bids_task": bids_path.task,
                "bids_run": bids_path.run,
                "bids_session": bids_path.session,
                "bids_dir": str(bids_dir),
                "bids_datatype": bids_path.datatype,
                "bids_suffix": bids_path.suffix,
                "bids_root": str(bids_path.root),
                "eegSystem": eeg_system,
                "configFile": str(config_file),
                "line_freq": 60.0
            }
        }

        manage_database(operation='update', update_record={
            'run_id': autoclean_dict['run_id'],
            'metadata': metadata
        })

        manage_database(operation='update_status', update_record={
            'run_id': autoclean_dict['run_id'],
            'status': 'bids_path_created'
        })

        return raw, autoclean_dict
    
    except Exception as e:
        message("error", f"Error converting raw to bids: {e}")
        raise e 


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
        sys.exit(1)

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
        "allow_preload": True
    }

    # Write BIDS data
    try:
        write_raw_bids(**bids_kwargs)
        message("success", f"Converted {fif_file.name} to BIDS format.")
        entries = {"Manufacturer": "Unknown", "PowerLineFrequency": line_freq}
        sidecar_path = bids_path.copy().update(extension=".json")
        update_sidecar_json(bids_path=sidecar_path, entries=entries)
    except Exception as e:
        message("error", f"Failed to write BIDS for {fif_file.name}: {e}")
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

def step_clean_bad_channels(raw, autoclean_dict):
    # Setup options
    options = {
        "random_state": 1337,
        "ransac": True,
        "channel_wise": False,
        "max_chunk_size": None,
        "threshold": 3.0
    }

    # Temporarily switch EOG channels to EEG type
    eog_picks = mne.pick_types(raw.info, eog=True, exclude=[])
    eog_ch_names = [raw.ch_names[idx] for idx in eog_picks]
    raw.set_channel_types({ch: 'eeg' for ch in eog_ch_names})

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])

    # Run noisy channels detection
    cleaned_raw = NoisyChannels(raw, random_state=options["random_state"])
    cleaned_raw.find_all_bads(
        ransac=options["ransac"], 
        channel_wise=options["channel_wise"],
        max_chunk_size=options["max_chunk_size"]
    )

    picks = mne.pick_types(raw.info, eeg=True, exclude=[])

    print(raw.info["bads"])
    raw.info["bads"].extend(cleaned_raw.get_bads())

    # Record metadata with options
    metadata = {
        "step_clean_bad_channels": {
            "creationDateTime": datetime.now().isoformat(),
            "method": "NoisyChannels",
            "options": options,
            "bads": raw.info["bads"],
            "channelCount": len(raw.ch_names),
            "durationSec": int(raw.n_times) / raw.info["sfreq"],
            "numberSamples": int(raw.n_times)
        }
    }

    manage_database(operation='update', update_record={
        'run_id': autoclean_dict['run_id'],
        'metadata': metadata
    })

    return raw

def step_run_pylossless(autoclean_dict):

    task                    = autoclean_dict["task"]
    bids_path               = autoclean_dict["bids_path"]
    config_path             = autoclean_dict["tasks"][task]["lossless_config"] 
    derivative_name         = "pylossless"
    raw = read_raw_bids(
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
                "numberSamples": int(pipeline.raw.n_times)
            }
        }

        manage_database(operation='update', update_record={
            'run_id': autoclean_dict['run_id'],
            'metadata': metadata
        })

    except Exception as e:
        message("error", f"Failed to load pylossless config: {str(e)}")
        raise e

    return pipeline

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from matplotlib.gridspec import GridSpec



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
        # Calculate how many components to show per page
        components_per_page = 20
        num_pages = int(np.ceil(len(component_indices) / components_per_page))

        # Create summary tables split across pages
        for page in range(num_pages):
            start_idx = page * components_per_page
            end_idx = min((page + 1) * components_per_page, len(component_indices))
            page_components = component_indices[start_idx:end_idx]

            fig_table = plt.figure(figsize=(11, 8.5))
            ax_table = fig_table.add_subplot(111)
            ax_table.axis('off')

            # Prepare table data for this page
            table_data = []
            colors = []
            for idx in page_components:
                comp_info = ic_labels.iloc[idx]
                table_data.append([
                    f"IC{idx + 1}",
                    comp_info['ic_type'],
                    f"{comp_info['confidence']:.2f}",
                    "Yes" if idx in ica.exclude else "No"
                ])
                
                # Define colors for different IC types
                color_map = {
                    'brain': '#d4edda',  # Light green
                    'eog': '#f9e79f',    # Light yellow
                    'muscle': '#f5b7b1',  # Light red
                    'ecg': '#d7bde2',    # Light purple,
                    'ch_noise': '#ffd700', # Light orange
                    'line_noise': '#add8e6', # Light blue
                    'other': '#f0f0f0'    # Light grey
                }
                colors.append([color_map.get(comp_info['ic_type'].lower(), 'white')] * 4)

            # Create and customize table
            table = ax_table.table(
                cellText=table_data,
                colLabels=['Component', 'Type', 'Confidence', 'Rejected'],
                loc='center',
                cellLoc='center',
                cellColours=colors,
                colWidths=[0.2, 0.3, 0.25, 0.25]
            )
            
            # Customize table appearance
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)  # Reduced vertical scaling
            
            # Add title with page information, filename and timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fig_table.suptitle(
                f'ICA Components Summary - {autoclean_dict["bids_path"].basename}\n'
                f'(Page {page + 1} of {num_pages})\n'
                f'Generated: {timestamp}', 
                fontsize=12, 
                y=0.95
            )
            # Add legend for colors
            legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, edgecolor='none') 
                             for color in color_map.values()]
            ax_table.legend(legend_elements, color_map.keys(), 
                           loc='upper right', title='Component Types')

            # Add margins
            plt.subplots_adjust(top=0.85, bottom=0.15)

            pdf.savefig(fig_table)
            plt.close(fig_table)

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
        return Path(pdf_path).name

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
    report_filename = plot_ica_components(pipeline, cleaned_raw, autoclean_dict, duration=duration, components='all')

    metadata = {
        "artifact_reports": {
            "creationDateTime": datetime.now().isoformat(),
            "ica_all_components": report_filename
        }
    }

    manage_database(operation='update', update_record={
        'run_id': autoclean_dict['run_id'],
        'metadata': metadata
    })

    # Generate report for rejected components
    report_filename = plot_ica_components(pipeline, cleaned_raw, autoclean_dict, duration=duration, components='rejected')

    metadata = {
        "artifact_reports": {
            "creationDateTime": datetime.now().isoformat(),
            "ica_rejected_components": report_filename
        }
    }

    manage_database(operation='update', update_record={
        'run_id': autoclean_dict['run_id'],
        'metadata': metadata
    })


def _plot_psd(fig, gs, freqs, psd_original_mean_mV2, psd_cleaned_mean_mV2, psd_original_rel, psd_cleaned_rel, num_bands):
    """Helper function to create PSD plots"""
    # First row: Two PSD plots side by side
    ax_abs_psd = fig.add_subplot(gs[0, :num_bands//2])
    ax_rel_psd = fig.add_subplot(gs[0, num_bands//2:])

    # Plot Absolute PSD with log scale
    ax_abs_psd.plot(freqs, psd_original_mean_mV2, color='red', label='Original')
    ax_abs_psd.plot(freqs, psd_cleaned_mean_mV2, color='black', label='Cleaned')
    ax_abs_psd.set_yscale('log')  # Set y-axis to log scale
    ax_abs_psd.set_xlabel('Frequency (Hz)')
    ax_abs_psd.set_ylabel('Power Spectral Density (mV²/Hz)')
    ax_abs_psd.set_title('Absolute Power Spectral Density (mV²/Hz)')
    ax_abs_psd.legend()
    ax_abs_psd.grid(True, which="both")  # Grid lines for both major and minor ticks

    # Add vertical lines and annotations for power bands on both PSDs
    for ax in [ax_abs_psd, ax_rel_psd]:
        for band_name, (f_start, f_end) in {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 12),
            'Beta': (12, 30),
            'Gamma': (30, 80)
        }.items():
            ax.axvline(f_start, color='grey', linestyle='--', linewidth=1)
            ax.axvline(f_end, color='grey', linestyle='--', linewidth=1)
            ax.fill_betweenx(ax.get_ylim(), f_start, f_end, color='grey', alpha=0.1)
            ax.text((f_start + f_end) / 2, ax.get_ylim()[1]*0.95, band_name,
                   horizontalalignment='center', verticalalignment='top', fontsize=9, color='grey')

    # Plot Relative PSD with log scale
    ax_rel_psd.plot(freqs, psd_original_rel, color='red', label='Original')
    ax_rel_psd.plot(freqs, psd_cleaned_rel, color='black', label='Cleaned')
    ax_rel_psd.set_yscale('log')  # Set y-axis to log scale
    ax_rel_psd.set_xlabel('Frequency (Hz)')
    ax_rel_psd.set_ylabel('Relative Power (%)')
    ax_rel_psd.set_title('Relative Power Spectral Density (%)')
    ax_rel_psd.legend()
    ax_rel_psd.grid(True, which="both")  # Grid lines for both major and minor ticks

def _plot_topomaps(fig, gs, bands, band_powers_orig, band_powers_clean, raw_original, raw_cleaned, outlier_channels_orig, outlier_channels_clean):
    """Helper function to create topographical maps"""
    # Second row: Topomaps for original data
    for i, (band, power) in enumerate(zip(bands, band_powers_orig)):
        band_name, l_freq, h_freq = band
        ax = fig.add_subplot(gs[1, i])
        mne.viz.plot_topomap(
            power, raw_original.info, axes=ax, show=False, contours=0, cmap='jet'
        )
        mean_power = np.mean(power)
        ax.set_title(f"Original: {band_name}\n({l_freq}-{h_freq} Hz)\nMean Power: {mean_power:.2e} mV²", fontsize=10)
        # Annotate outlier channels
        outliers = outlier_channels_orig[band_name]
        if outliers:
            ax.annotate(f"Outliers:\n{', '.join(outliers)}", xy=(0.5, -0.15), xycoords='axes fraction',
                       ha='center', va='top', fontsize=8, color='red')

    # Third row: Topomaps for cleaned data
    for i, (band, power) in enumerate(zip(bands, band_powers_clean)):
        band_name, l_freq, h_freq = band
        ax = fig.add_subplot(gs[2, i])
        mne.viz.plot_topomap(
            power, raw_cleaned.info, axes=ax, show=False, contours=0, cmap='jet'
        )
        mean_power = np.mean(power)
        ax.set_title(f"Cleaned: {band_name}\n({l_freq}-{h_freq} Hz)\nMean Power: {mean_power:.2e} mV²", fontsize=10)
        # Annotate outlier channels
        outliers = outlier_channels_clean[band_name]
        if outliers:
            ax.annotate(f"Outliers:\n{', '.join(outliers)}", xy=(0.5, -0.15), xycoords='axes fraction',
                       ha='center', va='top', fontsize=8, color='red')

def step_psd_topo_figure(raw_original, raw_cleaned, pipeline, autoclean_dict, bands=None, metadata=None):
    """
    Generate and save a single high-resolution image that includes:
    - Two PSD plots side by side: Absolute PSD (mV²) and Relative PSD (%).
    - Topographical maps for multiple EEG frequency bands arranged horizontally,
      showing both pre and post cleaning.
    - Annotations for average power and outlier channels.

    Parameters:
    -----------
    raw_original : mne.io.Raw
        Original raw EEG data before cleaning.
    raw_cleaned : mne.io.Raw
        Cleaned EEG data after preprocessing.
    pipeline : pylossless.Pipeline
        Pipeline object.
    autoclean_dict : dict
        Dictionary containing autoclean parameters and paths.
    bands : list of tuple, optional
        List of frequency bands to plot. Each tuple should contain 
        (band_name, lower_freq, upper_freq).
    metadata : dict, optional
        Additional metadata to include in the JSON sidecar.

    Returns:
    --------
    image_path : str
        Path to the saved combined figure.
    """

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
    
    # Create Artifact Report
    derivatives_path = pipeline.get_derivative_path(autoclean_dict["bids_path"])

    # Output figure path
    target_figure = str(derivatives_path.copy().update(
        suffix='step_psd_topo_figure',
        extension='.png',
        datatype='eeg'
    ))

    # Select all EEG channels
    picks = mne.pick_types(raw_original.info, eeg=True)
    if len(picks) == 0:
        raise ValueError("No EEG channels found in raw data.")

    # Parameters for PSD
    fmin = 0.5
    fmax = 80
    n_fft = int(raw_original.info['sfreq'] * 2)  # Window length of 2 seconds

    # Compute PSD for original and cleaned data
    psd_original = raw_original.compute_psd(
        method='welch', fmin=fmin, fmax=fmax, n_fft=n_fft,
        picks=picks, average='mean', verbose=False
    )
    psd_cleaned = raw_cleaned.compute_psd(
        method='welch', fmin=fmin, fmax=fmax, n_fft=n_fft,
        picks=picks, average='mean', verbose=False
    )

    freqs = psd_original.freqs
    df = freqs[1] - freqs[0]  # Frequency resolution

    # Convert PSDs to mV^2/Hz
    psd_original_mV2 = psd_original.get_data() * 1e6
    psd_cleaned_mV2 = psd_cleaned.get_data() * 1e6

    # Compute mean PSDs
    psd_original_mean_mV2 = np.mean(psd_original_mV2, axis=0)
    psd_cleaned_mean_mV2 = np.mean(psd_cleaned_mV2, axis=0)

    # Compute relative PSDs
    total_power_orig = np.sum(psd_original_mean_mV2 * df)
    total_power_clean = np.sum(psd_cleaned_mean_mV2 * df)
    psd_original_rel = (psd_original_mean_mV2 * df) / total_power_orig * 100
    psd_cleaned_rel = (psd_cleaned_mean_mV2 * df) / total_power_clean * 100

    # Compute band powers and identify outliers
    band_powers_orig = []
    band_powers_clean = []
    outlier_channels_orig = {}
    outlier_channels_clean = {}
    band_powers_metadata = {}

    for band_name, l_freq, h_freq in bands:
        # Get band powers
        band_power_orig = psd_original.get_data(fmin=l_freq, fmax=h_freq).mean(axis=-1) * df * 1e6
        band_power_clean = psd_cleaned.get_data(fmin=l_freq, fmax=h_freq).mean(axis=-1) * df * 1e6
        
        band_powers_orig.append(band_power_orig)
        band_powers_clean.append(band_power_clean)

        # Identify outliers
        for power, raw_data, outlier_dict in [
            (band_power_orig, raw_original, outlier_channels_orig),
            (band_power_clean, raw_cleaned, outlier_channels_clean)
        ]:
            mean_power = np.mean(power)
            std_power = np.std(power)
            if std_power > 0:
                z_scores = (power - mean_power) / std_power
                outliers = [ch for ch, z in zip(raw_data.ch_names, z_scores) if abs(z) > 3]
            else:
                outliers = []
            outlier_dict[band_name] = outliers

        # Store metadata
        band_powers_metadata[band_name] = {
            "frequency_band": f"{l_freq}-{h_freq} Hz",
            "band_power_mean_original_mV2": float(np.mean(band_power_orig)),
            "band_power_std_original_mV2": float(np.std(band_power_orig)),
            "band_power_mean_cleaned_mV2": float(np.mean(band_power_clean)),
            "band_power_std_cleaned_mV2": float(np.std(band_power_clean)),
            "outlier_channels_original": outlier_channels_orig[band_name],
            "outlier_channels_cleaned": outlier_channels_clean[band_name]
        }

    # Create figure and GridSpec
    fig = plt.figure(figsize=(15, 20))
    gs = GridSpec(4, len(bands), height_ratios=[2, 1, 1, 1.5], hspace=0.4, wspace=0.3)

    # Create PSD plots
    _plot_psd(fig, gs, freqs, psd_original_mean_mV2, psd_cleaned_mean_mV2, 
              psd_original_rel, psd_cleaned_rel, len(bands))

    # Create topographical maps
    _plot_topomaps(fig, gs, bands, band_powers_orig, band_powers_clean,
                   raw_original, raw_cleaned, outlier_channels_orig, outlier_channels_clean)

    # Add suptitle and adjust layout
    fig.suptitle(os.path.basename(raw_cleaned.filenames[0]), fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save and close figure
    fig.savefig(target_figure, dpi=300)
    plt.close(fig)

    print(f"Combined figure saved to {target_figure}")

    metadata = {
        "artifact_reports": {
            "creationDateTime": datetime.now().isoformat(),
            "plot_psd_topo_figure": Path(target_figure).name
        }
    }

    manage_database(operation='update', update_record={
        'run_id': autoclean_dict['run_id'],
        'metadata': metadata
    })

    return target_figure
    

def plot_bad_channels_with_topography(raw_original, raw_cleaned, pipeline, autoclean_dict, zoom_duration=30, zoom_start=0):
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
    import numpy as np
    import mne
    from matplotlib.gridspec import GridSpec

    # ----------------------------
    # 1. Collect Bad Channels
    # ----------------------------
    bad_channels_info = {}

    # Mapping from channel to reason(s)
    for reason, channels in pipeline.flags.get('ch', {}).items():
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

    data_original, times = raw_original.get_data(picks=picks_bad_original, return_times=True)
    data_cleaned = raw_cleaned.get_data(picks=picks_bad_cleaned)

    channel_labels = [raw_original.ch_names[i] for i in picks_bad_original]
    n_channels = len(channel_labels)

    # Debugging: Print number of bad channels being plotted
    print(f"Number of Bad Channels to Plot: {n_channels}")

    # ----------------------------
    # 4. Downsample Data if Necessary
    # ----------------------------
    sfreq = raw_original.info['sfreq']
    desired_sfreq = 100  # Target sampling rate
    downsample_factor = int(sfreq // desired_sfreq)
    if downsample_factor > 1:
        data_original = data_original[:, ::downsample_factor]
        data_cleaned = data_cleaned[:, ::downsample_factor]
        times = times[::downsample_factor]
        print(f"Data downsampled by a factor of {downsample_factor} to {desired_sfreq} Hz.")

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
        [int(raw_original.ch_names.index(ch)) for ch in bad_channels]
    ]
    colors = 'RdYlBu_r'

    # Plot again for the main figure subplot
    mne.viz.plot_sensors(
        raw_original.info,
        kind='topomap',
        ch_type='eeg',
        title='Sensor Topography: Good vs Bad Channels', 
        show_names=True,
        ch_groups=ch_groups,
        pointsize=75,
        linewidth=0,
        cmap=colors,
        show=False,
        axes=ax_topo
    )

    ax_topo.legend(['Good Channels', 'Bad Channels'], loc='upper right', fontsize=12)
    ax_topo.set_title('Topography of Good and Bad Channels', fontsize=16)

    # ----------------------------
    # 9. Full Duration Time Series Subplot
    # ----------------------------
    ax_full = fig.add_subplot(gs[1, 0])
    for idx in range(n_channels):
        # Plot original data
        ax_full.plot(times, data_original_scaled[idx] + offsets[idx], color='red', linewidth=1, linestyle='-')
        # Plot cleaned data
        ax_full.plot(times, data_cleaned_scaled[idx] + offsets[idx], color='black', linewidth=1, linestyle='-')

    ax_full.set_xlabel('Time (seconds)', fontsize=14)
    ax_full.set_ylabel('Bad Channels', fontsize=14)
    ax_full.set_title('Bad Channels: Original vs Interpolated (Full Duration)', fontsize=16)
    ax_full.set_xlim(times[0], times[-1])
    ax_full.set_ylim(-spacing, offsets[-1] + spacing)
    ax_full.set_yticks([])  # Hide y-ticks
    ax_full.invert_yaxis()

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, linestyle='-', label='Original Data'),
        Line2D([0], [0], color='black', lw=2, linestyle='-', label='Interpolated Data')
    ]
    ax_full.legend(handles=legend_elements, loc='upper right', fontsize=12)

    # ----------------------------
    # 10. Zoomed-In Time Series Subplot
    # ----------------------------
    ax_zoom = fig.add_subplot(gs[1, 1])
    for idx in range(n_channels):
        # Plot original data
        ax_zoom.plot(times, data_original_scaled[idx] + offsets[idx], color='red', linewidth=1, linestyle='-')
        # Plot cleaned data
        ax_zoom.plot(times, data_cleaned_scaled[idx] + offsets[idx], color='black', linewidth=1, linestyle='-')

    ax_zoom.set_xlabel('Time (seconds)', fontsize=14)
    ax_zoom.set_title(f'Bad Channels: Original vs Interpolated (Zoom: {zoom_start}-{zoom_end} s)', fontsize=16)
    ax_zoom.set_xlim(zoom_start, zoom_end)
    ax_zoom.set_ylim(-spacing, offsets[-1] + spacing)
    ax_zoom.set_yticks([])  # Hide y-ticks
    ax_zoom.invert_yaxis()

    # Add legend
    ax_zoom.legend(handles=legend_elements, loc='upper right', fontsize=12)

    # ----------------------------
    # 11. Add Channel Labels
    # ----------------------------
    for idx, ch in enumerate(channel_labels):
        label = f"{ch}\n({', '.join(bad_channels_info[ch])})"
        ax_full.text(times[0] - (0.05 * (times[-1] - times[0])), offsets[idx], label, 
                     horizontalalignment='right', fontsize=10, verticalalignment='center')

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
        target_figure = str(derivatives_path.copy().update(
            suffix='step_bad_channels_with_map',
            extension='.png',
            datatype='eeg'
        ))
    except AttributeError:
        # Fallback if copy or update is not implemented
        target_figure = os.path.join(derivatives_path, 'bad_channels_with_topography.png')

    # Save the figure
    fig.savefig(target_figure, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Bad channels with topography plot saved to {target_figure}")

    metadata = {
        "artifact_reports": {
            "creationDateTime": datetime.now().isoformat(),
            "plot_bad_channels_with_topography": Path(target_figure).name
        }
    }

    manage_database(operation='update', update_record={
        'run_id': autoclean_dict['run_id'],
        'metadata': metadata
    })

    return fig


def step_plot_raw_vs_cleaned_overlay(raw_original, raw_cleaned, pipeline, autoclean_dict, suffix=''):
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
        suffix='step_plot_raw_vs_cleaned_overlay',
        extension='.png',
        datatype='eeg'
    ))

    # Save as PNG with high DPI for quality
    fig.savefig(target_figure, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Raw channels overlay full duration plot saved to {target_figure}")

    metadata = {
        "artifact_reports": {
            "creationDateTime": datetime.now().isoformat(),
            "plot_raw_vs_cleaned_overlay": Path(target_figure).name
        }
    }

    manage_database(operation='update', update_record={
        'run_id': autoclean_dict['run_id'],
        'metadata': metadata
    })



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

    metadata = {
        "artifact_reports": {
            "creationDateTime": datetime.now().isoformat(),
            "ica_components_full_duration": Path(target_figure).name
        }
    }

    manage_database(operation='update', update_record={
        'run_id': autoclean_dict['run_id'],
        'metadata': metadata
    })

    return fig

def extended_BAD_LL_noisy_ICs_annotations(raw, pipeline, autoclean_dict, extra_duration=1):

    from collections import OrderedDict

    # Extend each annotation by 1 second on each side
    for ann in raw.annotations:
        new_annotation = OrderedDict([
            ('onset', np.float64(ann['onset'])),
            ('duration', np.float64(ann['duration'])),
            ('description', np.str_(ann['description'])),
            ('orig_time', ann.get('orig_time', None))
        ])
        print(f"'{new_annotation['description']}' goes from {new_annotation['onset']} to {new_annotation['onset'] + new_annotation['duration']}")

    from collections import OrderedDict

    updated_annotations = []
    for annotation in raw.annotations:
        if annotation['description'] == "BAD_LL_noisy_ICs":
            start = annotation['onset']  # Extend start by 1 second
            duration = annotation['duration'] + extra_duration  # Extend duration by extra_duration
            new_ann = mne.Annotations(onset=start, duration=duration, description=annotation['description'])
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
            if combined_orig_time is None and hasattr(ann, 'orig_time'):
                combined_orig_time = ann.orig_time
        else:
            # Handle individual annotation entries
            combined_onset.append(ann['onset'])
            combined_duration.append(ann['duration'])
            combined_description.append(ann['description'])
            if combined_orig_time is None and 'orig_time' in ann:
                combined_orig_time = ann['orig_time']

    # Create new consolidated Annotations object
    new_annotations = mne.Annotations(
        onset=np.array(combined_onset),
        duration=np.array(combined_duration), 
        description=np.array(combined_description),
        orig_time=combined_orig_time
    )

    raw.set_annotations(new_annotations)

    # Extract indices and info for BAD_LL_noisy_ICs after extension
    bad_indices = np.where(raw.annotations.description == "BAD_LL_noisy_ICs")[0]
    n_segments = len(bad_indices)

    if n_segments > 0:
        # Create figure with subplots for each segment
        fig, axes = plt.subplots(n_segments, 1, figsize=(15, 4 * n_segments), sharex=True)
        if n_segments == 1:
            axes = [axes]  # Ensure axes is iterable

        sfreq = raw.info['sfreq']
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
            data, times = raw.get_data(start=start_sample, stop=end_sample, return_times=True)

            # Plot the data
            axes[idx].plot(times, data.T, 'k', linewidth=0.5, alpha=0.5)

            # Highlight the annotation region
            axes[idx].axvspan(onset, onset + duration, color='red', alpha=0.2, label='BAD_LL_noisy_ICs')

            # Add vertical lines at annotation boundaries
            axes[idx].axvline(onset, color='red', linestyle='--', alpha=0.5)
            axes[idx].axvline(onset + duration, color='red', linestyle='--', alpha=0.5)

            axes[idx].set_title(f'Segment {idx + 1}: {onset:.1f}s - {(onset + duration):.1f}s', fontsize=10)
            axes[idx].set_ylabel('Amplitude')
            axes[idx].legend(loc='upper right')

        axes[-1].set_xlabel('Time (s)')

        plt.tight_layout()

        # Save figure
        derivatives_path = pipeline.get_derivative_path(autoclean_dict["bids_path"])
        target_figure = str(derivatives_path.copy().update(
            suffix='bad_ll_noisy_segments',
            extension='.pdf',
            datatype='eeg'
        ))

        fig.savefig(target_figure, dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Add catalog of BAD_LL_noisy_ICs to metadata
    bad_ll_noisy_ics_count = sum(1 for ann in updated_annotations if isinstance(ann, mne.Annotations) and ann.description == "BAD_LL_noisy_ICs") if updated_annotations else 0
    metadata = {
        "extended_BAD_LL_noisy_ICs_annotations": {
            "creationDateTime": datetime.now().isoformat(),
            "extended_BAD_LL_noisy_ICs_annotations": True,
            "extended_BAD_LL_noisy_ICs_annotations_figure": Path(target_figure).name,
            "extra_duration": extra_duration,
            "bad_LL_noisy_ICs_count": bad_ll_noisy_ics_count  # Count of BAD_LL_noisy_ICs
        }
    }

    manage_database(operation='update', update_record={
        'run_id': autoclean_dict['run_id'],
        'metadata': metadata
    })

    return raw


def detect_muscle_beta_focus_robust(epochs, pipeline, autoclean_dict, freq_band=(20, 30), scale_factor=3.0):
    """
    Detect muscle artifacts using a robust measure (median + MAD * scale_factor) 
    focusing only on electrodes labeled as 'OTHER'.
    This reduces forced removal of epochs in very clean data.
    """

    # Ensure data is loaded
    epochs.load_data()

    backup_epochs = epochs.copy()
    
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
    bad_epochs = np.where(max_p2p > threshold)[0].tolist()

    # Combine current bads and new bads
    current_bads = [i for i, log in enumerate(epochs.drop_log) if log]
    combined_bads = sorted(set(current_bads + bad_epochs))
    epochs.drop(bad_epochs, reason='BAD_MOVEMENT')


    metadata = {
        "muscle_beta_focus_robust": {
            "creationDateTime": datetime.now().isoformat(),
            "muscle_beta_focus_robust": True,
            "freq_band": freq_band,
            "scale_factor": scale_factor,
            "bad_epochs": bad_epochs
        }
    }

    manage_database(operation='update', update_record={
        'run_id': autoclean_dict['run_id'],
        'metadata': metadata
    })

    return bad_epochs


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

def step_run_ll_rejection_policy(pipeline, autoclean_dict):
    bids_path = autoclean_dict["bids_path"]

    rejection_policy = step_get_rejection_policy(autoclean_dict)
    cleaned_raw = rejection_policy.apply(pipeline)
    cleaned_raw = extended_BAD_LL_noisy_ICs_annotations(cleaned_raw, pipeline, autoclean_dict, extra_duration=1)

    # Calculate total duration of BAD annotations
    total_bad_duration = 0
    bad_annotation_count = 0
    distinct_annotation_types = set()

    if cleaned_raw.annotations:
        for annotation in cleaned_raw.annotations:
            if annotation['description'].startswith('BAD'):
                total_bad_duration += annotation['duration']
                bad_annotation_count += 1
                distinct_annotation_types.add(annotation['description'])

    plot_bad_channels_with_topography(
        raw_original=pipeline.raw,
        raw_cleaned=cleaned_raw,
        pipeline=pipeline,
        autoclean_dict=autoclean_dict,
        zoom_duration=30,  # Duration for the zoomed-in plot
        zoom_start=0        # Start time for the zoomed-in plot
    )

    metadata = {
        "step_run_ll_rejection_policy": {
            "creationDateTime": datetime.now().isoformat(),
            "rejection_policy": rejection_policy,
            "channelCount": len(cleaned_raw.ch_names),
            "durationSec": int(cleaned_raw.n_times) / cleaned_raw.info["sfreq"],
            "numberSamples": int(cleaned_raw.n_times),
            "bad_annotations_pending": {
                "number_of_annotations": bad_annotation_count,
                "total_duration_seconds": total_bad_duration,
                "total_duration_minutes": round(total_bad_duration / 60, 2),
                "percent_of_recording": round((total_bad_duration / cleaned_raw.times[-1]) * 100, 2) if cleaned_raw.times[-1] > 0 else 0,
                "distinct_annotation_types": list(distinct_annotation_types)
            }
        }
    }

    manage_database(operation='update', update_record={
        'run_id': autoclean_dict['run_id'],
        'metadata': metadata
    })

    return pipeline, cleaned_raw

def prepare_epochs_for_ica(epochs: mne.Epochs, pipeline, autoclean_dict) -> mne.Epochs:
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
    message("info", "Preliminary epoch rejection: ")

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
        "prepare_epochs_for_ica": {
            "creationDateTime": datetime.now().isoformat(),
            "initial_epochs": len(epochs),
            "final_epochs": len(epochs_faster),
            "rejected_epochs": len(bad_epochs),
            "rejection_percent": round((len(bad_epochs) / len(epochs)) * 100, 2),
            "bad_epochs_by_metric": bad_epochs_by_metric_dict,
            "total_bad_epochs": bad_epochs,
            "epoch_duration": epochs.times[-1] - epochs.times[0],
            "samples_per_epoch": epochs.times.shape[0],
            "total_duration_sec": (epochs.times[-1] - epochs.times[0])*len(epochs_faster),
            "total_samples": epochs.times.shape[0] * len(epochs_faster),
            "channel_count": len(epochs.ch_names)
        }
    }

    manage_database(operation='update', update_record={
        'run_id': autoclean_dict['run_id'],
        'metadata': metadata
    })

    return epochs_faster
    
def step_apply_autoreject(epochs, pipeline, autoclean_dict):
    """
    Apply AutoReject to clean epochs.
    
    Args:
        epochs (mne.Epochs): The input epoched EEG data
        pipeline: The pipeline object containing configuration and metadata
        autoclean_dict (dict): Dictionary containing pipeline configuration and metadata
        
    Returns:
        mne.Epochs: The cleaned epochs after AutoReject
    """
    message("info", "Applying AutoReject for artifact rejection.")
    ar = AutoReject()
    epochs_clean = ar.fit_transform(epochs)
    rejected_epochs = len(epochs) - len(epochs_clean)
    message("info", f"Artifacts rejected: {rejected_epochs} epochs removed by AutoReject.")

    metadata = {
        "step_apply_autoreject": {
            "creationDateTime": datetime.now().isoformat(),
            "initial_epochs": len(epochs),
            "final_epochs": len(epochs_clean),
            "rejected_epochs": rejected_epochs,
            "rejection_percent": round((rejected_epochs / len(epochs)) * 100, 2),
            "epoch_duration": epochs.times[-1] - epochs.times[0],
            "samples_per_epoch": epochs.times.shape[0],
            "total_duration_sec": (epochs.times[-1] - epochs.times[0])*len(epochs_clean),
            "total_samples": epochs.times.shape[0] * len(epochs_clean),
            "channel_count": len(epochs.ch_names)
        }
    }

    manage_database(operation='update', update_record={
        'run_id': autoclean_dict['run_id'],
        'metadata': metadata
    })

    return epochs_clean

def step_gfp_clean_epochs(
    epochs: mne.Epochs,
    pipeline,
    autoclean_dict,
    gfp_threshold=3.0,
    number_of_epochs=None,
    random_seed=None
):
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
    message("info", "Starting epoch cleaning process.")

    import random

    # Force preload to avoid RuntimeError
    if not epochs.preload:
        epochs.load_data()
    
    epochs_clean = epochs.copy()
    
    # Define non-scalp electrodes to exclude
    channel_region_map = {
        "E17":"OTHER", "E38":"OTHER", "E43":"OTHER", "E44":"OTHER", "E48":"OTHER", "E49":"OTHER",
        "E56":"OTHER", "E73":"OTHER", "E81":"OTHER", "E88":"OTHER", "E94":"OTHER", "E107":"OTHER",
        "E113":"OTHER", "E114":"OTHER", "E119":"OTHER", "E120":"OTHER", "E121":"OTHER", "E125":"OTHER",
        "E126":"OTHER", "E127":"OTHER", "E128":"OTHER"
    }

    # Get scalp electrode indices (all channels except those in channel_region_map)
    non_scalp_channels = list(channel_region_map.keys())
    all_channels = epochs_clean.ch_names
    scalp_channels = [ch for ch in all_channels if ch not in non_scalp_channels]
    scalp_indices = [epochs_clean.ch_names.index(ch) for ch in scalp_channels]
    
    # Step 2: Calculate Global Field Power (GFP) only for scalp electrodes
    message("info", "Calculating Global Field Power (GFP) for each epoch using only scalp electrodes.")
    gfp = np.sqrt(np.mean(epochs_clean.get_data()[:, scalp_indices, :] ** 2, axis=(1, 2)))  # Shape: (n_epochs,)
    
    # Step 3: Epoch Statistics
    epoch_stats = pd.DataFrame({
        'epoch': np.arange(len(gfp)),
        'gfp': gfp,
        'mean_amplitude': epochs_clean.get_data()[:, scalp_indices, :].mean(axis=(1, 2)),
        'max_amplitude': epochs_clean.get_data()[:, scalp_indices, :].max(axis=(1, 2)),
        'min_amplitude': epochs_clean.get_data()[:, scalp_indices, :].min(axis=(1, 2)),
        'std_amplitude': epochs_clean.get_data()[:, scalp_indices, :].std(axis=(1, 2))
    })
    # Step 4: Remove Outlier Epochs based on GFP
    message("info", "Removing outlier epochs based on GFP z-scores.")
    gfp_mean = epoch_stats['gfp'].mean()
    gfp_std = epoch_stats['gfp'].std()
    z_scores = np.abs((epoch_stats['gfp'] - gfp_mean) / gfp_std)
    good_epochs_mask = z_scores < gfp_threshold
    removed_by_gfp = np.sum(~good_epochs_mask)
    epochs_final = epochs_clean[good_epochs_mask]
    epoch_stats_final = epoch_stats[good_epochs_mask]
    message("info", f"Outlier epochs removed based on GFP: {removed_by_gfp}")
    
    # Step 5: Handle epoch selection with warning if needed
    requested_epochs_exceeded = False
    if number_of_epochs is not None:
        if len(epochs_final) < number_of_epochs:
            warning_msg = (f"Requested number_of_epochs={number_of_epochs} exceeds the available cleaned epochs={len(epochs_final)}. Using all available epochs.")
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
    annotation_types['KEEP'] = good_epochs
    annotation_types['TOTAL'] = total_epochs
    # Create GFP barplot
    plt.figure(figsize=(12, 4))

    # Plot all epochs in red first (marking removed epochs)
    plt.bar(epoch_stats.index, epoch_stats['gfp'], width=0.8, color='red', alpha=0.3)

    # Then overlay kept epochs in blue
    plt.bar(epoch_stats_final.index, epoch_stats_final['gfp'], width=0.8, color='blue')

    plt.xlabel('Epoch Number')
    plt.ylabel('Global Field Power (GFP)')
    plt.title('GFP Values by Epoch (Red = Removed, Blue = Kept)')

    # Save plot using BIDS derivative name
        # Create output path for the PDF report
    derivatives_path = pipeline.get_derivative_path(autoclean_dict["bids_path"])


    plot_fname = derivatives_path.copy().update(
        suffix='gfp',
        extension='png',
        check=False
    )
    plt.savefig(Path(plot_fname), dpi=150, bbox_inches='tight')
    plt.close()

    # Create GFP heatmap with larger figure size and improved readability
    plt.figure(figsize=(30, 18))
    
    # Calculate number of rows and columns for grid layout
    n_epochs = len(epoch_stats)
    n_cols = 8  # Reduced number of columns for larger cells
    n_rows = int(np.ceil(n_epochs / n_cols))
    
    # Create a grid of values
    grid = np.full((n_rows, n_cols), np.nan)
    for i, (idx, gfp) in enumerate(epoch_stats['gfp'].items()):
        row = i // n_cols
        col = i % n_cols
        grid[row, col] = gfp
    
    # Create heatmap with larger spacing between cells
    im = plt.imshow(grid, cmap='RdYlBu_r', aspect='auto')
    plt.colorbar(im, label='GFP Value (×10⁻⁶)', fraction=0.02, pad=0.04)
    
    # Add text annotations with increased font size and spacing
    for i, (idx, gfp) in enumerate(epoch_stats['gfp'].items()):
        row = i // n_cols
        col = i % n_cols
        kept = idx in epoch_stats_final.index
        color = 'black' if kept else 'red'
        plt.text(col, row, f'ID: {idx}\nGFP: {gfp:.1e}', 
                ha='center', va='center', color=color, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, pad=0.8))
    
    # Improve title and labels with larger font sizes
    plt.title('GFP Heatmap by Epoch (Red = Removed, Black = Kept)', fontsize=14, pad=20)
    plt.xlabel('Column', fontsize=12, labelpad=10)
    plt.ylabel('Row', fontsize=12, labelpad=10)
    
    # Adjust layout to prevent text overlap
    plt.tight_layout()
    
    # Save heatmap plot with higher DPI for better quality
    plot_fname = derivatives_path.copy().update(
        suffix='gfp-heatmap',
        extension='png',
        check=False
    )
    plt.savefig(Path(plot_fname), dpi=300, bbox_inches='tight')
    plt.close()

    
    
    metadata = {
        "step_gfp_clean_epochs": {
            "creationDateTime": datetime.now().isoformat(),
            'initial_epochs': len(epochs),
            'final_epochs': len(epochs_final),
            'removed_by_gfp': removed_by_gfp,
            'mean_amplitude': float(epoch_stats_final['mean_amplitude'].mean()),
            'max_amplitude': float(epoch_stats_final['max_amplitude'].max()),
            'min_amplitude': float(epoch_stats_final['min_amplitude'].min()),
            'std_amplitude': float(epoch_stats_final['std_amplitude'].mean()),
            'mean_gfp': float(epoch_stats_final['gfp'].mean()),
            'gfp_threshold': float(gfp_threshold),
            'removed_total': removed_by_gfp,
            'annotation_types': annotation_types,
            'epoch_duration': epochs.times[-1] - epochs.times[0],
            'samples_per_epoch': epochs.times.shape[0],
            'total_duration_sec': (epochs.times[-1] - epochs.times[0])*len(epochs_final),
            'total_samples': epochs.times.shape[0] * len(epochs_final),
            'channel_count': len(epochs.ch_names),
            'scalp_channels_used': scalp_channels,
            'requested_epochs_exceeded': requested_epochs_exceeded
        }
    }

    manage_database(operation='update', update_record={
        'run_id': autoclean_dict['run_id'],
        'metadata': metadata
    })
    
    message("info", "Epoch GFP cleaning process completed.")
    
    return epochs_final

def clean_artifacts_continuous(pipeline, autoclean_dict):

    bids_path = autoclean_dict["bids_path"]


    #step_psd_topo_figure(pipeline.raw, cleaned_raw, pipeline, autoclean_dict)

    # Generate ICA reports
    #generate_ica_reports(pipeline, cleaned_raw, autoclean_dict, duration=60)

    step_plot_raw_vs_cleaned_overlay(pipeline.raw, cleaned_raw, pipeline, autoclean_dict, suffix='')

    step_plot_ica_full(pipeline, autoclean_dict)

    cleaned_raw = extended_BAD_LL_noisy_ICs_annotations(cleaned_raw, pipeline, autoclean_dict, extra_duration=1)

    epochs = mne.make_fixed_length_epochs(cleaned_raw, duration=2, reject_by_annotation=True)
    bad_epochs = detect_muscle_beta_focus_robust(epochs, pipeline, autoclean_dict, freq_band=(20, 30), scale_factor=3.0)
    print(f"Detected {len(bad_epochs)} bad epochs")

    plot_epochs_to_pdf(epochs, pipeline=pipeline, autoclean_dict=autoclean_dict, bad_epochs=bad_epochs)

    # Remove bad epochs detected from muscle artifacts
    if len(bad_epochs) > 0:
        print(f"Removing {len(bad_epochs)} epochs with muscle artifacts...")
        epochs.drop(bad_epochs, reason='muscle')
        print(f"Remaining epochs: {len(epochs)}")
    
    # Store cleaned epochs for later usesave_epochs_to_set
    cleaned_epochs = epochs
    
    save_epochs_to_set(cleaned_epochs, autoclean_dict, 'post_clean_epochs')

    metadata = {
        "step_clean_artifacts_continuous": {
            "creationDateTime": datetime.now().isoformat(),
            "rejection_policy": rejection_policy
        }
    }

    return pipeline, autoclean_dict

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


    manage_database(operation='update', update_record={
        'run_id': autoclean_dict['run_id'],
        'metadata': metadata
    })

    # Log rejection policy details using messaging function
    message("info", "[bold blue]Rejection Policy Settings:[/bold blue]")
    message("info", f"Channel flags to reject: {rejection_policy['ch_flags_to_reject']}")
    message("info", f"Channel cleaning mode: {rejection_policy['ch_cleaning_mode']}")
    message("info", f"Interpolation method: {rejection_policy['interpolate_bads_kwargs']['method']}")
    message("info", f"IC flags to reject: {rejection_policy['ic_flags_to_reject']}")
    message("info", f"IC rejection threshold: {rejection_policy['ic_rejection_threshold']}")
    message("info", f"Remove flagged ICs: {rejection_policy['remove_flagged_ics']}")

    return rejection_policy

def step_create_regular_epochs(cleaned_raw, pipeline, autoclean_dict):

   # Detect Muscle Beta Focus
    epochs = mne.make_fixed_length_epochs(cleaned_raw, duration=2, reject_by_annotation=True)

    bad_epochs = detect_muscle_beta_focus_robust(epochs.copy(), pipeline, autoclean_dict, freq_band=(20, 100), scale_factor=2.0)

    epochs.drop_bad()

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

    message("info", f"\nEpoch Drop Log Summary:")
    message("info", f"Total epochs: {total_epochs}")
    message("info", f"Good epochs: {good_epochs}")
    for annotation, count in annotation_types.items():
        message("info", f"Epochs with {annotation}: {count}")

    # Add good and total to the annotation_types dictionary
    annotation_types['KEEP'] = good_epochs
    annotation_types['TOTAL'] = total_epochs

    metadata = {
        "make_fixed_length_epochs": {
            "creationDateTime": datetime.now().isoformat(),
            "duration": 2,
            "reject_by_annotation": True,
            "number_of_epochs": len(epochs),
            "single_epoch_duration": epochs.times[-1] - epochs.times[0],
            "single_epoch_samples": epochs.times.shape[0],
            "durationSec": (epochs.times[-1] - epochs.times[0])*len(epochs), 
            "numberSamples": epochs.times.shape[0] * len(epochs),
            "channelCount": len(epochs.ch_names),
            "annotation_types": annotation_types
        }
    }

    manage_database(operation='update', update_record={
        'run_id': autoclean_dict['run_id'],
        'metadata': metadata
    })

    return epochs

def process_resting_eyesopen(autoclean_dict: dict) -> None:
    message("info", "Processing resting_eyesopen data...")

    # Import and save raw EEG data
    raw = step_import_raw(autoclean_dict)
    save_raw_to_set(raw, autoclean_dict, 'post_import')

    # Run preprocessing pipeline and save intermediate result
    raw = pre_pipeline_processing(raw, autoclean_dict)
    save_raw_to_set(raw, autoclean_dict, 'post_prepipeline')

    # Create BIDS-compliant paths and filenames
    raw, autoclean_dict = create_bids_path(raw, autoclean_dict)

    raw = step_clean_bad_channels(raw, autoclean_dict)
    save_raw_to_set(raw, autoclean_dict, 'post_bad_channels')

    # Run PyLossless pipeline and save result
    pipeline = step_run_pylossless(autoclean_dict)
    save_raw_to_set(raw, autoclean_dict, 'post_pylossless')

    # Use PyLossless Rejection Policy
    pipeline, cleaned_raw = step_run_ll_rejection_policy(pipeline, autoclean_dict)

    # Plot raw data channels over the full duration, overlaying the original and cleaned data.
    step_plot_raw_vs_cleaned_overlay(pipeline.raw, cleaned_raw, pipeline, autoclean_dict, suffix='')

    # Plot ICA components
    step_plot_ica_full(pipeline, autoclean_dict)

    # Generate Full ICA Reports
    # generate_ica_reports(pipeline, cleaned_raw, autoclean_dict, duration=60)

    epochs = step_create_regular_epochs(cleaned_raw, pipeline, autoclean_dict)

    epochs = prepare_epochs_for_ica(epochs, pipeline, autoclean_dict)

    epochs = step_gfp_clean_epochs(
        epochs,
        pipeline,
        autoclean_dict,
        gfp_threshold=3.0,
        number_of_epochs=None
    ) 

    # epochs = step_apply_autoreject(epochs, pipeline, autoclean_dict)


    save_epochs_to_set(epochs, autoclean_dict, 'post_clean_epochs')




    # # Artifact Rejection
    # pipeline, autoclean_dict = clean_artifacts_continuous(pipeline, autoclean_dict)

    # console.print("[green]✓ Completed[/green]")

def get_run_record(run_id: str) -> dict:
    """Get a run record from the database by run ID.
    
    Args:
        run_id: String ID of the run to retrieve
        
    Returns:
        dict: The run record if found, None if not found
    """
    run_record = manage_database(operation='get_record', run_record={'run_id': run_id})
    return run_record


def entrypoint(unprocessed_file: Union[str, Path] = None, task: str = None, run_id: str = None) -> None:

    manage_database(operation='create_collection')

    if run_id is None:
        run_id = str(ULID())
        run_record = {
            'run_id': run_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'task': task,
            'unprocessed_file': str(unprocessed_file),
            'status': 'unprocessed',
            'success': False,
            'json_file': f"{unprocessed_file.stem}_autoclean_metadata.json",
            'metadata': {}
        }

        run_record['record_id'] = manage_database(operation='store', run_record=run_record)

    else:
        run_id = str(run_id)
        run_record = get_run_record(run_id)
        message("info", f"Resuming run {run_id}")
        message("info", f"Run record: {run_record}")
        return run_record


    try:
        autoclean_dir, autoclean_config_file = validate_environment_variables(run_id)

        autoclean_dict = validate_autoclean_config(autoclean_config_file)

        task = validate_task(task, step_list_tasks(autoclean_dict))
        eeg_system = validate_eeg_system(autoclean_dict['tasks'][task]['settings']['montage']['value'])

        validate_input_file(unprocessed_file)

        step_log_start(unprocessed_file, eeg_system, task, autoclean_config_file)
        
        autoclean_dir, bids_dir, metadata_dir, clean_dir, stage_dir, debug_dir = step_prepare_directories(task, run_id)

        autoclean_dict = {
            'run_id': run_record['run_id'],
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

        manage_database(operation='update', update_record={
            'run_id': run_record['run_id'],
            'metadata': {'entrypoint': autoclean_dict}
        })

        message("info", f"Starting processing for task: {task}")

        ##################################################
        #                                                #
        #         This is the start of the task        #
        #            specific code section               #
        #                                                #
        ##################################################

        if autoclean_dict['task'] == "rest_eyesopen":
            process_resting_eyesopen(autoclean_dict)
            
        run_record['status'] = 'completed'

        # Print record to console
        manage_database(operation='print_record', run_record=run_record)

        # Export run record to JSON file
        json_file = metadata_dir / run_record['json_file']
        with open(json_file, "w") as f:
            json.dump(run_record, f, indent=4)
        message("success", f"Run record exported to {json_file}")

    except Exception as e:
        run_record['status'] = 'failed'
        run_record['error'] = str(e)
        message("error", f"Run {run_record['run_id']} Pipeline failed: {e}")
        raise
    finally:
        manage_database(operation='update', update_record=run_record)
        message("header", f"Run {run_record['run_id']} completed")

def main() -> None:
    message("header", "Initializing Autoclean Pipeline")
    
    # Development Test File
    unprocessed_file = Path("/Users/ernie/Documents/GitHub/spg_analysis_redo/dataset_raw/0170_rest.raw")  
    unprocessed_file = Path("/Users/ernie/Documents/GitHub/spg_analysis_redo/dataset_raw/0006_rest.raw")
    task = "rest_eyesopen"

    try:
        entrypoint(unprocessed_file, task)
        #entrypoint(run_id="01JF8K91SWWYBEKNNZBVM1VJKS")
    except Exception as e:
        message("error", f"Pipeline failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
