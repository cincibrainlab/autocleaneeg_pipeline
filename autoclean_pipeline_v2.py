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

import mne
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids, update_sidecar_json
from mne.io.constants import FIFF

from pyprep.find_noisy_channels import NoisyChannels
import pylossless as ll


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
            "bads": raw.info["bads"]
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
                "pylossless_config": pylossless_config
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

def clean_artifacts_continuous(pipeline, autoclean_dict):

    bids_path = autoclean_dict["bids_path"]

    return pipeline, autoclean_dict
    
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

    # Artifact Rejection
    pipeline, autoclean_dict = clean_artifacts_continuous(pipeline, autoclean_dict)

    console.print("[green]✓ Completed[/green]")


def entrypoint(unprocessed_file: Union[str, Path], task: str) -> None:

    manage_database(operation='create_collection')

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

        if task == "rest_eyesopen":
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
    task = "rest_eyesopen"

    try:
        entrypoint(unprocessed_file, task)
    except Exception as e:
        message("error", f"Pipeline failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
