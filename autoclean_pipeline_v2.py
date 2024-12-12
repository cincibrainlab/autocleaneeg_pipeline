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

import mne

logger = logging.getLogger('autoclean')
console = Console()

load_dotenv()

# Single global database connection
db = UnQLite('autoclean.db')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('autoclean.log')]
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

def validate_environment_variables() -> tuple[str, str]:
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
        with open("validation_status.json", "w") as f:
            json.dump(status, f, indent=2)
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
        with open("validation_status.json", "w") as f:
            json.dump(status, f, indent=2)
        raise ValueError(error_msg)
    else:
        message("success", f"AUTOCLEAN_CONFIG: [dim cyan]{autoclean_config}[/dim cyan]")
        status["validate_environment_variables"]["variables"]["AUTOCLEAN_CONFIG"] = autoclean_config

    status["validate_environment_variables"]["status"] = "completed"
    with open("validation_status.json", "w") as f:
        json.dump(status, f, indent=2)
        
    message("success", "Environment variables validated")
    return autoclean_dir, autoclean_config

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

def step_prepare_directories(task: str) -> tuple[Path, Path, Path, Path, Path]:
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
    
    message("success", "Directories ready")
    return autoclean_dir, dirs["bids"], dirs["metadata"], dirs["clean"], dirs["stage"], dirs["debug"]

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
        logger.error(f"Failed to import raw EEG data: {str(e)}")
        console.print("[red]Error importing raw EEG data[/red]")
        raise

def process_resting_eyesopen(autoclean_dict: dict) -> None:
    message("info", "Processing resting_eyesopen data...")

    # Import and save raw EEG data
    raw = step_import_raw(autoclean_dict)
    # save_raw_to_set(raw, autoclean_dict, 'post_import')

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
        autoclean_dir, autoclean_config_file = validate_environment_variables()

        autoclean_dict = validate_autoclean_config(autoclean_config_file)

        task = validate_task(task, step_list_tasks(autoclean_dict))
        eeg_system = validate_eeg_system(autoclean_dict['tasks'][task]['settings']['montage']['value'])

        validate_input_file(unprocessed_file)

        step_log_start(unprocessed_file, eeg_system, task, autoclean_config_file)
        
        autoclean_dir, bids_dir, metadata_dir, clean_dir, stage_dir, debug_dir = step_prepare_directories(task)

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
