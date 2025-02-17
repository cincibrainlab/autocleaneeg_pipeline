# src/autoclean/utils/config.py
from pathlib import Path
from typing import Dict, Any
import yaml
import logging
from schema import Schema, Or
import hashlib
import base64
import zlib
from ..utils.logging import message
from .montage import VALID_MONTAGES


def load_config(config_file: Path) -> dict:
    message("info", f"Loading config: {config_file}")
    config_schema = Schema(
        {
            "tasks": {
                str: {
                    "mne_task": str,
                    "description": str,
                    "lossless_config": str,
                    "settings": {
                        "resample_step": {
                            "enabled": bool,
                            "value": Or(int, float, None),
                        },
                        "drop_outerlayer": {"enabled": bool, "value": Or(list, None)},
                        "eog_step": {"enabled": bool, "value": Or(list, None)},
                        "trim_step": {"enabled": bool, "value": Or(int, float)},
                        "crop_step": {
                            "enabled": bool,
                            "value": {
                                "start": Or(int, float),
                                "end": Or(int, float, None),
                            },
                        },
                        "reference_step": {
                            "enabled": bool,
                            "value": Or(str, list[str], None),
                        },
                        "montage": {"enabled": bool, "value": Or(str, None)},
                        "event_id": {"enabled": bool, "value": Or(dict, None)},
                        "epoch_settings": {
                            "enabled": bool,
                            "value": {
                                "tmin": Or(int, float, None),
                                "tmax": Or(int, float, None),
                            },
                            "remove_baseline": {
                                "enabled": bool,
                                "window": Or(list[float], None),
                            },
                            "threshold_rejection": {
                                "enabled": bool,
                                "volt_threshold": Or(dict, int, float),
                            },
                        },
                    },
                    "rejection_policy": {
                        "ch_flags_to_reject": list,
                        "ch_cleaning_mode": str,
                        "interpolate_bads_kwargs": {"method": str},
                        "ic_flags_to_reject": list,
                        "ic_rejection_threshold": float,
                        "remove_flagged_ics": bool,
                    },
                }
            },
            "stage_files": {str: {"enabled": bool, "suffix": str}},
        }
    )

    # Extract the config file path from the config_file
    config_file_path = Path(config_file)

    with open(config_file) as f:
        config = yaml.safe_load(f)
    autoclean_dict = config_schema.validate(config)

    # Validate signal processing parameters for each task
    for task in autoclean_dict["tasks"]:
        autoclean_dict = validate_pylossless_config(
            autoclean_dict, task, config_file_path
        )
        validate_signal_processing_params(autoclean_dict, task)

    return autoclean_dict


def validate_pylossless_config(
    autoclean_dict: dict, task: str, config_file_path: Path
) -> None:
    """Validate and locate the pylossless configuration file for a given task.

    This function checks if the pylossless configuration file exists at the specified path.
    If not found at the absolute path, it attempts to find it relative to the main config
    file directory. Updates the config dictionary with the full path if found.

    Args:
        autoclean_dict: Configuration dictionary containing task settings
        task: Name of the current processing task
        config_file_path: Path to the main configuration file

    Raises:
        FileNotFoundError: If the pylossless config file cannot be found at either the
            specified path or relative to the main config directory

    Example:
        >>> validate_pylossless_config(
        ...     autoclean_dict={"tasks": {"rest": {"lossless_config": "lossless.yaml"}}},
        ...     task="rest",
        ...     config_file_path=Path("/path/to/config.yaml")
        ... )
    """
    lossless_config_path = autoclean_dict["tasks"][task]["lossless_config"]
    message(
        "debug",
        f"[{task}] Validating pylossless config at path: {lossless_config_path}",
    )

    # Check if lossless config exists at specified path
    if not Path(lossless_config_path).exists():
        # Try finding it relative to main config file directory
        message(
            "debug",
            f"[{task}] Trying to find pylossless config relative to main config file directory",
        )
        relative_path = config_file_path.parent / Path(lossless_config_path).name
        if relative_path.exists():
            # Update config dict with full path
            autoclean_dict["tasks"][task]["lossless_config"] = str(relative_path)
            message("success", f"[{task}] Found pylossless config at: {relative_path}")
        else:
            message(
                "error",
                f"[{task}] Could not find pylossless config at {lossless_config_path} or {relative_path}",
            )
            raise FileNotFoundError(f"Missing pylossless config for task {task}")

    return autoclean_dict


def validate_signal_processing_params(autoclean_dict: dict, task: str) -> None:
    """Validate signal processing parameters for physical constraints.
    Args:
        autoclean_dict: Configuration dictionary
        task: Current processing task

    Raises:
        ValueError: If parameters violate signal processing constraints
    """
    # Get sampling rate
    resample_settings = autoclean_dict["tasks"][task]["settings"]["resample_step"]
    if not resample_settings["enabled"] or resample_settings["value"] is None:
        message("error", "Resampling must be enabled and have a valid value")
        raise ValueError("Invalid resampling configuration")

    sampling_rate = resample_settings["value"]
    nyquist_freq = sampling_rate / 2

    # Load lossless config to check filter settings
    lossless_config_path = autoclean_dict["tasks"][task]["lossless_config"]
    message("debug", f"Attempting to load config: {lossless_config_path}")
    try:
        with open(lossless_config_path) as f:
            lossless_config = yaml.safe_load(f)
    except Exception as e:
        message(
            "error",
            f"Failed to validate config ({lossless_config_path}) for task: {task}",
        )
        raise e

    # Validate filter settings
    filter_args = lossless_config.get("filtering", {}).get("filter_args", {})

    # High-pass filter validation (ensures cutoff is below Nyquist)
    h_freq = filter_args.get("h_freq", [])
    if isinstance(h_freq, (int, float)):
        h_freq = [h_freq]
    if any(freq >= nyquist_freq for freq in h_freq):
        message(
            "error",
            f"High-pass filter frequency {h_freq} must be below Nyquist frequency {nyquist_freq} Hz",
        )
        raise ValueError(
            f"Filter frequency {h_freq} exceeds Nyquist frequency {nyquist_freq} Hz"
        )

    # Low-pass filter validation (ensures cutoff is below Nyquist)
    l_freq = filter_args.get("l_freq", [])
    if isinstance(l_freq, (int, float)):
        l_freq = [l_freq]
    if any(freq >= nyquist_freq for freq in l_freq):
        message(
            "error",
            f"Low-pass filter frequency {l_freq} must be below Nyquist frequency {nyquist_freq} Hz",
        )
        raise ValueError(
            f"Filter frequency {l_freq} exceeds Nyquist frequency {nyquist_freq} Hz"
        )

    # Validate epoch settings if enabled
    epoch_settings = autoclean_dict["tasks"][task]["settings"]["epoch_settings"]
    if epoch_settings["enabled"]:
        tmin = epoch_settings["value"]["tmin"]
        tmax = epoch_settings["value"]["tmax"]
        if tmin is not None and tmax is not None:
            if tmax <= tmin:
                message(
                    "error", f"Epoch tmax ({tmax}s) must be greater than tmin ({tmin}s)"
                )
                raise ValueError(f"Invalid epoch times: tmax {tmax}s <= tmin {tmin}s")

    message("debug", f"Signal processing parameters validated for task {task}")


def validate_eeg_system(autoclean_dict: dict, task: str) -> str:
    eeg_system = autoclean_dict["tasks"][task]["settings"]["montage"]["value"]

    if eeg_system in VALID_MONTAGES:
        message("success", f"✓ EEG system validated: {eeg_system}")
        return eeg_system
    else:
        error_msg = f"Invalid EEG system: {eeg_system}. Supported: {', '.join(VALID_MONTAGES.keys())}"
        message("error", error_msg)
        raise ValueError(error_msg)


def hash_and_encode_yaml(content, is_file=True):
    if is_file:
        with open(content, "r") as f:
            yaml_str = f.read()
    else:
        yaml_str = yaml.safe_dump(content, sort_keys=True)

    data = yaml.safe_load(yaml_str)
    canonical_yaml = yaml.safe_dump(data, sort_keys=True)

    # Compute a secure hash of the canonical YAML.
    file_hash = hashlib.sha256(canonical_yaml.encode("utf-8")).hexdigest()

    # Compress and then base64 encode the canonical YAML.
    compressed = zlib.compress(canonical_yaml.encode("utf-8"))
    compressed_encoded = base64.b64encode(compressed).decode("utf-8")

    return file_hash, compressed_encoded


def decode_compressed_yaml(encoded_str):
    compressed_data = base64.b64decode(encoded_str)
    yaml_str = zlib.decompress(compressed_data).decode("utf-8")
    return yaml.safe_load(yaml_str)
