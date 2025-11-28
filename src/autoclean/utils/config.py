# src/autoclean/utils/config.py
"""
This module contains functions for loading and validating the autoclean configuration file.
"""

# pylint: disable=line-too-long
import base64
import hashlib
import zlib
from pathlib import Path

import yaml
from platformdirs import user_config_dir
from schema import And, Optional, Or, Schema

from autoclean.configkit.schema import (
    COMP_REJ_METHODS,
    ICA_METHODS,
    SCHEMA_VERSION,
    THRESHOLD_MODES,
    _ic_flags_valid,
    _is_valid_montage,
    _is_valid_wavelet,
)
from autoclean.utils.logging import message
from autoclean.utils.montage import VALID_MONTAGES


def load_config(config_file: Path = None) -> dict:
    """Deprecated: YAML pipeline configs are no longer supported.

    Use Python task modules with embedded `config` dictionaries.
    This function now raises at call sites to prevent silent fallback.
    """
    raise RuntimeError(
        "YAML-based pipeline configs are removed. Use Python task files with embedded "
        "`config` dicts and validate via autoclean.configkit.schema."
    )


# DEPRECATED: Legacy schema helpers retained temporarily for reference only
# These are superseded by autoclean.configkit.schema


def _legacy_build_task_settings_schema():
    """Schema for Python task module `config` dictionaries.

    Mirrors the canonical template and supports new features (wavelet, component_rejection).
    """
    # Common step helpers
    step_bool = {"enabled": bool}
    step_value_num = {**step_bool, "value": Or(int, float, None)}
    step_value_list = {**step_bool, "value": Or(list, None)}

    return Schema(
        {
            "schema_version": And(str, lambda v: v == SCHEMA_VERSION),
            Optional("ai_reporting"): Or(bool, None),
            # Basic preprocessing
            "resample_step": step_value_num,
            "filtering": {
                "enabled": bool,
                "value": {
                    "l_freq": Or(int, float, None),
                    "h_freq": Or(int, float, None),
                    "notch_freqs": Or(float, int, list[float], list[int], None),
                    "notch_widths": Or(float, int, list[float], list[int], None),
                },
            },
            "drop_outerlayer": step_value_list,
            "eog_step": step_value_list,
            "trim_step": {**step_bool, "value": Or(int, float)},
            "crop_step": {
                "enabled": bool,
                "value": {"start": Or(int, float), "end": Or(int, float, None)},
            },
            # Wavelet thresholding
            Optional("wavelet_threshold"): {
                "enabled": bool,
                "value": {
                    "wavelet": And(str, _is_valid_wavelet),
                    "level": Or(
                        And(Or(int, float), lambda v: v >= 0),
                        And(str, lambda v: v.lower() == "auto"),
                    ),
                    "threshold_mode": Or(*THRESHOLD_MODES),
                    "is_erp": bool,
                    Optional("threshold_scale"): Or(int, float),
                    Optional("psd_fmax"): Or(int, float, None),
                    Optional("bandpass"): Or(list, tuple, None),
                    Optional("filter_kwargs"): Or(dict, None),
                    Optional("picks"): Or(str, list, tuple, None),
                },
            },
            # Referencing and montage
            "reference_step": {"enabled": bool, "value": Or(str, list[str], None)},
            "montage": {
                "enabled": bool,
                "value": Or(And(str, _is_valid_montage), None),
            },
            # ICA
            "ICA": {
                "enabled": bool,
                "value": {
                    "method": Or(*ICA_METHODS),
                    Optional("n_components"): Or(int, float, None),
                    Optional("noise_cov"): Or(dict, None),
                    Optional("random_state"): Or(int, None),
                    Optional("fit_params"): Or(dict, None),
                    Optional("max_iter"): Or(int, str, None),
                    Optional("allow_ref_meg"): Or(bool, None),
                    Optional("decim"): Or(int, None),
                    Optional("temp_highpass_for_ica"): Or(float, None),
                },
            },
            # Unified component rejection
            "component_rejection": {
                "enabled": bool,
                "method": Or(*COMP_REJ_METHODS),
                "value": {
                    "ic_flags_to_reject": And(list, _ic_flags_valid),
                    "ic_rejection_threshold": Or(int, float),
                    Optional("psd_fmax"): Or(int, float, None),
                    Optional("ic_rejection_overrides"): Or(dict, None),
                    Optional("icvision_n_components"): Or(int, None),
                },
            },
            # Epochs
            "epoch_settings": {
                "enabled": bool,
                "value": {"tmin": Or(int, float, None), "tmax": Or(int, float, None)},
                "event_id": Or(dict, None),
                Optional("strip_other_events"): bool,
                Optional("allowed_events"): Or(list[str], None),
                "remove_baseline": {"enabled": bool, "window": Or(list[float], None)},
                "threshold_rejection": {
                    "enabled": bool,
                    "volt_threshold": Or(dict, int, float),
                },
            },
        }
    )


def _legacy_migrate_legacy_task_config(task_config: dict) -> dict:
    """Migrate legacy task config keys to the current schema in-place.

    - ICLabel block -> component_rejection with method="iclabel"
    """
    if "ICLabel" in task_config and "component_rejection" not in task_config:
        iclabel = task_config.pop("ICLabel")
        task_config["component_rejection"] = {
            "enabled": iclabel.get("enabled", True),
            "method": "iclabel",
            "value": {
                "ic_flags_to_reject": iclabel.get("value", {}).get(
                    "ic_flags_to_reject", []
                ),
                "ic_rejection_threshold": iclabel.get("value", {}).get(
                    "ic_rejection_threshold", 0.3
                ),
            },
        }
    return task_config


def _legacy_validate_task_module_config(task_config: dict) -> dict:
    """Validate a Python task module `config` dict against the unified schema.

    Returns the validated (possibly legacy-migrated) config or raises SchemaError.
    """
    migrated = _legacy_migrate_legacy_task_config(dict(task_config))
    schema = _legacy_build_task_settings_schema()
    return schema.validate(migrated)


def validate_signal_processing_params(autoclean_dict: dict, task: str) -> None:
    """Validate signal processing parameters for physical constraints.

    Parameters
    ----------
    autoclean_dict : dict
        Configuration dictionary
    task : str
        Current processing task

    Raises
    ------
    ValueError
        If parameters violate signal processing constraints
    """

    # Validate filtering settings
    filtering_settings = autoclean_dict["tasks"][task]["settings"]["filtering"]
    if filtering_settings["enabled"]:
        l_freq = filtering_settings["value"]["l_freq"]
        h_freq = filtering_settings["value"]["h_freq"]

        if l_freq is not None and h_freq is not None:
            if l_freq >= h_freq:
                message(
                    "error",
                    f"Low-pass filter frequency {l_freq} must be less than high-pass filter frequency {h_freq}",
                )
                raise ValueError(
                    f"Invalid filtering settings: l_freq {l_freq} >= h_freq {h_freq}"
                )

        resampling_settings = autoclean_dict["tasks"][task]["settings"]["resample_step"]
        if resampling_settings["enabled"]:
            resampling_rate = resampling_settings["value"]
            if resampling_rate is not None:
                if resampling_rate <= 0:
                    message(
                        "error",
                        f"Resampling rate {resampling_rate} Hz must be greater than 0",
                    )
                    raise ValueError(f"Invalid resampling rate: {resampling_rate} Hz")
                if l_freq is not None and h_freq is not None:
                    if l_freq >= resampling_rate / 2 or h_freq >= resampling_rate / 2:
                        message(
                            "error",
                            f"Filter frequencies {l_freq} Hz and {h_freq} Hz must be below Nyquist frequency {resampling_rate / 2} Hz",
                        )
                        raise ValueError(
                            f"Filter frequencies {l_freq} Hz and {h_freq} Hz must be below Nyquist frequency {resampling_rate / 2} Hz"
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
    # pylint: disable=line-too-long
    """Validate the EEG system for a given task. Checks if the EEG system is in the VALID_MONTAGES dictionary.

    Parameters
    ----------
    autoclean_dict : dict
        The autoclean configuration dictionary.
    task : str
        The task to validate the EEG system for.

    Returns
    -------
    eeg_system : str
        The validated EEG system.
    """

    # Handle both YAML-based and Python-based task configurations
    if task in autoclean_dict.get("tasks", {}):
        # YAML-based task configuration
        eeg_system = autoclean_dict["tasks"][task]["settings"]["montage"]["value"]
    else:
        # Python-based task - extract from task_config if available
        task_config = autoclean_dict.get("task_config", {})
        if "montage" in task_config and "value" in task_config["montage"]:
            eeg_system = task_config["montage"]["value"]
        else:
            # Default or skip validation for Python tasks without explicit montage
            message(
                "warning",
                f"No montage specified for Python task '{task}', skipping EEG system validation",
            )
            return None

    if eeg_system in VALID_MONTAGES:
        message("success", f"✓ EEG system validated: {eeg_system}")
        return eeg_system
    else:
        error_msg = (
            f"Invalid EEG system: {eeg_system}. Supported: {', '.join(VALID_MONTAGES.keys())}. "
            "To add a new montage, please edit configs/montage.yaml or request it on GitHub issues."
        )
        message("error", error_msg)
        raise ValueError(error_msg)


def hash_and_encode_yaml(content: str | dict, is_file: bool = True) -> tuple[str, str]:
    """Hash and encode a YAML file or dictionary.

    Parameters
    ----------
    content : str or dict
        The content to hash and encode.
    is_file : bool
        Whether the content is a file path.

    Returns
    -------
    file_hash : str
        The hash of the content.
    compressed_encoded : str
        The compressed and encoded content.
    """
    if is_file:
        with open(content, "r", encoding="utf-8") as f:
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


def load_user_config() -> dict:
    """Load user-level configuration including compliance settings.

    Returns
    -------
    user_config : dict
        User configuration dictionary with compliance settings
    """
    user_config_dir_path = Path(user_config_dir("autoclean"))
    user_config_file = user_config_dir_path / "user_config.yaml"

    # Default user configuration
    default_config = {
        "compliance": {
            "enabled": False,
            "auth_provider": None,
            "require_electronic_signatures": False,
        },
        "workspace": {"default_output_dir": None, "auto_backup": True},
    }

    if not user_config_file.exists():
        return default_config

    try:
        with open(user_config_file, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}

        # Merge with defaults to ensure all keys exist
        merged_config = default_config.copy()

        # Safely merge compliance settings
        if "compliance" in user_config and isinstance(user_config["compliance"], dict):
            merged_config["compliance"].update(user_config["compliance"])

        # Safely merge workspace settings
        if "workspace" in user_config and isinstance(user_config["workspace"], dict):
            merged_config["workspace"].update(user_config["workspace"])

        return merged_config

    except Exception as e:
        message("warning", f"Failed to load user config: {e}, using defaults")
        return default_config


def save_user_config(user_config: dict) -> None:
    """Save user-level configuration.

    Parameters
    ----------
    user_config : dict
        User configuration dictionary to save
    """
    user_config_dir_path = Path(user_config_dir("autoclean"))
    user_config_dir_path.mkdir(parents=True, exist_ok=True)
    user_config_file = user_config_dir_path / "user_config.yaml"

    try:
        with open(user_config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(user_config, f, default_flow_style=False, indent=2)
        message("debug", f"User config saved to {user_config_file}")
    except Exception as e:
        message("error", f"Failed to save user config: {e}")


def is_compliance_mode_enabled() -> bool:
    """Check if compliance mode is enabled in user configuration.

    Returns
    -------
    bool
        True if compliance mode is enabled, False otherwise
    """
    try:
        user_config = load_user_config()
        return user_config.get("compliance", {}).get("enabled", False)
    except Exception:
        return False


def is_compliance_mode_permanent() -> bool:
    """Check if compliance mode is permanently enabled (cannot be disabled).

    Returns
    -------
    bool
        True if compliance mode is permanent, False otherwise
    """
    try:
        user_config = load_user_config()
        compliance = user_config.get("compliance", {})
        return compliance.get("enabled", False) and compliance.get("permanent", False)
    except Exception:
        return False


def validate_compliance_mode_change(new_enabled_state: bool) -> tuple[bool, str]:
    """Validate if compliance mode can be changed.

    Parameters
    ----------
    new_enabled_state : bool
        The desired new state for compliance mode

    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message) - is_valid=False if change is not allowed
    """
    try:
        if is_compliance_mode_permanent() and not new_enabled_state:
            return False, "Compliance mode cannot be disabled once permanently enabled"
        return True, ""
    except Exception as e:
        return False, f"Failed to validate compliance mode change: {e}"


def enable_compliance_mode(permanent: bool = False) -> bool:
    """Enable compliance mode.

    Parameters
    ----------
    permanent : bool
        Whether to enable compliance mode permanently (cannot be disabled)

    Returns
    -------
    bool
        True if successfully enabled, False otherwise
    """
    try:
        user_config = load_user_config()
        user_config["compliance"]["enabled"] = True
        if permanent:
            user_config["compliance"]["permanent"] = True
        save_user_config(user_config)
        message(
            "info", f"✓ Compliance mode enabled{' (permanent)' if permanent else ''}"
        )
        return True
    except Exception as e:
        message("error", f"Failed to enable compliance mode: {e}")
        return False


def disable_compliance_mode() -> bool:
    """Disable compliance mode.

    Returns
    -------
    bool
        True if successfully disabled, False otherwise
    """
    try:
        is_valid, error_msg = validate_compliance_mode_change(False)
        if not is_valid:
            message("error", error_msg)
            return False

        user_config = load_user_config()
        user_config["compliance"]["enabled"] = False
        # Don't remove permanent flag - keep it for audit trail
        save_user_config(user_config)
        message("info", "✓ Compliance mode disabled")
        return True
    except Exception as e:
        message("error", f"Failed to disable compliance mode: {e}")
        return False


def get_compliance_status() -> dict:
    """Get current compliance mode status.

    Returns
    -------
    dict
        Dictionary with compliance mode status information
    """
    try:
        user_config = load_user_config()
        compliance = user_config.get("compliance", {})

        return {
            "enabled": compliance.get("enabled", False),
            "permanent": compliance.get("permanent", False),
            "auth_provider": compliance.get("auth_provider"),
            "require_electronic_signatures": compliance.get(
                "require_electronic_signatures", False
            ),
        }
    except Exception as e:
        message("error", f"Failed to get compliance status: {e}")
        return {
            "enabled": False,
            "permanent": False,
            "auth_provider": None,
            "require_electronic_signatures": False,
        }


def decode_compressed_yaml(encoded_str: str) -> dict:
    """Decode a compressed and encoded YAML string.

    Parameters
    ----------
    encoded_str : str
        The compressed and encoded YAML string.

    Returns
    -------
    yaml_dict : dict
        The decoded YAML dictionary.
    """

    compressed_data = base64.b64decode(encoded_str)
    yaml_str = zlib.decompress(compressed_data).decode("utf-8")
    return yaml.safe_load(yaml_str)
