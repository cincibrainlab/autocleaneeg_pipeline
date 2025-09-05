"""Task configuration schema and validator.

This module defines the schema for task-level configuration dictionaries
and provides a validation function with user-friendly logging output.  It
serves as the single source of truth for supported configuration options
and implements a simple versioning system.
"""

from __future__ import annotations

from schema import Optional, Or, Schema, SchemaError

from autoclean.utils.logging import message

# ---------------------------------------------------------------------------
# Configuration version
# ---------------------------------------------------------------------------
CONFIG_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------
STEP_SCHEMA = {"enabled": bool, "value": object}

TASK_CONFIG_SCHEMA = Schema(
    {
        "version": str,
        Optional("move_flagged_files"): bool,
        Optional("resample_step"): {"enabled": bool, "value": Or(int, float, None)},
        Optional("filtering"): {
            "enabled": bool,
            "value": {
                Optional("l_freq"): Or(int, float, None),
                Optional("h_freq"): Or(int, float, None),
                Optional("notch_freqs"): Or(list, int, float, None),
                Optional("notch_widths"): Or(list, int, float, None),
            },
        },
        Optional("drop_outerlayer"): {"enabled": bool, "value": Or(list, None)},
        Optional("eog_step"): {"enabled": bool, "value": Or(list, None)},
        Optional("trim_step"): {"enabled": bool, "value": Or(int, float, None)},
        Optional("crop_step"): {
            "enabled": bool,
            "value": {"start": Or(int, float, None), "end": Or(int, float, None)},
        },
        Optional("reference_step"): {
            "enabled": bool,
            "value": Or(str, list, None),
        },
        Optional("montage"): {"enabled": bool, "value": Or(str, None)},
        Optional("ICA"): {
            "enabled": bool,
            "value": {
                "method": str,
                Optional("n_components"): Or(int, float, None),
                Optional("fit_params"): Or(dict, None),
                Optional("temp_highpass_for_ica"): Or(int, float, None),
            },
        },
        Optional("component_rejection"): {
            "enabled": bool,
            Optional("method"): Or(str, None),
            "value": {
                Optional("ic_flags_to_reject"): Or(list, None),
                Optional("ic_rejection_threshold"): Or(int, float, None),
            },
        },
        Optional("epoch_settings"): {
            "enabled": bool,
            "value": {
                Optional("tmin"): Or(int, float, None),
                Optional("tmax"): Or(int, float, None),
            },
            Optional("event_id"): Or(dict, None),
            Optional("remove_baseline"): {
                "enabled": bool,
                "window": Or(list, None),
            },
            Optional("threshold_rejection"): {
                "enabled": bool,
                "volt_threshold": Or(dict, int, float, None),
            },
        },
    }
)


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------
def validate_task_config(config: dict) -> dict:
    """Validate a task configuration dictionary.

    Parameters
    ----------
    config : dict
        Configuration dictionary defined in a task file.

    Returns
    -------
    dict
        The validated configuration dictionary.

    Raises
    ------
    ValueError
        If the configuration version is missing or unsupported.
    schema.SchemaError
        If the configuration does not match the expected schema.
    """

    if config is None:
        message("warning", "No task configuration provided")
        return {}

    version = config.get("version")
    if version != CONFIG_VERSION:
        message(
            "error",
            f"Config version '{version}' is not supported. Expected '{CONFIG_VERSION}'.",
        )
        raise ValueError(
            f"Unsupported config version: {version}. Expected {CONFIG_VERSION}."
        )

    try:
        validated = TASK_CONFIG_SCHEMA.validate(config)
    except SchemaError as exc:  # pragma: no cover - schema handles messaging
        message("error", f"Task configuration validation failed: {exc}")
        raise

    message("success", "\u2713 Task configuration validated")
    return validated
