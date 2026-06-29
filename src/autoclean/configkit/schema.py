"""Task config schema and validation (single source of truth).

Exports enums and a validator for Python task module `config` dicts.
"""

from __future__ import annotations

from schema import And, Optional, Or, Schema

from autoclean.utils.montage import VALID_MONTAGES

SCHEMA_VERSION = "2025.09"
MATLAB_ENTRYPOINT_KINDS = ("function", "script")

# Optional: wavelet validation via PyWavelets
try:  # pragma: no cover - optional dep
    import pywt  # type: ignore

    _WAVELET_CHECK_AVAILABLE = True
except Exception:  # pragma: no cover - optional dep
    pywt = None
    _WAVELET_CHECK_AVAILABLE = False


# Allowed enums
THRESHOLD_MODES = ("soft", "hard")
COMP_REJ_METHODS = ("iclabel", "icvision", "hybrid")
ICA_METHODS = ("fastica", "infomax", "picard")
IC_FLAGS = (
    "brain",
    "muscle",
    "eye",
    "eog",
    "heart",
    "line_noise",
    "channel_noise",
    "ch_noise",
    "other",
)


def _is_valid_wavelet(name: str) -> bool:
    if not _WAVELET_CHECK_AVAILABLE:
        return True
    try:
        pywt.Wavelet(name)
        return True
    except Exception:
        return False


def _is_valid_montage(value: str) -> bool:
    return value == "auto" or value in VALID_MONTAGES


def _ic_flags_valid(flags: list) -> bool:
    try:
        return all(flag in IC_FLAGS for flag in flags)
    except Exception:
        return False


def _is_processing_block_key(key: str) -> bool:
    """Check if a key looks like a processing block configuration key.

    Processing blocks are dynamically discovered and can add their own
    config keys. This allows the schema to be flexible about block keys
    while still validating core pipeline config.

    Matches patterns like:
    - apply_*  (e.g., apply_zapline, apply_autoreject)
    - clean_*  (e.g., clean_bad_channels)
    - Other block-like patterns
    """
    if not isinstance(key, str):
        return False

    # Known processing block prefixes
    block_prefixes = ("apply_", "clean_", "run_")
    return key.startswith(block_prefixes)


def _step_bool_descriptor() -> dict:
    return {"enabled": "bool"}


def _step_value_num_descriptor() -> dict:
    return {"enabled": "bool", "value": "number|null"}


def _filtering_descriptor() -> dict:
    return {
        "enabled": "bool",
        "value": {
            "l_freq": "number|null",
            "h_freq": "number|null",
            "notch_freqs": "number|list[number]|None",
            "notch_widths": "number|list[number]|None (optional, defaults to 0.5)",
        },
    }


def _wavelet_descriptor() -> dict:
    return {
        "enabled": "bool",
        "value": {
            "wavelet": "string (PyWavelets name)",
            "level": "integer>=0 or 'auto'",
            "threshold_mode": "'soft'|'hard'",
            "is_erp": "bool",
            "threshold_scale": "number (default 1.0)",
            "psd_fmax": "number|null",
            "bandpass": "[low, high] or None",
            "filter_kwargs": "mapping|None",
            "picks": "str|sequence|None",
        },
    }


def _source_localization_descriptor() -> dict:
    return {
        "enabled": "bool",
        "value": {
            "method": "string (MNE|dSPM|sLORETA)",
            "lambda2": "number (regularization parameter)",
            "pick_ori": "string|null (normal|None)",
            "n_jobs": "integer (parallel jobs)",
            "convert_to_eeg": "bool (convert to 68-channel EEG format)",
        },
    }


def _source_psd_descriptor() -> dict:
    return {
        "enabled": "bool",
        "value": {
            "segment_duration": "number|null (seconds of data to analyze)",
            "n_jobs": "integer (parallel jobs)",
            "generate_plots": "bool (create diagnostic PSD plots)",
        },
    }


def _sensor_psd_descriptor() -> dict:
    return {
        "enabled": "bool",
        "value": {
            "method": "'welch'|'multitaper'",
            "fmin": "number (minimum frequency in Hz)",
            "fmax": "number (maximum frequency in Hz)",
            "picks": "str|sequence|None",
            "n_jobs": "integer (parallel jobs)",
            "n_fft": "integer|None (Welch only)",
            "n_overlap": "integer|None (Welch only)",
            "bandwidth": "number|None (multitaper only)",
            "adaptive": "bool (multitaper only)",
            "low_bias": "bool (multitaper only)",
            "normalization": "string (multitaper only)",
        },
    }


def _source_connectivity_descriptor() -> dict:
    return {
        "enabled": "bool",
        "value": {
            "epoch_length": "number (epoch duration in seconds)",
            "n_epochs": "integer (number of epochs for averaging)",
            "n_jobs": "integer (parallel jobs)",
        },
    }


def _fooof_aperiodic_descriptor() -> dict:
    return {
        "enabled": "bool",
        "value": {
            "fmin": "number (minimum frequency in Hz)",
            "fmax": "number (maximum frequency in Hz)",
            "n_jobs": "integer (parallel jobs)",
            "aperiodic_mode": "string ('fixed' or 'knee')",
        },
    }


def _fooof_periodic_descriptor() -> dict:
    return {
        "enabled": "bool",
        "value": {
            "freq_bands": "dict|None (e.g., {'alpha': (8, 13)})",
            "n_jobs": "integer (parallel jobs)",
            "aperiodic_mode": "string ('fixed' or 'knee')",
        },
    }


def _ica_descriptor() -> dict:
    return {
        "enabled": "bool",
        "value": {
            "method": "'fastica'|'infomax'|'picard'",
            "n_components": "number|null",
            "noise_cov": "dict|None",
            "random_state": "int|null",
            "fit_params": "dict|None",
            "max_iter": "int|'auto'|None",
            "allow_ref_meg": "bool|None",
            "decim": "int|None",
            "temp_highpass_for_ica": "float|None",
        },
    }


def _component_rejection_descriptor() -> dict:
    return {
        "enabled": "bool",
        "method": "'iclabel'|'icvision'|'hybrid'",
        "model_name": "str|None (optional, for icvision)",
        "base_url": "str|None (optional, custom OpenAI endpoint)",
        "value": {
            "ic_flags_to_reject": "list[str]",
            "ic_rejection_threshold": "number",
            "psd_fmax": "number|null",
            "ic_rejection_overrides": "dict|None",
            "icvision_n_components": "int|None",
        },
    }


def _is_non_negative_number(value: object) -> bool:
    try:
        return float(value) >= 0
    except Exception:
        return False


def _is_non_negative_int(value: object) -> bool:
    try:
        return int(value) >= 0 and float(value) == int(value)
    except Exception:
        return False


def _matlab_step_descriptor() -> dict:
    return {
        "enabled": "bool",
        "value": {
            "kind": "'function'|'script'",
            "entrypoint": "string",
            "args": "list|None",
            "paths": "list[str]|None",
            "startup_options": "string",
            "startup_timeout_seconds": "number>=0",
            "license_file": "string|None",
            "nargout": "integer>=0 (function only)",
            "toolbox_requirements": "list[str]|None",
            "outputs": {
                "artifacts_subdir": "string|None",
                "capture_stdout": "bool|None",
            },
        },
    }


def _matlab_fooof_descriptor() -> dict:
    return {
        "enabled": "bool",
        "value": {
            "vhtp_path": "string",
            "eeglab_path": "string",
            "spect_freqs": "[fmin, fmax]",
            "save_fooof_img": "bool",
            "parallel": "bool",
            "startup_options": "string",
            "startup_timeout_seconds": "number>=0",
            "license_file": "string|None",
            "artifacts_subdir": "string|None",
        },
    }


def _autoreject_descriptor() -> dict:
    return {
        "enabled": "bool",
        "value": {
            "n_interpolate": "list[int]",
            "consensus": "list[float]",
            "n_jobs": "int",
            "cv": "int",
            "random_state": "int|null",
            "picks": "str|sequence|None",
            "thresh_method": "str",
        },
    }


def _epoch_descriptor() -> dict:
    return {
        "enabled": "bool",
        "value": {"tmin": "number|null", "tmax": "number|null"},
        "event_id": "dict|null",
        Optional("strip_other_events"): "bool",
        Optional("allowed_events"): "list|null",
        "remove_baseline": {
            "enabled": "bool",
            "window": "[start, end]|None",
        },
        "threshold_rejection": {
            "enabled": "bool",
            "volt_threshold": "dict|number",
        },
    }


def _build_task_settings_schema() -> Schema:
    """Schema for Python task module `config` dictionaries.

    Mirrors the canonical template and supports new features (wavelet, component_rejection).
    """
    # Common step helpers
    step_bool = {"enabled": bool}
    step_value_num = {**step_bool, "value": Or(int, float, None)}
    step_value_list = {**step_bool, "value": Or(list, None)}

    # allow eog_step.value to be either a dict with keys or a simple list of indices
    eog_value_schema = Or(
        {"eog_indices": Or(list[int], None), "eog_drop": Or(bool, None)},
        list,
        None,
    )
    matlab_step_value = {
        "kind": Or(*MATLAB_ENTRYPOINT_KINDS),
        "entrypoint": And(str, len),
        Optional("args"): Or(list, tuple, None),
        Optional("paths"): Or(list[str], tuple[str, ...], None),
        Optional("startup_options"): str,
        Optional("startup_timeout_seconds"): And(
            Or(int, float), _is_non_negative_number
        ),
        Optional("license_file"): Or(str, None),
        Optional("nargout"): And(int, _is_non_negative_int),
        Optional("toolbox_requirements"): Or(list[str], tuple[str, ...], None),
        Optional("outputs"): {
            Optional("artifacts_subdir"): Or(str, None),
            Optional("capture_stdout"): Or(bool, None),
        },
    }
    matlab_fooof_value = {
        "vhtp_path": And(str, len),
        "eeglab_path": And(str, len),
        Optional("spect_freqs"): Or(
            [Or(int, float), Or(int, float)],
            (Or(int, float), Or(int, float)),
        ),
        Optional("save_fooof_img"): bool,
        Optional("parallel"): bool,
        Optional("startup_options"): str,
        Optional("startup_timeout_seconds"): And(
            Or(int, float), _is_non_negative_number
        ),
        Optional("license_file"): Or(str, None),
        Optional("artifacts_subdir"): Or(str, None),
    }

    return Schema(
        {
            "schema_version": And(str, lambda v: v == SCHEMA_VERSION),
            "montage": {
                "enabled": bool,
                "value": Or(And(str, _is_valid_montage), None),
            },
            Optional("ai_reporting"): Or(bool, None),
            Optional("llm_reporting"): {
                Optional("enabled"): Or(bool, None),
                Optional("api_key"): Or(str, None),
                Optional("base_url"): Or(str, None),
                Optional("model"): Or(str, None),
                Optional("temperature"): Or(int, float, None),
                Optional("seed"): Or(int, None),
            },
            Optional("move_flagged_files"): Or(bool, None),
            Optional("incremental_cleanup"): {
                Optional("enabled"): bool,
            },
            Optional("dataset_name"): Or(str, None),
            Optional("automation_mode"): Or(bool, str, None),
            # Basic preprocessing
            "resample_step": step_value_num,
            "filtering": {
                "enabled": bool,
                "value": {
                    "l_freq": Or(int, float, None),
                    "h_freq": Or(int, float, None),
                    "notch_freqs": Or(float, int, list[float], list[int], None),
                    Optional("notch_widths"): Or(
                        float, int, list[float], list[int], None
                    ),
                },
            },
            "drop_outerlayer": step_value_list,
            "eog_step": {**step_bool, "value": eog_value_schema},
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
                Optional("model_name"): Or(str, None),
                Optional("base_url"): Or(str, None),
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
            # AutoReject
            Optional("apply_autoreject"): {
                "enabled": bool,
                "value": {
                    "n_interpolate": list[int],
                    "consensus": list[float],
                    Optional("n_jobs"): int,
                    Optional("cv"): int,
                    Optional("random_state"): Or(int, None),
                    Optional("picks"): Or(str, list, tuple, None),
                    Optional("thresh_method"): str,
                },
            },
            # Source Localization
            Optional("apply_source_localization"): {
                "enabled": bool,
                "value": {
                    Optional("method"): str,
                    Optional("lambda2"): Or(int, float),
                    Optional("pick_ori"): Or(str, None),
                    Optional("n_jobs"): int,
                    Optional("convert_to_eeg"): bool,
                },
            },
            # Source PSD
            Optional("apply_source_psd"): {
                "enabled": bool,
                "value": {
                    Optional("segment_duration"): Or(int, float, None),
                    Optional("n_jobs"): int,
                    Optional("generate_plots"): bool,
                },
            },
            Optional("apply_sensor_psd"): {
                "enabled": bool,
                "value": {
                    Optional("method"): Or("welch", "multitaper"),
                    Optional("fmin"): Or(int, float),
                    Optional("fmax"): Or(int, float),
                    Optional("picks"): Or(str, list, tuple, None),
                    Optional("n_jobs"): int,
                    Optional("n_fft"): Or(int, None),
                    Optional("n_overlap"): Or(int, None),
                    Optional("bandwidth"): Or(int, float, None),
                    Optional("adaptive"): bool,
                    Optional("low_bias"): bool,
                    Optional("normalization"): str,
                },
            },
            # Source Connectivity
            Optional("apply_source_connectivity"): {
                "enabled": bool,
                "value": {
                    Optional("epoch_length"): Or(int, float),
                    Optional("n_epochs"): int,
                    Optional("n_jobs"): int,
                },
            },
            # Specparam Aperiodic
            Optional("apply_fooof_aperiodic"): {
                "enabled": bool,
                "value": {
                    Optional("fmin"): Or(int, float),
                    Optional("fmax"): Or(int, float),
                    Optional("n_jobs"): int,
                    Optional("aperiodic_mode"): str,
                },
            },
            # Specparam Periodic
            Optional("apply_fooof_periodic"): {
                "enabled": bool,
                "value": {
                    Optional("freq_bands"): Or(dict, None),
                    Optional("n_jobs"): int,
                    Optional("aperiodic_mode"): str,
                },
            },
            Optional("apply_matlab"): {
                "enabled": bool,
                "value": matlab_step_value,
            },
            Optional("run_matlab"): {
                "enabled": bool,
                "value": matlab_step_value,
            },
            Optional("apply_matlab_fooof"): {
                "enabled": bool,
                "value": matlab_fooof_value,
            },
            # Flexible processing block keys (apply_*, clean_*, run_*)
            # Allows dynamically discovered blocks to add their own config keys
            Optional(_is_processing_block_key): {
                "enabled": bool,
                "value": object,  # Block-specific, don't validate internal structure
            },
        }
    )


def export_task_schema_layout() -> dict:
    """Return a JSON-serialisable view of the task schema."""

    return {
        "schema_version": SCHEMA_VERSION,
        "tasks": {
            "montage": {
                "enabled": "bool",
                "value": "string (valid montage) | 'auto' | None",
            },
            "ai_reporting": "bool|null",
            "llm_reporting": {
                "enabled": "bool|null",
                "api_key": "str|null",
                "base_url": "str|null",
                "model": "str|null",
                "temperature": "number|null",
                "seed": "int|null",
            },
            "move_flagged_files": "bool|null",
            "resample_step": _step_value_num_descriptor(),
            "filtering": _filtering_descriptor(),
            "drop_outerlayer": {"enabled": "bool", "value": "list|None"},
            "eog_step": {
                "enabled": "bool",
                "value": "dict{eog_indices:list[int]|None, eog_drop:bool|None} | list | None",
            },
            "trim_step": {"enabled": "bool", "value": "number"},
            "crop_step": {
                "enabled": "bool",
                "value": {"start": "number", "end": "number|None"},
            },
            "wavelet_threshold": _wavelet_descriptor(),
            "reference_step": {
                "enabled": "bool",
                "value": "'average' | list[str] | None",
            },
            "ICA": _ica_descriptor(),
            "component_rejection": _component_rejection_descriptor(),
            "epoch_settings": _epoch_descriptor(),
            "apply_autoreject": _autoreject_descriptor(),
            "apply_source_localization": _source_localization_descriptor(),
            "apply_source_psd": _source_psd_descriptor(),
            "apply_sensor_psd": _sensor_psd_descriptor(),
            "apply_source_connectivity": _source_connectivity_descriptor(),
            "apply_fooof_aperiodic": _fooof_aperiodic_descriptor(),
            "apply_fooof_periodic": _fooof_periodic_descriptor(),
            "apply_matlab": _matlab_step_descriptor(),
            "run_matlab": _matlab_step_descriptor(),
            "apply_matlab_fooof": _matlab_fooof_descriptor(),
        },
    }


def migrate_legacy_task_config(task_config: dict) -> dict:
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


def validate_task_module_config(task_config: dict) -> dict:
    """Validate a Python task module `config` dict against the unified schema.

    Returns the validated (possibly legacy-migrated) config or raises SchemaError.
    """
    migrated = migrate_legacy_task_config(dict(task_config))
    schema = _build_task_settings_schema()
    return schema.validate(migrated)
