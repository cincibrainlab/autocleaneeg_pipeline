"""Task config schema and validation (single source of truth).

Exports enums and a validator for Python task module `config` dicts.
"""

from __future__ import annotations

from schema import And, Optional, Or, Schema

from autoclean.utils.montage import VALID_MONTAGES

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


def _build_task_settings_schema() -> Schema:
    """Schema for Python task module `config` dictionaries.

    Mirrors the canonical template and supports new features (wavelet, component_rejection).
    """
    # Common step helpers
    step_bool = {"enabled": bool}
    step_value_num = {**step_bool, "value": Or(int, float, None)}
    step_value_list = {**step_bool, "value": Or(list, None)}

    return Schema(
        {
            "montage": {"enabled": bool, "value": Or(And(str, _is_valid_montage), None)},
            Optional("ai_reporting"): Or(bool, None),
            Optional("move_flagged_files"): Or(bool, None),
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
            "eog_step": {
                "enabled": bool,
                "value": {
                    "eog_indices": Or(list[int], None),
                    "eog_drop": Or(bool, None),
                },
            },
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
                    "level": And(Or(int, float), lambda v: v >= 0),
                    "threshold_mode": Or(*THRESHOLD_MODES),
                    "is_erp": bool,
                    Optional("bandpass"): Or(list, tuple, None),
                    Optional("filter_kwargs"): Or(dict, None),
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
                "remove_baseline": {"enabled": bool, "window": Or(list[float], None)},
                "threshold_rejection": {
                    "enabled": bool,
                    "volt_threshold": Or(dict, int, float),
                },
            },
        }
    )


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
