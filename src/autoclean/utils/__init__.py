"""Utility functions and helpers.

Keep package exports lazy so importing a leaf utility module does not trigger
unrelated package-level import cycles.
"""

from importlib import import_module

_LAZY_EXPORTS = {
    "step_convert_to_bids": ("autoclean.utils.bids", "step_convert_to_bids"),
    "step_create_dataset_desc": (
        "autoclean.utils.bids",
        "step_create_dataset_desc",
    ),
    "step_create_participants_json": (
        "autoclean.utils.bids",
        "step_create_participants_json",
    ),
    "step_sanitize_id": ("autoclean.utils.bids", "step_sanitize_id"),
    "load_config": ("autoclean.utils.config", "load_config"),
    "validate_eeg_system": ("autoclean.utils.config", "validate_eeg_system"),
    "manage_database": ("autoclean.utils.database", "manage_database"),
    "get_run_record": ("autoclean.utils.database", "get_run_record"),
    "step_prepare_directories": (
        "autoclean.utils.file_system",
        "step_prepare_directories",
    ),
    "message": ("autoclean.utils.logging", "message"),
    "configure_logger": ("autoclean.utils.logging", "configure_logger"),
    "has_logged_errors": ("autoclean.utils.logging", "has_logged_errors"),
    "VALID_MONTAGES": ("autoclean.utils.montage", "VALID_MONTAGES"),
}

__all__ = [
    "step_convert_to_bids",
    "step_sanitize_id",
    "step_create_dataset_desc",
    "step_create_participants_json",
    "load_config",
    "validate_eeg_system",
    "manage_database",
    "get_run_record",
    "step_prepare_directories",
    "message",
    "configure_logger",
    "has_logged_errors",
    "VALID_MONTAGES",
]


def __getattr__(name: str):
    """Resolve package exports lazily to avoid import-time cycles."""
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
