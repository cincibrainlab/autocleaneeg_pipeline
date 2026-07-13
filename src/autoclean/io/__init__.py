"""Input/output helpers for the autoclean pipeline.

Keep package exports lazy so importing a leaf module, such as EEGLAB
provenance helpers, does not trigger unrelated logging or plugin side effects.
"""

from importlib import import_module

_EXPORT_NAMES = [
    "save_stc_to_file",
    "save_raw_to_set",
    "save_epochs_to_set",
    "save_epochs_to_set_chunked",
    "copy_final_files",
    "_get_stage_number",
]

_IMPORT_NAMES = [
    "import_eeg",
    "register_plugin",
    "BaseEEGPlugin",
    "register_format",
    "BaseEventProcessor",
    "register_event_processor",
    "get_event_processor_for_task",
    "normalize_montage_name",
]

_LAZY_EXPORTS = {
    **{name: ("autoclean.io.export", name) for name in _EXPORT_NAMES},
    **{name: ("autoclean.io.import_", name) for name in _IMPORT_NAMES},
}

__all__ = [*_EXPORT_NAMES, *_IMPORT_NAMES]


def __getattr__(name: str):
    """Resolve package exports lazily to avoid import-time side effects."""
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
