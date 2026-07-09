"""Lazy-exported standalone AutoClean functions.

This module intentionally avoids importing the full scientific stack at import
time so lightweight helpers such as the MATLAB wrapper layer remain usable in
minimal environments.
"""

from __future__ import annotations

from importlib import import_module

_FUNCTION_EXPORTS = {
    "autoreject_epochs": ("autoclean.functions.advanced", "autoreject_epochs"),
    "call_matlab": ("autoclean.functions.matlab", "call_matlab"),
    "execute_matlab_config": ("autoclean.functions.matlab", "execute_matlab_config"),
    "run_matlab_file": ("autoclean.functions.matlab", "run_matlab_file"),
    "compute_statistical_learning_itc": (
        "autoclean.functions.analysis",
        "compute_statistical_learning_itc",
    ),
    "detect_bad_channels": ("autoclean.functions.artifacts", "detect_bad_channels"),
    "interpolate_bad_channels": (
        "autoclean.functions.artifacts",
        "interpolate_bad_channels",
    ),
    "create_eventid_epochs": ("autoclean.functions.epoching", "create_eventid_epochs"),
    "create_regular_epochs": ("autoclean.functions.epoching", "create_regular_epochs"),
    "create_statistical_learning_epochs": (
        "autoclean.functions.epoching",
        "create_statistical_learning_epochs",
    ),
    "detect_outlier_epochs": ("autoclean.functions.epoching", "detect_outlier_epochs"),
    "gfp_clean_epochs": ("autoclean.functions.epoching", "gfp_clean_epochs"),
    "apply_ica_component_rejection": (
        "autoclean.functions.ica",
        "apply_ica_component_rejection",
    ),
    "apply_ica_rejection": ("autoclean.functions.ica", "apply_ica_rejection"),
    "classify_ica_components": ("autoclean.functions.ica", "classify_ica_components"),
    "fit_ica": ("autoclean.functions.ica", "fit_ica"),
    "assign_channel_types": (
        "autoclean.functions.preprocessing",
        "assign_channel_types",
    ),
    "crop_data": ("autoclean.functions.preprocessing", "crop_data"),
    "drop_channels": ("autoclean.functions.preprocessing", "drop_channels"),
    "filter_data": ("autoclean.functions.preprocessing", "filter_data"),
    "rereference_data": ("autoclean.functions.preprocessing", "rereference_data"),
    "resample_data": ("autoclean.functions.preprocessing", "resample_data"),
    "trim_edges": ("autoclean.functions.preprocessing", "trim_edges"),
    "detect_dense_oscillatory_artifacts": (
        "autoclean.functions.segment_rejection",
        "detect_dense_oscillatory_artifacts",
    ),
    "annotate_noisy_segments": (
        "autoclean.functions.segment_rejection",
        "annotate_noisy_segments",
    ),
    "annotate_uncorrelated_segments": (
        "autoclean.functions.segment_rejection",
        "annotate_uncorrelated_segments",
    ),
    "create_processing_summary": (
        "autoclean.functions.visualization",
        "create_processing_summary",
    ),
    "generate_processing_report": (
        "autoclean.functions.visualization",
        "generate_processing_report",
    ),
    "plot_ica_components": ("autoclean.functions.visualization", "plot_ica_components"),
    "plot_psd_topography": (
        "autoclean.functions.visualization",
        "plot_psd_topography",
    ),
    "plot_raw_comparison": ("autoclean.functions.visualization", "plot_raw_comparison"),
}

__all__ = list(_FUNCTION_EXPORTS)


def __getattr__(name: str):
    """Load exports on demand to avoid import-time dependency fan-out."""
    try:
        module_name, attr_name = _FUNCTION_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
