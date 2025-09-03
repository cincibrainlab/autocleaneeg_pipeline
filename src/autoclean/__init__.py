"""Automated EEG preprocessing pipeline.

This package provides tools for automated EEG data preprocessing,
supporting multiple experimental paradigms and processing workflows.

The package provides both a complete pipeline system (Pipeline) and
standalone signal processing functions that can be used independently.
"""

__version__ = "2.2.1"


def __getattr__(name):
    """Lazy import for Pipeline and standalone functions to avoid loading heavy dependencies."""
    if name == "Pipeline":
        from .core.pipeline import Pipeline

        return Pipeline

    # Lazy imports for standalone functions (will be enabled as functions are implemented)
    # Preprocessing functions
    elif name == "filter_data":
        from .functions.preprocessing import filter_data

        return filter_data
    elif name == "resample_data":
        from .functions.preprocessing import resample_data

        return resample_data
    elif name == "rereference_data":
        from .functions.preprocessing import rereference_data

        return rereference_data
    elif name == "drop_channels":
        from .functions.preprocessing import drop_channels

        return drop_channels
    elif name == "crop_data":
        from .functions.preprocessing import crop_data

        return crop_data
    elif name == "trim_edges":
        from .functions.preprocessing import trim_edges

        return trim_edges
    elif name == "assign_channel_types":
        from .functions.preprocessing import assign_channel_types

        return assign_channel_types

    # Epoching functions
    elif name == "create_regular_epochs":
        from .functions.epoching import create_regular_epochs

        return create_regular_epochs
    elif name == "create_eventid_epochs":
        from .functions.epoching import create_eventid_epochs

        return create_eventid_epochs
    elif name == "create_sl_epochs":
        from .functions.epoching import create_sl_epochs

        return create_sl_epochs
    elif name == "detect_outlier_epochs":
        from .functions.epoching import detect_outlier_epochs

        return detect_outlier_epochs
    elif name == "gfp_clean_epochs":
        from .functions.epoching import gfp_clean_epochs

        return gfp_clean_epochs

    # Artifact functions
    elif name == "detect_bad_channels":
        from .functions.artifacts import detect_bad_channels

        return detect_bad_channels
    elif name == "interpolate_bad_channels":
        from .functions.artifacts import interpolate_bad_channels

        return interpolate_bad_channels
    elif name == "fit_ica":
        from .functions.ica import fit_ica

        return fit_ica
    elif name == "classify_ica_components":
        from .functions.ica import classify_ica_components

        return classify_ica_components
    elif name == "apply_ica_rejection":
        from .functions.ica import apply_ica_rejection

        return apply_ica_rejection
    elif name == "apply_iclabel_rejection":
        from .functions.ica import apply_iclabel_rejection

        return apply_iclabel_rejection

    # Advanced functions
    elif name == "autoreject_epochs":
        from .functions.advanced import autoreject_epochs

        return autoreject_epochs
    elif name == "annotate_noisy_segments":
        from .functions.segment_rejection import annotate_noisy_segments

        return annotate_noisy_segments
    elif name == "annotate_uncorrelated_segments":
        from .functions.segment_rejection import annotate_uncorrelated_segments

        return annotate_uncorrelated_segments

    # Visualization functions
    # elif name == "plot_raw_comparison":
    #     from .functions.visualization import plot_raw_comparison
    #     return plot_raw_comparison
    # elif name == "plot_ica_components":
    #     from .functions.visualization import plot_ica_components
    #     return plot_ica_components
    # elif name == "plot_psd_topography":
    #     from .functions.visualization import plot_psd_topography
    #     return plot_psd_topography
    # elif name == "generate_processing_report":
    #     from .functions.visualization import generate_processing_report
    #     return generate_processing_report

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "Pipeline",
    # Standalone functions (will be enabled as implemented)
    # Preprocessing functions
    "filter_data",
    "resample_data",
    "rereference_data",
    "drop_channels",
    "crop_data",
    "trim_edges",
    "assign_channel_types",
    # Epoching functions
    "create_regular_epochs",
    "create_eventid_epochs",
    "create_sl_epochs",
    "detect_outlier_epochs",
    "gfp_clean_epochs",
    # Artifact functions
    "detect_bad_channels",
    "interpolate_bad_channels",
    # ICA functions
    "fit_ica",
    "classify_ica_components",
    "apply_ica_rejection",
    "apply_iclabel_rejection",
    # Advanced functions
    "autoreject_epochs",
    "annotate_noisy_segments",
    "annotate_uncorrelated_segments",
    # Visualization functions
    # "plot_raw_comparison",
    # "plot_ica_components",
    # "plot_psd_topography",
    # "generate_processing_report",
]
