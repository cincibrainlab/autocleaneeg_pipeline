"""AutoClean Standalone Functions.

This module provides standalone signal processing functions that can be used
independently of the AutoClean pipeline infrastructure. All functions accept
MNE data objects and explicit parameters, making them suitable for use in
custom processing workflows.

The functions are organized by category:
- preprocessing: Basic signal processing (filtering, resampling, referencing)
- epoching: Epoch creation and management
- artifacts: Channel detection, ICA, artifact removal
- visualization: Plotting and report generation

Examples
--------
Basic usage of standalone functions:

>>> from autoclean import filter_data, resample_data, create_regular_epochs
>>> filtered_raw = filter_data(raw, l_freq=1.0, h_freq=40.0)
>>> resampled_raw = resample_data(filtered_raw, sfreq=250)
>>> epochs = create_regular_epochs(resampled_raw, tmin=-1, tmax=1)

All functions can also be imported from their specific modules:

>>> from autoclean.functions.preprocessing import filter_data, resample_data
>>> from autoclean.functions.epoching import create_regular_epochs
"""

# Import all standalone functions for top-level access
# Note: These imports will be added as functions are implemented

# Advanced functions
from .advanced import (
    autoreject_epochs,
)

# Analysis functions
from .analysis import (
    compute_statistical_learning_itc,
)

# Artifact functions
from .artifacts import (
    detect_bad_channels,
    interpolate_bad_channels,
)

# Epoching functions
from .epoching import (
    create_eventid_epochs,
    create_regular_epochs,
    create_statistical_learning_epochs,
    detect_outlier_epochs,
    gfp_clean_epochs,
)

# ICA functions
from .ica import (
    apply_ica_component_rejection,
    apply_ica_rejection,
    classify_ica_components,
    fit_ica,
    process_ica_control_sheet,
)

# Preprocessing functions
from .preprocessing import (
    assign_channel_types,
    crop_data,
    drop_channels,
    filter_data,
    rereference_data,
    resample_data,
    trim_edges,
)

# Segment rejection functions
from .segment_rejection import (
    annotate_noisy_segments,
    annotate_uncorrelated_segments,
    detect_dense_oscillatory_artifacts,
)

# Visualization functions
from .visualization import (
    create_processing_summary,
    generate_processing_report,
    plot_ica_components,
    plot_psd_topography,
    plot_raw_comparison,
)

# Define what gets imported with "from autoclean.functions import *"
__all__ = [
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
    "create_statistical_learning_epochs",
    "detect_outlier_epochs",
    "gfp_clean_epochs",
    # Analysis functions
    "compute_statistical_learning_itc",
    # Artifact functions
    "detect_bad_channels",
    "interpolate_bad_channels",
    # Advanced functions
    "autoreject_epochs",
    # Segment rejection functions
    "detect_dense_oscillatory_artifacts",
    "annotate_noisy_segments",
    "annotate_uncorrelated_segments",
    # ICA functions
    "fit_ica",
    "classify_ica_components",
    "apply_ica_component_rejection",
    "apply_ica_rejection",
    "process_ica_control_sheet",
    "apply_iclabel_rejection",
    # Visualization functions
    "plot_raw_comparison",
    "plot_ica_components",
    "plot_psd_topography",
    "generate_processing_report",
    "create_processing_summary",
    # Will be populated as more functions are implemented
]
