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

# Preprocessing functions
from .preprocessing import (
    filter_data,
    resample_data,
    rereference_data,
    drop_channels,
    crop_data,
    trim_edges,
    assign_channel_types
)

# Epoching functions  
# from .epoching import (
#     create_regular_epochs,
#     create_eventid_epochs,
#     create_sl_epochs,
#     detect_outlier_epochs,
#     gfp_clean_epochs
# )

# Artifact functions
# from .artifacts import (
#     detect_bad_channels,
#     interpolate_bad_channels,
#     fit_ica,
#     classify_ica_components,
#     apply_ica_rejection
# )

# Visualization functions
# from .visualization import (
#     plot_raw_comparison,
#     plot_ica_components,
#     plot_psd_topography,
#     generate_processing_report
# )

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
    # Will be populated as more functions are implemented
]