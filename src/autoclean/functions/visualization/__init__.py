"""Visualization and Reporting Functions.

This module contains standalone functions for creating plots and reports
from EEG data processing results. Includes comparison plots, component
visualizations, and summary reports.

Functions
---------
plot_raw_comparison : Plot before/after raw data comparison
plot_ica_components : Visualize ICA components
plot_psd_topography : Create power spectral density topography plots
generate_processing_report : Generate HTML processing report
create_processing_summary : Create JSON processing summary
"""

from .icvision_layouts import (
    plot_component_for_classification,
    plot_components_batch,
    plot_ica_topographies_overview,
    save_ica_data,
)
from .plotting import plot_ica_components, plot_psd_topography, plot_raw_comparison
from .reports import create_processing_summary, generate_processing_report

__all__ = [
    "plot_raw_comparison",
    "plot_ica_components",
    "plot_psd_topography",
    "generate_processing_report",
    "create_processing_summary",
    "plot_component_for_classification",
    "plot_components_batch",
    "plot_ica_topographies_overview",
    "save_ica_data",
]
