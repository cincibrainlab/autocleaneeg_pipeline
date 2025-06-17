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

from .plotting import plot_raw_comparison, plot_ica_components, plot_psd_topography
from .reports import generate_processing_report, create_processing_summary

__all__ = [
    "plot_raw_comparison",
    "plot_ica_components",
    "plot_psd_topography", 
    "generate_processing_report",
    "create_processing_summary"
]