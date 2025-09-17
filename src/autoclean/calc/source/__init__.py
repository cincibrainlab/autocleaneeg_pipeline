"""Modular source-space analysis utilities."""
from __future__ import annotations

from ._compat import FOOOF_AVAILABLE, NETWORK_ANALYSIS_AVAILABLE
from .connectivity import (
    calculate_aec_connectivity,
    calculate_source_connectivity,
    calculate_source_connectivity_list,
    test_connectivity_function,
    test_connectivity_function_list,
)
from .conversion import convert_stc_list_to_eeg, convert_stc_to_eeg
from .estimation import (
    estimate_source_function_epochs,
    estimate_source_function_raw,
)
from .fooof import (
    calculate_fooof_aperiodic,
    calculate_fooof_periodic,
    calculate_vertex_peak_frequencies,
    visualize_fooof_results,
    visualize_peak_frequencies,
)
from .pac import calculate_source_pac
from .pipeline import (
    PipelineContext,
    PipelineData,
    PipelineStep,
    SourceAnalysisPipeline,
    aec_step,
    connectivity_step,
    conversion_step,
    fooof_aperiodic_step,
    fooof_peak_step,
    fooof_periodic_step,
    load_source_estimates_from_directory,
    pac_step,
    psd_step,
    vertex_power_step,
    vertex_psd_step,
)
from .psd import (
    calculate_source_psd,
    calculate_source_psd_list,
    visualize_psd_results,
)
from .vertex import (
    apply_spatial_smoothing,
    calculate_vertex_level_spectral_power,
    calculate_vertex_level_spectral_power_list,
    calculate_vertex_psd_for_fooof,
)

__all__ = [
    "FOOOF_AVAILABLE",
    "NETWORK_ANALYSIS_AVAILABLE",
    "calculate_aec_connectivity",
    "calculate_source_connectivity",
    "calculate_source_connectivity_list",
    "test_connectivity_function",
    "test_connectivity_function_list",
    "convert_stc_list_to_eeg",
    "convert_stc_to_eeg",
    "estimate_source_function_epochs",
    "estimate_source_function_raw",
    "calculate_fooof_aperiodic",
    "calculate_fooof_periodic",
    "calculate_vertex_peak_frequencies",
    "visualize_fooof_results",
    "visualize_peak_frequencies",
    "calculate_source_pac",
    "PipelineContext",
    "PipelineData",
    "PipelineStep",
    "SourceAnalysisPipeline",
    "aec_step",
    "connectivity_step",
    "conversion_step",
    "fooof_aperiodic_step",
    "fooof_peak_step",
    "fooof_periodic_step",
    "load_source_estimates_from_directory",
    "pac_step",
    "psd_step",
    "vertex_power_step",
    "vertex_psd_step",
    "calculate_source_psd",
    "calculate_source_psd_list",
    "visualize_psd_results",
    "apply_spatial_smoothing",
    "calculate_vertex_level_spectral_power",
    "calculate_vertex_level_spectral_power_list",
    "calculate_vertex_psd_for_fooof",
]
