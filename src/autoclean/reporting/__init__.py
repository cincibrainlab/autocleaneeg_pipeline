"""Reporting utilities for AutoClean EEG pipeline."""

from autoclean.functions.preprocessing.wavelet_thresholding import (
    WaveletReportResult,
    generate_wavelet_report,
)

from .llm_reporting import (
    EpochStats,
    FilterParams,
    ICAStats,
    LLMClient,
    RunContext,
    create_reports,
    render_methods,
    run_context_from_dict,
)

__all__ = [
    "ICAStats",
    "EpochStats",
    "FilterParams",
    "RunContext",
    "run_context_from_dict",
    "LLMClient",
    "render_methods",
    "create_reports",
    "WaveletReportResult",
    "generate_wavelet_report",
]
