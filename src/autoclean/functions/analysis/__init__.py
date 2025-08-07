"""Analysis Functions.

This module contains standalone functions for analyzing processed EEG data.
Includes spectral analysis, connectivity measures, and statistical tests.

Functions
---------
compute_statistical_learning_itc : Compute inter-trial coherence for statistical learning epochs
analyze_itc_bands : Analyze ITC values within specific frequency bands
validate_itc_significance : Test ITC significance using Rayleigh test
compute_itc_confidence_intervals : Compute confidence intervals for ITC values
calculate_word_learning_index : Calculate Word Learning Index (WLI) for statistical learning
extract_itc_at_frequencies : Extract ITC values at specific target frequencies
"""

from .statistical_learning import (
    compute_statistical_learning_itc,
    analyze_itc_bands,
    validate_itc_significance,
    compute_itc_confidence_intervals,
    calculate_word_learning_index,
    extract_itc_at_frequencies,
)

__all__ = [
    "compute_statistical_learning_itc",
    "analyze_itc_bands",
    "validate_itc_significance",
    "compute_itc_confidence_intervals",
    "calculate_word_learning_index",
    "extract_itc_at_frequencies",
]