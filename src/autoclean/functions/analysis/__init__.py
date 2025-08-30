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
    analyze_itc_bands,
    calculate_word_learning_index,
    compute_itc_confidence_intervals,
    compute_statistical_learning_itc,
    extract_itc_at_frequencies,
    validate_itc_significance,
)

__all__ = [
    "compute_statistical_learning_itc",
    "analyze_itc_bands",
    "validate_itc_significance",
    "compute_itc_confidence_intervals",
    "calculate_word_learning_index",
    "extract_itc_at_frequencies",
]
