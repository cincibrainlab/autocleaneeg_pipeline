"""Analysis Functions.

This module contains standalone functions for analyzing processed EEG data.
Includes inter-trial coherence analysis.

Functions
---------
compute_itc : Compute inter-trial coherence for statistical learning epochs
plot_itc : Plot inter-trial coherence for statistical learning epochs
export_itc_csv : Export inter-trial coherence for statistical learning epochs
"""

from .intertrial_coherence import compute_itc, plot_itc, export_itc_csv

__all__ = ["compute_itc", "plot_itc", "export_itc_csv"]