"""Epoching Functions.

This module contains standalone functions for creating and processing epochs
from continuous EEG data. Includes regular epoching, event-based epoching,
and epoch quality assessment.

Functions
---------
create_regular_epochs : Create fixed-length epochs
create_eventid_epochs : Create epochs around specific events
create_sl_epochs : Create statistical learning epochs
detect_outlier_epochs : Identify outlier epochs
gfp_clean_epochs : Clean epochs using global field power
"""

# Import implemented functions
from .regular import create_regular_epochs
from .eventid import create_eventid_epochs
from .statistical import create_sl_epochs
from .quality import detect_outlier_epochs, gfp_clean_epochs

__all__ = [
    "create_regular_epochs",
    "create_eventid_epochs", 
    "create_sl_epochs",
    "detect_outlier_epochs",
    "gfp_clean_epochs"
]