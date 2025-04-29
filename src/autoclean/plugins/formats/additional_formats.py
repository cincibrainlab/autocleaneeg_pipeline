# src/autoclean/plugins/formats/additional_formats.py
"""Additional format registrations for AutoClean.

This module registers additional file formats that aren't included
in the core package.
"""

from autoclean.io.import_ import register_format

# Register additional file formats
register_format("xdf", "XDF_FORMAT")  # XDF files (BIDS format)
register_format("edf", "EDF_FORMAT")  # European Data Format
register_format("gdf", "GDF_FORMAT")  # General Data Format
register_format("sqd", "KIT_FORMAT")  # KIT/Yokogawa MEG data
register_format("rda", "RDA_FORMAT")  # RecView Data Acquisition files
