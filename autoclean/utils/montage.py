"""Utility functions for handling EEG montage mappings and conversions."""

import os
from typing import Dict, List

import yaml

from autoclean.utils.logging import message


def load_valid_montages() -> Dict[str, str]:
    """Load valid montages from configuration file."""
    config_path = os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        ),
        "configs",
        "montages.yaml",
    )
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            return config["valid_montages"]
    except Exception as e:
        message("error", f"Failed to load montages config: {e}")
        return {}


VALID_MONTAGES = load_valid_montages()

# Standard 10-20 to GSN-HydroCel mapping
# Based on official EGI GSN-HydroCel channel maps
GSN_TO_1020_MAPPING = {
    # Frontal midline
    "Fz": "E11",
    "FCz": "E6",
    "Cz": "E129",  # Reference electrode in 129 montage
    # Left frontal
    "F3": "E24",
    "F7": "E33",
    "FC3": "E20",
    # Right frontal
    "F4": "E124",
    "F8": "E122",
    "FC4": "E118",
    # Left central/temporal
    "C3": "E36",
    "T7": "E45",
    "CP3": "E42",
    # Right central/temporal
    "C4": "E104",
    "T8": "E108",
    "CP4": "E93",
    # Parietal midline
    "Pz": "E62",
    "POz": "E68",
    # Left parietal/occipital
    "P3": "E52",
    "P7": "E58",
    "O1": "E70",
    # Right parietal/occipital
    "P4": "E92",
    "P8": "E96",
    "O2": "E83",
}

# Create reverse mapping
_1020_TO_GSN_MAPPING = {v: k for k, v in GSN_TO_1020_MAPPING.items()}


def get_10_20_to_gsn_mapping() -> Dict[str, str]:
    """Get mapping from 10-20 system to GSN-HydroCel channel names."""
    return GSN_TO_1020_MAPPING.copy()


def get_gsn_to_10_20_mapping() -> Dict[str, str]:
    """Get mapping from GSN-HydroCel to 10-20 system channel names."""
    return _1020_TO_GSN_MAPPING.copy()


def convert_channel_names(channels: List[str], montage_type: str) -> List[str]:
    """Convert between 10-20 and GSN-HydroCel channel names."""
    message("info", f"Converting channels: {channels}")
    message("info", f"Montage type: {montage_type}")

    # Always convert from 10-20 to GSN since we're working with standard sets
    mapping = get_10_20_to_gsn_mapping()
    message("info", f"Using 10-20 to GSN mapping: {mapping}")

    # Handle special case for Cz in 124 montage
    if "124" in montage_type and "Cz" in channels:
        message("info", "Using 124 montage, adjusting Cz mapping")
        mapping["Cz"] = "E31"  # Cz is E31 in 124 montage

    converted = []
    for ch in channels:
        if ch in mapping:
            converted.append(mapping[ch])
            message("info", f"Converted {ch} to {mapping[ch]}")
        else:
            message("warning", f"No mapping found for channel {ch}")
            converted.append(ch)  # Keep original if no mapping exists

    message("info", f"Final converted channels: {converted}")
    return converted


def get_standard_set_in_montage(roi_set: str, montage_type: str) -> List[str]:
    """Get standard channel set converted to appropriate montage type.

    Args:
        roi_set: Name of standard channel set ('frontal', 'frontocentral', etc.)
        montage_type: Type of montage ('GSN-HydroCel-128', 'GSN-HydroCel-129', '10-20', etc.)

    Returns:
        List of channel names in appropriate montage format
    """
    # Standard ROI sets in 10-20 system
    STANDARD_SETS = {
        "frontal": ["Fz", "F3", "F4"],
        "frontocentral": ["Fz", "FCz", "Cz", "F3", "F4"],
        "central": ["Cz", "C3", "C4"],
        "temporal": ["T7", "T8"],
        "parietal": ["Pz", "P3", "P4"],
        "occipital": ["O1", "O2"],
        "mmn_standard": ["Fz", "FCz", "Cz", "F3", "F4"],  # Standard MMN analysis set
    }

    if roi_set not in STANDARD_SETS:
        raise ValueError(
            f"Unknown ROI set: {roi_set}. Available sets: {list(STANDARD_SETS.keys())}"
        )

    channels = STANDARD_SETS[roi_set]
    return convert_channel_names(channels, montage_type)


def validate_channel_set(
    channels: List[str], available_channels: List[str]
) -> List[str]:
    """Validate and filter channel list based on available channels.

    Args:
        channels: List of requested channel names
        available_channels: List of actually available channel names

    Returns:
        List of valid channel names
    """
    valid_channels = [ch for ch in channels if ch in available_channels]
    if len(valid_channels) != len(channels):
        missing = set(channels) - set(valid_channels)
        message("warning", f"Some requested channels not found in data: {missing}")

    return valid_channels
