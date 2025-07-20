"""Utility functions for handling EEG montage mappings and conversions."""

import os
from typing import Dict, List

import yaml

from autoclean.utils.logging import message

# Handle importlib.resources compatibility
try:
    from importlib import resources

    IMPORTLIB_RESOURCES_AVAILABLE = True
except ImportError:
    # Python < 3.9 compatibility
    try:
        import importlib_resources as resources

        IMPORTLIB_RESOURCES_AVAILABLE = True
    except ImportError:
        IMPORTLIB_RESOURCES_AVAILABLE = False
        resources = None

# Handle configs import - this may be optional
try:
    import configs

    CONFIGS_AVAILABLE = True
except ImportError:
    CONFIGS_AVAILABLE = False
    configs = None


def load_valid_montages() -> Dict[str, str]:
    """Load valid montages from configuration file.

    Returns
    -------
    Dict[str, str]
        Dictionary of valid montages
    """
    try:
        config_data = None
        # Try to load from package resources first (for installed package)
        if IMPORTLIB_RESOURCES_AVAILABLE and CONFIGS_AVAILABLE:
            try:
                config_data = (
                    resources.files(configs)
                    .joinpath("montages.yaml")
                    .read_text(encoding="utf-8")
                )
            except FileNotFoundError:
                pass  # Will fall back to file path

        # Fallback to relative path (for development)
        if config_data is None:
            config_path = os.path.join(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                ),
                "configs",
                "montages.yaml",
            )
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = f.read()

        config = yaml.safe_load(config_data)
        return config["valid_montages"]
    except Exception as e:  # pylint: disable=broad-except
        message("error", f"Failed to load montages config: {e}")
        raise


# Standard montage mappings and validation
#: List of valid montages loaded from MNE-Python
VALID_MONTAGES = load_valid_montages()

#: Standard 10-20 to GSN-HydroCel mapping based on official EGI GSN-HydroCel channel maps
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

#: Reverse mapping from GSN-HydroCel to 10-20 system
_1020_TO_GSN_MAPPING = {v: k for k, v in GSN_TO_1020_MAPPING.items()}


def get_10_20_to_gsn_mapping() -> Dict[str, str]:
    """Get mapping from 10-20 system to GSN-HydroCel channel names.

    Returns
    -------
    Dict[str, str]
        Mapping from 10-20 system to GSN-HydroCel channel names
    """

    return GSN_TO_1020_MAPPING.copy()


def get_gsn_to_10_20_mapping() -> Dict[str, str]:
    """Get mapping from GSN-HydroCel to 10-20 system channel names.

    Returns
    -------
    Dict[str, str]
        Mapping from GSN-HydroCel to 10-20 system channel names
    """
    return _1020_TO_GSN_MAPPING.copy()


def convert_channel_names(channels: List[str], montage_type: str) -> List[str]:
    """Convert between 10-20 and GSN-HydroCel channel names.

    Parameters
    ----------
    channels : List[str]
        List of channel names to convert
    montage_type : str
        Type of montage to convert to

    Returns
    -------
    List[str]
        List of converted channel names
    """
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

    Parameters
    ----------
    roi_set : str
        Name of standard channel set ('frontal', 'frontocentral', etc.)
    montage_type : str
        Type of montage ('GSN-HydroCel-128', 'GSN-HydroCel-129', '10-20', etc.)

    Returns
    -------
    List[str]
        List of channel names in appropriate montage format
    """
    # Standard ROI sets in 10-20 system
    standard_sets = {
        "frontal": ["Fz", "F3", "F4"],
        "frontocentral": ["Fz", "FCz", "Cz", "F3", "F4"],
        "central": ["Cz", "C3", "C4"],
        "temporal": ["T7", "T8"],
        "parietal": ["Pz", "P3", "P4"],
        "occipital": ["O1", "O2"],
        "mmn_standard": ["Fz", "FCz", "Cz", "F3", "F4"],  # Standard MMN analysis set
    }

    if roi_set not in standard_sets:
        raise ValueError(
            f"Unknown ROI set: {roi_set}. Available sets: {list(standard_sets.keys())}"
        )

    channels = standard_sets[roi_set]
    return convert_channel_names(channels, montage_type)


def validate_channel_set(
    channels: List[str], available_channels: List[str]
) -> List[str]:
    """Validate and filter channel list based on available channels.

    Parameters
    ----------
    channels : List[str]
        List of requested channel names
    available_channels : List[str]
        List of actually available channel names

    Returns
    -------
    List[str]
        List of valid channel names
    """
    valid_channels = [ch for ch in channels if ch in available_channels]
    if len(valid_channels) != len(channels):
        missing = set(channels) - set(valid_channels)
        message("warning", f"Some requested channels not found in data: {missing}")

    return valid_channels
