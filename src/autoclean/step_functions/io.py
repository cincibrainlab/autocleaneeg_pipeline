# src/autoclean/step_functions/io.py
"""Input/Output functions for EEG data.

This module provides a unified plugin-based architecture for loading and processing EEG data.
Each plugin handles both the data import and montage configuration as a single unit,
making it easier to extend functionality without modifying core code.
"""

import abc
from datetime import datetime
from pathlib import Path
import importlib
import pkgutil
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import mne
import numpy as np
import pandas as pd
import scipy.io as sio

from autoclean.utils.database import manage_database
from autoclean.utils.logging import message

__all__ = [
    "import_eeg", 
    "save_raw_to_set", 
    "save_epochs_to_set", 
    "register_plugin",
    "BaseEEGPlugin",
    "register_format"
]

# Registry to store format mappings and plugins
_FORMAT_REGISTRY = {}  # Maps extensions to format IDs
_PLUGIN_REGISTRY = {}  # Maps (format_id, montage_name) tuples to plugin classes

# Core built-in formats
_CORE_FORMATS = {
    'set': 'EEGLAB_SET',
    'raw': 'EGI_RAW',
    'fif': 'GENERIC_FIF',
    'vhdr': 'BRAINVISION_VHDR',
    'bdf': 'BIOSEMI_BDF',
    'cnt': 'NEUROSCAN_CNT'
}


def register_format(extension: str, format_id: str) -> None:
    """Register a new file format.
    
    Args:
        extension: File extension without dot (e.g., 'xyz')
        format_id: Unique identifier for the format (e.g., 'XYZ_FORMAT')
    """
    extension = extension.lower().lstrip('.')
    if extension in _FORMAT_REGISTRY or extension in _CORE_FORMATS:
        message("warning", f"Overriding existing format for extension: {extension}")
    
    _FORMAT_REGISTRY[extension] = format_id
    message("info", f"Registered file format: {format_id} for extension .{extension}")


def get_format_from_extension(extension: str) -> Optional[str]:
    """Get format ID from file extension."""
    extension = extension.lower().lstrip('.')
    formats = {**_CORE_FORMATS, **_FORMAT_REGISTRY}
    return formats.get(extension)


class BaseEEGPlugin(abc.ABC):
    """Abstract base class for unified EEG data plugins.
    
    Each plugin handles both importing data and configuring the montage
    for a specific combination of file format and EEG system.
    """
    
    @classmethod
    @abc.abstractmethod
    def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
        """Check if this plugin supports the given format and montage combination.
        
        Args:
            format_id: Format identifier
            montage_name: Name of the EEG montage/system
            
        Returns:
            bool: True if this plugin can handle the combination, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def import_and_configure(self, 
                           file_path: Path, 
                           autoclean_dict: dict, 
                           preload: bool = True) -> Union[mne.io.Raw, mne.Epochs]:
        """Import data and configure montage in a single step.
        
        Args:
            file_path: Path to the EEG data file
            autoclean_dict: Configuration dictionary
            preload: Whether to load data into memory
            
        Returns:
            mne.io.Raw or mne.Epochs: Processed EEG data
            
        Raises:
            RuntimeError: If processing fails
        """
        pass
    
    def process_events(self, 
                     raw: mne.io.Raw, 
                     autoclean_dict: dict) -> Tuple[Optional[np.ndarray], 
                                                   Optional[Dict], 
                                                   Optional[pd.DataFrame]]:
        """Process events and annotations in the EEG data.
        
        Args:
            raw: Raw EEG data
            autoclean_dict: Configuration dictionary
            
        Returns:
            tuple: (events, event_id, events_df) - processed events information
        """
        # Default implementation - override for format-specific event processing
        message("info", "Using default event processing")
        try:
            events, event_id = mne.events_from_annotations(raw)
            return events, event_id, None
        except Exception as e:
            message("warning", f"Default event processing failed: {str(e)}")
            return None, None, None
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get additional metadata about this plugin.
        
        Returns:
            dict: Additional metadata to include in the import record
        """
        # Default implementation - override to add custom metadata
        return {
            "plugin_name": self.__class__.__name__,
            "plugin_version": getattr(self, "VERSION", "1.0.0")
        }


def register_plugin(plugin_class: Type[BaseEEGPlugin]) -> None:
    """Register a new EEG plugin.
    
    Args:
        plugin_class: Plugin class to register (must inherit from BaseEEGPlugin)
    
    Raises:
        TypeError: If plugin_class is not a subclass of BaseEEGPlugin
    """
    if not issubclass(plugin_class, BaseEEGPlugin):
        raise TypeError(f"Plugin must inherit from BaseEEGPlugin, got {plugin_class}")
    
    # Create an instance to test supported combinations
    plugin_instance = plugin_class()
    
    # Check each format and montage combination
    for format_id in list(_CORE_FORMATS.values()) + list(_FORMAT_REGISTRY.values()):
        # Test some common montages plus check any custom ones that might be registered
        # In a real implementation, we might want a more systematic way to enumerate supported montages
        test_montages = [
            "GSN-HydroCel-129", "GSN-HydroCel-124", "standard_1020", "biosemi64", 
            "MEA30", "BioSemi-256", "CustomCap-64"
        ]
        
        for montage_name in test_montages:
            if plugin_class.supports_format_montage(format_id, montage_name):
                key = (format_id, montage_name)
                if key in _PLUGIN_REGISTRY:
                    message("warning", f"Overriding existing plugin for {format_id}, {montage_name}")
                _PLUGIN_REGISTRY[key] = plugin_class
                message("info", f"Registered {plugin_class.__name__} for {format_id}, {montage_name}")


def discover_plugins() -> None:
    """Discover and register all available plugins."""
    # Discover format registrations
    try:
        import autoclean.plugins.formats as formats_pkg
        
        for _, name, is_pkg in pkgutil.iter_modules(formats_pkg.__path__):
            if not is_pkg:
                # Simply importing the module will trigger the format registrations
                importlib.import_module(f"autoclean.plugins.formats.{name}")
    except ImportError:
        message("info", "No format registration plugins found")

    # Discover plugins
    try:
        import autoclean.plugins.eeg_plugins as plugins_pkg
        
        for _, name, is_pkg in pkgutil.iter_modules(plugins_pkg.__path__):
            if not is_pkg:
                module = importlib.import_module(f"autoclean.plugins.eeg_plugins.{name}")
                for item_name in dir(module):
                    item = getattr(module, item_name)
                    if (isinstance(item, type) and 
                        issubclass(item, BaseEEGPlugin) and 
                        item is not BaseEEGPlugin):
                        register_plugin(item)
    except ImportError:
        message("warning", "No EEG plugins package found, using built-in plugins only")


def get_plugin_for_combination(format_id: str, montage_name: str) -> BaseEEGPlugin:
    """Get appropriate plugin for the given format and montage.
    
    Args:
        format_id: Format identifier
        montage_name: Name of the EEG montage/system
        
    Returns:
        BaseEEGPlugin: An instance of the appropriate plugin class
        
    Raises:
        ValueError: If no suitable plugin is found
    """
    # Ensure plugins are discovered
    if not _PLUGIN_REGISTRY:
        discover_plugins()
    
    # Try to get an exact match
    key = (format_id, montage_name)
    if key in _PLUGIN_REGISTRY:
        plugin_class = _PLUGIN_REGISTRY[key]
        return plugin_class()
    
    # If no exact match, look for plugins that claim they can handle this combination
    for plugin_class in set(_PLUGIN_REGISTRY.values()):
        if plugin_class.supports_format_montage(format_id, montage_name):
            return plugin_class()
    
    # If still no match, try to find a plugin that supports this format with any montage
    format_plugins = [
        plugin_class for key, plugin_class in _PLUGIN_REGISTRY.items() 
        if key[0] == format_id
    ]
    
    if format_plugins:
        message("warning", f"No exact plugin match for {format_id}, {montage_name}. Using compatible format plugin.")
        return format_plugins[0]()
    
    raise ValueError(f"No plugin found for format '{format_id}' and montage '{montage_name}'")


def import_eeg(autoclean_dict: dict, preload: bool = True) -> Union[mne.io.Raw, mne.Epochs]:
    """Import EEG data using the appropriate plugin.
    
    This function replaces the original import_eeg function but uses a 
    unified plugin-based architecture.
    
    Args:
        autoclean_dict: Configuration dictionary
        preload: Whether to load data into memory
        
    Returns:
        mne.io.Raw or mne.Epochs: Imported and configured EEG data
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If no suitable plugin is found
        RuntimeError: If loading or validation fails
    """
    unprocessed_file = Path(autoclean_dict["unprocessed_file"])
    
    if not unprocessed_file.exists():
        raise FileNotFoundError(f"Input file not found: {unprocessed_file}")
    
    try:
        # Determine file format
        format_id = get_format_from_extension(unprocessed_file.suffix)
        if not format_id:
            raise ValueError(f"Unsupported file format: {unprocessed_file.suffix}")
        
        # Get montage name from configuration
        montage_name = autoclean_dict["eeg_system"]
        
        # Get appropriate plugin
        plugin = get_plugin_for_combination(format_id, montage_name)
        message("info", f"Using plugin: {plugin.__class__.__name__}")
        
        # Import and configure the data
        eeg_data = plugin.import_and_configure(unprocessed_file, autoclean_dict, preload)
        
        # Determine if we have Raw or Epochs
        is_epochs = isinstance(eeg_data, mne.Epochs)
        
        # Process events if we have Raw data
        events = event_dict = events_df = None
        if not is_epochs:
            events, event_dict, events_df = plugin.process_events(eeg_data, autoclean_dict)
            
            # Apply task-specific processing if needed
            if "task" in autoclean_dict and autoclean_dict["task"]:
                task = autoclean_dict["task"]
                message("info", f"Applying task-specific processing for: {task}")
                eeg_data = _apply_task_specific_processing(eeg_data, events, events_df, autoclean_dict)
        
        # Get plugin metadata
        plugin_metadata = plugin.get_metadata()
        
        # Prepare metadata
        metadata = {
            "import_eeg": {
                "import_function": "import_eeg",
                "plugin_used": plugin.__class__.__name__,
                "file_format": format_id,
                "montage_name": montage_name,
                "creationDateTime": datetime.now().isoformat(),
                "unprocessedFile": str(unprocessed_file.name),
                "eegSystem": autoclean_dict["eeg_system"],
                "sampleRate": eeg_data.info["sfreq"],
                "channelCount": len(eeg_data.ch_names),
                "data_type": "epochs" if is_epochs else "raw",
                **plugin_metadata  # Include any plugin-specific metadata
            }
        }
        
        # Add additional metadata for Raw data
        if not is_epochs:
            metadata["import_eeg"].update({
                "durationSec": int(eeg_data.n_times) / eeg_data.info["sfreq"],
                "numberSamples": int(eeg_data.n_times),
                "hasEvents": events is not None,
            })
            
            # Add event information to metadata if present
            if events is not None:
                metadata["import_eeg"].update({
                    "event_dict": event_dict,
                    "event_count": len(events),
                    "unique_event_types": list(set(events[:, 2])),
                })
                
            if events_df is not None:
                metadata["import_eeg"]["additional_event_info"] = {
                    "variables": list(events_df.columns),
                    "event_count": len(events_df)
                }
        else:
            # Add epoch-specific metadata
            metadata["import_eeg"].update({
                "n_epochs": len(eeg_data),
                "tmin": eeg_data.tmin,
                "tmax": eeg_data.tmax,
                "baseline": eeg_data.baseline
            })
            
        # Update database
        manage_database(
            operation="update",
            update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
        )
        
        message("success", f"✓ EEG data imported successfully as {metadata['import_eeg']['data_type']}")
        return eeg_data
        
    except Exception as e:
        message("error", f"Failed to import EEG data: {str(e)}")
        raise


def _apply_task_specific_processing(raw, events, events_df, autoclean_dict):
    """Apply task-specific processing to raw data."""
    task = autoclean_dict["task"]
    
    if task == "p300_grael4k":
        message("info", "Processing P300 task-specific annotations...")
        mapping = {"13": "Standard", "14": "Target"}
        raw.annotations.rename(mapping)
        
    elif task == "hbcd_mmn":
        message("info", "Processing HBCD MMN task-specific annotations...")
        if events_df is not None:
            subset_events_df = events_df[["Task", "type", "onset", "Condition"]]
            new_annotations = mne.Annotations(
                onset=subset_events_df["onset"].values,
                duration=np.zeros(len(subset_events_df)),
                description=[
                    f"{row['Task']}/{row['type']}/{row['Condition']}"
                    for _, row in subset_events_df.iterrows()
                ],
            )
            raw.set_annotations(new_annotations)
            message("success", "Successfully processed HBCD MMN annotations")
            
    # Add more task-specific processing as needed
    
    return raw


# Keep the existing save functions with minor updates to ensure backward compatibility
def _get_stage_number(stage: str, autoclean_dict: Dict[str, Any]) -> str:
    """Get two-digit number based on enabled stages order."""
    enabled_stages = [
        s for s, cfg in autoclean_dict["stage_files"].items() if cfg["enabled"]
    ]
    return f"{enabled_stages.index(stage) + 1:02d}"


def save_raw_to_set(
    raw: mne.io.Raw,
    autoclean_dict: Dict[str, Any],
    stage: str = "post_import",
    output_path: Optional[Path] = None,
) -> Path:
    """Save raw EEG data to file.

    This function saves raw EEG data at various processing stages.
    
    Args:
        raw: Raw EEG data to save
        autoclean_dict: Configuration dictionary
        stage: Processing stage identifier (e.g., "post_import")
        output_path: Optional custom output path. If None, uses config

    Returns:
        Path: Path to the saved file (stage path)

    Raises:
        ValueError: If stage is not configured
        RuntimeError: If saving fails
    """
    if stage not in autoclean_dict["stage_files"]:
        raise ValueError(f"Stage not configured: {stage}")
        
    if not autoclean_dict["stage_files"][stage]["enabled"]:
        message("info", f"Saving disabled for stage: {stage}")
        return None

    suffix = autoclean_dict["stage_files"][stage]["suffix"]
    basename = Path(autoclean_dict["unprocessed_file"]).stem
    stage_num = _get_stage_number(stage, autoclean_dict)

    # Save to stage directory
    if output_path is None:
        output_path = autoclean_dict["stage_dir"]
    subfolder = output_path / f"{stage_num}{suffix}"
    subfolder.mkdir(exist_ok=True)
    stage_path = subfolder / f"{basename}{suffix}_raw.set"

    # Save to both locations for post_comp
    paths = [stage_path]
    if stage == "post_comp":
        clean_path = autoclean_dict["clean_dir"] / f"{basename}{suffix}.set"
        autoclean_dict["clean_dir"].mkdir(exist_ok=True)
        paths.append(clean_path)

    # Save to all paths
    raw.info["description"] = autoclean_dict["run_id"]
    for path in paths:
        try:
            raw.export(path, fmt="eeglab", overwrite=True)
            message("success", f"✓ Saved {stage} file to: {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save {stage} file to {path}: {str(e)}")

    metadata = {
        "save_raw_to_set": {
            "creationDateTime": datetime.now().isoformat(),
            "stage": stage,
            "stage_number": stage_num,
            "outputPath": str(stage_path),
            "suffix": suffix,
            "basename": basename,
            "format": "eeglab",
        }
    }

    run_id = autoclean_dict["run_id"]
    manage_database(
        operation="update", update_record={"run_id": run_id, "metadata": metadata}
    )
    manage_database(
        operation="update_status",
        update_record={"run_id": run_id, "status": f"{stage} completed"},
    )

    return paths[0]  # Return stage path for consistency


def save_epochs_to_set(
    epochs: mne.Epochs,
    autoclean_dict: Dict[str, Any],
    stage: str = "post_clean_epochs",
    output_path: Optional[Path] = None,
) -> Path:
    """Save epoched EEG data to file.

    This function saves epoched EEG data, typically after processing.
    
    Args:
        epochs: Epoched EEG data to save
        autoclean_dict: Configuration dictionary
        stage: Processing stage identifier (default: "post_clean_epochs")
        output_path: Optional custom output path. If None, uses config

    Returns:
        Path: Path to the saved file (stage path)

    Raises:
        ValueError: If stage is not configured
        RuntimeError: If saving fails
    """
    if stage not in autoclean_dict["stage_files"]:
        raise ValueError(f"Stage not configured: {stage}")
        
    if not autoclean_dict["stage_files"][stage]["enabled"]:
        message("info", f"Saving disabled for stage: {stage}")
        return None

    suffix = autoclean_dict["stage_files"][stage]["suffix"]
    basename = Path(autoclean_dict["unprocessed_file"]).stem
    stage_num = _get_stage_number(stage, autoclean_dict)

    # Save to stage directory
    if output_path is None:
        output_path = autoclean_dict["stage_dir"]
    subfolder = output_path / f"{stage_num}{suffix}"
    subfolder.mkdir(exist_ok=True)
    stage_path = subfolder / f"{basename}{suffix}_epo.set"

    # Save to both locations for post_comp
    paths = [stage_path]
    if stage == "post_comp":
        clean_path = autoclean_dict["clean_dir"] / f"{basename}{suffix}.set"
        autoclean_dict["clean_dir"].mkdir(exist_ok=True)
        paths.append(clean_path)

    # Save to all paths
    epochs.info["description"] = autoclean_dict["run_id"]
    epochs.apply_proj()
    for path in paths:
        try:
            epochs.export(path, fmt="eeglab", overwrite=True)
            # Add run_id to each file
            EEG = sio.loadmat(path)
            EEG["etc"] = {}
            EEG["etc"]["run_id"] = autoclean_dict["run_id"]
            sio.savemat(path, EEG, do_compression=False)
            message("success", f"✓ Saved {stage} file to: {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save {stage} file to {path}: {str(e)}")

    metadata = {
        "save_epochs_to_set": {
            "creationDateTime": datetime.now().isoformat(),
            "stage": stage,
            "stage_number": stage_num,
            "outputPaths": [str(p) for p in paths],
            "suffix": suffix,
            "basename": basename,
            "format": "eeglab",
            "n_epochs": len(epochs),
            "tmin": epochs.tmin,
            "tmax": epochs.tmax,
        }
    }

    run_id = autoclean_dict["run_id"]
    manage_database(
        operation="update", update_record={"run_id": run_id, "metadata": metadata}
    )
    manage_database(
        operation="update_status",
        update_record={"run_id": run_id, "status": f"{stage} completed"},
    )

    return paths[0]  # Return stage path for consistency