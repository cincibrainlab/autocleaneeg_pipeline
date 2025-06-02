# src/autoclean/io/import.py
"""Input functions for EEG data.

This module provides a unified plugin-based architecture for loading and processing EEG data.
Each plugin handles both the data import and montage configuration as a single unit,
making it easier to extend functionality without modifying core code.
"""

import abc
import importlib
import pkgutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, Union

import mne
import numpy as np
import pandas as pd

from autoclean.utils.database import manage_database
from autoclean.utils.logging import message

__all__ = [
    "import_eeg",
    "register_plugin",
    "BaseEEGPlugin",
    "register_format",
    "BaseEventProcessor",
    "register_event_processor",
    "get_event_processor_for_task",
]

# Registry to store format mappings and plugins
_FORMAT_REGISTRY = {}  # Maps extensions to format IDs
_PLUGIN_REGISTRY = {}  # Maps (format_id, montage_name) tuples to plugin classes

# Core built-in formats
_CORE_FORMATS = {
    "set": "EEGLAB_SET",
    "raw": "EGI_RAW",
    "mff": "EGI_RAW",
    "fif": "GENERIC_FIF",
    "vhdr": "BRAINVISION_VHDR",
    "bdf": "BIOSEMI_BDF",
    "cnt": "NEUROSCAN_CNT",
}


def register_format(extension: str, format_id: str) -> None:
    """Register a new file format.

    Args:
        extension: File extension without dot (e.g., 'xyz')
        format_id: Unique identifier for the format (e.g., 'XYZ_FORMAT')
    """
    extension = extension.lower().lstrip(".")
    if extension in _FORMAT_REGISTRY or extension in _CORE_FORMATS:
        message("warning", f"Overriding existing format for extension: {extension}")

    _FORMAT_REGISTRY[extension] = format_id
    message("info", f"Registered file format: {format_id} for extension .{extension}")


def get_format_from_extension(extension: str) -> Optional[str]:
    """Get format ID from file extension."""
    extension = extension.lower().lstrip(".")
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

    @abc.abstractmethod
    def import_and_configure(
        self, file_path: Path, autoclean_dict: dict, preload: bool = True
    ) -> Union[mne.io.Raw, mne.Epochs]:
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

    def process_events(
        self, raw: mne.io.Raw
    ) -> Tuple[Optional[np.ndarray], Optional[Dict], Optional[pd.DataFrame]]:
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
            "plugin_version": getattr(self, "VERSION", "1.0.0"),
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
    plugin_instance = plugin_class()  # noqa: F841

    # Check each format and montage combination
    for format_id in list(_CORE_FORMATS.values()) + list(_FORMAT_REGISTRY.values()):
        # Test some common montages plus check any custom ones that might be registered
        # In a real implementation, we might want a more systematic way to enumerate supported montages
        test_montages = [
            "GSN-HydroCel-129",
            "GSN-HydroCel-124",
            "standard_1020",
            "biosemi64",
            "MEA30",
            "BioSemi-256",
            "CustomCap-64",
        ]

        for montage_name in test_montages:
            if plugin_class.supports_format_montage(format_id, montage_name):
                key = (format_id, montage_name)
                if key in _PLUGIN_REGISTRY:
                    message(
                        "warning",
                        f"Overriding existing plugin for {format_id}, {montage_name}",
                    )
                _PLUGIN_REGISTRY[key] = plugin_class
                message(
                    "info",
                    f"Registered {plugin_class.__name__} for {format_id}, {montage_name}",
                )


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
                module = importlib.import_module(
                    f"autoclean.plugins.eeg_plugins.{name}"
                )
                for item_name in dir(module):
                    item = getattr(module, item_name)
                    if (
                        isinstance(item, type)
                        and issubclass(item, BaseEEGPlugin)
                        and item is not BaseEEGPlugin
                    ):
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
        plugin_class
        for key, plugin_class in _PLUGIN_REGISTRY.items()
        if key[0] == format_id
    ]

    if format_plugins:
        message(
            "warning",
            f"No exact plugin match for {format_id}, {montage_name}. Using compatible format plugin.",
        )
        return format_plugins[0]()

    raise ValueError(
        f"No plugin found for format '{format_id}' and montage '{montage_name}'"
    )


def import_eeg(
    autoclean_dict: dict, preload: bool = True
) -> Union[mne.io.Raw, mne.Epochs]:
    """Import EEG data using the appropriate plugin.

    This function uses a unified plugin-based architecture to import EEG data.

    Parameters
    ----------
        autoclean_dict : dict
            Configuration dictionary
        preload : bool
            Whether to load data into memory

    Returns
    -------
        eeg_data : mne.io.Raw or mne.Epochs
            Imported and configured EEG data

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
        message("header", f"Using plugin: {plugin.__class__.__name__}")

        # Import and configure the data
        eeg_data = plugin.import_and_configure(
            unprocessed_file, autoclean_dict, preload
        )

        is_epochs = isinstance(eeg_data, mne.BaseEpochs)

        # Process events if we have Raw data
        events = event_dict = events_df = None
        if not is_epochs:
            # Basic event extraction from annotations
            events, event_dict, events_df = plugin.process_events(eeg_data)

            # Apply task-specific event processing if specified
            if "task" in autoclean_dict and autoclean_dict["task"]:
                task = autoclean_dict["task"]
                message("info", f"Applying task-specific event processing for: {task}")
                eeg_data = _apply_task_specific_processing(
                    eeg_data, events, events_df, autoclean_dict
                )

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
                **plugin_metadata,  # Include any plugin-specific metadata
            }
        }

        # Add additional metadata for Raw data
        if not is_epochs:
            metadata["import_eeg"].update(
                {
                    "durationSec": int(eeg_data.n_times) / eeg_data.info["sfreq"],
                    "numberSamples": int(eeg_data.n_times),
                    "hasEvents": events is not None,
                }
            )

            # Add event information to metadata if present
            if events is not None:
                metadata["import_eeg"].update(
                    {
                        "event_dict": event_dict,
                        "event_count": len(events),
                        "unique_event_types": list(set(events[:, 2])),
                    }
                )

            if events_df is not None:
                metadata["import_eeg"]["additional_event_info"] = {
                    "variables": list(events_df.columns),
                    "event_count": len(events_df),
                }
        else:
            # Add epoch-specific metadata
            metadata["import_eeg"].update(
                {
                    "n_epochs": len(eeg_data),
                    "tmin": eeg_data.tmin,
                    "tmax": eeg_data.tmax,
                    "baseline": eeg_data.baseline,
                    "durationSec": len(eeg_data) * (eeg_data.tmax - eeg_data.tmin),
                }
            )

        # Update database
        manage_database(
            operation="update",
            update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
        )

        message(
            "success",
            f"✓ EEG data imported successfully as {metadata['import_eeg']['data_type']}",
        )
        return eeg_data

    except Exception as e:
        message("error", f"Failed to import EEG data: {str(e)}")
        raise


# Event processor plugin system
_EVENT_PROCESSOR_REGISTRY = {}  # Maps task names to event processor classes


class BaseEventProcessor(abc.ABC):
    """Abstract base class for event processing plugins.

    Each plugin handles task-specific event processing logic for a particular
    experimental paradigm or data format.
    """

    @classmethod
    @abc.abstractmethod
    def supports_task(cls, task_name: str) -> bool:
        """Check if this processor supports the given task.

        Args:
            task_name: Name of the task

        Returns:
            bool: True if this processor can handle the task, False otherwise
        """

    def _check_config_enabled(
        self, step_name: str, autoclean_dict: dict, default: bool = True
    ) -> bool:
        """Check if a specific processing step is enabled in configuration.

        Args:
            step_name: Name of the step in configuration
            autoclean_dict: Configuration dictionary
            default: Default value if not specified in config

        Returns:
            bool: True if enabled, False if disabled
        """
        # Configuration can be specified in several ways, check all of them
        if step_name in autoclean_dict:
            return autoclean_dict[step_name]
        elif (
            "processing_steps" in autoclean_dict
            and step_name in autoclean_dict["processing_steps"]
        ):
            return autoclean_dict["processing_steps"][step_name]
        return default

    @abc.abstractmethod
    def process_events(
        self,
        raw: mne.io.Raw,
        events: Optional[np.ndarray],
        events_df: Optional[pd.DataFrame],
        autoclean_dict: dict,
    ) -> mne.io.Raw:
        """Process events for a specific task.

        Args:
            raw: Raw EEG data
            events: Event array from MNE
            events_df: DataFrame containing event information
            autoclean_dict: Configuration dictionary

        Returns:
            mne.io.Raw: Raw data with processed events/annotations
        """


def register_event_processor(processor_class: Type[BaseEventProcessor]) -> None:
    """Register a new event processor plugin.

    Args:
        processor_class: Event processor class to register (must inherit from BaseEventProcessor)

    Raises:
        TypeError: If processor_class is not a subclass of BaseEventProcessor
    """
    if not issubclass(processor_class, BaseEventProcessor):
        raise TypeError(
            f"Event processor must inherit from BaseEventProcessor, got {processor_class}"
        )

    # Create an instance to test supported tasks
    processor_instance = processor_class()  # noqa: F841

    # Test with known tasks
    test_tasks = [
        "p300_grael4k",
        "hbcd_mmn",
        "resting_eyes_open",
        "custom_task",
        "bb_long",
        "mouse_xdat_assr",
        "mouse_xdat_chirp",
    ]

    for task_name in test_tasks:
        if processor_class.supports_task(task_name):
            if task_name in _EVENT_PROCESSOR_REGISTRY:
                message(
                    "warning",
                    f"Overriding existing event processor for task: {task_name}",
                )
            _EVENT_PROCESSOR_REGISTRY[task_name] = processor_class
            message(
                "info", f"Registered {processor_class.__name__} for task: {task_name}"
            )


def discover_event_processors() -> None:
    """Discover and register all available event processor plugins."""
    try:
        import autoclean.plugins.event_processors as processors_pkg

        for _, name, is_pkg in pkgutil.iter_modules(processors_pkg.__path__):
            if not is_pkg:
                module = importlib.import_module(
                    f"autoclean.plugins.event_processors.{name}"
                )
                for item_name in dir(module):
                    item = getattr(module, item_name)
                    if (
                        isinstance(item, type)
                        and issubclass(item, BaseEventProcessor)
                        and item is not BaseEventProcessor
                    ):
                        register_event_processor(item)
    except ImportError:
        message(
            "info", "No event processor plugins found, using built-in processors only"
        )


def get_event_processor_for_task(task_name: str) -> Optional[BaseEventProcessor]:
    """Get appropriate event processor for the given task.

    Args:
        task_name: Name of the task

    Returns:
        BaseEventProcessor or None: An instance of the appropriate processor class, or None if not found
    """
    # Ensure processors are discovered
    if not _EVENT_PROCESSOR_REGISTRY:
        discover_event_processors()

    # Try to get an exact match
    if task_name in _EVENT_PROCESSOR_REGISTRY:
        processor_class = _EVENT_PROCESSOR_REGISTRY[task_name]
        return processor_class()

    # If no exact match, try to find a processor that supports the task
    for processor_class in _EVENT_PROCESSOR_REGISTRY.values():
        if processor_class.supports_task(task_name):
            return processor_class()

    return None


# Built-in event processors
class P300EventProcessor(BaseEventProcessor):
    """Event processor for P300 tasks."""

    @classmethod
    def supports_task(cls, task_name: str) -> bool:
        return task_name == "p300_grael4k"

    def process_events(self, raw, events, events_df, autoclean_dict):
        message("info", "Processing P300 task-specific annotations...")
        mapping = {"13": "Standard", "14": "Target"}
        raw.annotations.rename(mapping)
        return raw


class HBCDEventProcessor(BaseEventProcessor):
    """Event processor for HBCD paradigm tasks."""

    @classmethod
    def supports_task(cls, task_name: str) -> bool:
        # Support all HBCD tasks including MMN, VEP, etc.
        task_name = task_name.lower()
        return task_name.startswith("hbcd_") or task_name in ["mmn", "vep"]

    def process_events(self, raw, events, events_df, autoclean_dict):
        task = autoclean_dict.get("task", "").lower()
        message("info", f"Processing {task} task-specific annotations...")

        # For backwards compatibility, direct to the dedicated processor
        # This will be removed once the full implementation is completed
        message("info", f"Using generic HBCD event processor for {task}")

        # Import the dedicated processor to avoid circular imports
        try:
            from autoclean.plugins.event_processors.hbcd_processor import (
                HBCDEventProcessor as DedicatedProcessor,
            )

            processor = DedicatedProcessor()
            return processor.process_events(raw, events, events_df, autoclean_dict)
        except ImportError:
            message(
                "warning",
                "Could not import dedicated HBCD processor, using legacy implementation",
            )

            # Legacy implementation for MMN
            if events_df is not None and "hbcd_mmn" in task:
                if all(
                    col in events_df.columns
                    for col in ["Task", "type", "onset", "Condition"]
                ):
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
                    message(
                        "success",
                        "Successfully processed HBCD annotations using legacy method",
                    )

        return raw


# Register built-in processors
register_event_processor(P300EventProcessor)
register_event_processor(HBCDEventProcessor)


def _apply_task_specific_processing(raw, events, events_df, autoclean_dict):
    """Apply task-specific processing to raw data using the plugin system.

    This function respects configuration toggles from autoclean_config.yaml.
    If 'event_processing_step' is set to False in the config, event processing
    will be skipped.
    """
    # Check if event processing is enabled in config
    event_processing_enabled = autoclean_dict.get("event_processing_step", True)
    if not event_processing_enabled:
        message("info", "✗ Event processing disabled in configuration")
        return raw

    # Check if task is specified
    if "task" not in autoclean_dict or not autoclean_dict["task"]:
        message("info", "No task specified for event processing")
        return raw

    task = autoclean_dict["task"]
    message("info", f"✓ Processing events for task: {task}")

    # Try to get a task-specific processor
    processor = get_event_processor_for_task(task)

    if processor:
        message("info", f"Using event processor: {processor.__class__.__name__}")
        return processor.process_events(raw, events, events_df, autoclean_dict)
    else:
        message("warning", f"No event processor found for task: {task}")
        return raw
