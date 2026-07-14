# src/autoclean/io/import.py
"""Input functions for EEG data.

This module provides a unified plugin-based architecture for loading and processing EEG data.
Each plugin handles both the data import and montage configuration as a single unit,
making it easier to extend functionality without modifying core code.
"""

import abc
import importlib
import pkgutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, Union

import mne
import numpy as np
import pandas as pd

from autoclean.io.eeglab_provenance import (
    extract_eeglab_provenance,
    resolve_eeglab_provenance_dir,
    write_eeglab_dataset_artifacts,
    write_eeglab_provenance_artifacts,
)
from autoclean.utils.database import manage_database_conditionally
from autoclean.utils.logging import message

# Optional import for HBCD processor
try:
    from autoclean.plugins.event_processors.hbcd_processor import (
        HBCDEventProcessor as DedicatedProcessor,
    )

    HBCD_PROCESSOR_AVAILABLE = True
except ImportError:
    HBCD_PROCESSOR_AVAILABLE = False

__all__ = [
    "import_eeg",
    "register_plugin",
    "BaseEEGPlugin",
    "register_format",
    "BaseEventProcessor",
    "register_event_processor",
    "get_event_processor_for_task",
    "normalize_montage_name",
]

# Registry to store format mappings and plugins
_FORMAT_REGISTRY = {}  # Maps extensions to format IDs
_PLUGIN_REGISTRY = {}  # Maps (format_id, montage_name) tuples to plugin classes
_PLUGINS_DISCOVERED = False  # Track if plugin discovery has been run
_DISCOVERY_LOCK = threading.Lock()  # Thread-safe plugin discovery

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

_BIOSEMI_MONTAGE_ALIASES = {
    "biosemi-32": "biosemi32",
    "biosemi32": "biosemi32",
    "biosemi-64": "biosemi64",
    "biosemi64": "biosemi64",
    "biosemi-128": "biosemi128",
    "biosemi128": "biosemi128",
    "biosemi-256": "biosemi256",
    "biosemi256": "biosemi256",
}


def normalize_montage_name(montage_name: str) -> str:
    """Normalize supported montage aliases while preserving unknown names."""
    normalized_name = str(montage_name).strip()
    normalized_key = normalized_name.lower()
    return _BIOSEMI_MONTAGE_ALIASES.get(normalized_key, normalized_name)


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
    message("debug", f"Registered file format: {format_id} for extension .{extension}")


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

    def generate_montage_report(
        self,
        raw_before: mne.io.Raw,
        raw_after: mne.io.Raw,
        montage: mne.channels.DigMontage,
        report_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate plugin-specific montage validation report data.

        This method is called by 'montage test' to generate custom
        report sections for plugins that perform complex transformations
        (e.g., channel remapping, dropping, coordinate transformations).

        Args:
            raw_before: Raw data BEFORE plugin processing (original file)
            raw_after: Raw data AFTER plugin processing (transformed)
            montage: The montage that was applied
            report_data: Standard report data dict (read-only reference)

        Returns:
            dict: Additional report sections with optional keys:
                - 'html_sections': List[str] - HTML strings to insert in report
                - 'summary_stats': Dict - Additional statistics to display
                - 'warnings': List[str] - Warning messages
                - 'info_messages': List[str] - Informational messages

        Note:
            If not overridden, returns empty dict (no custom report content).
            Plugins only need to override this if they perform transformations
            that should be explained in the montage validation report.
        """
        # Default implementation - no custom report
        return {}


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

    # Check each format and montage combination (deduplicate format IDs)
    all_formats = set(_CORE_FORMATS.values()) | set(_FORMAT_REGISTRY.values())
    for format_id in all_formats:
        # Test some common montages plus check any custom ones that might be registered
        # In a real implementation, we might want a more systematic way to enumerate supported montages
        test_montages = [
            "GSN-HydroCel-129",
            "GSN-HydroCel-124",
            "standard_1020",
            "BioSemi-32",
            "BioSemi-64",
            "BioSemi-128",
            "BioSemi-256",
            "biosemi32",
            "biosemi64",
            "biosemi128",
            "biosemi256",
            "MEA30",
            "MEA30_EDF",
            "MouseEEGv2_H32",
            "MEA30_MNI",
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
                    "debug",
                    f"Registered {plugin_class.__name__} for {format_id}, {montage_name}",
                )


def discover_plugins() -> None:
    """Discover and register all available plugins in a thread-safe manner."""
    global _PLUGINS_DISCOVERED

    # Use double-checked locking pattern for thread safety
    if _PLUGINS_DISCOVERED:
        return

    with _DISCOVERY_LOCK:
        # Check again inside the lock in case another thread completed discovery
        if _PLUGINS_DISCOVERED:
            return

        message("debug", "Starting thread-safe plugin discovery...")

        # Mark as discovered to prevent re-entry
        _PLUGINS_DISCOVERED = True

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

    message(
        "debug",
        f"Plugin discovery completed. Registered {len(_PLUGIN_REGISTRY)} plugin combinations.",
    )


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
    # Ensure plugins are discovered (will only run once)
    discover_plugins()
    normalized_montage_name = normalize_montage_name(montage_name)

    # Try to get an exact match
    key = (format_id, normalized_montage_name)
    if key in _PLUGIN_REGISTRY:
        plugin_class = _PLUGIN_REGISTRY[key]
        return plugin_class()

    # If no exact match, look for plugins that claim they can handle this combination
    for plugin_class in set(_PLUGIN_REGISTRY.values()):
        if plugin_class.supports_format_montage(format_id, normalized_montage_name):
            return plugin_class()

    # If still no match, try to find a plugin that supports this format with any montage
    format_plugins = [
        plugin_class
        for key, plugin_class in _PLUGIN_REGISTRY.items()
        if key[0] == format_id
    ]

    if format_plugins and format_id == "BIOSEMI_BDF":
        supported_montages = sorted(
            montage
            for plugin_format, montage in _PLUGIN_REGISTRY
            if plugin_format == format_id
        )
        raise ValueError(
            f"No plugin found for format '{format_id}' and montage '{montage_name}' "
            f"(normalized to '{normalized_montage_name}'). "
            f"Supported montages for {format_id}: {', '.join(supported_montages)}"
        )

    if format_plugins:
        message(
            "warning",
            f"No exact plugin match for {format_id}, {montage_name}. Using compatible format plugin.",
        )
        return format_plugins[0]()

    raise ValueError(
        f"No plugin found for format '{format_id}' and montage '{montage_name}'"
    )


def find_plugin_for_combination(
    format_id: str, montage_name: str
) -> Optional[BaseEEGPlugin]:
    """Find a plugin for the given format and montage combination.

    Similar to get_plugin_for_combination, but returns None if no plugin
    is found instead of raising an exception. Useful for optional plugin
    features like montage test reports.

    Args:
        format_id: Format identifier
        montage_name: Name of the EEG montage/system

    Returns:
        BaseEEGPlugin instance if found, None otherwise
    """
    try:
        return get_plugin_for_combination(format_id, montage_name)
    except ValueError:
        return None


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
        eeglab_provenance = None
        dataset_artifact_paths = {}
        if format_id == "EEGLAB_SET":
            try:
                eeglab_provenance = extract_eeglab_provenance(unprocessed_file)
                provenance_dir = resolve_eeglab_provenance_dir(autoclean_dict)
                artifact_paths = write_eeglab_provenance_artifacts(
                    eeglab_provenance,
                    provenance_dir,
                    unprocessed_file.stem,
                )
                message(
                    "success",
                    f"✓ EEGLAB provenance summary written to {artifact_paths['json']}",
                )
            except Exception as provenance_error:  # pylint: disable=broad-except
                message(
                    "warning",
                    f"EEGLAB provenance summary unavailable: {provenance_error}",
                )
            else:
                try:
                    dataset_artifact_paths = write_eeglab_dataset_artifacts(
                        provenance_dir
                    )
                except Exception as dataset_error:  # pylint: disable=broad-except
                    message(
                        "warning",
                        f"EEGLAB dataset provenance summary unavailable: {dataset_error}",
                    )

        # Prepare metadata
        metadata = {
            "import_eeg": {
                "import_function": "import_eeg",
                "plugin_used": plugin.__class__.__name__,
                "file_format": format_id,
                "montage_name": normalize_montage_name(montage_name),
                "creationDateTime": datetime.now().isoformat(),
                "unprocessedFile": str(unprocessed_file.name),
                "eegSystem": autoclean_dict["eeg_system"],
                "sampleRate": eeg_data.info["sfreq"],
                "channelCount": len(eeg_data.ch_names),
                "originalChannelNames": list(
                    eeg_data.ch_names
                ),  # Store original channel names before any removals
                "data_type": "epochs" if is_epochs else "raw",
                **plugin_metadata,  # Include any plugin-specific metadata
            }
        }
        if format_id == "EEGLAB_SET":
            metadata["import_eeg"]["eeglab_provenance"] = {
                "available": eeglab_provenance is not None,
                "schema_version": (
                    eeglab_provenance.get("schema_version")
                    if eeglab_provenance
                    else None
                ),
                "artifact_paths": (
                    eeglab_provenance.get("artifact_paths") if eeglab_provenance else {}
                ),
                "dataset_artifact_paths": dataset_artifact_paths,
                "summary_row": (
                    eeglab_provenance.get("summary_row") if eeglab_provenance else {}
                ),
            }
            per_file_paths = metadata["import_eeg"]["eeglab_provenance"][
                "artifact_paths"
            ]
            metadata["import_eeg"]["artifact_reports"] = {
                **{
                    f"eeglab_provenance_{key}": value
                    for key, value in per_file_paths.items()
                },
                **{
                    f"eeglab_dataset_provenance_{key}": value
                    for key, value in dataset_artifact_paths.items()
                },
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
        manage_database_conditionally(
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
_EVENT_PROCESSORS_DISCOVERED = False  # Track if event processor discovery has been run
_EVENT_DISCOVERY_LOCK = threading.Lock()  # Thread-safe event processor discovery


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
                "debug", f"Registered {processor_class.__name__} for task: {task_name}"
            )


def discover_event_processors() -> None:
    """Discover and register all available event processor plugins in a thread-safe manner."""
    global _EVENT_PROCESSORS_DISCOVERED

    # Use double-checked locking pattern for thread safety
    if _EVENT_PROCESSORS_DISCOVERED:
        return

    with _EVENT_DISCOVERY_LOCK:
        # Check again inside the lock in case another thread completed discovery
        if _EVENT_PROCESSORS_DISCOVERED:
            return

        message("debug", "Starting thread-safe event processor discovery...")

        # Mark as discovered to prevent re-entry
        _EVENT_PROCESSORS_DISCOVERED = True

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

    message(
        "debug",
        f"Event processor discovery completed. Registered {len(_EVENT_PROCESSOR_REGISTRY)} processors.",
    )

    # Built-in processors are now handled through plugin discovery
    # P300EventProcessor and HBCDEventProcessor are defined as plugins and auto-discovered


def get_event_processor_for_task(task_name: str) -> Optional[BaseEventProcessor]:
    """Get appropriate event processor for the given task.

    Args:
        task_name: Name of the task

    Returns:
        BaseEventProcessor or None: An instance of the appropriate processor class, or None if not found
    """
    # Ensure processors are discovered (will only run once)
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
        if HBCD_PROCESSOR_AVAILABLE:
            processor = DedicatedProcessor()
            return processor.process_events(raw, events, events_df, autoclean_dict)
        else:
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


# Built-in processors will be registered during discovery


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

    # Try to get a task-specific processor
    processor = get_event_processor_for_task(task)

    if processor:
        message("info", f"Using event processor: {processor.__class__.__name__}")
        return processor.process_events(raw, events, events_df, autoclean_dict)
    else:
        message("warning", f"No event processor found for task: {task}")
        return raw
