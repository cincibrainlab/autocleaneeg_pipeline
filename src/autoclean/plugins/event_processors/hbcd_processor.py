"""Event processor for HBCD paradigm EEG data.

This processor handles all HBCD (HEALthy Brain and Child Development) paradigms,
including MMN, VEP, and other task variants.
"""

from typing import Optional, List

import mne
import numpy as np
import pandas as pd

from autoclean.io.import_ import BaseEventProcessor
from autoclean.utils.logging import message


class HBCDEventProcessor(BaseEventProcessor):
    """Event processor for HBCD tasks.
    
    This processor handles all HBCD (HEALthy Brain and Child Development) paradigms,
    creating rich annotations that include task, type, and condition information
    for MMN, VEP, and other HBCD tasks.
    """
    
    VERSION = "1.1.0"
    
    # List of all supported HBCD tasks
    SUPPORTED_TASKS = [
        "hbcd_mmn", "mmn",  # Mismatch Negativity variants
        "hbcd_vep", "vep",  # Visual Evoked Potential variants
        # Add other HBCD tasks here as they are implemented
    ]
    
    @classmethod
    def supports_task(cls, task_name: str) -> bool:
        """Check if this processor supports the given task."""
        # Convert task name to lowercase for case-insensitive matching
        task_lower = task_name.lower()
        
        # Check if task is in the supported tasks list or starts with "hbcd_"
        return (task_lower in [t.lower() for t in cls.SUPPORTED_TASKS] or 
                task_lower.startswith("hbcd_"))
    
    def process_events(self, 
                     raw: mne.io.Raw,
                     events: Optional[np.ndarray],
                     events_df: Optional[pd.DataFrame],
                     autoclean_dict: dict) -> mne.io.Raw:
        """Process HBCD task-specific annotations.
        
        This method respects configuration settings and can be disabled via
        the 'hbcd_event_processing' configuration parameter.
        
        Args:
            raw: Raw EEG data
            events: Event array from MNE
            events_df: DataFrame containing event information
            autoclean_dict: Configuration dictionary
            
        Returns:
            mne.io.Raw: Raw data with processed events/annotations
        """
        # Get the task name
        task = autoclean_dict.get("task", "").lower()
        
        # Check if this specific processor is enabled
        # Support both generic and task-specific configuration
        processing_enabled = True
        config_keys = [
            "hbcd_event_processing",           # Generic for all HBCD
            f"{task}_event_processing",        # Task-specific (e.g., hbcd_mmn_event_processing)
            "hbcd_mmn_event_processing"        # Legacy support for older config files
        ]
        
        for key in config_keys:
            if key in autoclean_dict:
                processing_enabled = autoclean_dict[key]
                break
        
        if not processing_enabled:
            message("info", f"✗ {task.upper()} event processing disabled in configuration")
            return raw
            
        message("info", f"✓ Processing {task.upper()} task-specific annotations...")
        
        if events_df is not None:
            # Get columns to include from config or use defaults
            default_columns = ["Task", "type", "onset", "Condition"]
            
            # Support both generic and task-specific configuration
            columns_keys = [
                f"{task}_event_columns",     # e.g., hbcd_vep_event_columns
                "hbcd_event_columns",        # Generic
                "hbcd_mmn_event_columns"     # Legacy support
            ]
            
            columns = None
            for key in columns_keys:
                if key in autoclean_dict:
                    columns = autoclean_dict[key]
                    break
                    
            # Fall back to defaults if no configuration found
            if columns is None:
                columns = default_columns
            
            # Make sure we always have onset
            if "onset" not in columns:
                columns.append("onset")
                
            # Extract relevant columns (with error handling)
            try:
                subset_events_df = events_df[columns]
            except KeyError as e:
                message("warning", f"Missing column in events_df: {e}. Using available columns.")
                # Fall back to columns that exist
                available_columns = [col for col in default_columns if col in events_df.columns]
                if "onset" not in available_columns and "onset" in events_df.columns:
                    available_columns.append("onset")
                if "onset" not in available_columns:
                    message("error", "Required 'onset' column not found in events_df")
                    return raw
                subset_events_df = events_df[available_columns]
            
            # Get format template from config or use default
            # Support both generic and task-specific formats
            format_keys = [
                f"{task}_description_format",  # e.g., hbcd_vep_description_format
                "hbcd_description_format",     # Generic
                "hbcd_mmn_description_format"  # Legacy support
            ]
            
            description_format = None
            for key in format_keys:
                if key in autoclean_dict:
                    description_format = autoclean_dict[key]
                    break
                    
            # Fall back to default if no configuration found
            if description_format is None:
                description_format = "{Task}/{type}/{Condition}" if "Condition" in subset_events_df.columns else "{Task}/{type}"
            
            # Create rich annotations with task/type/condition information
            try:
                # Safely format descriptions, handling missing columns
                descriptions = []
                for _, row in subset_events_df.iterrows():
                    row_dict = row.to_dict()
                    try:
                        desc = description_format.format(**row_dict)
                    except KeyError:
                        # If formatting fails, create a simpler description with available data
                        parts = []
                        for col in row_dict:
                            if col != "onset" and col != "duration":
                                parts.append(f"{row_dict[col]}")
                        desc = "/".join(parts)
                    descriptions.append(desc)
                
                new_annotations = mne.Annotations(
                    onset=subset_events_df["onset"].values,
                    duration=np.zeros(len(subset_events_df)),
                    description=descriptions,
                )
                
                # Apply new annotations to the raw data
                raw.set_annotations(new_annotations)
                message("success", f"Successfully processed {task.upper()} annotations")
            except Exception as e:
                message("error", f"Error creating annotations: {e}")
        else:
            message("warning", f"No events dataframe available for {task.upper()} processing")
            
        return raw 