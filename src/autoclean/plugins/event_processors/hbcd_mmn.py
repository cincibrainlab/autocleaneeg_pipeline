"""Event processor for HBCD MMN paradigm EEG data."""

from typing import Optional

import mne
import numpy as np
import pandas as pd

from autoclean.step_functions.io import BaseEventProcessor
from autoclean.utils.logging import message


class HBCDMMNEventProcessor(BaseEventProcessor):
    """Event processor for HBCD MMN tasks.
    
    This processor handles HBCD MMN (Mismatch Negativity) paradigm data,
    creating rich annotations that include task, type, and condition information.
    """
    
    VERSION = "1.0.0"
    
    @classmethod
    def supports_task(cls, task_name: str) -> bool:
        """Check if this processor supports the given task."""
        return task_name in ["hbcd_mmn", "mmn"]
    
    def process_events(self, 
                     raw: mne.io.Raw,
                     events: Optional[np.ndarray],
                     events_df: Optional[pd.DataFrame],
                     autoclean_dict: dict) -> mne.io.Raw:
        """Process HBCD MMN task-specific annotations.
        
        This method respects configuration settings and can be disabled via
        the 'hbcd_mmn_event_processing' configuration parameter.
        
        Args:
            raw: Raw EEG data
            events: Event array from MNE
            events_df: DataFrame containing event information
            autoclean_dict: Configuration dictionary
            
        Returns:
            mne.io.Raw: Raw data with processed events/annotations
        """
        # Check if this specific processor is enabled
        if not self._check_config_enabled("hbcd_mmn_event_processing", autoclean_dict):
            message("info", "✗ HBCD MMN event processing disabled in configuration")
            return raw
            
        message("info", "✓ Processing HBCD MMN task-specific annotations...")
        
        if events_df is not None:
            # Get columns to include from config or use defaults
            default_columns = ["Task", "type", "onset", "Condition"]
            columns = autoclean_dict.get("hbcd_mmn_event_columns", default_columns)
            
            # Make sure we always have onset
            if "onset" not in columns:
                columns.append("onset")
                
            # Extract relevant columns (with error handling)
            try:
                subset_events_df = events_df[columns]
            except KeyError as e:
                message("warning", f"Missing column in events_df: {e}. Using default columns.")
                # Fall back to columns that exist
                available_columns = [col for col in default_columns if col in events_df.columns]
                if "onset" not in available_columns:
                    message("error", "Required 'onset' column not found in events_df")
                    return raw
                subset_events_df = events_df[available_columns]
            
            # Get format template from config or use default
            description_format = autoclean_dict.get("hbcd_mmn_description_format", "{Task}/{type}/{Condition}")
            
            # Create rich annotations with task/type/condition information
            try:
                new_annotations = mne.Annotations(
                    onset=subset_events_df["onset"].values,
                    duration=np.zeros(len(subset_events_df)),
                    description=[
                        description_format.format(**row.to_dict())
                        for _, row in subset_events_df.iterrows()
                    ],
                )
                
                # Apply new annotations to the raw data
                raw.set_annotations(new_annotations)
                message("success", "Successfully processed HBCD MMN annotations")
            except Exception as e:
                message("error", f"Error creating annotations: {e}")
        else:
            message("warning", "No events dataframe available for HBCD MMN processing")
            
        return raw
