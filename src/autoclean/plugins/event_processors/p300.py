"""Event processor for P300 paradigm EEG data."""

from typing import Optional

import mne
import numpy as np
import pandas as pd

from autoclean.io.import_ import BaseEventProcessor
from autoclean.utils.logging import message


class P300EventProcessor(BaseEventProcessor):
    """Event processor for P300 tasks.
    
    This processor handles P300 oddball paradigm data, converting
    numeric annotations (13, 14) to descriptive labels ("Standard", "Target").
    """
    
    VERSION = "1.0.0"
    
    @classmethod
    def supports_task(cls, task_name: str) -> bool:
        """Check if this processor supports the given task."""
        return task_name in ["p300_grael4k", "p300"]
    
    def process_events(self, 
                     raw: mne.io.Raw,
                     events: Optional[np.ndarray],
                     events_df: Optional[pd.DataFrame],
                     autoclean_dict: dict) -> mne.io.Raw:
        """Process P300 task-specific annotations.
        
        This method respects configuration settings and can be disabled via
        the 'p300_event_processing' configuration parameter.
        
        Args:
            raw: Raw EEG data
            events: Event array from MNE
            events_df: DataFrame containing event information
            autoclean_dict: Configuration dictionary
            
        Returns:
            mne.io.Raw: Raw data with processed events/annotations
        """
        # Check if this specific processor is enabled
        if not self._check_config_enabled("p300_event_processing", autoclean_dict):
            message("info", "✗ P300 event processing disabled in configuration")
            return raw
            
        message("info", "✓ Processing P300 task-specific annotations...")
        
        # Get mapping from config or use default
        default_mapping = {"13": "Standard", "14": "Target"}
        mapping = autoclean_dict.get("p300_event_mapping", default_mapping)
        
        # Apply mapping to annotations
        raw.annotations.rename(mapping)
        
        # If needed, we could also create/modify events array here
        
        message("success", "Successfully processed P300 annotations")
        return raw
