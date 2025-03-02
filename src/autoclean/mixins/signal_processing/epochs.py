"""Epochs creation and handling mixin for autoclean tasks."""

from typing import Union, Dict, Optional, List
import mne
import numpy as np

from autoclean.utils.logging import message

class EpochsMixin:
    """Mixin class providing epochs creation and handling functionality for EEG data."""
    
    def create_epochs(self, data: Union[mne.io.BaseRaw, None] = None,
                     event_id: Optional[Dict[str, int]] = None,
                     tmin: float = -0.2,
                     tmax: float = 0.5,
                     baseline: Optional[tuple] = (None, 0),
                     threshold: Optional[float] = None,
                     stage_name: str = "epochs") -> mne.Epochs:
        """Create epochs from raw data based on events.
        
        This method creates epochs from raw data based on specified events.
        
        Args:
            data: Optional MNE Raw object. If None, uses self.raw
            event_id: Dictionary mapping event names to event IDs
            tmin: Start time of the epoch relative to the event in seconds
            tmax: End time of the epoch relative to the event in seconds
            baseline: Baseline correction (tuple of start, end)
            threshold: Rejection threshold in microvolts
            stage_name: Name for saving and metadata
            
        Returns:
            The created epochs object
            
        Raises:
            AttributeError: If self.raw doesn't exist when needed
            TypeError: If data is not a Raw object
            ValueError: If no events are found
            RuntimeError: If epoch creation fails
        """
        # Check if this step is enabled in the configuration
        is_enabled, config_value = self._check_step_enabled("epoch_settings")
            
        if not is_enabled:
            message("info", "Epoch creation step is disabled in configuration")
            return None
            
        # Get parameters from config if available
        if config_value and isinstance(config_value, dict):
            # Get epoch settings
            epoch_value = config_value.get("value", {})
            if isinstance(epoch_value, dict):
                tmin = epoch_value.get("tmin", tmin)
                tmax = epoch_value.get("tmax", tmax)
            
            # Get baseline settings
            baseline_settings = config_value.get("remove_baseline", {})
            if isinstance(baseline_settings, dict) and baseline_settings.get("enabled", False):
                baseline = baseline_settings.get("window", baseline)
            
            # Get threshold settings
            threshold_settings = config_value.get("threshold_rejection", {})
            if isinstance(threshold_settings, dict) and threshold_settings.get("enabled", False):
                volt_threshold = threshold_settings.get("volt_threshold", {})
                if isinstance(volt_threshold, dict):
                    threshold = volt_threshold.get("eeg", threshold)
                    
        # Get event_id from config if not provided
        if event_id is None:
            event_id_enabled, event_id_config = self._check_step_enabled("event_id")
            if event_id_enabled and event_id_config:
                event_id = event_id_config
        
        # Determine which data to use
        data = self._get_data_object(data)
        
        # Type checking
        if not isinstance(data, mne.io.BaseRaw):
            raise TypeError("Data must be an MNE Raw object for epoch creation")
            
        try:
            # Find events if not already present
            if not hasattr(self, 'events') or self.events is None:
                message("info", "Finding events in raw data...")
                events = mne.find_events(data)
                if hasattr(self, 'config') and self.config.get("run_id"):
                    self.events = events
            else:
                events = self.events
                
            # Check if events were found
            if events is None or len(events) == 0:
                message("warning", "No events found in raw data")
                raise ValueError("No events found in raw data")
                
            # Create event_id if not provided
            if event_id is None:
                # Get unique event IDs
                unique_ids = np.unique(events[:, 2])
                event_id = {f"event_{id}": id for id in unique_ids}
                
            # Create epochs
            message("header", f"Creating epochs from {len(events)} events...")
            
            # Set up rejection parameters
            reject = None
            if threshold is not None:
                reject = {"eeg": threshold * 1e-6}  # Convert µV to V
                
            # Create epochs
            epochs = mne.Epochs(
                data,
                events,
                event_id=event_id,
                tmin=tmin,
                tmax=tmax,
                baseline=baseline,
                reject=reject,
                preload=True,
            )
            
            message("info", f"Created {len(epochs)} epochs from {len(events)} events")
            
            # Update metadata
            metadata = {
                "event_count": len(events),
                "epoch_count": len(epochs),
                "tmin": tmin,
                "tmax": tmax,
                "baseline": baseline,
                "threshold": threshold,
                "event_id": event_id
            }
            
            self._update_metadata("create_epochs", metadata)
            
            # Store epochs
            if hasattr(self, 'config') and self.config.get("run_id"):
                self.epochs = epochs
                
            return epochs
            
        except Exception as e:
            message("error", f"Error during epoch creation: {str(e)}")
            raise RuntimeError(f"Failed to create epochs: {str(e)}") from e
