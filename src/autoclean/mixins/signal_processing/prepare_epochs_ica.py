"""Prepare epochs for ICA mixin for autoclean tasks."""

from typing import Union, Dict, Optional, List, Any
import mne
import numpy as np
import pandas as pd
from datetime import datetime

from autoclean.utils.logging import message

class PrepareEpochsICAMixin:
    """Mixin class providing functionality to prepare epochs for ICA analysis."""
    
    def prepare_epochs_for_ica(self, epochs: Union[mne.Epochs, None] = None,
                              threshold: float = 3.0,
                              stage_name: str = "prepare_epochs_ica") -> mne.Epochs:
        """Prepare epochs for ICA by dropping epochs marked as bad based on global outlier detection.
        
        This implementation is based on the Python implementation of the FASTER algorithm
        from Marijn van Vliet (https://gist.github.com/wmvanvliet/d883c3fe1402c7ced6fc).
        
        Args:
            epochs: Optional MNE Epochs object. If None, uses self.epochs
            threshold: Z-score threshold for outlier detection
            stage_name: Name for saving and metadata
            
        Returns:
            The prepared epochs object
            
        Raises:
            AttributeError: If self.epochs doesn't exist when needed
            TypeError: If epochs is not an Epochs object
            RuntimeError: If preparation fails
        """
        # Check if this step is enabled in the configuration
        is_enabled, config_value = self._check_step_enabled("prepare_epochs_ica")
            
        if not is_enabled:
            message("info", "Prepare epochs for ICA step is disabled in configuration")
            return None
            
        # Get parameters from config if available
        if config_value and isinstance(config_value, dict):
            threshold = config_value.get("threshold", threshold)
        
        # Determine which data to use
        epochs = self._get_epochs_object(epochs)
        
        # Type checking
        if not isinstance(epochs, mne.Epochs):
            raise TypeError("Data must be an MNE Epochs object for ICA preparation")
            
        try:
            message("header", "Preparing epochs for ICA by removing outliers")
            
            # Force preload to avoid RuntimeError
            if not epochs.preload:
                epochs.load_data()
                
            # Create a copy to work with
            epochs_clean = epochs.copy()
            
            # Get the data and reshape to channels x timepoints
            data = epochs.get_data()
            data_flat = data.reshape(data.shape[0], -1)
            
            # Calculate statistics across epochs
            channel_means = np.mean(data_flat, axis=1)
            channel_stds = np.std(data_flat, axis=1)
            channel_max = np.max(np.abs(data_flat), axis=1)
            channel_ranges = np.max(data_flat, axis=1) - np.min(data_flat, axis=1)
            
            # Calculate z-scores for each statistic
            z_means = np.abs((channel_means - np.mean(channel_means)) / np.std(channel_means))
            z_stds = np.abs((channel_stds - np.mean(channel_stds)) / np.std(channel_stds))
            z_max = np.abs((channel_max - np.mean(channel_max)) / np.std(channel_max))
            z_ranges = np.abs((channel_ranges - np.mean(channel_ranges)) / np.std(channel_ranges))
            
            # Find epochs with z-scores above threshold for any statistic
            bad_epochs = np.unique(np.concatenate([
                np.where(z_means > threshold)[0],
                np.where(z_stds > threshold)[0],
                np.where(z_max > threshold)[0],
                np.where(z_ranges > threshold)[0]
            ]))
            
            # Drop bad epochs
            if len(bad_epochs) > 0:
                epochs_clean.drop(bad_epochs)
                message("info", f"Dropped {len(bad_epochs)} epochs with z-scores above {threshold}")
            else:
                message("info", f"No epochs with z-scores above {threshold} found")
                
            # Update metadata
            metadata = {
                "initial_epoch_count": len(epochs),
                "final_epoch_count": len(epochs_clean),
                "dropped_epoch_count": len(bad_epochs),
                "threshold": threshold,
                "bad_epochs": bad_epochs.tolist() if len(bad_epochs) > 0 else [],
                "z_score_metrics": ["mean", "std", "max", "range"],
                "single_epoch_duration": epochs.times[-1] - epochs.times[0],
                "single_epoch_samples": epochs.times.shape[0],
                "total_duration_sec": (epochs.times[-1] - epochs.times[0]) * len(epochs_clean),
                "total_samples": epochs.times.shape[0] * len(epochs_clean),
                "channel_count": len(epochs.ch_names),
            }
            
            self._update_metadata("prepare_epochs_for_ica", metadata)
            
            # Store epochs
            if hasattr(self, 'config') and self.config.get("run_id"):
                self.epochs = epochs_clean
                
            # Save epochs
            from autoclean.step_functions.io import save_epochs_to_set
            if hasattr(self, 'config'):
                save_epochs_to_set(epochs_clean, self.config, "post_ica_prep")
                
            return epochs_clean
            
        except Exception as e:
            message("error", f"Error during epochs preparation for ICA: {str(e)}")
            raise RuntimeError(f"Failed to prepare epochs for ICA: {str(e)}") from e
