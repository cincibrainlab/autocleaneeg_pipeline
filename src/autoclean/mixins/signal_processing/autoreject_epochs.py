"""AutoReject epochs cleaning mixin for autoclean tasks."""

from typing import Union, Dict, Optional, List, Any
import mne
import numpy as np
from datetime import datetime

from autoclean.utils.logging import message
from autoclean.utils.database import manage_database

class AutoRejectEpochsMixin:
    """Mixin class providing functionality to clean epochs using AutoReject."""
    
    def apply_autoreject(self, epochs: Union[mne.Epochs, None] = None,
                        n_interpolate: Optional[List[int]] = None,
                        consensus: Optional[List[float]] = None,
                        n_jobs: int = 1,
                        stage_name: str = "apply_autoreject") -> mne.Epochs:
        """Apply AutoReject to clean epochs.
        
        AutoReject is a machine learning-based method for automatic artifact rejection
        in EEG data. It identifies and removes bad epochs and interpolates bad channels
        within epochs.
        
        Args:
            epochs: Optional MNE Epochs object. If None, uses self.epochs
            n_interpolate: List of number of channels to interpolate
            consensus: List of consensus percentages
            n_jobs: Number of parallel jobs to run
            stage_name: Name for saving and metadata
            
        Returns:
            The cleaned epochs object
            
        Raises:
            AttributeError: If self.epochs doesn't exist when needed
            TypeError: If epochs is not an Epochs object
            RuntimeError: If AutoReject fails
            ImportError: If autoreject package is not installed
        """
        # Check if this step is enabled in the configuration
        is_enabled, config_value = self._check_step_enabled("apply_autoreject")
            
        if not is_enabled:
            message("info", "AutoReject step is disabled in configuration")
            return None
            
        # Get parameters from config if available
        if config_value and isinstance(config_value, dict):
            n_interpolate = config_value.get("n_interpolate", n_interpolate)
            consensus = config_value.get("consensus", consensus)
            n_jobs = config_value.get("n_jobs", n_jobs)
        
        # Determine which data to use
        epochs = self._get_epochs_object(epochs)
        
        # Type checking
        if not isinstance(epochs, mne.Epochs):
            raise TypeError("Data must be an MNE Epochs object for AutoReject")
            
        try:
            # Import AutoReject
            try:
                from autoreject import AutoReject
            except ImportError:
                message("error", "AutoReject package is not installed. Please install it with 'pip install autoreject'")
                raise ImportError("AutoReject package is not installed")
                
            message("header", "Applying AutoReject for artifact rejection")
            
            # Create AutoReject object with parameters if provided
            if n_interpolate is not None and consensus is not None:
                ar = AutoReject(n_interpolate=n_interpolate, consensus=consensus, n_jobs=n_jobs)
            else:
                ar = AutoReject(n_jobs=n_jobs)
                
            # Fit and transform epochs
            epochs_clean = ar.fit_transform(epochs)
            
            # Calculate statistics
            rejected_epochs = len(epochs) - len(epochs_clean)
            rejection_percent = round((rejected_epochs / len(epochs)) * 100, 2) if len(epochs) > 0 else 0
            
            message("info", f"Artifacts rejected: {rejected_epochs} epochs removed by AutoReject ({rejection_percent}%)")
            
            # Update metadata
            metadata = {
                "initial_epochs": len(epochs),
                "final_epochs": len(epochs_clean),
                "rejected_epochs": rejected_epochs,
                "rejection_percent": rejection_percent,
                "epoch_duration": epochs.times[-1] - epochs.times[0],
                "samples_per_epoch": epochs.times.shape[0],
                "total_duration_sec": (epochs.times[-1] - epochs.times[0]) * len(epochs_clean),
                "total_samples": epochs.times.shape[0] * len(epochs_clean),
                "channel_count": len(epochs.ch_names),
                "n_interpolate": n_interpolate,
                "consensus": consensus,
                "n_jobs": n_jobs,
            }
            
            self._update_metadata("apply_autoreject", metadata)
            
            # Store epochs
            if hasattr(self, 'config') and self.config.get("run_id"):
                self.epochs = epochs_clean
                
            # Save epochs
            from autoclean.step_functions.io import save_epochs_to_set
            if hasattr(self, 'config'):
                save_epochs_to_set(epochs_clean, self.config, "post_autoreject")
                
            return epochs_clean
            
        except Exception as e:
            message("error", f"Error during AutoReject: {str(e)}")
            raise RuntimeError(f"Failed to apply AutoReject: {str(e)}") from e
