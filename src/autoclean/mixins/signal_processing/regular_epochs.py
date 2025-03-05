"""Regular epochs creation mixin for autoclean tasks.

This module provides functionality for creating regular fixed-length epochs from
continuous EEG data. Regular epochs are time segments of equal duration that are
created at fixed intervals throughout the recording, regardless of event markers.

The RegularEpochsMixin class implements methods for creating these epochs and
handling annotations, allowing users to either automatically reject epochs that
overlap with bad annotations or just mark them in the metadata for later processing.

Regular epoching is particularly useful for resting-state data analysis, where
there are no specific events of interest, but the data needs to be segmented
into manageable chunks for further processing and analysis.

Updates:
- Added reject_by_annotation parameter to create_regular_epochs to control how bad annotations are handled
- Added functionality to detect muscle artifacts in continuous Raw data using a sliding window approach
"""

from typing import Union, Dict, Optional, List
import mne
import numpy as np
import pandas as pd
from datetime import datetime

from autoclean.utils.logging import message

class RegularEpochsMixin:
    """Mixin class providing regular (fixed-length) epochs creation functionality for EEG data.
    
    This mixin provides methods for creating regular fixed-length epochs from continuous
    EEG data. It includes functionality for handling annotations and amplitude-based artifact
    rejection.
    
    Regular epochs are time segments of equal duration that are created at fixed intervals
    throughout the recording, regardless of event markers. This approach is particularly
    useful for resting-state data analysis, where there are no specific events of interest.
    
    The mixin respects configuration settings from the autoclean_config.yaml file,
    allowing users to customize the epoching parameters and artifact detection thresholds.
    
    Methods:
        create_regular_epochs: Create regular fixed-length epochs from raw data with options
                              to either reject or just mark epochs with bad annotations.
        detect_muscle_artifacts: Detect muscle artifacts in continuous Raw data and add annotations.
    """
    
    def create_regular_epochs(self, data: Union[mne.io.BaseRaw, None] = None,
                             tmin: float = -1,
                             tmax: float = 1,
                             baseline: Optional[tuple] = None,
                             volt_threshold: Optional[Dict[str, float]] = None,
                             stage_name: str = "post_epochs",
                             reject_by_annotation: bool = False) -> mne.Epochs:
        """Create regular fixed-length epochs from raw data.
        
        This method creates fixed-length epochs from continuous EEG data at regular
        intervals. It supports optional baseline correction and amplitude-based artifact
        rejection.
        
        The epoching parameters can be customized through the configuration file
        (autoclean_config.yaml) under the "epoch_settings" section. If enabled, the
        configuration values will override the default parameters.
        
        Args:
            data: Optional MNE Raw object. If None, uses self.raw
            tmin: Start time of the epoch in seconds
            tmax: End time of the epoch in seconds
            baseline: Baseline correction (tuple of start, end)
            volt_threshold: Dictionary mapping channel types to rejection thresholds in microvolts
                           (e.g., {"eeg": 100, "eog": 200})
            stage_name: Name for saving and metadata tracking
            reject_by_annotation: Whether to automatically reject epochs that overlap with
                                 annotations starting with "bad" or "BAD", or just mark them
                                 in the metadata for later processing
            
        Returns:
            mne.Epochs: The created epochs object with bad epochs marked (and dropped if reject_by_annotation=True)
            
        Raises:
            AttributeError: If self.raw doesn't exist when needed
            TypeError: If data is not a Raw object
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
                threshold_config = threshold_settings.get("volt_threshold", {})
                if isinstance(threshold_config, (int, float)):
                    volt_threshold = {"eeg": float(threshold_config)}
                elif isinstance(threshold_config, dict):
                    volt_threshold = {k: float(v) for k, v in threshold_config.items()}
        
        # Determine which data to use
        data = self._get_data_object(data)
        
        # Type checking
        if not isinstance(data, mne.io.BaseRaw):
            raise TypeError("Data must be an MNE Raw object for epoch creation")
            
        try:
            # Create initial epochs with reject_by_annotation parameter
            message("header", f"Creating regular epochs from {tmin}s to {tmax}s...")
            events = mne.make_fixed_length_events(
                data, duration=tmax - tmin, overlap=0, start=abs(tmin)
            )
            
            epochs = mne.Epochs(
                data,
                events,
                tmin=tmin,
                tmax=tmax,
                baseline=baseline,
                reject=volt_threshold,
                preload=True,
                reject_by_annotation=reject_by_annotation,
            )
            
            # Initialize metadata DataFrame
            epochs.metadata = pd.DataFrame(index=range(len(epochs)))
            
            # If not using reject_by_annotation, manually track bad annotations
            if not reject_by_annotation:
                # Find epochs that overlap with any "bad" or "BAD" annotations
                bad_epochs = []
                bad_annotations = {}  # To track which annotation affected each epoch
                
                for ann in data.annotations:
                    # Check if annotation description starts with "bad" or "BAD"
                    if ann["description"].lower().startswith("bad"):
                        ann_start = ann["onset"]
                        ann_end = ann["onset"] + ann["duration"]
                        
                        # Check each epoch
                        for idx, event in enumerate(epochs.events):
                            epoch_start = event[0] / epochs.info["sfreq"]  # Convert to seconds
                            epoch_end = epoch_start + (tmax - tmin)
                            
                            # Check for overlap
                            if (epoch_start <= ann_end) and (epoch_end >= ann_start):
                                bad_epochs.append(idx)
                                
                                # Track which annotation affected this epoch
                                if idx not in bad_annotations:
                                    bad_annotations[idx] = []
                                bad_annotations[idx].append(ann["description"])
                
                # Remove duplicates and sort
                bad_epochs = sorted(list(set(bad_epochs)))
                
                # Mark bad epochs in metadata
                epochs.metadata["BAD_ANNOTATION"] = [
                    idx in bad_epochs for idx in range(len(epochs))
                ]
                
                # Add specific annotation types to metadata
                for idx, annotations in bad_annotations.items():
                    for annotation in annotations:
                        col_name = annotation.upper()
                        if col_name not in epochs.metadata.columns:
                            epochs.metadata[col_name] = False
                        epochs.metadata.loc[idx, col_name] = True
                
                message("info", f"Marked {len(bad_epochs)} epochs with bad annotations (not dropped)")
            
            # Save epochs with bad epochs marked but not dropped
            from autoclean.step_functions.io import save_epochs_to_set
            if hasattr(self, 'config'):
                save_epochs_to_set(epochs, self.config, stage_name)
            
            # Create a copy for dropping if using amplitude thresholds
            epochs_clean = epochs.copy()
            
            # Drop bad epochs based on amplitude thresholds
            epochs_clean.drop_bad()
            
            # Analyze drop log to tally different annotation types
            drop_log = epochs_clean.drop_log
            total_epochs = len(drop_log)
            good_epochs = sum(1 for log in drop_log if len(log) == 0)
            
            # Dynamically collect all unique annotation types
            annotation_types = {}
            for log in drop_log:
                if len(log) > 0:  # If epoch was dropped
                    for annotation in log:
                        # Convert numpy string to regular string if needed
                        annotation = str(annotation)
                        annotation_types[annotation] = annotation_types.get(annotation, 0) + 1
            
            message("info", "\nEpoch Drop Log Summary:")
            message("info", f"Total epochs: {total_epochs}")
            message("info", f"Good epochs: {good_epochs}")
            for annotation, count in annotation_types.items():
                message("info", f"Epochs with {annotation}: {count}")
            
            # Add good and total to the annotation_types dictionary
            annotation_types["KEEP"] = good_epochs
            annotation_types["TOTAL"] = total_epochs
            
            # Update metadata
            metadata = {
                "duration": tmax - tmin,
                "reject_by_annotation": reject_by_annotation,
                "initial_epoch_count": len(epochs),
                "final_epoch_count": len(epochs_clean),
                "single_epoch_duration": epochs.times[-1] - epochs.times[0],
                "single_epoch_samples": epochs.times.shape[0],
                "durationSec": (epochs.times[-1] - epochs.times[0]) * len(epochs_clean),
                "numberSamples": epochs.times.shape[0] * len(epochs_clean),
                "channelCount": len(epochs.ch_names),
                "annotation_types": annotation_types,
                "marked_epochs_file": "post_epochs",
                "cleaned_epochs_file": "post_drop_bads",
                "tmin": tmin,
                "tmax": tmax,
            }
            
            self._update_metadata("step_create_regular_epochs", metadata)
            
            # Store epochs
            if hasattr(self, 'config') and self.config.get("run_id"):
                self.epochs = epochs_clean

            # Save epochs
            if hasattr(self, 'config'):
                save_epochs_to_set(epochs_clean, self.config, "post_drop_bad_epochs")
                
            return epochs_clean
            
        except Exception as e:
            message("error", f"Error during regular epoch creation: {str(e)}")
            raise RuntimeError(f"Failed to create regular epochs: {str(e)}") from e
            
    