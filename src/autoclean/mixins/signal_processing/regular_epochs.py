"""Regular epochs creation mixin for autoclean tasks.

This module provides functionality for creating regular fixed-length epochs from
continuous EEG data. Regular epochs are time segments of equal duration that are
created at fixed intervals throughout the recording, regardless of event markers.

The RegularEpochsMixin class implements methods for creating these epochs and
detecting artifacts within them, particularly focusing on reference and muscle
artifacts that can contaminate the data.

Regular epoching is particularly useful for resting-state data analysis, where
there are no specific events of interest, but the data needs to be segmented
into manageable chunks for further processing and analysis.
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
    EEG data. It includes functionality for artifact detection, specifically focusing on
    reference and muscle artifacts that can contaminate the data.
    
    Regular epochs are time segments of equal duration that are created at fixed intervals
    throughout the recording, regardless of event markers. This approach is particularly
    useful for resting-state data analysis, where there are no specific events of interest.
    
    The mixin respects configuration settings from the autoclean_config.yaml file,
    allowing users to customize the epoching parameters and artifact detection thresholds.
    """
    
    def create_regular_epochs(self, data: Union[mne.io.BaseRaw, None] = None,
                             tmin: float = -1,
                             tmax: float = 1,
                             baseline: Optional[tuple] = None,
                             volt_threshold: Optional[Dict[str, float]] = None,
                             stage_name: str = "post_epochs") -> mne.Epochs:
        """Create regular fixed-length epochs from raw data.
        
        This method creates fixed-length epochs from continuous EEG data at regular
        intervals. It supports optional baseline correction and amplitude-based artifact
        rejection. The method also detects reference and muscle artifacts using specialized
        algorithms.
        
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
            
        Returns:
            mne.Epochs: The created epochs object with bad epochs marked
            
        Raises:
            AttributeError: If self.raw doesn't exist when needed
            TypeError: If data is not a Raw object
            RuntimeError: If epoch creation fails
            
        Example:
            ```python
            # Create regular epochs with default parameters
            epochs = task.create_regular_epochs()
            
            # Create regular epochs with custom parameters
            epochs = task.create_regular_epochs(
                tmin=-0.5,
                tmax=1.0,
                baseline=(-0.5, 0),
                volt_threshold={"eeg": 75, "eog": 150}
            )
            ```
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
            # Create initial epochs with reject_by_annotation=False to handle annotations manually
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
                reject_by_annotation=False,
            )
            
            # Initialize metadata DataFrame
            epochs.metadata = pd.DataFrame(index=range(len(epochs)))
            
            # Find epochs that overlap with BAD_REF_AF annotations
            bad_ref_epochs = []
            for ann in data.annotations:
                if ann["description"] == "BAD_REF_AF":
                    # Find epochs that overlap with this annotation
                    ann_start = ann["onset"]
                    ann_end = ann["onset"] + ann["duration"]
                    
                    # Check each epoch
                    for idx, event in enumerate(epochs.events):
                        epoch_start = event[0] / epochs.info["sfreq"]  # Convert to seconds
                        epoch_end = epoch_start + (tmax - tmin)
                        
                        # Check for overlap
                        if (epoch_start <= ann_end) and (epoch_end >= ann_start):
                            bad_ref_epochs.append(idx)
            
                    # Remove duplicates and sort
                    bad_ref_epochs = sorted(list(set(bad_ref_epochs)))
                    
                    # Mark bad reference epochs in metadata
                    epochs.metadata["BAD_REF_AF"] = [
                        idx in bad_ref_epochs for idx in range(len(epochs))
                    ]
                    message("info", f"Marked {len(bad_ref_epochs)} unique epochs as BAD_REF_AF")
            
            # Detect Muscle Beta Focus
            bad_muscle_epochs = self._detect_muscle_beta_focus_robust(
                epochs.copy(), freq_band=(20, 100), scale_factor=2.0
            )
            
            # Remove duplicates and sort
            bad_muscle_epochs = sorted(list(set(bad_muscle_epochs)))
            
            # Add muscle artifact information to metadata
            epochs.metadata["BAD_MOVEMENT"] = [
                idx in bad_muscle_epochs for idx in range(len(epochs))
            ]
            message("info", f"Marked {len(bad_muscle_epochs)} unique epochs as BAD_MOVEMENT")
            
            # Add annotations for visualization
            for idx in bad_muscle_epochs:
                onset = epochs.events[idx, 0] / epochs.info["sfreq"]
                duration = tmax - tmin
                description = "BAD_MOVEMENT"
                epochs.annotations.append(onset, duration, description)
            
            # Save epochs with bad epochs marked but not dropped
            from autoclean.step_functions.io import save_epochs_to_set
            if hasattr(self, 'config'):
                save_epochs_to_set(epochs, self.config, stage_name)
            
            # Create a copy for dropping
            epochs_clean = epochs.copy()
            
            # Combine all bad epochs and remove duplicates
            all_bad_epochs = sorted(list(set(bad_ref_epochs + bad_muscle_epochs)))
            
            # Drop all bad epochs at once
            if all_bad_epochs:
                epochs_clean.drop(all_bad_epochs)
                message("info", f"Dropped {len(all_bad_epochs)} unique bad epochs")
            
            # Drop remaining bad epochs
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
                "reject_by_annotation": False,  # We handled annotations manually
                "initial_epoch_count": len(epochs),
                "final_epoch_count": len(epochs_clean),
                "single_epoch_duration": epochs.times[-1] - epochs.times[0],
                "single_epoch_samples": epochs.times.shape[0],
                "durationSec": (epochs.times[-1] - epochs.times[0]) * len(epochs_clean),
                "numberSamples": epochs.times.shape[0] * len(epochs_clean),
                "channelCount": len(epochs.ch_names),
                "annotation_types": annotation_types,
                "unique_bad_ref_epochs": len(bad_ref_epochs),
                "unique_bad_muscle_epochs": len(bad_muscle_epochs),
                "total_unique_bad_epochs": len(all_bad_epochs),
                "marked_epochs_file": "post_epochs",
                "cleaned_epochs_file": "post_drop_bads",
                "tmin": tmin,
                "tmax": tmax,
            }
            
            self._update_metadata("create_regular_epochs", metadata)
            
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
            
    def _detect_muscle_beta_focus_robust(self, epochs, freq_band=(20, 30), scale_factor=3.0):
        """Detect muscle artifacts using a robust measure (median + MAD * scale_factor).
        
        This method focuses only on electrodes labeled as 'OTHER' to reduce forced
        removal of epochs in very clean data.
        
        Args:
            epochs: MNE Epochs object
            freq_band: Frequency band for filtering (min, max)
            scale_factor: Scale factor for threshold calculation
            
        Returns:
            List of indices of epochs with muscle artifacts
        """
        # Ensure data is loaded
        epochs.load_data()
        
        # Filter in beta band
        epochs_beta = epochs.copy().filter(
            l_freq=freq_band[0], h_freq=freq_band[1], verbose=False
        )
        
        # Get channel names
        ch_names = epochs_beta.ch_names
        
        # Build channel_region_map from the provided channel data
        # Make sure all "OTHER" electrodes are listed here
        channel_region_map = {
            "E17": "OTHER",
            "E38": "OTHER",
            "E43": "OTHER",
            "E44": "OTHER",
            "E48": "OTHER",
            "E49": "OTHER",
            "E56": "OTHER",
            "E73": "OTHER",
            "E81": "OTHER",
            "E88": "OTHER",
            "E94": "OTHER",
            "E107": "OTHER",
            "E113": "OTHER",
            "E114": "OTHER",
            "E119": "OTHER",
            "E120": "OTHER",
            "E121": "OTHER",
            "E125": "OTHER",
            "E126": "OTHER",
            "E127": "OTHER",
            "E128": "OTHER",
        }
        
        # Select only OTHER channels
        selected_ch_indices = [
            i for i, ch in enumerate(ch_names) if channel_region_map.get(ch, "") == "OTHER"
        ]
        
        # If no OTHER channels are found, return empty
        if not selected_ch_indices:
            return []
        
        # Extract data from OTHER channels only
        data = epochs_beta.get_data()[
            :, selected_ch_indices, :
        ]  # shape: (n_epochs, n_sel_channels, n_times)
        
        # Compute peak-to-peak amplitude per epoch and selected channels
        p2p = data.max(axis=2) - data.min(axis=2)
        
        # Compute maximum peak-to-peak amplitude across the selected channels
        max_p2p = p2p.max(axis=1)
        
        # Compute median and MAD
        med = np.median(max_p2p)
        mad = np.median(np.abs(max_p2p - med))
        
        # Robust threshold
        threshold = med + scale_factor * mad
        
        # Identify bad epochs
        bad_epochs = np.where(max_p2p > threshold)[0].tolist()
        
        # Update metadata
        metadata = {
            "freq_band": freq_band,
            "scale_factor": scale_factor,
            "bad_epochs": bad_epochs,
        }
        
        self._update_metadata("muscle_beta_focus_robust", metadata)
        
        return bad_epochs
