"""Mixin classes for data processing operations in autoclean tasks.

This module contains mixin classes that provide common functionality for
data processing operations that can be used across different task types.
These mixins are designed to be included in task classes to provide
standardized implementations of common operations.

Examples:
    ```python
    from autoclean.mixins.data_processing import SignalProcessingMixin
    from autoclean.core.task import Task

    class MyTask(Task, SignalProcessingMixin):
        def run(self):
            # ... task initialization ...
            
            # Use the mixin method
            self.raw = self.resample_data(self.raw)
            
            # ... continue processing ...
    ```
"""

from typing import Union, Dict, Any, Optional, List, Tuple
from datetime import datetime

import mne
import numpy as np

from autoclean.utils.logging import message
from autoclean.utils.database import manage_database, get_run_record
from autoclean.step_functions.io import save_raw_to_set

from pyprep.find_noisy_channels import NoisyChannels



class SignalProcessingMixin:
    """Mixin class providing signal processing functionality for EEG data.
    
    This mixin provides methods for common signal processing operations such as
    resampling, filtering, and other preprocessing steps for MNE Raw or Epochs objects.
    It respects the configuration toggles in the task configuration.
    
    Attributes:
        config: Task configuration dictionary (provided by the parent class)
    """
    
    def resample_data(self, data: Union[mne.io.Raw, mne.Epochs, None] = None, target_sfreq: float = None, 
                    stage_name: str = "resampled", use_epochs: bool = False) -> Union[mne.io.Raw, mne.Epochs]:
        """Resample raw or epoched data based on configuration settings.
        
        This method can work with self.raw, self.epochs, or a provided data object.
        It checks the resample_step toggle in the configuration if no target_sfreq is provided.
        
        Args:
            data: Optional MNE Raw or Epochs object to resample. If None, uses self.raw or self.epochs
            target_sfreq: Optional target sampling frequency. If None, reads from config
            stage_name: Name for saving the resampled data (default: "resampled")
            use_epochs: If True and data is None, uses self.epochs instead of self.raw
            
        Returns:
            The resampled data object (same type as input)
            
        Raises:
            TypeError: If data is not a Raw or Epochs object
            RuntimeError: If resampling fails
            AttributeError: If self.raw or self.epochs doesn't exist when needed
        """
        # Determine which data to use
        if data is None:
            if use_epochs:
                if not hasattr(self, 'epochs') or self.epochs is None:
                    raise AttributeError("No epochs data available for resampling")
                data = self.epochs
            else:
                if not hasattr(self, 'raw') or self.raw is None:
                    raise AttributeError("No raw data available for resampling")
                data = self.raw
        
        # Type checking
        if not isinstance(data, (mne.io.BaseRaw, mne.BaseEpochs)):
            raise TypeError("Data must be an MNE Raw or Epochs object")
            
        # Access configuration if needed
        if not hasattr(self, 'config'):
            raise AttributeError("SignalProcessingMixin requires a 'config' attribute")
            
        # If target_sfreq is not provided, get it from config
        if target_sfreq is None:
            task = self.config.get("task")
            
            # Check if resampling is enabled in the configuration
            resample_enabled = self.config.get("tasks", {}).get(task, {}).get("settings", {}).get(
                "resample_step", {}).get("enabled", False)
                
            if not resample_enabled:
                message("info", "Resampling step is disabled in configuration")
                return data
                
            # Get target sampling frequency from config
            target_sfreq = self.config.get("tasks", {}).get(task, {}).get("settings", {}).get(
                "resample_step", {}).get("value")
                
            if target_sfreq is None:
                message("warning", "Target sampling frequency not specified, skipping resampling")
                return data
            
        # Check if we need to resample (avoid unnecessary resampling)
        current_sfreq = data.info["sfreq"]
        if abs(current_sfreq - target_sfreq) < 0.01:  # Small threshold to account for floating point errors
            message("info", f"Data already at target frequency ({target_sfreq} Hz), skipping resampling")
            return data
            
        message("header", f"Resampling data from {current_sfreq} Hz to {target_sfreq} Hz...")
        
        try:
            # Resample based on data type
            if isinstance(data, mne.io.Raw):
                resampled_data = data.copy().resample(target_sfreq)
                # Save resampled raw data if it's a Raw object
                save_raw_to_set(resampled_data, self.config, f"post_{stage_name}")
            else:  # Epochs
                resampled_data = data.copy().resample(target_sfreq)
                
            message("info", f"Data successfully resampled to {target_sfreq} Hz")
            
            # Update metadata
            metadata = {
                "resample_data": {
                    "creationDateTime": datetime.now().isoformat(),
                    "original_sfreq": current_sfreq,
                    "target_sfreq": target_sfreq,
                    "data_type": "raw" if isinstance(data, mne.io.Raw) else "epochs"
                }
            }
            
            # Update the database if we have a run_id
            if hasattr(self, 'config') and self.config.get("run_id"):
                run_id = self.config.get("run_id")
                manage_database(
                    operation="update", update_record={"run_id": run_id, "metadata": metadata}
                )
            
            # Update self.raw or self.epochs if we're using those
            if data is None and hasattr(self, 'raw') and self.raw is not None and not use_epochs:
                self.raw = resampled_data
            elif data is None and hasattr(self, 'epochs') and self.epochs is not None and use_epochs:
                self.epochs = resampled_data
            breakpoint()
            return resampled_data
            
        except Exception as e:
            message("error", f"Error during resampling: {str(e)}")
            raise RuntimeError(f"Failed to resample data: {str(e)}") from e
            
    def detect_dense_oscillatory_artifacts(self, data: Union[mne.io.BaseRaw, None] = None,
                                         window_size_ms: int = 100, 
                                         channel_threshold_uv: float = 50,
                                         min_channels: int = 75, 
                                         padding_ms: float = 500,
                                         annotation_label: str = "BAD_REF_AF",
                                         stage_name: str = "artifact_detection") -> mne.io.BaseRaw:
        """Detect smaller, dense oscillatory multichannel artifacts.
        
        This method identifies oscillatory artifacts that affect multiple channels simultaneously,
        while excluding large single deflections. It adds annotations to the raw data.
        
        Args:
            data: Optional MNE Raw object. If None, uses self.raw
            window_size_ms: Window size in milliseconds for artifact detection
            channel_threshold_uv: Threshold for peak-to-peak amplitude in microvolts
            min_channels: Minimum number of channels that must exhibit oscillations
            padding_ms: Amount of padding in milliseconds to add before and after each detected artifact
            annotation_label: Label to use for the annotations
            stage_name: Name for saving and metadata
            
        Returns:
            The raw data object with updated artifact annotations
            
        Raises:
            AttributeError: If self.raw doesn't exist when needed
            TypeError: If data is not a Raw object
            RuntimeError: If artifact detection fails
        """
        # Determine which data to use
        if data is None:
            if not hasattr(self, 'raw') or self.raw is None:
                raise AttributeError("No raw data available for artifact detection")
            data = self.raw
        
        # Type checking
        if not isinstance(data, mne.io.BaseRaw):
            raise TypeError("Data must be an MNE Raw object for artifact detection")
            
        try:
            # Convert parameters to samples and volts
            sfreq = data.info["sfreq"]
            window_size = int(window_size_ms * sfreq / 1000)
            channel_threshold = channel_threshold_uv * 1e-6  # Convert µV to V
            padding_sec = padding_ms / 1000.0  # Convert padding to seconds
            
            # Get data and times
            raw_data, times = data.get_data(return_times=True)
            n_channels, n_samples = raw_data.shape
            
            artifact_annotations = []
            
            # Sliding window detection
            for start_idx in range(0, n_samples - window_size, window_size):
                window = raw_data[:, start_idx : start_idx + window_size]
                
                # Compute peak-to-peak amplitude for each channel in the window
                ptp_amplitudes = np.ptp(window, axis=1)  # Peak-to-peak amplitude per channel
                
                # Count channels exceeding the threshold
                num_channels_exceeding = np.sum(ptp_amplitudes > channel_threshold)
                
                # Check if artifact spans multiple channels with oscillatory behavior
                if num_channels_exceeding >= min_channels:
                    start_time = times[start_idx] - padding_sec  # Add padding before
                    end_time = times[start_idx + window_size] + padding_sec  # Add padding after
                    
                    # Ensure we don't go beyond recording bounds
                    start_time = max(start_time, times[0])
                    end_time = min(end_time, times[-1])
                    
                    artifact_annotations.append(
                        [start_time, end_time - start_time, annotation_label]
                    )
            
            # Create a copy of the raw data
            result_raw = data.copy()
            
            # Add annotations to the raw data
            if artifact_annotations:
                for annotation in artifact_annotations:
                    result_raw.annotations.append(
                        onset=annotation[0], duration=annotation[1], description=annotation[2]
                    )
                message("info", f"Added {len(artifact_annotations)} artifact annotations")
            else:
                message("info", "No reference artifacts detected")
                
            # Update metadata if we have a config
            if hasattr(self, 'config') and self.config.get("run_id"):
                metadata = {
                    "detect_dense_oscillatory_artifacts": {
                        "creationDateTime": datetime.now().isoformat(),
                        "window_size_ms": window_size_ms,
                        "channel_threshold_uv": channel_threshold_uv,
                        "min_channels": min_channels,
                        "padding_ms": padding_ms,
                        "annotation_label": annotation_label,
                        "artifacts_detected": len(artifact_annotations)
                    }
                }
                
                run_id = self.config.get("run_id")
                manage_database(
                    operation="update", update_record={"run_id": run_id, "metadata": metadata}
                )
                
                # Save the result if we have a config
                save_raw_to_set(result_raw, self.config, f"post_{stage_name}")
            
            # Update self.raw if we're using it
            if data is self.raw:
                self.raw = result_raw
                
            return result_raw
            
        except Exception as e:
            message("error", f"Error during artifact detection: {str(e)}")
            raise RuntimeError(f"Failed to detect artifacts: {str(e)}") from e
            
    def reject_bad_segments(self, data: Union[mne.io.BaseRaw, None] = None,
                            bad_label: Optional[str] = None,
                            stage_name: str = "bad_segment_rejection") -> mne.io.BaseRaw:
        """Remove all time spans annotated with a specific label or all 'BAD' segments.
        
        This method removes segments marked as bad and concatenates the remaining good segments.
        
        Args:
            data: Optional MNE Raw object. If None, uses self.raw
            bad_label: Specific label of annotations to reject. If None, rejects all segments
                      where description starts with 'BAD'
            stage_name: Name for saving and metadata
            
        Returns:
            A new Raw object with the bad segments removed
            
        Raises:
            AttributeError: If self.raw doesn't exist when needed
            TypeError: If data is not a Raw object
            RuntimeError: If segment rejection fails
        """
        # Determine which data to use
        if data is None:
            if not hasattr(self, 'raw') or self.raw is None:
                raise AttributeError("No raw data available for segment rejection")
            data = self.raw
        
        # Type checking
        if not isinstance(data, mne.io.BaseRaw):
            raise TypeError("Data must be an MNE Raw object for segment rejection")
            
        try:
            # Get annotations
            annotations = data.annotations
            
            # Identify bad intervals based on label matching strategy
            bad_intervals = [
                (onset, onset + duration)
                for onset, duration, desc in zip(
                    annotations.onset, annotations.duration, annotations.description
                )
                if (bad_label is None and desc.startswith("BAD"))
                or (bad_label is not None and desc == bad_label)
            ]
            
            # Define good intervals (non-bad spans)
            good_intervals = []
            prev_end = 0  # Start of the first good interval
            for start, end in sorted(bad_intervals):
                if prev_end < start:
                    good_intervals.append((prev_end, start))  # Add non-bad span
                prev_end = end
            if prev_end < data.times[-1]:  # Add final good interval if it exists
                good_intervals.append((prev_end, data.times[-1]))
                
            # Crop and concatenate good intervals
            if not good_intervals:
                message("warning", "No good segments found after rejection")
                return data.copy()
                
            raw_segments = [
                data.copy().crop(tmin=start, tmax=end) for start, end in good_intervals
            ]
            
            raw_cleaned = mne.concatenate_raws(raw_segments)
            
            # Update metadata if we have a config
            if hasattr(self, 'config') and self.config.get("run_id"):
                metadata = {
                    "reject_bad_segments": {
                        "creationDateTime": datetime.now().isoformat(),
                        "bad_label": bad_label if bad_label else "All BAD*",
                        "segments_removed": len(bad_intervals),
                        "segments_kept": len(good_intervals),
                        "original_duration": data.times[-1],
                        "cleaned_duration": raw_cleaned.times[-1]
                    }
                }
                
                run_id = self.config.get("run_id")
                manage_database(
                    operation="update", update_record={"run_id": run_id, "metadata": metadata}
                )
                
                # Save the result if we have a config
                save_raw_to_set(raw_cleaned, self.config, f"post_{stage_name}")
            
            # Update self.raw if we're using it
            if data is self.raw:
                self.raw = raw_cleaned
                
            return raw_cleaned
            
        except Exception as e:
            message("error", f"Error during segment rejection: {str(e)}")
            raise RuntimeError(f"Failed to reject bad segments: {str(e)}") from e
            
    def set_eeg_reference(self, data: Union[mne.io.BaseRaw, None] = None,
                          ref_type: str = "average", 
                          projection: bool = False,
                          stage_name: str = "reference") -> mne.io.BaseRaw:
        """Apply EEG reference to the data.
        
        This method applies a reference to the EEG data, such as average reference.
        
        Args:
            data: Optional MNE Raw object. If None, uses self.raw
            ref_type: Type of reference to apply (e.g., 'average')
            projection: Whether to use projection (for average reference)
            stage_name: Name for saving and metadata
            
        Returns:
            The raw data object with reference applied
            
        Raises:
            AttributeError: If self.raw doesn't exist when needed
            TypeError: If data is not a Raw object
            RuntimeError: If reference application fails
        """
        # Determine which data to use
        if data is None:
            if not hasattr(self, 'raw') or self.raw is None:
                raise AttributeError("No raw data available for referencing")
            data = self.raw
        
        # Type checking
        if not isinstance(data, mne.io.BaseRaw):
            raise TypeError("Data must be an MNE Raw object for referencing")
            
        try:
            # Apply reference
            message("header", f"Applying {ref_type} reference...")
            if ref_type == "average":
                result_raw = data.copy().set_eeg_reference(ref_type, projection=projection)
            else:
                result_raw = data.copy().set_eeg_reference(ref_type)
                
            message("info", f"Applied {ref_type} reference")
            
            # Update metadata if we have a config
            if hasattr(self, 'config') and self.config.get("run_id"):
                metadata = {
                    "set_eeg_reference": {
                        "creationDateTime": datetime.now().isoformat(),
                        "reference_type": ref_type,
                        "projection": projection
                    }
                }
                
                run_id = self.config.get("run_id")
                manage_database(
                    operation="update", update_record={"run_id": run_id, "metadata": metadata}
                )
                
                # Save the result if we have a config
                save_raw_to_set(result_raw, self.config, f"post_{stage_name}")
            
            # Update self.raw if we're using it
            if data is self.raw:
                self.raw = result_raw
                
            return result_raw
            
        except Exception as e:
            message("error", f"Error during referencing: {str(e)}")
            raise RuntimeError(f"Failed to apply reference: {str(e)}") from e
            
    def drop_channels(self, data: Union[mne.io.BaseRaw, mne.BaseEpochs, None] = None,
                      channels: List[str] = None,
                      stage_name: str = "drop_channels",
                      use_epochs: bool = False) -> Union[mne.io.BaseRaw, mne.BaseEpochs]:
        """Drop specified channels from the data.
        
        This method removes specified channels from the data.
        
        Args:
            data: Optional MNE Raw or Epochs object. If None, uses self.raw or self.epochs
            channels: List of channel names to drop
            stage_name: Name for saving and metadata
            use_epochs: If True and data is None, uses self.epochs instead of self.raw
            
        Returns:
            The data object with channels dropped
            
        Raises:
            AttributeError: If self.raw or self.epochs doesn't exist when needed
            TypeError: If data is not a Raw or Epochs object
            ValueError: If channels is None or empty
            RuntimeError: If channel dropping fails
        """
        # Check if channels is provided
        if channels is None or len(channels) == 0:
            raise ValueError("No channels specified for dropping")
            
        # Determine which data to use
        if data is None:
            if use_epochs:
                if not hasattr(self, 'epochs') or self.epochs is None:
                    raise AttributeError("No epochs data available for dropping channels")
                data = self.epochs
            else:
                if not hasattr(self, 'raw') or self.raw is None:
                    raise AttributeError("No raw data available for dropping channels")
                data = self.raw
        
        # Type checking
        if not isinstance(data, (mne.io.BaseRaw, mne.BaseEpochs)):
            raise TypeError("Data must be an MNE Raw or Epochs object for dropping channels")
            
        try:
            # Drop channels
            message("header", "Dropping channels...")
            result_data = data.copy().drop_channels(channels)
            message("info", f"Dropped {len(channels)} channels: {channels}")
            
            # Update metadata if we have a config
            if hasattr(self, 'config') and self.config.get("run_id"):
                metadata = {
                    "drop_channels": {
                        "creationDateTime": datetime.now().isoformat(),
                        "channels_dropped": channels,
                        "channels_remaining": len(result_data.ch_names)
                    }
                }
                
                run_id = self.config.get("run_id")
                manage_database(
                    operation="update", update_record={"run_id": run_id, "metadata": metadata}
                )
                
                # Save the result if we have a config and it's a Raw object
                if isinstance(result_data, mne.io.BaseRaw):
                    save_raw_to_set(result_data, self.config, f"post_{stage_name}")
            
            # Update self.raw or self.epochs if we're using them
            if data is None and hasattr(self, 'raw') and self.raw is not None and not use_epochs:
                self.raw = result_data
            elif data is None and hasattr(self, 'epochs') and self.epochs is not None and use_epochs:
                self.epochs = result_data
                
            return result_data
            
        except Exception as e:
            message("error", f"Error during channel dropping: {str(e)}")
            raise RuntimeError(f"Failed to drop channels: {str(e)}") from e
            
    def set_channel_types(self, data: Union[mne.io.BaseRaw, mne.BaseEpochs, None] = None,
                           ch_types_dict: Dict[str, str] = None,
                           stage_name: str = "set_channel_types",
                           use_epochs: bool = False) -> Union[mne.io.BaseRaw, mne.BaseEpochs]:
        """Set channel types for specific channels.
        
        This method sets the type of specific channels (e.g., marking channels as EOG).
        
        Args:
            data: Optional MNE Raw or Epochs object. If None, uses self.raw or self.epochs
            ch_types_dict: Dictionary mapping channel names to types (e.g., {'E1': 'eog'})
            stage_name: Name for saving and metadata
            use_epochs: If True and data is None, uses self.epochs instead of self.raw
            
        Returns:
            The data object with updated channel types
            
        Raises:
            AttributeError: If self.raw or self.epochs doesn't exist when needed
            TypeError: If data is not a Raw or Epochs object
            ValueError: If ch_types_dict is None or empty
            RuntimeError: If setting channel types fails
        """
        # Check if ch_types_dict is provided
        if ch_types_dict is None or len(ch_types_dict) == 0:
            raise ValueError("No channel types specified")
            
        # Determine which data to use
        if data is None:
            if use_epochs:
                if not hasattr(self, 'epochs') or self.epochs is None:
                    raise AttributeError("No epochs data available for setting channel types")
                data = self.epochs
            else:
                if not hasattr(self, 'raw') or self.raw is None:
                    raise AttributeError("No raw data available for setting channel types")
                data = self.raw
        
        # Type checking
        if not isinstance(data, (mne.io.BaseRaw, mne.BaseEpochs)):
            raise TypeError("Data must be an MNE Raw or Epochs object for setting channel types")
            
        try:
            # Set channel types
            message("header", "Setting channel types...")
            result_data = data.copy().set_channel_types(ch_types_dict)
            message("info", f"Set types for {len(ch_types_dict)} channels")
            
            # Update metadata if we have a config
            if hasattr(self, 'config') and self.config.get("run_id"):
                metadata = {
                    "set_channel_types": {
                        "creationDateTime": datetime.now().isoformat(),
                        "channel_types": ch_types_dict
                    }
                }
                
                run_id = self.config.get("run_id")
                manage_database(
                    operation="update", update_record={"run_id": run_id, "metadata": metadata}
                )
                
                # Save the result if we have a config and it's a Raw object
                if isinstance(result_data, mne.io.BaseRaw):
                    save_raw_to_set(result_data, self.config, f"post_{stage_name}")
            
            # Update self.raw or self.epochs if we're using them
            if data is None and hasattr(self, 'raw') and self.raw is not None and not use_epochs:
                self.raw = result_data
            elif data is None and hasattr(self, 'epochs') and self.epochs is not None and use_epochs:
                self.epochs = result_data
                
            return result_data
            
        except Exception as e:
            message("error", f"Error during setting channel types: {str(e)}")
            raise RuntimeError(f"Failed to set channel types: {str(e)}") from e
            
    def crop_data(self, data: Union[mne.io.BaseRaw, mne.BaseEpochs, None] = None,
                  tmin: float = None, tmax: float = None,
                  stage_name: str = "crop",
                  use_epochs: bool = False) -> Union[mne.io.BaseRaw, mne.BaseEpochs]:
        """Crop data to a specific time range.
        
        This method crops the data to a specific time range.
        
        Args:
            data: Optional MNE Raw or Epochs object. If None, uses self.raw or self.epochs
            tmin: Start time for cropping in seconds
            tmax: End time for cropping in seconds
            stage_name: Name for saving and metadata
            use_epochs: If True and data is None, uses self.epochs instead of self.raw
            
        Returns:
            The cropped data object
            
        Raises:
            AttributeError: If self.raw or self.epochs doesn't exist when needed
            TypeError: If data is not a Raw or Epochs object
            RuntimeError: If cropping fails
        """
        # Determine which data to use
        if data is None:
            if use_epochs:
                if not hasattr(self, 'epochs') or self.epochs is None:
                    raise AttributeError("No epochs data available for cropping")
                data = self.epochs
            else:
                if not hasattr(self, 'raw') or self.raw is None:
                    raise AttributeError("No raw data available for cropping")
                data = self.raw
        
        # Type checking
        if not isinstance(data, (mne.io.BaseRaw, mne.BaseEpochs)):
            raise TypeError("Data must be an MNE Raw or Epochs object for cropping")
            
        try:
            # Get current time range if not specified
            if tmin is None:
                tmin = data.times[0]
            if tmax is None:
                tmax = data.times[-1]
                
            # Crop data
            message("header", "Cropping data...")
            result_data = data.copy().crop(tmin=tmin, tmax=tmax)
            message("info", f"Cropped data to {tmin}s - {tmax}s")
            
            # Update metadata if we have a config
            if hasattr(self, 'config') and self.config.get("run_id"):
                metadata = {
                    "crop_data": {
                        "creationDateTime": datetime.now().isoformat(),
                        "tmin": tmin,
                        "tmax": tmax,
                        "duration": tmax - tmin
                    }
                }
                
                run_id = self.config.get("run_id")
                manage_database(
                    operation="update", update_record={"run_id": run_id, "metadata": metadata}
                )
                
                # Save the result if we have a config and it's a Raw object
                if isinstance(result_data, mne.io.BaseRaw):
                    save_raw_to_set(result_data, self.config, f"post_{stage_name}")
            
            # Update self.raw or self.epochs if we're using them
            if data is None and hasattr(self, 'raw') and self.raw is not None and not use_epochs:
                self.raw = result_data
            elif data is None and hasattr(self, 'epochs') and self.epochs is not None and use_epochs:
                self.epochs = result_data
                
            return result_data
            
        except Exception as e:
            message("error", f"Error during cropping: {str(e)}")
            raise RuntimeError(f"Failed to crop data: {str(e)}") from e
            
    def trim_data_edges(self, data: Union[mne.io.BaseRaw, None] = None,
                        trim_amount: float = 1.0,
                        stage_name: str = "trim") -> mne.io.BaseRaw:
        """Trim a specified amount from the beginning and end of the data.
        
        This method removes a specified amount of time from both the beginning and end of the data.
        
        Args:
            data: Optional MNE Raw object. If None, uses self.raw
            trim_amount: Amount to trim from each end in seconds
            stage_name: Name for saving and metadata
            
        Returns:
            The trimmed data object
            
        Raises:
            AttributeError: If self.raw doesn't exist when needed
            TypeError: If data is not a Raw object
            ValueError: If trim_amount is too large
            RuntimeError: If trimming fails
        """
        # Determine which data to use
        if data is None:
            if not hasattr(self, 'raw') or self.raw is None:
                raise AttributeError("No raw data available for trimming")
            data = self.raw
        
        # Type checking
        if not isinstance(data, mne.io.BaseRaw):
            raise TypeError("Data must be an MNE Raw object for trimming")
            
        try:
            # Check if trim amount is valid
            total_duration = data.times[-1] - data.times[0]
            if trim_amount * 2 >= total_duration:
                raise ValueError(f"Trim amount ({trim_amount}s) is too large for data duration ({total_duration}s)")
                
            # Trim data
            message("header", "Trimming data edges...")
            start_time = data.times[0] + trim_amount
            end_time = data.times[-1] - trim_amount
            result_data = data.copy().crop(tmin=start_time, tmax=end_time)
            message("info", f"Trimmed {trim_amount}s from each end of the data")
            
            # Update metadata if we have a config
            if hasattr(self, 'config') and self.config.get("run_id"):
                metadata = {
                    "trim_data_edges": {
                        "creationDateTime": datetime.now().isoformat(),
                        "trim_amount": trim_amount,
                        "original_duration": total_duration,
                        "trimmed_duration": end_time - start_time
                    }
                }
                
                run_id = self.config.get("run_id")
                manage_database(
                    operation="update", update_record={"run_id": run_id, "metadata": metadata}
                )
                
                # Save the result if we have a config
                save_raw_to_set(result_data, self.config, f"post_{stage_name}")
            
            # Update self.raw if we're using it
            if data is self.raw:
                self.raw = result_data
                
            return result_data
            
        except Exception as e:
            message("error", f"Error during edge trimming: {str(e)}")
            raise RuntimeError(f"Failed to trim data edges: {str(e)}") from e
            
    def clean_bad_channels(self, data: Union[mne.io.BaseRaw, None] = None,
                           correlation_thresh: float = 0.35,
                           deviation_thresh: float = 6.0,
                           ransac_corr_thresh: float = 0.5,
                           ransac_channel_wise: bool = False,
                           random_state: int = 1337,
                           stage_name: str = "clean_bad_channels") -> mne.io.BaseRaw:
        """Detect and mark bad channels using various methods.
        
        This method uses the MNE NoisyChannels class to detect bad channels using SNR,
        correlation, deviation, and RANSAC methods.
        
        Args:
            data: Optional MNE Raw object. If None, uses self.raw
            correlation_thresh: Threshold for correlation-based detection
            deviation_thresh: Threshold for deviation-based detection
            ransac_corr_thresh: Threshold for RANSAC-based detection
            ransac_channel_wise: Whether to use channel-wise RANSAC
            random_state: Random state for reproducibility
            stage_name: Name for saving and metadata
            
        Returns:
            The raw data object with bad channels marked
            
        Raises:
            AttributeError: If self.raw doesn't exist when needed
            TypeError: If data is not a Raw object
            RuntimeError: If bad channel detection fails
        """
        # Determine which data to use
        if data is None:
            if not hasattr(self, 'raw') or self.raw is None:
                raise AttributeError("No raw data available for bad channel detection")
            data = self.raw
        
        # Type checking
        if not isinstance(data, mne.io.BaseRaw):
            raise TypeError("Data must be an MNE Raw object for bad channel detection")
            
        try:
            # Check if "eog" is in channel types and handle EOG channels if needed
            if hasattr(self, 'config') and self.config.get("task") and "eog" in data.get_channel_types():
                task = self.config.get("task")
                if not self.config.get("tasks", {}).get(task, {}).get("settings", {}).get("eog_step", {}).get("enabled", True):
                    # If EOG step is disabled, temporarily set EOG channels to EEG type
                    eog_picks = mne.pick_types(data.info, eog=True)
                    eog_ch_names = [data.ch_names[idx] for idx in eog_picks]
                    data.set_channel_types({ch: "eeg" for ch in eog_ch_names})
            
            # Create a copy of the data
            result_raw = data.copy()
            
            # Setup options
            options = {
                "random_state": random_state,
                "correlation_thresh": correlation_thresh,
                "deviation_thresh": deviation_thresh,
                "ransac_corr_thresh": ransac_corr_thresh,
                "ransac_channel_wise": ransac_channel_wise,
            }
            
            # Run noisy channels detection
            message("header", "Detecting bad channels...")
            cleaned_raw = NoisyChannels(result_raw, random_state=options["random_state"])
            cleaned_raw.find_bad_by_SNR()
            cleaned_raw.find_bad_by_correlation(
                correlation_secs=5.0,
                correlation_threshold=options["correlation_thresh"],
                frac_bad=0.1
            )
            cleaned_raw.find_bad_by_deviation(deviation_threshold=options["deviation_thresh"])
            cleaned_raw.find_bad_by_ransac(
                n_samples=100,
                sample_prop=0.5,
                corr_thresh=options["ransac_corr_thresh"],
                frac_bad=0.5,
                corr_window_secs=4.0,
                channel_wise=options["ransac_channel_wise"],
                max_chunk_size=None,
            )
            
            # Get bad channels and add them to the raw object
            bad_channels = cleaned_raw.get_bads(as_dict=True)
            result_raw.info["bads"].extend([str(ch) for ch in bad_channels["bad_by_ransac"]])
            result_raw.info["bads"].extend([str(ch) for ch in bad_channels["bad_by_deviation"]])
            result_raw.info["bads"].extend([str(ch) for ch in bad_channels["bad_by_correlation"]])
            result_raw.info["bads"].extend([str(ch) for ch in bad_channels["bad_by_SNR"]])
            
            # Remove duplicates
            result_raw.info["bads"] = list(set(result_raw.info["bads"]))
            
            message("info", f"Detected {len(result_raw.info['bads'])} bad channels: {result_raw.info['bads']}")
            
            # Update metadata if we have a config
            if hasattr(self, 'config') and self.config.get("run_id"):
                metadata = {
                    "clean_bad_channels": {
                        "creationDateTime": datetime.now().isoformat(),
                        "method": "NoisyChannels",
                        "options": options,
                        "bads": result_raw.info["bads"],
                        "channelCount": len(result_raw.ch_names),
                        "durationSec": int(result_raw.n_times) / result_raw.info["sfreq"],
                        "numberSamples": int(result_raw.n_times),
                    }
                }
                
                run_id = self.config.get("run_id")
                manage_database(
                    operation="update", update_record={"run_id": run_id, "metadata": metadata}
                )
                
                # Save the result if we have a config
                save_raw_to_set(result_raw, self.config, f"post_{stage_name}")
            
            # Update self.raw if we're using it
            if data is self.raw:
                self.raw = result_raw
                
            return result_raw
            
        except Exception as e:
            message("error", f"Error during bad channel detection: {str(e)}")
            raise RuntimeError(f"Failed to detect bad channels: {str(e)}") from e
