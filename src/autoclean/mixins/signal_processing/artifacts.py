"""Artifacts detection and rejection mixin for autoclean tasks."""

from typing import Union, Optional, List, Tuple
import mne
import numpy as np

from autoclean.utils.logging import message

class ArtifactsMixin:
    """Mixin class providing artifact detection and rejection functionality for EEG data."""
    
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
        data = self._get_data_object(data)
        
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
                
            # Update metadata
            metadata = {
                "window_size_ms": window_size_ms,
                "channel_threshold_uv": channel_threshold_uv,
                "min_channels": min_channels,
                "padding_ms": padding_ms,
                "annotation_label": annotation_label,
                "artifacts_detected": len(artifact_annotations)
            }
            
            self._update_metadata("detect_dense_oscillatory_artifacts", metadata)
            
            # Save the result
            self._save_raw_result(result_raw, stage_name)
            
            # Update self.raw if we're using it
            self._update_instance_data(data, result_raw)
                
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
        data = self._get_data_object(data)
        
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
            
            # Update metadata
            metadata = {
                "bad_label": bad_label if bad_label else "All BAD*",
                "segments_removed": len(bad_intervals),
                "segments_kept": len(good_intervals),
                "original_duration": data.times[-1],
                "cleaned_duration": raw_cleaned.times[-1]
            }
            
            self._update_metadata("reject_bad_segments", metadata)
            
            # Save the result
            self._save_raw_result(raw_cleaned, stage_name)
            
            # Update self.raw if we're using it
            self._update_instance_data(data, raw_cleaned)
                
            return raw_cleaned
            
        except Exception as e:
            message("error", f"Error during segment rejection: {str(e)}")
            raise RuntimeError(f"Failed to reject bad segments: {str(e)}") from e
