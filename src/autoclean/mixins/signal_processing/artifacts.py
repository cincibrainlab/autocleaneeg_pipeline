"""Artifacts detection and rejection mixin for autoclean tasks."""

from typing import Optional, Union

import mne
import numpy as np

from autoclean.utils.logging import message


class ArtifactsMixin:
    """Mixin class providing artifact detection and rejection functionality for EEG data."""

    def detect_dense_oscillatory_artifacts(
        self,
        data: Union[mne.io.Raw, None] = None,
        window_size_ms: int = 100,
        channel_threshold_uv: float = 45,
        min_channels: int = 75,
        padding_ms: float = 500,
        annotation_label: str = "BAD_REF_AF",
    ) -> mne.io.Raw:
        """Detect smaller, dense oscillatory multichannel artifacts.

        This method identifies oscillatory artifacts that affect multiple channels simultaneously,
        while excluding large single deflections.

        Parameters
        ----------
        data : mne.io.Raw, Optional
            The raw data to detect artifacts from. If None, uses self.raw.
        window_size_ms : int, Optional
            Window size in milliseconds for artifact detection, by default 100.
        channel_threshold_uv : float, Optional
            Threshold for peak-to-peak amplitude in microvolts, by default 45.
        min_channels : int, Optional
            Minimum number of channels that must exhibit oscillations, by default 75.
        padding_ms : float, Optional
            Amount of padding in milliseconds to add before and after each detected artifact, 
            by default 500.
        annotation_label : str, Optional
            Label to use for the annotations, by default "BAD_REF_AF".
        stage_name : str, Optional
            Name for saving and metadata, by default "detect_dense_oscillatory_artifacts".

        Returns
        -------
        result_raw : instance of mne.io.Raw
            The raw data object with updated artifact annotations. 
            *Note the self.raw is updated in place. So the return value is optional.*

        Examples
        --------
        >>> #Inside a task class that uses the autoclean framework
        >>> self.detect_dense_oscillatory_artifacts()
        >>> #Or with custom parameters
        >>> self.detect_dense_oscillatory_artifacts(window_size_ms=200, channel_threshold_uv=50, 
        min_channels=100, padding_ms=1000, annotation_label="BAD_CUSTOM_ARTIFACT")

        Notes
        -----
        This method is intended to find reference artifacts 
        but may also be triggered by other artifacts.
        """
        # Determine which data to use
        data = self._get_data_object(data)

        # Type checking
        if not isinstance(data, mne.io.Raw) and not isinstance(
            data, mne.io.base.BaseRaw
        ):
            raise TypeError("Data must be an MNE Raw object for artifact detection")

        try:
            # Convert parameters to samples and volts
            sfreq = data.info["sfreq"]
            window_size = int(window_size_ms * sfreq / 1000)
            channel_threshold = channel_threshold_uv * 1e-6  # Convert µV to V
            padding_sec = padding_ms / 1000.0  # Convert padding to seconds

            # Get data and times
            raw_data, times = data.get_data(return_times=True)
            _, n_samples = raw_data.shape

            artifact_annotations = []

            # Sliding window detection
            for start_idx in range(0, n_samples - window_size, window_size):
                window = raw_data[:, start_idx : start_idx + window_size]

                # Compute peak-to-peak amplitude for each channel in the window
                ptp_amplitudes = np.ptp(
                    window, axis=1
                )  # Peak-to-peak amplitude per channel

                # Count channels exceeding the threshold
                num_channels_exceeding = np.sum(ptp_amplitudes > channel_threshold)

                # Check if artifact spans multiple channels with oscillatory behavior
                if num_channels_exceeding >= min_channels:
                    start_time = times[start_idx] - padding_sec  # Add padding before
                    end_time = (
                        times[start_idx + window_size] + padding_sec
                    )  # Add padding after

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
                        onset=annotation[0],
                        duration=annotation[1],
                        description=annotation[2],
                    )
                message(
                    "info", f"Added {len(artifact_annotations)} potential reference artifact annotations"
                )
            else:
                message("info", "No reference artifacts detected")

            # Add flags if needed
            if len(artifact_annotations) > self.REFERENCE_ARTIFACT_THRESHOLD:
                flagged_reason = f"WARNING: {len(artifact_annotations)} potential reference artifacts detected" # pylint: disable=line-too-long
                self._update_flagged_status(flagged=True, reason=flagged_reason)

            # Update metadata
            metadata = {
                "window_size_ms": window_size_ms,
                "channel_threshold_uv": channel_threshold_uv,
                "min_channels": min_channels,
                "padding_ms": padding_ms,
                "annotation_label": annotation_label,
                "artifacts_detected": len(artifact_annotations),
            }

            self._update_metadata("step_detect_dense_oscillatory_artifacts", metadata)

            # Save the result
            # self._save_raw_result(result_raw, stage_name)

            # Update self.raw if we're using it
            self._update_instance_data(data, result_raw)

            return result_raw

        except Exception as e:
            message("error", f"Error during artifact detection: {str(e)}")
            raise RuntimeError(f"Failed to detect artifacts: {str(e)}") from e

    def detect_muscle_beta_focus(
        self,
        data: Union[mne.io.Raw, None] = None,
        freq_band: tuple = (20, 30),
        scale_factor: float = 3.0,
        window_length: float = 1.0,
        window_overlap: float = 0.5,
        annotation_description: str = "BAD_MOVEMENT",
    ) -> mne.io.Raw:
        """Detect muscle artifacts in continuous Raw data and add annotations.

        This method detects muscle artifacts in continuous EEG data by analyzing
        high-frequency activity in peripheral electrodes. It automatically adds
        annotations to the Raw object marking segments with detected artifacts.

        Parameters
        ----------
        data : mne.io.Raw, Optional
            The raw data to detect artifacts from. If None, uses self.raw.
        freq_band : tuple, Optional
            Frequency band for filtering (min, max), by default (20, 30).
        scale_factor : float, Optional
            Scale factor for threshold calculation, by default 3.0.
        window_length : float, Optional
            Length of sliding window in seconds, by default 1.0.
        window_overlap : float, Optional
            Overlap between windows as a fraction (0-1), by default 0.5.
        annotation_description : str, Optional
            Description for the annotations, by default "BAD_MOVEMENT".
        Returns
        -------
        results_raw : instance of mne.io.Raw
            The raw data object with updated artifact annotations. 
            *Note the self.raw is updated in place. So the return value is optional.*

        Examples
        --------
        >>> #Inside a task class that uses the autoclean framework
        >>> self.detect_muscle_beta_focus()
        >>> #Or with custom parameters
        >>> self.detect_muscle_beta_focus(freq_band=(20, 30), scale_factor=4.0, window_length=2.0, 
        window_overlap=0.7, annotation_description="BAD_CUSTOM_ARTIFACT")
        """
        # Determine which data to use
        data = self._get_data_object(data)

        # Type checking
        if not isinstance(data, mne.io.Raw) and not isinstance(
            data, mne.io.base.BaseRaw
        ):
            raise TypeError(
                "Data must be an MNE Raw object for muscle artifact detection"
            )

        # Ensure data is loaded
        data.load_data()

        # Create a copy to work with
        results_raw = data.copy()

        # Filter in beta/gamma band
        raw_beta = data.copy().filter(
            l_freq=freq_band[0], h_freq=freq_band[1], verbose=False
        )

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

        # Get channel names
        ch_names = raw_beta.ch_names

        # Select only OTHER channels
        selected_ch_indices = [
            i
            for i, ch in enumerate(ch_names)
            if channel_region_map.get(ch, "") == "OTHER"
        ]

        # If no OTHER channels are found, return
        if not selected_ch_indices:
            message("info", "No 'OTHER' channels found for muscle artifact detection")
            return None

        # Calculate window parameters
        sfreq = raw_beta.info["sfreq"]
        n_samples = len(raw_beta.times)
        window_samples = int(window_length * sfreq)
        step_samples = int(window_samples * (1 - window_overlap))

        # Create sliding windows
        n_windows = max(1, int((n_samples - window_samples) / step_samples) + 1)

        # Store peak-to-peak values for each window
        max_p2p_values = []
        window_times = []

        # Process each window
        for i in range(n_windows):
            start_sample = i * step_samples
            end_sample = min(start_sample + window_samples, n_samples)

            # Skip if window is too small
            if end_sample - start_sample < window_samples / 2:
                continue

            # Extract data for this window (only selected channels)
            window_data = raw_beta.get_data(
                picks=selected_ch_indices, start=start_sample, stop=end_sample
            )

            # Compute peak-to-peak amplitude per channel
            p2p = window_data.max(axis=1) - window_data.min(axis=1)

            # Compute maximum peak-to-peak amplitude across channels
            max_p2p = np.max(p2p)
            max_p2p_values.append(max_p2p)

            # Store window time boundaries
            start_time = start_sample / sfreq
            end_time = end_sample / sfreq
            window_times.append((start_time, end_time))

        # Compute median and MAD
        max_p2p_values = np.array(max_p2p_values)
        med = np.median(max_p2p_values)
        mad = np.median(np.abs(max_p2p_values - med))

        # Robust threshold
        threshold = med + scale_factor * mad

        # Identify bad windows
        bad_window_indices = np.where(max_p2p_values > threshold)[0].tolist()
        bad_windows = [window_times[i] for i in bad_window_indices]

        # Add annotations
        if bad_windows:
            # Merge overlapping windows
            merged_windows = self._merge_overlapping_windows(bad_windows)

            # Add annotations
            for start, end in merged_windows:
                results_raw.annotations.append(
                    onset=start,
                    duration=end - start,
                    description=annotation_description,
                )

            message(
                "info",
                f"Added {len(merged_windows)} {annotation_description} annotations to Raw data",
            )

            # Update the original data with the new annotations
            self._update_instance_data(data, results_raw)
        else:
            message("info", "No muscle artifacts detected")

        # Update metadata
        metadata = {
            "freq_band": freq_band,
            "scale_factor": scale_factor,
            "window_length": window_length,
            "window_overlap": window_overlap,
            "annotation_description": annotation_description,
        }

        self._update_metadata("step_detect_muscle_artifacts", metadata)

        return results_raw

    def _merge_overlapping_windows(self, windows):
        """Merge overlapping time windows.

        Args:
            windows : List of tuples (start_time, end_time) in seconds

        Returns
        -------
            List of merged tuples (start_time, end_time) with no overlaps
        """
        if not windows:
            return []

        # Sort windows by start time
        sorted_windows = sorted(windows, key=lambda x: x[0])

        # Initialize with the first window
        merged = [sorted_windows[0]]

        # Iterate through remaining windows
        for current in sorted_windows[1:]:
            previous = merged[-1]

            # If current window overlaps with previous, merge them
            if current[0] <= previous[1]:
                merged[-1] = (previous[0], max(previous[1], current[1]))
            else:
                merged.append(current)

        return merged

    def reject_bad_segments(
        self,
        data: Union[mne.io.Raw, None] = None,
        bad_label: Optional[str] = None,
        stage_name: str = "bad_segment_rejection",
    ) -> mne.io.Raw:
        """Remove all time spans annotated with a specific label or all 'BAD' segments.

        This method removes segments marked as bad and concatenates the remaining good segments.

        Parameters
        ----------
        data : mne.io.Raw, Optional
            The raw data to detect artifacts from. If None, uses self.raw.
        bad_label : str, Optional
            Specific label of annotations to reject. If None, rejects all segments
                      where description starts with 'BAD'
        stage_name : str, Optional
            Name for saving and metadata, by default "bad_segment_rejection".

        Returns
        -------
        raw_cleaned : instance of mne.io.Raw
            The raw data object with updated artifact annotations. 
            *Note the self.raw is updated in place. So the return value is optional.*.

        Examples
        --------
        >>> #Inside a task class that uses the autoclean framework
        >>> self.reject_bad_segments()
        >>> #Or with custom label
        >>> self.reject_bad_segments(bad_label="BAD_CUSTOM_ARTIFACT")

        """
        # Determine which data to use
        data = self._get_data_object(data)

        # Type checking
        if not isinstance(data, mne.io.Raw):
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
                "cleaned_duration": raw_cleaned.times[-1],
            }

            self._update_metadata("step_reject_bad_segments", metadata)

            # Save the result
            self._save_raw_result(raw_cleaned, stage_name)

            # Update self.raw if we're using it
            self._update_instance_data(data, raw_cleaned)

            return raw_cleaned

        except Exception as e:
            message("error", f"Error during segment rejection: {str(e)}")
            raise RuntimeError(f"Failed to reject bad segments: {str(e)}") from e
