"""Dense oscillatory artifact detection for EEG data.

This module provides standalone functions for identifying and annotating dense
oscillatory multichannel artifacts in continuous EEG data.
"""

from typing import Optional

import mne
import numpy as np


def detect_dense_oscillatory_artifacts(
    raw: mne.io.Raw,
    window_size_ms: int = 100,
    channel_threshold_uv: float = 45,
    min_channels: int = 75,
    padding_ms: float = 500,
    annotation_label: str = "BAD_REF_AF",
    verbose: Optional[bool] = None,
) -> mne.io.Raw:
    """Detect smaller, dense oscillatory multichannel artifacts.

    This function identifies oscillatory artifacts that affect multiple channels
    simultaneously, while excluding large single deflections. It uses a sliding
    window approach to detect periods where many channels simultaneously show
    high peak-to-peak amplitudes.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data to analyze for oscillatory artifacts.
    window_size_ms : int, default 100
        Window size in milliseconds for artifact detection.
    channel_threshold_uv : float, default 45
        Threshold for peak-to-peak amplitude in microvolts.
    min_channels : int, default 75
        Minimum number of channels that must exhibit oscillations.
    padding_ms : float, default 500
        Amount of padding in milliseconds to add before and after each detected
        artifact.
    annotation_label : str, default "BAD_REF_AF"
        Label to use for the annotations.
    verbose : bool or None, default None
        Control verbosity of output.

    Returns
    -------
    raw_annotated : mne.io.Raw
        Copy of the input Raw object with added annotations for detected
        oscillatory artifacts.

    Raises
    ------
    TypeError
        If raw is not an MNE Raw object.
    ValueError
        If parameters are outside valid ranges.
    RuntimeError
        If processing fails due to insufficient data or other errors.

    Notes
    -----
    **Detection Algorithm:**
    1. Uses a sliding window across the continuous data
    2. For each window, calculates peak-to-peak amplitude for each channel
    3. Counts channels exceeding the amplitude threshold
    4. Marks windows where channel count exceeds min_channels threshold
    5. Adds padding around detected artifacts to ensure complete removal

    **Parameter Guidelines:**
    - window_size_ms: 50-200ms typical. Shorter for transient artifacts
    - channel_threshold_uv: 30-60µV typical. Depends on preprocessing
    - min_channels: 50-100 typical. Should be ~50-75% of total channels
    - padding_ms: 200-1000ms typical. Ensures artifact boundaries captured

    **Performance Considerations:**
    - Processing time scales with data length and window overlap
    - Memory usage depends on number of channels and sampling rate
    - Larger windows are more computationally efficient but less precise

    Examples
    --------
    Basic oscillatory artifact detection:

    >>> from autoclean.functions.segment_rejection import detect_dense_oscillatory_artifacts
    >>> raw_clean = detect_dense_oscillatory_artifacts(raw)
    >>> artifacts = [ann for ann in raw_clean.annotations
    ...              if 'REF_AF' in ann['description']]
    >>> print(f"Found {len(artifacts)} oscillatory artifacts")

    Conservative detection for clean data:

    >>> raw_clean = detect_dense_oscillatory_artifacts(
    ...     raw,
    ...     window_size_ms=150,
    ...     channel_threshold_uv=60,
    ...     min_channels=100,
    ...     padding_ms=1000
    ... )

    Sensitive detection for noisy data:

    >>> raw_clean = detect_dense_oscillatory_artifacts(
    ...     raw,
    ...     window_size_ms=75,
    ...     channel_threshold_uv=30,
    ...     min_channels=50,
    ...     annotation_label="BAD_oscillatory"
    ... )

    See Also
    --------
    annotate_noisy_segments : General segment-level noise detection
    annotate_uncorrelated_segments : Correlation-based segment rejection
    mne.Annotations : MNE annotations system

    References
    ----------
    This implementation is designed to detect reference artifacts and other
    dense oscillatory patterns that affect multiple channels simultaneously.
    """
    # Input validation
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(f"Data must be an MNE Raw object, got {type(raw).__name__}")

    if window_size_ms <= 0:
        raise ValueError(f"window_size_ms must be positive, got {window_size_ms}")

    if channel_threshold_uv <= 0:
        raise ValueError(
            f"channel_threshold_uv must be positive, got {channel_threshold_uv}"
        )

    if min_channels <= 0:
        raise ValueError(f"min_channels must be positive, got {min_channels}")

    if padding_ms < 0:
        raise ValueError(f"padding_ms must be non-negative, got {padding_ms}")

    try:
        # Convert parameters to samples and volts
        sfreq = raw.info["sfreq"]
        window_size = int(window_size_ms * sfreq / 1000)
        channel_threshold = channel_threshold_uv * 1e-6  # Convert µV to V
        padding_sec = padding_ms / 1000.0  # Convert padding to seconds

        # Get data and times
        raw_data, times = raw.get_data(return_times=True)
        _, n_samples = raw_data.shape

        artifact_annotations = []

        # Sliding window detection
        for start_idx in range(0, n_samples - window_size, window_size):
            window = raw_data[:, start_idx : start_idx + window_size]

            # Compute peak-to-peak amplitude for each channel in the window
            ptp_amplitudes = np.ptp(window, axis=1)

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
        raw_annotated = raw.copy()

        # Add annotations to the raw data
        if artifact_annotations:
            for annotation in artifact_annotations:
                raw_annotated.annotations.append(
                    onset=annotation[0],
                    duration=annotation[1],
                    description=annotation[2],
                )
            if verbose:
                print(
                    f"Added {len(artifact_annotations)} oscillatory artifact annotations"
                )
        else:
            if verbose:
                print("No oscillatory artifacts detected")

        return raw_annotated

    except Exception as e:
        raise RuntimeError(f"Failed to detect oscillatory artifacts: {str(e)}") from e
