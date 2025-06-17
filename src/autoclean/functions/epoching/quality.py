"""Epoch quality assessment functions for EEG data.

This module provides standalone functions for assessing and improving epoch quality
through outlier detection and Global Field Power (GFP) cleaning.
"""

import random
from typing import Optional

import mne
import numpy as np


def detect_outlier_epochs(
    epochs: mne.Epochs,
    threshold: float = 3.0,
    measures: Optional[list] = None,
    return_scores: bool = False,
    verbose: Optional[bool] = None,
) -> mne.Epochs:
    """Detect and mark outlier epochs based on statistical measures.

    This function identifies epochs that are statistical outliers based on
    multiple measures, following principles from the FASTER algorithm. It
    calculates z-scores for various epoch properties and marks epochs as bad
    if they exceed the specified threshold in any measure.

    The default statistical measures include:
    - Mean amplitude across channels and time
    - Variance across channels and time
    - Maximum amplitude range (max - min)
    - Mean gradient (rate of change)

    This is particularly useful for removing epochs with extreme amplitude
    characteristics that could negatively impact subsequent processing steps
    like ICA or averaging.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs object to analyze for outliers.
    threshold : float, default 3.0
        The z-score threshold for outlier detection. Epochs with z-scores
        exceeding this threshold in any measure will be marked as bad.
    measures : list of str or None, default None
        List of measures to compute. If None, uses all available measures:
        ['mean', 'variance', 'range', 'gradient']. Available options:
        - 'mean': Mean amplitude across channels and time
        - 'variance': Variance across channels and time
        - 'range': Maximum amplitude range (max - min)
        - 'gradient': Mean gradient (rate of change)
    return_scores : bool, default False
        If True, returns a tuple of (epochs, scores_dict) where scores_dict
        contains the computed z-scores for each measure.
    verbose : bool or None, default None
        Control verbosity of output.

    Returns
    -------
    epochs_clean : mne.Epochs
        Copy of input epochs with outlier epochs marked as bad.
    scores : dict, optional
        Dictionary of z-scores for each measure (only if return_scores=True).
        Keys are measure names, values are arrays of z-scores for each epoch.

    Raises
    ------
    TypeError
        If epochs is not an MNE Epochs object.
    ValueError
        If threshold is not positive or measures contains invalid options.
    RuntimeError
        If outlier detection fails due to processing errors.

    Notes
    -----
    **Statistical Approach:**
    The function computes multiple statistical measures for each epoch and
    converts them to z-scores. Epochs with z-scores exceeding the threshold
    are considered outliers and marked as bad.

    **FASTER Algorithm:**
    This implementation is inspired by the FASTER (Fully Automated Statistical
    Thresholding for EEG artifact Rejection) algorithm, which uses statistical
    thresholding to identify artifacts in EEG data.

    **Measure Descriptions:**
    - **Mean**: Average amplitude across all channels and time points
    - **Variance**: Measure of amplitude variability
    - **Range**: Difference between maximum and minimum amplitudes
    - **Gradient**: Average rate of change (temporal derivative)

    **Threshold Selection:**
    - threshold=3.0: Conservative (marks ~0.3% of epochs as outliers)
    - threshold=2.5: Moderate (marks ~1.2% of epochs as outliers)
    - threshold=2.0: Liberal (marks ~4.6% of epochs as outliers)

    Examples
    --------
    Detect outliers with default parameters:

    >>> from autoclean import detect_outlier_epochs
    >>> clean_epochs = detect_outlier_epochs(epochs, threshold=3.0)

    Detect outliers with custom measures and return scores:

    >>> clean_epochs, scores = detect_outlier_epochs(
    ...     epochs,
    ...     threshold=2.5,
    ...     measures=['mean', 'variance'],
    ...     return_scores=True
    ... )

    More conservative outlier detection:

    >>> clean_epochs = detect_outlier_epochs(epochs, threshold=2.0)

    See Also
    --------
    mne.Epochs.drop_bad : Drop bad epochs
    autoclean.gfp_clean_epochs : Clean epochs using Global Field Power

    References
    ----------
    Nolan, H., Whelan, R., & Reilly, R. B. (2010). FASTER: fully automated
    statistical thresholding for EEG artifact rejection. Journal of
    neuroscience methods, 192(1), 152-162.
    """
    # Input validation
    if not isinstance(epochs, (mne.Epochs, mne.EpochsArray)):
        raise TypeError(
            f"epochs must be an MNE Epochs or EpochsArray object, got {type(epochs).__name__}"
        )

    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    if measures is None:
        measures = ["mean", "variance", "range", "gradient"]

    valid_measures = ["mean", "variance", "range", "gradient"]
    invalid_measures = [m for m in measures if m not in valid_measures]
    if invalid_measures:
        raise ValueError(
            f"Invalid measures: {invalid_measures}. Valid options: {valid_measures}"
        )

    try:
        # Create a copy to avoid modifying the original
        epochs_clean = epochs.copy()

        # Get epoch data
        data = epochs_clean.get_data()  # Shape: (n_epochs, n_channels, n_times)
        n_epochs, n_channels, n_times = data.shape

        if n_epochs == 0:
            # No epochs to process
            if return_scores:
                return epochs_clean, {}
            return epochs_clean

        # Initialize scores dictionary
        scores = {}
        outlier_epochs = set()

        # Compute statistical measures
        if "mean" in measures:
            # Mean amplitude across channels and time
            mean_vals = np.mean(np.abs(data), axis=(1, 2))
            z_scores = np.abs((mean_vals - np.mean(mean_vals)) / np.std(mean_vals))
            scores["mean"] = z_scores
            outliers = np.where(z_scores > threshold)[0]
            outlier_epochs.update(outliers)

        if "variance" in measures:
            # Variance across channels and time
            var_vals = np.var(data, axis=(1, 2))
            z_scores = np.abs((var_vals - np.mean(var_vals)) / np.std(var_vals))
            scores["variance"] = z_scores
            outliers = np.where(z_scores > threshold)[0]
            outlier_epochs.update(outliers)

        if "range" in measures:
            # Amplitude range (max - min)
            range_vals = np.max(data, axis=(1, 2)) - np.min(data, axis=(1, 2))
            z_scores = np.abs((range_vals - np.mean(range_vals)) / np.std(range_vals))
            scores["range"] = z_scores
            outliers = np.where(z_scores > threshold)[0]
            outlier_epochs.update(outliers)

        if "gradient" in measures:
            # Mean gradient (rate of change)
            gradients = np.diff(data, axis=2)  # Temporal gradient
            mean_gradient = np.mean(np.abs(gradients), axis=(1, 2))
            z_scores = np.abs(
                (mean_gradient - np.mean(mean_gradient)) / np.std(mean_gradient)
            )
            scores["gradient"] = z_scores
            outliers = np.where(z_scores > threshold)[0]
            outlier_epochs.update(outliers)

        # Mark outlier epochs as bad
        outlier_list = sorted(list(outlier_epochs))
        if outlier_list:
            epochs_clean.drop(outlier_list, reason="OUTLIER")
            if verbose:
                print(
                    f"Marked {len(outlier_list)} epochs as outliers (threshold={threshold})"
                )
        elif verbose:
            print("No outlier epochs detected")

        if return_scores:
            return epochs_clean, scores
        return epochs_clean

    except Exception as e:
        raise RuntimeError(f"Failed to detect outlier epochs: {str(e)}") from e


def gfp_clean_epochs(
    epochs: mne.Epochs,
    gfp_threshold: float = 3.0,
    number_of_epochs: Optional[int] = None,
    random_seed: Optional[int] = None,
    return_gfp_values: bool = False,
    verbose: Optional[bool] = None,
) -> mne.Epochs:
    """Clean epochs based on Global Field Power (GFP) outlier detection.

    This function removes epochs with abnormal Global Field Power values,
    which represent the spatial standard deviation across all electrodes
    at each time point. GFP is useful for identifying epochs with widespread
    artifacts affecting multiple channels simultaneously.

    Optionally, the function can randomly subsample epochs after cleaning
    to achieve a target number of epochs for analysis.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs object to clean.
    gfp_threshold : float, default 3.0
        The z-score threshold for GFP-based outlier detection. Epochs with
        mean GFP z-scores exceeding this threshold will be removed.
    number_of_epochs : int or None, default None
        If specified, randomly selects this number of epochs from the cleaned
        data. Useful for ensuring consistent epoch counts across conditions.
    random_seed : int or None, default None
        Seed for random number generator when selecting epochs. Ensures
        reproducible results when number_of_epochs is specified.
    return_gfp_values : bool, default False
        If True, returns a tuple of (epochs_clean, gfp_values) where gfp_values
        contains the computed GFP z-scores for each original epoch.
    verbose : bool or None, default None
        Control verbosity of output.

    Returns
    -------
    epochs_clean : mne.Epochs
        The cleaned epochs object with GFP outliers removed and optionally
        randomly subsampled.
    gfp_values : np.ndarray, optional
        Array of GFP z-scores for each original epoch (only if return_gfp_values=True).

    Raises
    ------
    TypeError
        If epochs is not an MNE Epochs object.
    ValueError
        If threshold is not positive, number_of_epochs is invalid, or
        insufficient epochs remain after cleaning.
    RuntimeError
        If GFP cleaning fails due to processing errors.

    Notes
    -----
    **Global Field Power (GFP):**
    GFP is calculated as the spatial standard deviation across all electrodes
    at each time point. It provides a reference-free measure of global brain
    activity and is particularly sensitive to artifacts that affect multiple
    channels simultaneously.

    **GFP Formula:**
    GFP(t) = sqrt(sum((V_i(t) - V_mean(t))^2) / N)
    where V_i(t) is the voltage at electrode i and time t, V_mean(t) is the
    mean voltage across all electrodes at time t, and N is the number of electrodes.

    **Outlier Detection:**
    The function computes the mean GFP for each epoch, converts to z-scores,
    and removes epochs exceeding the threshold. This approach effectively
    identifies epochs with abnormal global activity patterns.

    **Random Subsampling:**
    When number_of_epochs is specified, the function randomly selects epochs
    from the cleaned data. This is useful for:
    - Equalizing epoch counts across conditions
    - Reducing computational load for subsequent analyses
    - Creating balanced datasets for machine learning

    Examples
    --------
    Clean epochs with default GFP threshold:

    >>> from autoclean import gfp_clean_epochs
    >>> clean_epochs = gfp_clean_epochs(epochs, gfp_threshold=3.0)

    Clean and subsample to specific number of epochs:

    >>> clean_epochs = gfp_clean_epochs(
    ...     epochs,
    ...     gfp_threshold=2.5,
    ...     number_of_epochs=40,
    ...     random_seed=42
    ... )

    Clean epochs and return GFP values for analysis:

    >>> clean_epochs, gfp_scores = gfp_clean_epochs(
    ...     epochs,
    ...     return_gfp_values=True
    ... )

    See Also
    --------
    mne.Epochs.drop_bad : Drop bad epochs
    autoclean.detect_outlier_epochs : Detect outliers using multiple measures

    References
    ----------
    Lehmann, D., & Skrandies, W. (1980). Reference-free identification of
    components of checkerboard-evoked multichannel potential fields.
    Electroencephalography and clinical neurophysiology, 48(6), 609-621.
    """
    # Input validation
    if not isinstance(epochs, (mne.Epochs, mne.EpochsArray)):
        raise TypeError(
            f"epochs must be an MNE Epochs or EpochsArray object, got {type(epochs).__name__}"
        )

    if gfp_threshold <= 0:
        raise ValueError(f"gfp_threshold must be positive, got {gfp_threshold}")

    if number_of_epochs is not None and number_of_epochs <= 0:
        raise ValueError(f"number_of_epochs must be positive, got {number_of_epochs}")

    try:
        # Create a copy to avoid modifying the original
        epochs_clean = epochs.copy()

        # Get epoch data
        data = epochs_clean.get_data()  # Shape: (n_epochs, n_channels, n_times)
        n_epochs, n_channels, n_times = data.shape

        if n_epochs == 0:
            # No epochs to process
            if return_gfp_values:
                return epochs_clean, np.array([])
            return epochs_clean

        # Calculate Global Field Power for each epoch
        gfp_values = []

        for epoch_idx in range(n_epochs):
            epoch_data = data[epoch_idx]  # Shape: (n_channels, n_times)

            # Calculate GFP at each time point
            # GFP = sqrt(mean((V - V_mean)^2)) where V_mean is spatial mean
            spatial_mean = np.mean(epoch_data, axis=0)  # Mean across channels
            gfp_timepoints = np.sqrt(np.mean((epoch_data - spatial_mean) ** 2, axis=0))

            # Use mean GFP across time as the epoch's GFP value
            mean_gfp = np.mean(gfp_timepoints)
            gfp_values.append(mean_gfp)

        gfp_values = np.array(gfp_values)

        # Calculate z-scores for GFP values
        gfp_mean = np.mean(gfp_values)
        gfp_std = np.std(gfp_values)

        if gfp_std == 0:
            # All GFP values are identical, no outliers
            gfp_z_scores = np.zeros_like(gfp_values)
        else:
            gfp_z_scores = np.abs((gfp_values - gfp_mean) / gfp_std)

        # Identify outlier epochs
        outlier_indices = np.where(gfp_z_scores > gfp_threshold)[0]

        # Remove outlier epochs
        if len(outlier_indices) > 0:
            epochs_clean.drop(outlier_indices, reason="GFP_OUTLIER")
            if verbose:
                print(
                    f"Removed {len(outlier_indices)} epochs based on GFP outlier detection"
                )
        elif verbose:
            print("No GFP outlier epochs detected")

        # Random subsampling if requested
        if number_of_epochs is not None:
            remaining_epochs = len(epochs_clean)

            if remaining_epochs < number_of_epochs:
                raise ValueError(
                    f"Insufficient epochs after GFP cleaning. "
                    f"Requested {number_of_epochs}, but only {remaining_epochs} remain."
                )

            # Set random seed for reproducibility
            if random_seed is not None:
                random.seed(random_seed)
                np.random.seed(random_seed)

            # Randomly select epochs
            selected_indices = np.random.choice(
                remaining_epochs, size=number_of_epochs, replace=False
            )
            selected_indices = np.sort(selected_indices)

            # Create new epochs object with selected epochs
            epochs_clean = epochs_clean[selected_indices]

            if verbose:
                print(
                    f"Randomly selected {number_of_epochs} epochs from {remaining_epochs} available"
                )

        if return_gfp_values:
            return epochs_clean, gfp_z_scores
        return epochs_clean

    except Exception as e:
        raise RuntimeError(f"Failed to clean epochs using GFP: {str(e)}") from e
