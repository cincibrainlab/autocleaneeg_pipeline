"""AutoReject-based epoch cleaning functions for EEG data.

This module provides standalone functions for cleaning epochs using AutoReject,
a machine learning-based method for automatic artifact rejection in EEG data.
AutoReject automatically identifies and removes bad epochs and interpolates 
bad channels within epochs.
"""

from typing import Dict, List, Optional, Tuple

import mne
from autoreject import AutoReject


def autoreject_epochs(
    epochs: mne.Epochs,
    n_interpolate: Optional[List[int]] = None,
    consensus: Optional[List[float]] = None,
    n_jobs: int = 1,
    cv: int = 4,
    random_state: Optional[int] = None,
    picks: Optional[List[str]] = None,
    thresh_method: str = "bayesian_optimization",
    verbose: Optional[bool] = None
) -> Tuple[mne.Epochs, Dict]:
    """Apply AutoReject for automatic epoch cleaning and channel interpolation.
    
    This function applies the AutoReject algorithm to clean epochs by identifying
    and removing bad epochs and interpolating bad channels within epochs. AutoReject
    is a machine learning-based method that automatically determines optimal 
    thresholds for artifact rejection, reducing the need for manual inspection.
    
    The method uses a cross-validation approach to determine the optimal parameters
    for artifact rejection, including the number of channels to interpolate and the
    consensus threshold. These parameters can be customized through the function
    arguments or determined automatically by the algorithm.
    
    AutoReject works by:
    1. Creating a grid of rejection thresholds and interpolation parameters
    2. Using cross-validation to find optimal parameters for each channel
    3. Applying the learned thresholds to identify bad epochs and channels
    4. Interpolating bad channels and rejecting bad epochs
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epoched EEG data to clean. Must have at least 4 epochs for 
        cross-validation to work properly.
    n_interpolate : list of int or None, default None
        List of number of channels to interpolate for parameter search.
        If None, uses [1, 4, 8] as default values. Higher values allow
        more channel interpolation but may reduce data quality.
    consensus : list of float or None, default None
        List of consensus percentages for parameter search (0.0-1.0).
        If None, uses [0.1, 0.25, 0.5, 0.75, 0.9] as default values.
        Higher values are more conservative (fewer rejections).
    n_jobs : int, default 1
        Number of parallel jobs to run for cross-validation. Set to -1 
        to use all available CPU cores. Higher values speed up computation
        but use more memory.
    cv : int, default 4
        Number of cross-validation folds for parameter optimization.
        Must be at least 2. Higher values provide more robust parameter
        estimates but increase computation time.
    random_state : int or None, default None
        Random seed for reproducible results in cross-validation splits.
        Set to an integer for reproducible results across runs.
    picks : list of str or None, default None
        Channel names to include in the analysis. If None, uses all EEG
        channels. Non-EEG channels are automatically excluded.
    thresh_method : str, default 'bayesian_optimization'
        Method for threshold optimization. Options:
        - 'bayesian_optimization': Uses Bayesian optimization (recommended)
        - 'random_search': Uses random search (faster but less optimal)
    verbose : bool or None, default None
        Control verbosity of output during processing.
        
    Returns
    -------
    epochs_clean : mne.Epochs
        The cleaned epochs object with bad epochs removed and bad channels
        interpolated. May contain fewer epochs than the input.
    metadata : dict
        Dictionary containing detailed information about the cleaning process:
        - 'initial_epochs': Number of epochs before cleaning
        - 'final_epochs': Number of epochs after cleaning  
        - 'rejected_epochs': Number of epochs rejected
        - 'rejection_percent': Percentage of epochs rejected
        - 'epoch_duration': Duration of each epoch in seconds
        - 'samples_per_epoch': Number of time samples per epoch
        - 'total_duration_sec': Total duration of cleaned data
        - 'total_samples': Total number of samples in cleaned data
        - 'channel_count': Number of channels
        - 'interpolated_channels': Channels that were interpolated
        - 'n_interpolate': Parameter values used
        - 'consensus': Parameter values used
        - 'cv_scores': Cross-validation scores for parameter selection
        
    Raises
    ------
    TypeError
        If epochs is not an MNE Epochs object.
    ValueError
        If parameters are outside valid ranges or insufficient data for CV.
    ImportError
        If AutoReject package is not installed.
    RuntimeError
        If AutoReject processing fails.
        
    Notes
    -----
    **Algorithm Overview:**
    AutoReject uses a cross-validation approach to learn optimal rejection
    thresholds for each channel individually. It creates a grid search over
    possible numbers of channels to interpolate and consensus thresholds,
    then uses CV to find the best combination.
    
    **Parameter Guidelines:**
    - n_interpolate: Start with [1, 4, 8]. For high-density arrays, consider
      [1, 4, 8, 16]. For low-density arrays, use [1, 2, 4].
    - consensus: [0.1, 0.25, 0.5, 0.75, 0.9] covers range from liberal to
      conservative rejection. Lower values = more aggressive rejection.
    - n_jobs: Use -1 for maximum speed on multi-core systems.
    - cv: 4-5 folds typical. Higher values more robust but slower.
    
    **Memory and Performance:**
    - Memory usage scales with (n_epochs × n_channels × n_times × cv)
    - For large datasets, consider reducing cv or chunking epochs
    - Processing time: ~1-10 minutes for typical datasets (64 channels, 100+ epochs)
    
    **Quality Considerations:**
    - Requires minimum 20-30 epochs for reliable parameter estimation
    - Best results with 100+ epochs for robust cross-validation
    - Interpolated channels maintain spatial relationships
    - Aggressive rejection (>50% epochs) may indicate poor data quality
    
    Examples
    --------
    Basic usage with default parameters:
    
    >>> from autoclean import autoreject_epochs
    >>> clean_epochs, metadata = autoreject_epochs(epochs)
    >>> print(f"Rejected {metadata['rejection_percent']:.1f}% of epochs")
    
    Conservative cleaning for high-quality data:
    
    >>> clean_epochs, metadata = autoreject_epochs(
    ...     epochs,
    ...     n_interpolate=[1, 2, 4],
    ...     consensus=[0.5, 0.75, 0.9],
    ...     n_jobs=4
    ... )
    
    Aggressive cleaning for noisy data:
    
    >>> clean_epochs, metadata = autoreject_epochs(
    ...     epochs,
    ...     n_interpolate=[1, 4, 8, 16],
    ...     consensus=[0.1, 0.25, 0.5],
    ...     random_state=42
    ... )
    
    Processing specific channels only:
    
    >>> clean_epochs, metadata = autoreject_epochs(
    ...     epochs,
    ...     picks=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4'],
    ...     n_jobs=-1
    ... )
    
    See Also
    --------
    autoreject.AutoReject : Underlying AutoReject implementation
    mne.preprocessing.ICA : Alternative artifact removal method
    autoclean.detect_outlier_epochs : Simpler statistical epoch rejection
    
    References
    ----------
    Jas, M., Engemann, D. A., Bekhti, Y., Raimondo, F., & Gramfort, A. (2017).
    Autoreject: Automated artifact rejection for MEG and EEG data. NeuroImage, 159, 417-429.
    
    Jas, M., Engemann, D. A., Raimondo, F., Bekhti, Y., & Gramfort, A. (2016).
    Automated rejection and repair of bad trials in MEG/EEG. In 2016 international 
    workshop on pattern recognition in neuroimaging (PRNI) (pp. 1-4). IEEE.
    """
    # Input validation
    if not isinstance(epochs, mne.Epochs):
        raise TypeError(
            f"Data must be an MNE Epochs object, got {type(epochs).__name__}"
        )
    
    if len(epochs) < cv:
        raise ValueError(
            f"Need at least {cv} epochs for {cv}-fold cross-validation, got {len(epochs)}"
        )
    
    if n_interpolate is not None:
        if not isinstance(n_interpolate, list) or not all(isinstance(x, int) and x >= 0 for x in n_interpolate):
            raise ValueError("n_interpolate must be a list of non-negative integers")
        if max(n_interpolate) >= len(epochs.ch_names):
            raise ValueError(f"Cannot interpolate more channels ({max(n_interpolate)}) than available ({len(epochs.ch_names)})")
    
    if consensus is not None:
        if not isinstance(consensus, list) or not all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in consensus):
            raise ValueError("consensus must be a list of values between 0 and 1")
    
    if cv < 2:
        raise ValueError("cv must be at least 2")
    
    if picks is not None:
        # Validate picks exist in epochs
        missing_picks = [ch for ch in picks if ch not in epochs.ch_names]
        if missing_picks:
            raise ValueError(f"Picks not found in data: {missing_picks}")
    
    # Set default parameters
    if n_interpolate is None:
        n_interpolate = [1, 4, 8]
    
    if consensus is None:
        consensus = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    try:
        # Create a copy to avoid modifying original data
        epochs_copy = epochs.copy()
        
        # Apply picks if specified
        if picks is not None:
            epochs_copy = epochs_copy.pick(picks)
        
        # Initialize AutoReject with specified parameters
        ar = AutoReject(
            n_interpolate=n_interpolate,
            consensus=consensus,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
            thresh_method=thresh_method,
            verbose=verbose
        )
        
        # Fit and transform epochs
        epochs_clean = ar.fit_transform(epochs_copy)
        
        # Calculate statistics
        initial_epochs = len(epochs_copy)
        final_epochs = len(epochs_clean)
        rejected_epochs = initial_epochs - final_epochs
        rejection_percent = (rejected_epochs / initial_epochs * 100) if initial_epochs > 0 else 0
        
        # Get information about interpolated channels
        interpolated_channels = []
        if hasattr(ar, 'bad_segments_'):
            # Extract channels that were interpolated in any epoch
            for epoch_idx in range(ar.bad_segments_.shape[0]):
                for ch_idx in range(ar.bad_segments_.shape[1]):
                    if ar.bad_segments_[epoch_idx, ch_idx]:
                        ch_name = epochs_clean.ch_names[ch_idx]
                        if ch_name not in interpolated_channels:
                            interpolated_channels.append(ch_name)
        
        # Get cross-validation scores if available
        cv_scores = None
        if hasattr(ar, 'loss_'):
            cv_scores = ar.loss_.copy()
        
        # Create metadata dictionary
        metadata = {
            'initial_epochs': initial_epochs,
            'final_epochs': final_epochs,
            'rejected_epochs': rejected_epochs,
            'rejection_percent': round(rejection_percent, 2),
            'epoch_duration': epochs_clean.times[-1] - epochs_clean.times[0] if len(epochs_clean) > 0 else 0,
            'samples_per_epoch': len(epochs_clean.times) if len(epochs_clean) > 0 else 0,
            'total_duration_sec': ((epochs_clean.times[-1] - epochs_clean.times[0]) * final_epochs) if len(epochs_clean) > 0 else 0,
            'total_samples': len(epochs_clean.times) * final_epochs if len(epochs_clean) > 0 else 0,
            'channel_count': len(epochs_clean.ch_names) if len(epochs_clean) > 0 else 0,
            'interpolated_channels': interpolated_channels,
            'n_interpolate': n_interpolate,
            'consensus': consensus,
            'cv_folds': cv,
            'n_jobs': n_jobs,
            'thresh_method': thresh_method,
            'cv_scores': cv_scores
        }
        
        return epochs_clean, metadata
        
    except ImportError as e:
        raise ImportError(
            "AutoReject package is required for this function. "
            "Install it with: pip install autoreject"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to apply AutoReject: {str(e)}") from e