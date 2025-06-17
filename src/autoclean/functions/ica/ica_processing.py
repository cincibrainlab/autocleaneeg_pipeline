"""ICA processing functions for EEG data.

This module provides standalone functions for Independent Component Analysis (ICA)
including component fitting, classification, and artifact rejection.
"""

from typing import Dict, List, Optional, Tuple, Union

import mne
import mne_icalabel
import numpy as np
import pandas as pd
from mne.preprocessing import ICA


def fit_ica(
    raw: mne.io.Raw,
    n_components: Optional[int] = None,
    method: str = "fastica",
    max_iter: Union[int, str] = "auto",
    random_state: Optional[int] = 97,
    picks: Optional[Union[List[str], str]] = None,
    verbose: Optional[bool] = None,
    **kwargs
) -> ICA:
    """Fit Independent Component Analysis (ICA) to EEG data.

    This function creates and fits an ICA decomposition on the provided EEG data.
    ICA is commonly used to identify and remove artifacts like eye movements,
    muscle activity, and heartbeat from EEG recordings.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data to decompose with ICA.
    n_components : int or None, default None
        Number of principal components to use. If None, uses all available
        components based on the data rank.
    method : str, default "fastica"
        The ICA algorithm to use. Options: "fastica", "infomax", "picard".
    max_iter : int or "auto", default "auto"
        Maximum number of iterations for the ICA algorithm.
    random_state : int or None, default 97
        Random state for reproducible results.
    picks : list of str, str, or None, default None
        Channels to include in ICA. If None, uses all available channels.
    verbose : bool or None, default None
        Control verbosity of output.
    **kwargs
        Additional keyword arguments passed to mne.preprocessing.ICA.

    Returns
    -------
    ica : mne.preprocessing.ICA
        The fitted ICA object containing the decomposition.

    Raises
    ------
    TypeError
        If raw is not an MNE Raw object.
    ValueError
        If parameters are invalid.
    RuntimeError
        If ICA fitting fails.

    Notes
    -----
    **ICA Methods:**
    - "fastica": Fast Fixed-Point Algorithm, good general purpose choice
    - "infomax": Information Maximization, robust to outliers
    - "picard": Preconditioned ICA, fastest convergence

    **Parameter Guidelines:**
    - n_components: Typically 15-25 for dense arrays, fewer for sparse
    - For artifact removal, components should capture major noise sources
    - Higher component counts provide more detailed decomposition

    **Performance Considerations:**
    - Fitting time scales with number of components and data length
    - "picard" method is typically fastest
    - Consider downsampling data to 250-500 Hz before ICA for speed

    Examples
    --------
    Basic ICA fitting:

    >>> from autoclean.functions.ica import fit_ica
    >>> ica = fit_ica(raw)
    >>> print(f"Fitted {ica.n_components_} components")

    Custom ICA parameters:

    >>> ica = fit_ica(
    ...     raw,
    ...     n_components=20,
    ...     method="picard",
    ...     max_iter=1000,
    ...     picks="eeg"
    ... )

    See Also
    --------
    classify_ica_components : Classify ICA components using ICLabel
    apply_ica_rejection : Apply ICA to remove artifact components
    mne.preprocessing.ICA : MNE ICA implementation

    References
    ----------
    Makeig, S., et al. (1996). Independent component analysis of
    electroencephalographic data. Advances in neural information processing
    systems, 8.
    """
    # Input validation
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(
            f"Data must be an MNE Raw object, got {type(raw).__name__}"
        )

    if method not in ["fastica", "infomax", "picard"]:
        raise ValueError(f"method must be 'fastica', 'infomax', or 'picard', got '{method}'")

    if n_components is not None and n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}")

    try:
        # Create ICA object
        ica_kwargs = {
            "n_components": n_components,
            "method": method,
            "max_iter": max_iter,
            "random_state": random_state,
            **kwargs
        }

        ica = ICA(**ica_kwargs)

        # Fit ICA to the data
        ica.fit(raw, picks=picks, verbose=verbose)

        return ica

    except Exception as e:
        raise RuntimeError(f"Failed to fit ICA: {str(e)}") from e


def classify_ica_components(
    raw: mne.io.Raw,
    ica: ICA,
    method: str = "iclabel",
    verbose: Optional[bool] = None
) -> pd.DataFrame:
    """Classify ICA components using automated algorithms.

    This function uses automated classification methods to identify the likely
    source of each ICA component (brain, eye, muscle, heart, etc.). The most
    common method is ICLabel, which uses deep learning to classify components.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data used for ICA fitting.
    ica : mne.preprocessing.ICA
        The fitted ICA object to classify.
    method : str, default "iclabel"
        Classification method to use. Currently supports "iclabel".

    Returns
    -------
    component_labels : pd.DataFrame
        DataFrame with columns:
        - "component": Component index
        - "ic_type": Predicted component type (brain, eye, muscle, etc.)
        - "confidence": Confidence score (0-1) for the prediction
        - Additional columns with probabilities for each component type

    Raises
    ------
    TypeError
        If inputs are not correct MNE objects.
    ValueError
        If classification method is not supported.
    RuntimeError
        If classification fails.

    Notes
    -----
    **ICLabel Component Types:**
    - "brain": Neural brain activity
    - "eye": Eye movement artifacts (EOG)
    - "muscle": Muscle activity artifacts (EMG)
    - "heart": Cardiac artifacts (ECG)
    - "line_noise": Power line noise (50/60 Hz)
    - "ch_noise": Channel-specific noise
    - "other": Other artifacts

    **Classification Confidence:**
    - Values range from 0 to 1
    - Higher values indicate more confident predictions
    - Typical thresholds: 0.7-0.8 for artifact rejection

    Examples
    --------
    Basic component classification:

    >>> from autoclean.functions.ica import classify_ica_components
    >>> labels = classify_ica_components(raw, ica)
    >>> print(labels[["component", "ic_type", "confidence"]])

    Find high-confidence artifact components:

    >>> artifacts = labels[
    ...     (labels["ic_type"].isin(["eye", "muscle", "heart"])) &
    ...     (labels["confidence"] > 0.8)
    ... ]
    >>> print(f"Found {len(artifacts)} artifact components")

    See Also
    --------
    fit_ica : Fit ICA decomposition to EEG data
    apply_ica_rejection : Apply ICA to remove artifact components
    mne_icalabel.label_components : ICLabel implementation

    References
    ----------
    Pion-Tonachini, L., et al. (2019). ICLabel: An automated
    electroencephalographic independent component classifier, dataset, and
    website. NeuroImage, 198, 181-197.
    """
    # Input validation
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(
            f"Raw data must be an MNE Raw object, got {type(raw).__name__}"
        )

    if not isinstance(ica, ICA):
        raise TypeError(
            f"ICA must be an MNE ICA object, got {type(ica).__name__}"
        )

    if method != "iclabel":
        raise ValueError(f"Currently only 'iclabel' method is supported, got '{method}'")

    try:
        # Run ICLabel classification
        mne_icalabel.label_components(raw, ica, method=method)

        # Extract results into a DataFrame
        component_labels = _icalabel_to_dataframe(ica)

        return component_labels

    except Exception as e:
        raise RuntimeError(f"Failed to classify ICA components: {str(e)}") from e


def apply_ica_rejection(
    raw: mne.io.Raw,
    ica: ICA,
    components_to_reject: List[int],
    copy: bool = True,
    verbose: Optional[bool] = None
) -> mne.io.Raw:
    """Apply ICA to remove specified components from EEG data.

    This function applies the ICA transformation to remove specified artifact
    components from the EEG data, effectively cleaning the signal.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data to clean.
    ica : mne.preprocessing.ICA
        The fitted ICA object.
    components_to_reject : list of int
        List of component indices to remove from the data.
    copy : bool, default True
        If True, returns a copy of the data. If False, modifies in place.
    verbose : bool or None, default None
        Control verbosity of output.

    Returns
    -------
    raw_cleaned : mne.io.Raw
        The cleaned EEG data with artifact components removed.

    Raises
    ------
    TypeError
        If inputs are not correct MNE objects.
    ValueError
        If component indices are invalid.
    RuntimeError
        If ICA application fails.

    Notes
    -----
    **Component Rejection:**
    - ICA removes components by subtracting their contribution from the data
    - Rejected components are zeroed out in the mixing matrix
    - Original data rank is preserved

    **Best Practices:**
    - Always validate components before rejection (visual inspection)
    - Reject conservatively - removing brain components degrades signal
    - Document which components were rejected for reproducibility

    Examples
    --------
    Remove specific components:

    >>> from autoclean.functions.ica import apply_ica_rejection
    >>> # Remove components 0, 2, and 5
    >>> raw_clean = apply_ica_rejection(raw, ica, [0, 2, 5])

    Remove components based on classification:

    >>> # Get artifact components from classification
    >>> artifacts = labels[
    ...     (labels["ic_type"] == "eye") & 
    ...     (labels["confidence"] > 0.8)
    ... ]["component"].tolist()
    >>> raw_clean = apply_ica_rejection(raw, ica, artifacts)

    See Also
    --------
    fit_ica : Fit ICA decomposition to EEG data
    classify_ica_components : Classify ICA components
    mne.preprocessing.ICA.apply : Apply ICA transformation

    References
    ----------
    Jung, T. P., et al. (2000). Removing electroencephalographic artifacts by
    blind source separation. Psychophysiology, 37(2), 163-178.
    """
    # Input validation
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(
            f"Raw data must be an MNE Raw object, got {type(raw).__name__}"
        )

    if not isinstance(ica, ICA):
        raise TypeError(
            f"ICA must be an MNE ICA object, got {type(ica).__name__}"
        )

    if not isinstance(components_to_reject, list):
        components_to_reject = list(components_to_reject)

    # Validate component indices
    max_components = ica.n_components_
    invalid_components = [c for c in components_to_reject if c < 0 or c >= max_components]
    if invalid_components:
        raise ValueError(
            f"Invalid component indices {invalid_components}. "
            f"Must be between 0 and {max_components - 1}"
        )

    try:
        # Set components to exclude - simple approach matching original mixin
        ica_copy = ica.copy()
        ica_copy.exclude = components_to_reject

        # Apply ICA
        raw_cleaned = ica_copy.apply(raw, copy=copy, verbose=verbose)

        return raw_cleaned

    except Exception as e:
        raise RuntimeError(f"Failed to apply ICA rejection: {str(e)}") from e


def _icalabel_to_dataframe(ica: ICA) -> pd.DataFrame:
    """Convert ICLabel results to a pandas DataFrame.
    
    Helper function to extract ICLabel classification results from an ICA object
    and format them into a convenient DataFrame structure.
    
    This matches the format used in the original AutoClean ICA mixin.
    """
    # Initialize ic_type array with empty strings
    ic_type = [""] * ica.n_components_
    
    # Fill in the component types based on labels
    for label, comps in ica.labels_.items():
        for comp in comps:
            ic_type[comp] = label
    
    # Create DataFrame matching the original format with component index as DataFrame index
    results = pd.DataFrame({
        "component": getattr(ica, '_ica_names', list(range(ica.n_components_))),
        "annotator": ["ic_label"] * ica.n_components_,
        "ic_type": ic_type,
        "confidence": ica.labels_scores_.max(1) if hasattr(ica, 'labels_scores_') else [1.0] * ica.n_components_,
    }, index=range(ica.n_components_))  # Ensure index is component indices
    
    return results


def apply_iclabel_rejection(
    raw: mne.io.Raw,
    ica: ICA,
    labels_df: pd.DataFrame,
    ic_flags_to_reject: List[str] = ["eog", "muscle", "ecg"],
    ic_rejection_threshold: float = 0.8,
    verbose: Optional[bool] = None
) -> tuple[mne.io.Raw, List[int]]:
    """Apply ICA rejection based on ICLabel classifications and criteria.

    This function combines the classification results with rejection criteria
    to automatically identify and remove artifact components, similar to the
    original AutoClean mixin behavior.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data to clean.
    ica : mne.preprocessing.ICA
        The fitted ICA object with ICLabel classifications.
    labels_df : pd.DataFrame
        DataFrame with ICLabel results from classify_ica_components().
    ic_flags_to_reject : list of str, default ["eog", "muscle", "ecg"]
        Component types to consider for rejection.
    ic_rejection_threshold : float, default 0.8
        Confidence threshold for rejecting components.
    verbose : bool or None, default None
        Control verbosity of output.

    Returns
    -------
    raw_cleaned : mne.io.Raw
        The cleaned EEG data with artifact components removed.
    rejected_components : list of int
        List of component indices that were rejected.

    Examples
    --------
    Complete ICA workflow with automatic rejection:

    >>> # Fit ICA and classify components
    >>> ica = fit_ica(raw)
    >>> labels = classify_ica_components(raw, ica)
    >>> 
    >>> # Apply automatic rejection
    >>> raw_clean, rejected = apply_iclabel_rejection(
    ...     raw, ica, labels,
    ...     ic_flags_to_reject=["eog", "muscle"],
    ...     ic_rejection_threshold=0.8
    ... )
    >>> print(f"Rejected components: {rejected}")

    Conservative rejection:

    >>> raw_clean, rejected = apply_iclabel_rejection(
    ...     raw, ica, labels,
    ...     ic_flags_to_reject=["eog"],
    ...     ic_rejection_threshold=0.9
    ... )
    """
    # Find components that meet rejection criteria - use DataFrame index like original mixin
    rejected_components = []
    for idx, row in labels_df.iterrows():
        if (
            row["ic_type"] in ic_flags_to_reject
            and row["confidence"] > ic_rejection_threshold
        ):
            rejected_components.append(idx)

    # Match original mixin logic exactly
    if not rejected_components:
        if verbose:
            print("No new components met ICLabel rejection criteria in this step.")
        return raw, rejected_components
    else:
        if verbose:
            print(f"Identified {len(rejected_components)} components for rejection based on ICLabel: {rejected_components}")
        
        # Combine with any existing exclusions like original mixin
        ica_copy = ica.copy()
        if ica_copy.exclude is None:
            ica_copy.exclude = []
        
        current_exclusions = set(ica_copy.exclude)
        for idx in rejected_components:
            current_exclusions.add(idx)
        ica_copy.exclude = sorted(list(current_exclusions))
        
        if verbose:
            print(f"Total components now marked for exclusion: {ica_copy.exclude}")
        
        if not ica_copy.exclude:
            if verbose:
                print("No components are marked for exclusion. Skipping ICA apply.")
            return raw, rejected_components
        else:
            # Apply ICA to remove the excluded components (modifies in place like original mixin)
            ica_copy.apply(raw, verbose=verbose)
            if verbose:
                print(f"Applied ICA, removing/attenuating {len(ica_copy.exclude)} components.")

    return raw, rejected_components