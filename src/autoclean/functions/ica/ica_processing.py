"""ICA processing functions for EEG data.

This module provides standalone functions for Independent Component Analysis (ICA)
including component fitting, classification, and artifact rejection.
"""

from typing import Dict, List, Optional, Union

import mne
import mne_icalabel
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
    **kwargs,
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

    Examples
    --------
    >>> ica = fit_ica(raw)
    >>> ica = fit_ica(raw, n_components=20, method="picard")

    See Also
    --------
    classify_ica_components : Classify ICA components using ICLabel
    apply_ica_rejection : Apply ICA to remove artifact components
    mne.preprocessing.ICA : MNE ICA implementation
    """
    # Input validation
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(f"Data must be an MNE Raw object, got {type(raw).__name__}")

    if method not in ["fastica", "infomax", "picard"]:
        raise ValueError(
            f"method must be 'fastica', 'infomax', or 'picard', got '{method}'"
        )

    if n_components is not None and n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}")

    try:
        # Create ICA object
        ica_kwargs = {
            "n_components": n_components,
            "method": method,
            "max_iter": max_iter,
            "random_state": random_state,
            **kwargs,
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
    verbose: Optional[bool] = None,
    **kwargs
) -> pd.DataFrame:
    """Classify ICA components using automated algorithms.

    This function uses automated classification methods to identify the likely
    source of each ICA component (brain, eye, muscle, heart, etc.). Supports
    both ICLabel and ICVision methods for component classification.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data used for ICA fitting.
    ica : mne.preprocessing.ICA
        The fitted ICA object to classify.
    method : str, default "iclabel"
        Classification method to use. Options: "iclabel", "icvision".
    verbose : bool or None, default None
        Control verbosity of output.
    **kwargs
        Additional keyword arguments passed to the classification method.
        For icvision method, supports 'psd_fmax' to limit PSD plot frequency range.

    Returns
    -------
    component_labels : pd.DataFrame
        DataFrame with columns:
        - "component": Component index
        - "ic_type": Predicted component type (brain, eye, muscle, etc.)
        - "confidence": Confidence score (0-1) for the prediction
        - Additional columns with probabilities for each component type

    Examples
    --------
    >>> labels = classify_ica_components(raw, ica, method="iclabel")
    >>> labels = classify_ica_components(raw, ica, method="icvision")
    >>> artifacts = labels[(labels["ic_type"] == "eye") & (labels["confidence"] > 0.8)]

    See Also
    --------
    fit_ica : Fit ICA decomposition to EEG data
    apply_ica_rejection : Apply ICA to remove artifact components
    mne_icalabel.label_components : ICLabel implementation
    """
    # Input validation
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(f"Raw data must be an MNE Raw object, got {type(raw).__name__}")

    if not isinstance(ica, ICA):
        raise TypeError(f"ICA must be an MNE ICA object, got {type(ica).__name__}")

    if method not in ["iclabel", "icvision"]:
        raise ValueError(f"method must be 'iclabel' or 'icvision', got '{method}'")

    try:
        if method == "iclabel":
            # Run ICLabel classification
            mne_icalabel.label_components(raw, ica, method=method)
            # Extract results into a DataFrame
            component_labels = _icalabel_to_dataframe(ica)

        elif method == "icvision":
            # Run ICVision classification
            try:
                from icvision.compat import label_components
            except ImportError as e:
                raise ImportError(
                    "autoclean-icvision package is required for icvision method. "
                    "Install it with: pip install autoclean-icvision"
                ) from e

            # Use ICVision as drop-in replacement, passing through any extra kwargs
            label_components(raw, ica, **kwargs)
            # Extract results into a DataFrame using the same format
            component_labels = _icalabel_to_dataframe(ica)

        return component_labels

    except Exception as e:
        raise RuntimeError(
            f"Failed to classify ICA components with {method}: {str(e)}"
        ) from e


def apply_ica_rejection(
    raw: mne.io.Raw,
    ica: ICA,
    components_to_reject: List[int],
    copy: bool = True,
    verbose: Optional[bool] = None,
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

    Examples
    --------
    >>> raw_clean = apply_ica_rejection(raw, ica, [0, 2, 5])

    See Also
    --------
    fit_ica : Fit ICA decomposition to EEG data
    classify_ica_components : Classify ICA components
    mne.preprocessing.ICA.apply : Apply ICA transformation
    """
    # Input validation
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(f"Raw data must be an MNE Raw object, got {type(raw).__name__}")

    if not isinstance(ica, ICA):
        raise TypeError(f"ICA must be an MNE ICA object, got {type(ica).__name__}")

    if not isinstance(components_to_reject, list):
        components_to_reject = list(components_to_reject)

    # Validate component indices
    max_components = ica.n_components_
    invalid_components = [
        c for c in components_to_reject if c < 0 or c >= max_components
    ]
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
    results = pd.DataFrame(
        {
            "component": getattr(ica, "_ica_names", list(range(ica.n_components_))),
            "annotator": ["ic_label"] * ica.n_components_,
            "ic_type": ic_type,
            "confidence": (
                ica.labels_scores_.max(1)
                if hasattr(ica, "labels_scores_")
                else [1.0] * ica.n_components_
            ),
        },
        index=range(ica.n_components_),
    )  # Ensure index is component indices

    return results


def apply_ica_component_rejection(
    raw: mne.io.Raw,
    ica: ICA,
    labels_df: pd.DataFrame,
    ic_flags_to_reject: List[str] = ["eog", "muscle", "ecg"],
    ic_rejection_threshold: float = 0.8,
    ic_rejection_overrides: Optional[Dict[str, float]] = None,
    verbose: Optional[bool] = None,
) -> tuple[mne.io.Raw, List[int]]:
    """Apply ICA rejection based on component classifications and criteria.

    This function combines the classification results with rejection criteria
    to automatically identify and remove artifact components. Works with both
    ICLabel and ICVision classification results.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data to clean.
    ica : mne.preprocessing.ICA
        The fitted ICA object with component classifications.
    labels_df : pd.DataFrame
        DataFrame with classification results from classify_ica_components().
    ic_flags_to_reject : list of str, default ["eog", "muscle", "ecg"]
        Component types to consider for rejection.
    ic_rejection_threshold : float, default 0.8
        Global confidence threshold for rejecting components.
    ic_rejection_overrides : dict of str to float, optional
        Per-component-type confidence thresholds that override the global threshold.
        Keys are IC types (e.g., 'muscle', 'heart'), values are confidence thresholds.
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
    >>> raw_clean, rejected = apply_ica_component_rejection(raw, ica, labels)

    See Also
    --------
    fit_ica : Fit ICA decomposition to EEG data
    classify_ica_components : Classify ICA components
    apply_ica_rejection : Apply ICA to remove specific components
    """
    # Find components that meet rejection criteria - use DataFrame index like original mixin
    if ic_rejection_overrides is None:
        ic_rejection_overrides = {}
    
    rejected_components = []
    for idx, row in labels_df.iterrows():
        ic_type = row["ic_type"]
        confidence = row["confidence"]
        
        if ic_type in ic_flags_to_reject:
            # Use override threshold if available, otherwise global threshold
            threshold = ic_rejection_overrides.get(ic_type, ic_rejection_threshold)
            
            if confidence > threshold:
                rejected_components.append(idx)

    # Match original mixin logic exactly
    if not rejected_components:
        if verbose:
            print("No new components met rejection criteria in this step.")
        return raw, rejected_components
    else:
        if verbose:
            print(
                f"Identified {len(rejected_components)} components for rejection: {rejected_components}"
            )

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
                print(
                    f"Applied ICA, removing/attenuating {len(ica_copy.exclude)} components."
                )

    return raw, rejected_components
