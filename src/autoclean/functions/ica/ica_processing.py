"""ICA processing functions for EEG data.

This module provides standalone functions for Independent Component Analysis (ICA)
including component fitting, classification, and artifact rejection.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import mne
import mne_icalabel
import pandas as pd
from mne.preprocessing import ICA, read_ica
from autoclean.utils.bids import step_sanitize_id
from autoclean.io.export import save_raw_to_set

# Optional import for ICVision
try:
    from icvision.compat import label_components

    ICVISION_AVAILABLE = True
except ImportError:
    ICVISION_AVAILABLE = False


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
    >>> ica = fit_ica(raw, n_components=20, method="infomax")

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
        # Remove 'ortho' from fit_params if method is 'infomax' and 'ortho' is in kwargs
        if (
            method == "infomax"
            and "fit_params" in kwargs
            and "ortho" in kwargs["fit_params"]
        ):
            kwargs["fit_params"].pop("ortho")

        if verbose:
            print(f"Running ICA with method: '{method}'")

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
    **kwargs,
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
            if not ICVISION_AVAILABLE:
                raise ImportError(
                    "autoclean-icvision package is required for icvision method. "
                    "Install it with: pip install autoclean-icvision"
                )

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

        # Apply ICA, falling back if copy parameter is unsupported
        try:
            raw_cleaned = ica_copy.apply(raw, copy=copy, verbose=verbose)
        except TypeError:
            raw_cleaned = ica_copy.apply(raw, verbose=verbose)

        return raw_cleaned

    except Exception as e:
        raise RuntimeError(f"Failed to apply ICA rejection: {str(e)}") from e


def _icalabel_to_dataframe(ica: ICA) -> pd.DataFrame:
    """Convert ICLabel results to a pandas DataFrame."""

    labels_attr = getattr(ica, "labels_", {}) or {}
    n_components = ica.n_components_

    ic_type = [""] * n_components
    confidence = [0.0] * n_components

    if "iclabel" in labels_attr and isinstance(labels_attr["iclabel"], dict):
        proba = labels_attr["iclabel"].get("y_pred_proba")
        if proba is not None and len(proba) == n_components:
            confidence = proba.max(axis=1).tolist()
    else:
        for label, comps in labels_attr.items():
            if isinstance(comps, (list, tuple, set)):
                for comp in comps:
                    ic_type[comp] = label
        confidence = (
            ica.labels_scores_.max(1).tolist()
            if hasattr(ica, "labels_scores_")
            else [1.0] * n_components
        )

    results = pd.DataFrame(
        {
            "component": getattr(ica, "_ica_names", list(range(n_components))),
            "annotator": ["ic_label"] * n_components,
            "ic_type": ic_type,
            "confidence": confidence,
        },
        index=range(n_components),
    )

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

        # Also update the original ICA object so the mixin can access the excluded components
        ica.exclude = ica_copy.exclude.copy()

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


def _parse_component_str(value: str) -> set[int]:
    """Parse a comma-separated component string into a set of ints."""
    if value is None or str(value).strip() == "" or pd.isna(value):
        return set()
    return {int(item) for item in str(value).split(",") if item.strip().isdigit()}


def _format_component_list(values: List[int]) -> str:
    """Format a list of component integers as a comma-separated string."""
    return ",".join(str(v) for v in values)


def _discover_pre_ica_path(
    autoclean_dict: Dict[str, Union[str, Path]], basename: str
) -> Optional[Path]:
    """Discover the pre-ICA stage file path from the stage directory structure.

    Looks under the pipeline's intermediate stage directory for a folder
    matching the pre-ICA stage (e.g., 'NN_pre_ica') and returns the expected
    filename pattern '<basename>_pre_ica_raw.set'. Returns None if not found.
    """
    try:
        # 1) Prefer subject-level derivatives (per-subject organization)
        metadata_dir = Path(autoclean_dict.get("metadata_dir", ""))
        if metadata_dir:
            pipeline_deriv_root = metadata_dir.parent
            subject_id = step_sanitize_id(f"{basename}")
            subj_eeg = pipeline_deriv_root / f"sub-{subject_id}" / "eeg"
            candidate_fif = subj_eeg / f"{basename}_pre_ica_raw.fif"
            if candidate_fif.exists():
                return candidate_fif
            candidate_set = subj_eeg / f"{basename}_pre_ica_raw.set"
            if candidate_set.exists():
                return candidate_set

        # 2) Fallback to stage directory discovery (intermediate NN_pre_ica)
        stage_dir = Path(autoclean_dict.get("stage_dir") or Path(autoclean_dict["derivatives_dir"]) / "intermediate")
        fif_candidates = sorted(
            stage_dir.glob(f"**/*_pre_ica/{basename}_pre_ica_raw.fif"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if fif_candidates:
            return fif_candidates[0]
        set_candidates = sorted(
            stage_dir.glob(f"**/*_pre_ica/{basename}_pre_ica_raw.set"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return set_candidates[0] if set_candidates else None
    except Exception:  # best-effort discovery; ignore errors
        return None


def update_ica_control_sheet(
    autoclean_dict: Dict[str, Union[str, Path]], auto_exclude: List[int]
) -> List[int]:
    """Update the ICA control sheet and return the final exclusion list.

    Parameters
    ----------
    autoclean_dict : dict
        Configuration dictionary containing at minimum ``metadata_dir``,
        ``derivatives_dir`` and ``unprocessed_file`` keys.
    auto_exclude : list of int
        Components automatically selected for exclusion in the current run.

    Returns
    -------
    list of int
        Final list of components to exclude after applying manual edits from
        the control sheet.
    """

    metadata_dir = Path(autoclean_dict["metadata_dir"])
    metadata_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = metadata_dir / "ica_control_sheet.csv"

    derivatives_dir = Path(autoclean_dict.get("derivatives_dir", metadata_dir))
    original_file = Path(autoclean_dict["unprocessed_file"]).name
    ica_fif = derivatives_dir / f"{Path(original_file).stem}-ica.fif"
    auto_initial_str = _format_component_list(sorted(auto_exclude))
    now_iso = datetime.now().isoformat()

    columns = [
        "original_file",
        "ica_fif",
        "pre_ica_path",
        "post_ica_path",
        "auto_initial",
        "final_removed",
        "manual_add",
        "manual_drop",
        "status",
        "last_run_iso",
    ]

    if sheet_path.exists():
        df = pd.read_csv(sheet_path, dtype=str, keep_default_na=False)
    else:
        df = pd.DataFrame(columns=columns)

    # Ensure new columns exist for backward compatibility
    for new_col in ["pre_ica_path", "post_ica_path"]:
        if new_col not in df.columns:
            df[new_col] = ""

    if original_file not in df.get("original_file", []).tolist():
        # First run for this file: create new row with auto selections
        discovered_pre_ica = _discover_pre_ica_path(autoclean_dict, Path(original_file).stem)
        new_row = {
            "original_file": original_file,
            "ica_fif": str(ica_fif),
            "pre_ica_path": str(discovered_pre_ica) if discovered_pre_ica else "",
            "post_ica_path": "",
            "auto_initial": auto_initial_str,
            "final_removed": auto_initial_str,
            "manual_add": "",
            "manual_drop": "",
            "status": "auto",
            "last_run_iso": now_iso,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(sheet_path, index=False)
        return sorted(auto_exclude)

    # Existing row: ensure columns present and apply manual edits if any
    idx = df.index[df["original_file"] == original_file][0]

    # Backfill pre_ica_path if empty
    if not str(df.loc[idx, "pre_ica_path"]).strip():
        discovered_pre_ica = _discover_pre_ica_path(autoclean_dict, Path(original_file).stem)
        if discovered_pre_ica and discovered_pre_ica.exists():
            df.loc[idx, "pre_ica_path"] = str(discovered_pre_ica)

    final_removed_set = _parse_component_str(df.loc[idx, "final_removed"])
    manual_add_set = _parse_component_str(df.loc[idx, "manual_add"])
    manual_drop_set = _parse_component_str(df.loc[idx, "manual_drop"])

    if manual_add_set or manual_drop_set:
        final_removed_set = (final_removed_set | manual_add_set) - manual_drop_set
        df.loc[idx, "final_removed"] = _format_component_list(
            sorted(final_removed_set)
        )
        df.loc[idx, "manual_add"] = ""
        df.loc[idx, "manual_drop"] = ""
        df.loc[idx, "status"] = (
            "auto"
            if df.loc[idx, "auto_initial"] == df.loc[idx, "final_removed"]
            else "applied"
        )

    # Always update paths and timestamp
    # Preserve existing ica_fif if already set to a non-empty value
    if str(df.loc[idx, "ica_fif"]).strip() == "":
        df.loc[idx, "ica_fif"] = str(ica_fif)
    df.loc[idx, "last_run_iso"] = now_iso

    df.to_csv(sheet_path, index=False)

    return sorted(final_removed_set)


def process_ica_control_sheet(autoclean_dict: dict) -> None:
    """Apply manual ICA edits from the control sheet and update artifacts.

    This function reloads the original data and previously computed ICA
    decomposition, applies any manual inclusion or exclusion edits recorded
    in the ``ica_control_sheet.csv`` file, and saves the cleaned data back to
    the derivatives directory. The control sheet is updated to record the
    applied changes and clear any manual edit fields.

    Parameters
    ----------
    autoclean_dict : dict
        Configuration dictionary containing ``metadata_dir``,
        ``derivatives_dir`` and ``unprocessed_file`` keys.

    Returns
    -------
    None
        The function has side effects of updating the control sheet and
        overwriting the cleaned data file in the derivatives directory.
    """

    metadata_dir = Path(autoclean_dict["metadata_dir"])
    sheet_path = metadata_dir / "ica_control_sheet.csv"
    original_file = Path(autoclean_dict["unprocessed_file"]).name

    if not sheet_path.exists():
        raise FileNotFoundError(f"ICA control sheet not found: {sheet_path}")

    df = pd.read_csv(sheet_path, dtype=str, keep_default_na=False)
    if original_file not in df.get("original_file", []).tolist():
        raise ValueError(f"{original_file} not found in ICA control sheet")

    # Merge any manual edits and get final exclusion list
    final_exclude = update_ica_control_sheet(autoclean_dict, [])

    derivatives_dir = Path(autoclean_dict.get("derivatives_dir", metadata_dir))
    stage_dir = Path(autoclean_dict.get("stage_dir", derivatives_dir / "intermediate"))
    basename = Path(original_file).stem

    # Read control sheet row for this file
    df = pd.read_csv(sheet_path, dtype=str, keep_default_na=False)
    idx = df.index[df["original_file"] == original_file][0]

    # Determine ICA FIF path: prefer recorded sheet value
    ica_fif_str = df.loc[idx, "ica_fif"] if "ica_fif" in df.columns else ""
    ica_fif = Path(ica_fif_str) if ica_fif_str else None
    if not ica_fif or not ica_fif.exists():
        # Recompute subject-specific derivatives path and try again
        # bids_root = <...>/bids (three levels up from metadata_dir)
        subject_id = step_sanitize_id(original_file)
        # Use pipeline derivatives root from metadata_dir's parent
        pipeline_deriv_root = Path(metadata_dir).parent
        candidate = pipeline_deriv_root / f"sub-{subject_id}" / "eeg" / f"{basename}-ica.fif"
        if candidate.exists():
            ica_fif = candidate
        else:
            # Last fallback: original (possibly incorrect) path
            ica_fif = Path(autoclean_dict.get("derivatives_dir", metadata_dir)) / f"{basename}-ica.fif"
    # If still missing, raise a helpful error
    if not ica_fif.exists():
        raise FileNotFoundError(
            f"ICA FIF not found for {original_file}. Expected at: {ica_fif}. "
            "Run the ICA step first to create and save the decomposition."
        )
    pre_path_str = df.loc[idx, "pre_ica_path"] if "pre_ica_path" in df.columns else ""
    pre_ica_path = Path(pre_path_str) if pre_path_str else _discover_pre_ica_path(autoclean_dict, basename)
    if not pre_ica_path or not Path(pre_ica_path).exists():
        # Try discovery one more time
        pre_ica_path = _discover_pre_ica_path(autoclean_dict, basename)
    if not pre_ica_path or not Path(pre_ica_path).exists():
        raise FileNotFoundError(
            f"Could not find pre-ICA stage file for {original_file}. Run an ICA processing pass first."
        )

    # Read pre-ICA data (prefer FIF if provided)
    if str(pre_ica_path).lower().endswith(".fif"):
        raw = mne.io.read_raw_fif(str(pre_ica_path), preload=True, verbose=False)
    else:
        raw = mne.io.read_raw_eeglab(str(pre_ica_path), preload=True)
    ica = read_ica(ica_fif)

    # Apply exclusions and save cleaned data as a new stage
    raw_clean, _ = apply_ica_rejection(raw, ica, final_exclude, copy=True)
    autoclean_dict_with_stage = dict(autoclean_dict)
    autoclean_dict_with_stage["stage_dir"] = stage_dir
    post_path = save_raw_to_set(raw_clean, autoclean_dict_with_stage, stage="post_ica_manual")

    # Update control sheet with timestamp, ensure manual fields are cleared, and set post_ica_path
    final_exclude = update_ica_control_sheet(autoclean_dict, final_exclude)
    df = pd.read_csv(sheet_path, dtype=str, keep_default_na=False)
    idx = df.index[df["original_file"] == original_file][0]
    if "post_ica_path" not in df.columns:
        df["post_ica_path"] = ""
    df.loc[idx, "post_ica_path"] = str(post_path)
    df.to_csv(sheet_path, index=False)

    return None
