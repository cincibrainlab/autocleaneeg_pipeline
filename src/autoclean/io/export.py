from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import mne
import scipy.io as sio

from autoclean.utils.database import manage_database
from autoclean.utils.logging import message

__all__ = [
    "save_stc_to_file",
    "save_raw_to_set",
    "save_epochs_to_set",
    "_get_stage_number",
]

def save_stc_to_file(
    stc: mne.SourceEstimate,
    autoclean_dict: Dict[str, Any],
    stage: str = "post_source_localization",
    output_path: Optional[Path] = None,
) -> Path:
    """Save source estimate (STC) data to file.

    This function saves an MNE SourceEstimate object at a specified processing stage,
    consistent with the pipeline's directory structure and configuration.

    Parameters
    ----------
        stc : mne.SourceEstimate
            SourceEstimate object to save
        autoclean_dict : dict
            Configuration dictionary
        stage : str
            Processing stage identifier (default: "post_source_localization")
        output_path : Optional[Path]
            Optional custom output path. If None, uses config

    Returns
    -------
        Path: Path
            Path to the saved file (stage path)

    """
    # Validate stage configuration
    if stage not in autoclean_dict["stage_files"]:
        raise ValueError(f"Stage not configured: {stage}")
        
    if not autoclean_dict["stage_files"][stage]["enabled"]:
        message("info", f"Saving disabled for stage: {stage}")
        return None

    # Extract configuration details
    suffix = autoclean_dict["stage_files"][stage]["suffix"]
    basename = Path(autoclean_dict["unprocessed_file"]).stem
    stage_num = _get_stage_number(stage, autoclean_dict)

    # Determine output path
    if output_path is None:
        output_path = autoclean_dict["stage_dir"]
    subfolder = output_path / f"{stage_num}{suffix}"
    subfolder.mkdir(exist_ok=True)
    stage_path = subfolder / f"{basename}{suffix}-stc.h5"

    # Handle dual saving for "post_comp" stage (if applicable)
    paths = [stage_path]
    if stage == "post_comp":
        clean_path = autoclean_dict["clean_dir"] / f"{basename}{suffix}-stc.h5"
        autoclean_dict["clean_dir"].mkdir(exist_ok=True)
        paths.append(clean_path)

    # Save the STC to all specified paths
    for path in paths:
        try:
            stc.save(fname=path, ftype='h5', overwrite=True, verbose=False)
            message("success", f"✓ Saved {stage} STC file to: {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save {stage} STC file to {path}: {str(e)}")

    # Create metadata for database logging
    metadata = {
        "save_stc_to_file": {
            "creationDateTime": datetime.now().isoformat(),
            "stage": stage,
            "stage_number": stage_num,
            "outputPaths": [str(p) for p in paths],
            "suffix": suffix,
            "basename": basename,
            "format": "h5",
            "n_vertices": stc.data.shape[0],
            "n_times": stc.data.shape[1],
            "tmin": stc.tmin,
            "tstep": stc.tstep,
        }
    }

    # Update database
    run_id = autoclean_dict["run_id"]
    manage_database(
        operation="update", update_record={"run_id": run_id, "metadata": metadata}
    )
    manage_database(
        operation="update_status",
        update_record={"run_id": run_id, "status": f"{stage} completed"},
    )

    return paths[0]  # Return stage path for consistency

def save_raw_to_set(
    raw: mne.io.Raw,
    autoclean_dict: Dict[str, Any],
    stage: str = "post_import",
    output_path: Optional[Path] = None,
    flagged: bool = False,
) -> Path:
    """Save continuous EEG data to file.

    This function saves raw EEG data at various processing stages.
    
    Parameters
    ----------
        raw : mne.io.Raw
            Raw EEG data to save
        autoclean_dict : dict
            Configuration dictionary
        stage : str
            Processing stage identifier (e.g., "post_import")
        output_path : Optional[Path]
            Optional custom output path. If None, uses config
        flagged : bool
            Whether to save to flagged directory

    Returns
    -------
        Path: Path
            Path to the saved file (stage path)

    """

    if stage not in autoclean_dict["stage_files"]:
        raise ValueError(f"Stage not configured: {stage}")
        
    if not autoclean_dict["stage_files"][stage]["enabled"]:
        message("info", f"Saving disabled for stage: {stage}")
        return None

    suffix = autoclean_dict["stage_files"][stage]["suffix"]
    basename = Path(autoclean_dict["unprocessed_file"]).stem
    stage_num = _get_stage_number(stage, autoclean_dict)

    # Save to stage directory
    if flagged:
        output_path = autoclean_dict["flagged_dir"]
        subfolder = output_path / f"{basename}"
    elif output_path is None:
        output_path = autoclean_dict["stage_dir"]
        subfolder = output_path / f"{stage_num}{suffix}"
    subfolder.mkdir(exist_ok=True)
    stage_path = subfolder / f"{basename}{suffix}_raw.set"

    # Save to both locations for post_comp
    paths = [stage_path]
    if stage == "post_comp" and not flagged:
        clean_path = autoclean_dict["clean_dir"] / f"{basename}{suffix}.set"
        autoclean_dict["clean_dir"].mkdir(exist_ok=True)
        paths.append(clean_path)

    # Save to all paths
    raw.info["description"] = autoclean_dict["run_id"]
    for path in paths:
        try:
            raw.export(path, fmt="eeglab", overwrite=True)
            message("success", f"✓ Saved {stage} file to: {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save {stage} file to {path}: {str(e)}")

    metadata = {
        "save_raw_to_set": {
            "creationDateTime": datetime.now().isoformat(),
            "stage": stage,
            "stage_number": stage_num,
            "outputPath": str(stage_path),
            "suffix": suffix,
            "basename": basename,
            "format": "eeglab",
        }
    }

    run_id = autoclean_dict["run_id"]
    manage_database(
        operation="update", update_record={"run_id": run_id, "metadata": metadata}
    )
    manage_database(
        operation="update_status",
        update_record={"run_id": run_id, "status": f"{stage} completed"},
    )

    return paths[0]  # Return stage path for consistency


def save_epochs_to_set(
    epochs: mne.Epochs,
    autoclean_dict: Dict[str, Any],
    stage: str = "post_clean_epochs",
    output_path: Optional[Path] = None,
    flagged: bool = False,
) -> Path:
    """Save epoched EEG data to file.

    This function saves epoched EEG data, typically after processing.
    
    Parameters
    ----------
        epochs : mne.Epochs
            Epoched EEG data to save
        autoclean_dict : dict
            Configuration dictionary
        stage : str
            Processing stage identifier (default: "post_clean_epochs")
        output_path : Optional[Path]
            Optional custom output path. If None, uses config
        flagged : bool
            Whether to save to flagged directory

    Returns
    -------
        Path: Path
            Path to the saved file (stage path)

    """
    
    if stage not in autoclean_dict["stage_files"]:
        raise ValueError(f"Stage not configured: {stage}")
        
    if not autoclean_dict["stage_files"][stage]["enabled"]:
        message("info", f"Saving disabled for stage: {stage}")
        return None

    suffix = autoclean_dict["stage_files"][stage]["suffix"]
    basename = Path(autoclean_dict["unprocessed_file"]).stem
    stage_num = _get_stage_number(stage, autoclean_dict)

    # Save to stage directory
    if flagged:
        output_path = autoclean_dict["flagged_dir"]
        subfolder = output_path / f"{basename}"
    elif output_path is None:
        output_path = autoclean_dict["stage_dir"]
        subfolder = output_path / f"{stage_num}{suffix}"
    subfolder.mkdir(exist_ok=True)
    stage_path = subfolder / f"{basename}{suffix}_epo.set"

    # Save to both locations for post_comp
    paths = [stage_path]
    if stage == "post_comp" and not flagged:
        clean_path = autoclean_dict["clean_dir"] / f"{basename}{suffix}.set"
        autoclean_dict["clean_dir"].mkdir(exist_ok=True)
        paths.append(clean_path)

    # Save to all paths
    epochs.info["description"] = autoclean_dict["run_id"]
    epochs.apply_proj()
    for path in paths:
        try:
            epochs.export(path, fmt="eeglab", overwrite=True)
            # Add run_id to each file
            EEG = sio.loadmat(path)
            EEG["etc"] = {}
            EEG["etc"]["run_id"] = autoclean_dict["run_id"]
            sio.savemat(path, EEG, do_compression=False)
            message("success", f"✓ Saved {stage} file to: {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save {stage} file to {path}: {str(e)}")

    metadata = {
        "save_epochs_to_set": {
            "creationDateTime": datetime.now().isoformat(),
            "stage": stage,
            "stage_number": stage_num,
            "outputPaths": [str(p) for p in paths],
            "suffix": suffix,
            "basename": basename,
            "format": "eeglab",
            "n_epochs": len(epochs),
            "tmin": epochs.tmin,
            "tmax": epochs.tmax,
        }
    }

    run_id = autoclean_dict["run_id"]
    manage_database(
        operation="update", update_record={"run_id": run_id, "metadata": metadata}
    )
    manage_database(
        operation="update_status",
        update_record={"run_id": run_id, "status": f"{stage} completed"},
    )

    return paths[0]  # Return stage path for consistency

def save_ica_to_fif(pipeline, autoclean_dict, pre_ica_raw):
    """Save ICA results to FIF files.

    This function saves ICA results to FIF files in the derivatives directory.

    Parameters
    ----------
        pipeline : dict
            Pipeline dictionary
        autoclean_dict : dict
            Autoclean dictionary
        pre_ica_raw : mne.io.Raw
            Raw data before ICA
    """
    try:
        derivatives_dir = pipeline.get_derivative_path(autoclean_dict["bids_path"]).directory
        derivatives_dir.mkdir(parents=True, exist_ok=True)
        basename = Path(autoclean_dict["unprocessed_file"]).stem
    except Exception as e:
        message("error", f"Failed to save ICA to FIF files: {str(e)}")

    components = []

    if pipeline.ica1 is not None:
        ica1_path = derivatives_dir / f"{basename}ica1-ica.fif"
        pipeline.ica1.save(ica1_path, overwrite=True)
        components.append(ch for ch in pipeline.ica1.exclude)
    if pipeline.ica2 is not None:
        ica2_path = derivatives_dir / f"{basename}ica2-ica.fif"
        pipeline.ica2.save(ica2_path, overwrite=True)
        components.append(ch for ch in pipeline.ica2.exclude)
    pre_ica_path = derivatives_dir / f"{basename}_pre_ica.set"
    pre_ica_raw.export(pre_ica_path, fmt="eeglab", overwrite=True)



    metadata = {
        "save_ica_to_fif": {
            "creationDateTime": datetime.now().isoformat(),
            "components": components,
            "ica1_path": ica1_path.name,
            "ica2_path": ica2_path.name,
            "pre_ica_path": pre_ica_path.name,
        }
    }

    run_id = autoclean_dict["run_id"]

    manage_database(
        operation="update", update_record={"run_id": run_id, "metadata": metadata}
    )


# Keep the existing save functions with minor updates to ensure backward compatibility
def _get_stage_number(stage: str, autoclean_dict: Dict[str, Any]) -> str:
    """Get two-digit number based on enabled stages order."""
    enabled_stages = [
        s for s, cfg in autoclean_dict["stage_files"].items() if cfg["enabled"]
    ]
    return f"{enabled_stages.index(stage) + 1:02d}"