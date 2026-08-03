"""Shared path/loading helpers for Exclude review endpoints and services.

Split out of ``exclude.py`` so that other service modules (e.g. the ICA
rerun service) can reuse the same file-resolution conventions without
importing the full FastAPI router module and risking a circular import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mne
from fastapi import HTTPException

_SUFFIXES = ["_comp_epo", "_comp", "_epo", "_postedit", "_preproc", "_raw", "_clean"]


def _strip_suffixes(stem: str) -> str:
    for suffix in sorted(_SUFFIXES, key=len, reverse=True):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _postedit_path(task_root: Path, file_path: Path) -> Path:
    stem = _strip_suffixes(file_path.stem)
    return task_root / "postedit" / f"{stem}_postedit.set"


def _load_epochs(file_path: Path) -> mne.BaseEpochs:
    try:
        return mne.read_epochs_eeglab(str(file_path), verbose="ERROR")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not load epochs: {exc}")


def _related_paths(root: Path, file_path: Path) -> dict[str, Optional[Path]]:
    stem = _strip_suffixes(file_path.stem)
    task_root = root.parent
    candidates = {
        "run_report": task_root
        / "reports"
        / "run_reports"
        / f"{stem}_autoclean_report.pdf",
        "ica_report": task_root
        / "reports"
        / "ica_components"
        / f"{stem}_ica_components_all.pdf",
        "psd": task_root / "reports" / "psd_topo" / f"{stem}_psd_topo_figure.png",
        "metadata": task_root
        / "reports"
        / "run_reports"
        / f"{stem}_autoclean_metadata.json",
        "processing_log": task_root / "exports" / f"{stem}_processing_log.csv",
        "postedit": _postedit_path(task_root, file_path),
    }
    return {k: v if v.exists() else None for k, v in candidates.items()}


__all__ = [
    "_SUFFIXES",
    "_strip_suffixes",
    "_postedit_path",
    "_load_epochs",
    "_related_paths",
]
