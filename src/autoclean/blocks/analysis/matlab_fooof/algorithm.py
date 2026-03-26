"""Helpers for the bundled MATLAB FOOOF block."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_FOOOF_FREQS = (1.0, 55.0)
DEFAULT_ARTIFACTS_SUBDIR = "matlab/fooof"


def resolve_matlab_fooof_context(
    task_config: dict[str, Any],
    step_params: dict[str, Any],
    *,
    module_dir: Path,
) -> dict[str, Any]:
    """Resolve filesystem context for the MATLAB FOOOF block."""
    input_file = Path(task_config["unprocessed_file"]).expanduser().resolve()
    derivatives_dir = Path(task_config["derivatives_dir"]).expanduser().resolve()
    subject_id = input_file.stem

    artifacts_subdir = str(
        step_params.get("artifacts_subdir") or DEFAULT_ARTIFACTS_SUBDIR
    )
    block_root = derivatives_dir / artifacts_subdir / subject_id
    block_root.mkdir(parents=True, exist_ok=True)

    freq_range = step_params.get("spect_freqs") or DEFAULT_FOOOF_FREQS
    if len(freq_range) != 2:
        raise ValueError("spect_freqs must contain exactly two numbers.")

    matlab_assets_dir = module_dir / "matlab"

    return {
        "subject_id": subject_id,
        "input_file": input_file,
        "block_root": block_root,
        "matlab_assets_dir": matlab_assets_dir,
        "manifest_path": block_root / f"{subject_id}_fooof_manifest.json",
        "summary_csv": block_root / f"{subject_id}_fooof_summary.csv",
        "aperiodic_csv": block_root / f"{subject_id}_fooof_aperiodic.csv",
        "freq_range": (float(freq_range[0]), float(freq_range[1])),
    }


def load_matlab_fooof_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Load the JSON manifest written by the MATLAB wrapper."""
    path = Path(manifest_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"MATLAB FOOOF manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid MATLAB FOOOF manifest payload: {path}")
    return payload
