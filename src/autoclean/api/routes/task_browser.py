"""Task browser endpoints — detailed task discovery for the web UI.

Provides enriched task metadata including config summary and pipeline
step visualization, so operators can understand what each task does
before assigning it to a route.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import re
import sys
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Response model ─────────────────────────────────────────────────


class TaskConfig(BaseModel):
    """Flattened, UI-friendly task configuration summary."""

    montage: str = ""
    sample_rate: Optional[float] = None
    filter_low: Optional[float] = None
    filter_high: Optional[float] = None
    notch_freqs: list[float] = []
    ica_method: str = ""
    ica_threshold: Optional[float] = None
    epoch_tmin: Optional[float] = None
    epoch_tmax: Optional[float] = None
    event_id: Optional[dict[str, Any]] = None


class TaskDetail(BaseModel):
    """Full task detail for the task browser UI."""

    name: str
    description: str
    source: str
    category: str
    config: TaskConfig
    pipeline: list[str]
    source_code: str = ""


# ── Method-name → human label mapping ─────────────────────────────

_METHOD_LABELS: dict[str, str] = {
    "import_raw": "Import Raw",
    "resample_data": "Resample",
    "filter_data": "Filter",
    "drop_outer_layer": "Drop Outer Layer",
    "assign_eog_channels": "Assign EOG Channels",
    "trim_edges": "Trim Edges",
    "crop_duration": "Crop Duration",
    "clean_bad_channels": "Clean Bad Channels",
    "rereference_data": "Re-reference",
    "annotate_noisy_epochs": "Annotate Noisy",
    "annotate_uncorrelated_epochs": "Annotate Uncorrelated",
    "detect_dense_oscillatory_artifacts": "Detect Artifacts",
    "run_ica": "ICA",
    "classify_ica_components": "Classify Components",
    "create_regular_epochs": "Create Epochs",
    "create_eventid_epochs": "Create Event Epochs",
    "detect_outlier_epochs": "Detect Outliers",
    "gfp_clean_epochs": "GFP Clean",
    "generate_reports": "Generate Reports",
    "step_psd_topo_figure": "PSD Topography",
    "plot_raw_vs_cleaned_overlay": "Plot Raw vs Cleaned",
    "apply_wavelet_threshold": "Wavelet Threshold",
    "import_epochs": "Import Epochs",
    "apply_sensor_psd": "Sensor PSD Analysis",
}

# Steps that should be omitted from the pipeline display (internal bookkeeping)
_SKIP_METHODS = {"original_raw"}


def _parse_pipeline_steps(run_source: str, task_config: dict[str, Any]) -> list[str]:
    """Parse the run() method source and return human-readable pipeline step names.

    Args:
        run_source: Source code of the run() method.
        task_config: The task's config dict for parameter annotations.

    Returns:
        Ordered list of human-readable step labels.
    """
    # Filter out comment lines so the regex cannot match commented-out calls.
    lines = [ln for ln in run_source.splitlines() if not ln.strip().startswith("#")]
    filtered_source = "\n".join(lines)

    # Match self.method_name() calls, ignoring self.attribute = assignments
    method_pattern = re.compile(r"self\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
    steps: list[str] = []
    seen: set[str] = set()

    # Extract config values for annotation
    sample_rate = _get_nested(task_config, "resample_step", "value") or ""
    filter_val = _get_nested(task_config, "filtering", "value") or {}
    l_freq = filter_val.get("l_freq", "")
    h_freq = filter_val.get("h_freq", "")
    notch = filter_val.get("notch_freqs", [])
    ica_val = _get_nested(task_config, "ICA", "value") or {}
    ica_method = ica_val.get("method", "")
    ica_extended = ica_val.get("fit_params", {}).get("extended", False)
    threshold = _get_nested(
        task_config, "component_rejection", "value", "ic_rejection_threshold"
    )
    tmin = _get_nested(task_config, "epoch_settings", "value", "tmin")
    tmax = _get_nested(task_config, "epoch_settings", "value", "tmax")
    ref_val = _get_nested(task_config, "reference_step", "value") or ""

    for match in method_pattern.finditer(filtered_source):
        method_name = match.group(1)

        # Skip internal assignments and duplicate calls
        if method_name in _SKIP_METHODS or method_name in seen:
            continue
        if method_name not in _METHOD_LABELS:
            continue

        seen.add(method_name)
        label = _METHOD_LABELS[method_name]

        # Annotate with key parameters
        if method_name == "resample_data" and sample_rate:
            label = f"Resample ({sample_rate} Hz)"
        elif method_name == "filter_data" and (l_freq or h_freq):
            notch_str = ", ".join(str(int(f)) for f in notch) if notch else ""
            notch_part = f", notch: {notch_str}" if notch_str else ""
            label = f"Filter ({l_freq}-{h_freq} Hz{notch_part})"
        elif method_name == "rereference_data" and ref_val:
            label = f"Re-reference ({ref_val})"
        elif method_name == "run_ica" and ica_method:
            extended_part = " extended" if ica_extended else ""
            label = f"ICA ({ica_method}{extended_part})"
        elif method_name == "classify_ica_components":
            threshold_part = f", threshold: {threshold:.2f}" if threshold else ""
            label = f"Classify Components (iclabel{threshold_part})"
        elif method_name in ("create_regular_epochs", "create_eventid_epochs"):
            if tmin is not None and tmax is not None:
                label = f"Create Epochs ({tmin}s to {tmax}s)"
            else:
                label = "Create Epochs"

        steps.append(label)

    return steps


def _get_nested(d: dict[str, Any], *keys: str) -> Any:
    """Safely traverse nested dict keys."""
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _derive_category(source_path: str) -> str:
    """Derive task category from its source file path.

    Looks for known category folder names in the path.
    Falls back to 'custom' for user-defined tasks.
    """
    parts = Path(source_path).parts
    known_categories = {"resting", "auditory", "motor", "cognitive", "clinical"}
    for part in reversed(parts[:-1]):  # exclude the filename itself
        if part.lower() in known_categories:
            return part.lower()
    # If inside the builtins folder but no recognised category, use 'builtin'
    if "builtins" in parts:
        return "builtin"
    return "custom"


def _is_builtin_source(source_path: str) -> str:
    """Return 'builtin' if the task is from the installed package, else the file path."""
    try:
        import autoclean

        pkg_root = Path(inspect.getfile(autoclean)).parent
        if Path(source_path).is_relative_to(pkg_root):
            return "builtin"
    except Exception:
        pass
    return source_path


def _load_task_module(source_path: str) -> Any:
    """Load a task module from its file path, returning the module object."""
    module_name = f"_task_browser_{Path(source_path).stem}_{abs(hash(source_path))}"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec from {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    finally:
        sys.modules.pop(module_name, None)
    return module


def _build_task_detail(discovered_task: Any) -> Optional[TaskDetail]:
    """Build a TaskDetail from a DiscoveredTask object.

    Returns None if the task cannot be enriched (load error).
    """
    try:
        source_path = discovered_task.source
        module = _load_task_module(source_path)

        raw_config: dict[str, Any] = getattr(module, "config", {})

        # Flatten config for UI
        filter_val: dict[str, Any] = _get_nested(raw_config, "filtering", "value") or {}
        ica_val: dict[str, Any] = _get_nested(raw_config, "ICA", "value") or {}
        epoch_val: dict[str, Any] = (
            _get_nested(raw_config, "epoch_settings", "value") or {}
        )
        comp_val: dict[str, Any] = (
            _get_nested(raw_config, "component_rejection", "value") or {}
        )
        event_id: Any = _get_nested(raw_config, "epoch_settings", "event_id")

        task_config = TaskConfig(
            montage=_get_nested(raw_config, "montage", "value") or "",
            sample_rate=_get_nested(raw_config, "resample_step", "value"),
            filter_low=filter_val.get("l_freq"),
            filter_high=filter_val.get("h_freq"),
            notch_freqs=filter_val.get("notch_freqs") or [],
            ica_method=ica_val.get("method") or "",
            ica_threshold=comp_val.get("ic_rejection_threshold"),
            epoch_tmin=epoch_val.get("tmin"),
            epoch_tmax=epoch_val.get("tmax"),
            event_id=event_id if isinstance(event_id, dict) else None,
        )

        # Parse pipeline from the task class's run() method
        pipeline: list[str] = []
        task_class = discovered_task.class_obj
        if task_class is not None:
            try:
                run_method = getattr(task_class, "run", None)
                if run_method is not None:
                    run_src = inspect.getsource(run_method)
                    pipeline = _parse_pipeline_steps(run_src, raw_config)
            except (OSError, TypeError):
                pass

        # Read raw source code
        source_code = ""
        try:
            source_code = Path(source_path).read_text(encoding="utf-8")
        except Exception:
            pass

        return TaskDetail(
            name=discovered_task.name,
            description=discovered_task.description,
            source=_is_builtin_source(source_path),
            category=_derive_category(source_path),
            config=task_config,
            pipeline=pipeline,
            source_code=source_code,
        )

    except Exception as exc:
        logger.warning("Could not enrich task '%s': %s", discovered_task.name, exc)
        return None


# ── Endpoints ──────────────────────────────────────────────────────


@router.get("", response_model=list[TaskDetail])
async def list_task_details() -> list[TaskDetail]:
    """Return detailed info for all discovered tasks.

    Includes flattened config summary and pipeline step visualization.
    Does not require a workspace to be configured — tasks are discovered
    from the installed package plus the user's custom tasks directory.
    """
    try:
        from autoclean.utils.task_discovery import safe_discover_tasks

        valid_tasks, _invalid, _skipped = safe_discover_tasks()
    except Exception as exc:
        logger.error("Task discovery failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Task discovery failed: {exc}")

    results: list[TaskDetail] = []
    for discovered in valid_tasks:
        detail = _build_task_detail(discovered)
        if detail is not None:
            results.append(detail)

    return results


@router.get("/{task_name}", response_model=TaskDetail)
async def get_task_detail(task_name: str) -> TaskDetail:
    """Return detailed info for a single task by name."""
    try:
        from autoclean.utils.task_discovery import safe_discover_tasks

        valid_tasks, _invalid, _skipped = safe_discover_tasks()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Task discovery failed: {exc}")

    for discovered in valid_tasks:
        if discovered.name == task_name:
            detail = _build_task_detail(discovered)
            if detail is not None:
                return detail
            raise HTTPException(
                status_code=500, detail=f"Could not load task '{task_name}'"
            )

    raise HTTPException(status_code=404, detail=f"Task '{task_name}' not found")
