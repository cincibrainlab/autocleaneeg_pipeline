"""Output Results Viewer endpoints.

Scans all automation output directories for pipeline run databases, aggregates
run records, and serves processed artifacts (PDFs, PNGs, JSON).

Includes review-decision workflow: operators can mark runs as Pass/Fail/Review
with notes, persisted to decisions.json and decisions.csv in the workspace.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import logging
import sqlite3
import statistics
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from autoclean.api.state import api_state

logger = logging.getLogger(__name__)

router = APIRouter()

# Minimum event count to classify a recording as event-related vs resting-state
_MIN_EVENT_RELATED = 10


# ── Suffix table (longest-first for greedy matching) ─────────────────

_SUFFIXES = ["_comp_epo", "_comp", "_epo", "_postedit", "_preproc", "_raw", "_clean"]


def _extract_stem(filename: str) -> str:
    """Strip known processing suffixes from a filename stem.

    Example: '201001_D1BL_EC_comp_epo.set' → '201001_D1BL_EC'
    """
    stem = Path(filename).stem
    for suffix in sorted(_SUFFIXES, key=len, reverse=True):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


# ── DB scanning ──────────────────────────────────────────────────────

_runs_cache: list[tuple[dict[str, Any], Path]] = []
_runs_cache_time: float = 0.0
_runs_cache_lock = threading.Lock()
_RUNS_CACHE_TTL = 5.0  # seconds
_RUN_DB_FILENAMES = ("pipeline.db", "run_database.db")


def _find_all_runs(workspace: Path) -> list[tuple[dict[str, Any], Path]]:
    """Scan all automation dirs for supported run DBs. Cached for 5 seconds."""
    global _runs_cache, _runs_cache_time

    now = time.time()
    with _runs_cache_lock:
        if _runs_cache and (now - _runs_cache_time) < _RUNS_CACHE_TTL:
            return _runs_cache

    results: list[tuple[dict[str, Any], Path]] = []
    automations = workspace / "automations"
    if not automations.exists():
        with _runs_cache_lock:
            _runs_cache = results
            _runs_cache_time = now
        return results

    seen_db_paths: set[Path] = set()
    for auto_dir in automations.iterdir():
        if not auto_dir.is_dir():
            continue
        for db_name in _RUN_DB_FILENAMES:
            for db_path in auto_dir.rglob(db_name):
                resolved_db_path = db_path.resolve()
                if resolved_db_path in seen_db_paths:
                    continue
                seen_db_paths.add(resolved_db_path)
                task_root = db_path.parent
                try:
                    conn = sqlite3.connect(str(db_path))
                    try:
                        conn.row_factory = sqlite3.Row
                        rows = conn.execute(
                            "SELECT * FROM pipeline_runs ORDER BY created_at DESC"
                        ).fetchall()
                        for row in rows:
                            results.append((dict(row), task_root))
                    finally:
                        conn.close()
                except Exception as exc:
                    logger.debug("Skipping %s: %s", db_path, exc)
                    continue

    deduped_by_run_id: dict[str, tuple[dict[str, Any], Path]] = {}
    for row, task_root in results:
        run_id = str(row.get("run_id") or "").strip()
        if not run_id:
            continue
        existing = deduped_by_run_id.get(run_id)
        if existing is None or _prefer_task_root(task_root, existing[1]):
            deduped_by_run_id[run_id] = (row, task_root)

    results = list(deduped_by_run_id.values())
    results.sort(key=lambda t: t[0].get("created_at", ""), reverse=True)
    with _runs_cache_lock:
        _runs_cache = results
        _runs_cache_time = now
    return results


def _prefer_task_root(candidate: Path, current: Path) -> bool:
    """Return True when candidate is the better root for serving run assets."""
    candidate_reports = (candidate / "reports").exists()
    current_reports = (current / "reports").exists()
    if candidate_reports != current_reports:
        return candidate_reports
    return len(candidate.parts) > len(current.parts)


def _find_run(run_id: str, workspace: Path) -> tuple[dict[str, Any], Path] | None:
    """Return (row_dict, task_root) for a specific run_id, or None."""
    # Try cache first
    for row, task_root in _find_all_runs(workspace):
        if row.get("run_id") == run_id:
            return row, task_root
    return None


# ── Asset path resolution ────────────────────────────────────────────


def _resolve_asset(task_root: Path, stem: str, asset: str) -> Path | None:
    """Return the Path for a named asset, or None if it does not exist."""
    candidates: dict[str, Path] = {
        "report": task_root
        / "reports"
        / "run_reports"
        / f"{stem}_autoclean_report.pdf",
        "ica_report": task_root
        / "reports"
        / "ica_components"
        / f"{stem}_ica_components_all.pdf",
        "psd": task_root / "reports" / "psd_topo" / f"{stem}_psd_topo_figure.png",
        "overlay": task_root
        / "reports"
        / "raw_vs_cleaned_overlay"
        / f"{stem}_raw_vs_cleaned_overlay.png",
        "metadata": task_root
        / "reports"
        / "run_reports"
        / f"{stem}_autoclean_metadata.json",
        "channels": task_root
        / "reports"
        / "run_reports"
        / f"{stem}_autoclean_report_flagged_channels.tsv",
    }
    path = candidates.get(asset)
    if path and path.exists():
        return path
    return None


# ── Metrics extraction ───────────────────────────────────────────────


def _extract_metrics(meta: dict[str, Any]) -> dict[str, Any]:
    """Pull processing metrics out of the pipeline metadata dict defensively."""
    # Channel counts
    import_eeg = meta.get("import_eeg", {})
    channels_original: int = import_eeg.get("channelCount", 0) or 0

    save_epochs = meta.get("save_epochs_to_set", {})
    channels_retained: int = save_epochs.get("n_channels", 0) or 0

    # Bad channels from channel_removals list
    raw_removals = meta.get("channel_removals", [])
    bad_channels: list[dict[str, str]] = []
    if isinstance(raw_removals, list):
        for entry in raw_removals:
            if isinstance(entry, dict):
                ch = entry.get("channel", "")
                reason = entry.get("reason", "")
                if ch:
                    bad_channels.append({"channel": ch, "reason": reason})

    # Epochs
    epoch_step = meta.get("step_create_regular_epochs", {})
    epochs_total: int | None = epoch_step.get("initial_epoch_count")
    epochs_kept_raw: int | None = save_epochs.get("n_epochs")
    epochs_kept: int | None = (
        epochs_kept_raw
        if epochs_kept_raw is not None
        else epoch_step.get("final_epoch_count")
    )

    # ICA
    ica_step = meta.get("step_run_ica", {})
    ica_inner = ica_step.get("ica", {}) if isinstance(ica_step, dict) else {}
    ica_n_components_raw = ica_inner.get("ica_components")
    ica_n_components: int | None = (
        int(ica_n_components_raw) if ica_n_components_raw is not None else None
    )
    ica_method: str = ""
    ica_kwargs = ica_inner.get("ica_kwargs", {})
    if isinstance(ica_kwargs, dict):
        ica_method = ica_kwargs.get("method", "")

    rejection_step = meta.get("step_apply_ica_component_rejection", {})
    rejection_ica = (
        rejection_step.get("ica", {}) if isinstance(rejection_step, dict) else {}
    )
    ica_removed: list[int] = []
    raw_excluded = rejection_ica.get("final_excluded_indices", [])
    if isinstance(raw_excluded, list):
        ica_removed = [int(i) for i in raw_excluded if isinstance(i, (int, float))]

    # Duration
    filter_step = meta.get("step_filter_data", {})
    trim_step = meta.get("step_trim_edges", {})
    duration_raw: float | None = None
    duration_post: float | None = None

    # Try to get raw duration from import or filter step
    raw_duration_candidates = [
        import_eeg.get("durationSec"),
        filter_step.get("durationSec"),
        trim_step.get("original_duration"),
    ]
    for c in raw_duration_candidates:
        if c is not None:
            try:
                duration_raw = float(c)
                break
            except (TypeError, ValueError):
                pass

    # Post duration from save_epochs
    post_dur_raw = save_epochs.get("actual_duration")
    if post_dur_raw is not None:
        try:
            duration_post = float(post_dur_raw)
        except (TypeError, ValueError):
            pass

    # Filter params
    filter_low: float | None = None
    filter_high: float | None = None
    notch_freqs: list[float] = []
    if isinstance(filter_step, dict):
        try:
            lf = filter_step.get("applied_l_freq")
            if lf is not None:
                filter_low = float(lf)
        except (TypeError, ValueError):
            pass
        try:
            hf = filter_step.get("applied_h_freq")
            if hf is not None:
                filter_high = float(hf)
        except (TypeError, ValueError):
            pass
        raw_notch = filter_step.get("applied_notch_freqs", [])
        if isinstance(raw_notch, list):
            for nf in raw_notch:
                try:
                    notch_freqs.append(float(nf))
                except (TypeError, ValueError):
                    pass

    # Sample rate
    sample_rate: float | None = None
    sr_raw = import_eeg.get("sampleRate") or filter_step.get("filtered_sfreq")
    if sr_raw is not None:
        try:
            sample_rate = float(sr_raw)
        except (TypeError, ValueError):
            pass

    return {
        "channels_original": channels_original,
        "channels_retained": channels_retained,
        "bad_channels": bad_channels,
        "epochs_total": epochs_total,
        "epochs_kept": epochs_kept,
        "ica_n_components": ica_n_components,
        "ica_removed": ica_removed,
        "ica_method": ica_method,
        "duration_raw": duration_raw,
        "duration_post": duration_post,
        "filter_low": filter_low,
        "filter_high": filter_high,
        "notch_freqs": notch_freqs,
        "sample_rate": sample_rate,
    }


# ── Pydantic models ──────────────────────────────────────────────────


class RunSummary(BaseModel):
    run_id: str
    created_at: str
    task: str
    filename: str
    status: str
    success: bool
    automation_dir: str
    route_id: str | None = None


class ResultsListResponse(BaseModel):
    runs: list[RunSummary]
    total: int


class ProcessingMetrics(BaseModel):
    channels_original: int
    channels_retained: int
    bad_channels: list[dict[str, str]]
    epochs_total: int | None
    epochs_kept: int | None
    ica_n_components: int | None
    ica_removed: list[int]
    ica_method: str
    duration_raw: float | None
    duration_post: float | None
    filter_low: float | None
    filter_high: float | None
    notch_freqs: list[float]
    sample_rate: float | None


class AssetAvailability(BaseModel):
    report: bool
    ica_report: bool
    psd: bool
    overlay: bool
    metadata: bool
    channels: bool


class RunDetail(BaseModel):
    run_id: str
    created_at: str
    task: str
    filename: str
    status: str
    success: bool
    error: str | None
    metrics: ProcessingMetrics
    assets: AssetAvailability
    user_context: dict[str, Any] | None
    route_id: str | None = None


# ── Helpers ──────────────────────────────────────────────────────────


def _require_workspace() -> Path:
    if not api_state.workspace_dir:
        raise HTTPException(status_code=409, detail="Workspace not configured")
    return api_state.workspace_dir


def _route_output_map(workspace: Path) -> dict[str, Path]:
    try:
        from autoclean.utils.ingestion import build_workspace_name
        from autoclean.utils.serve_routes import load_route_specs
    except Exception:
        return {}

    mapping: dict[str, Path] = {}
    for spec in load_route_specs(workspace):
        route_id = str(spec.get("id") or "").strip()
        taskfile = str(spec.get("taskfile") or "").strip()
        montage = str(spec.get("montage") or "").strip()
        if not route_id or not taskfile or not montage:
            continue
        taskfile_label = Path(taskfile).name.replace(".py", "")
        workspace_name = build_workspace_name(
            spec.get("workspace_name", "taskfile-montage-version"),
            taskfile=taskfile_label,
            montage=montage,
            version=spec.get("version"),
        )
        automation_root = workspace / str(spec.get("automation_root", "automations"))
        mapping[route_id] = (automation_root / workspace_name).resolve()
    return mapping


def _resolve_route_id(task_root: Path, route_outputs: dict[str, Path]) -> str | None:
    resolved = task_root.resolve()
    for route_id, output_path in route_outputs.items():
        if resolved == output_path:
            return route_id
        try:
            resolved.relative_to(output_path)
            return route_id
        except ValueError:
            continue
    return None


def _row_to_summary(
    row: dict[str, Any], task_root: Path, route_outputs: dict[str, Path]
) -> RunSummary:
    unprocessed = row.get("unprocessed_file") or ""
    filename = Path(unprocessed).name if unprocessed else ""
    return RunSummary(
        run_id=row.get("run_id", ""),
        created_at=row.get("created_at", ""),
        task=row.get("task", ""),
        filename=filename,
        status=row.get("status", ""),
        success=bool(row.get("success", False)),
        automation_dir=str(task_root),
        route_id=_resolve_route_id(task_root, route_outputs),
    )


def _get_run_or_404(run_id: str, workspace: Path) -> tuple[dict[str, Any], Path]:
    found = _find_run(run_id, workspace)
    if not found:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return found


def _get_stem_and_asset(
    run_id: str, workspace: Path, asset: str
) -> tuple[dict[str, Any], Path, str, Path]:
    """Locate a run, extract the stem, find the asset path or 404."""
    row, task_root = _get_run_or_404(run_id, workspace)
    unprocessed = row.get("unprocessed_file") or ""
    stem = _extract_stem(Path(unprocessed).name)
    asset_path = _resolve_asset(task_root, stem, asset)
    if not asset_path:
        raise HTTPException(
            status_code=404, detail=f"Asset '{asset}' not available for run '{run_id}'"
        )
    return row, task_root, stem, asset_path


# ── Endpoints ────────────────────────────────────────────────────────


@router.get("", response_model=ResultsListResponse)
async def list_results(
    route_id: str | None = Query(default=None, description="Filter by route ID"),
) -> ResultsListResponse:
    """List all processed runs across all automation output directories."""
    workspace = _require_workspace()
    all_runs = _find_all_runs(workspace)
    route_outputs = _route_output_map(workspace)
    summaries = [
        _row_to_summary(row, task_root, route_outputs) for row, task_root in all_runs
    ]
    if route_id:
        summaries = [summary for summary in summaries if summary.route_id == route_id]
    return ResultsListResponse(runs=summaries, total=len(summaries))


@router.get("/{run_id}", response_model=RunDetail)
async def get_run_detail(run_id: str) -> RunDetail:
    """Full run detail with extracted processing metrics and asset availability."""
    workspace = _require_workspace()
    row, task_root = _get_run_or_404(run_id, workspace)

    unprocessed = row.get("unprocessed_file") or ""
    stem = _extract_stem(Path(unprocessed).name)
    filename = Path(unprocessed).name

    # Parse metadata
    meta: dict[str, Any] = {}
    raw_meta = row.get("metadata")
    if raw_meta:
        try:
            meta = json.loads(raw_meta)
        except Exception:
            logger.debug("Failed to parse metadata for run %s", run_id)

    metrics_dict = _extract_metrics(meta)

    # Parse user_context
    user_context: dict[str, Any] | None = None
    raw_uc = row.get("user_context")
    if raw_uc:
        try:
            user_context = json.loads(raw_uc)
        except Exception:
            pass

    # Asset availability
    assets = AssetAvailability(
        report=_resolve_asset(task_root, stem, "report") is not None,
        ica_report=_resolve_asset(task_root, stem, "ica_report") is not None,
        psd=_resolve_asset(task_root, stem, "psd") is not None,
        overlay=_resolve_asset(task_root, stem, "overlay") is not None,
        metadata=_resolve_asset(task_root, stem, "metadata") is not None,
        channels=_resolve_asset(task_root, stem, "channels") is not None,
    )

    # Error field: normalize "None" string to actual None
    error_raw = row.get("error")
    error: str | None = None
    if error_raw and str(error_raw).strip().lower() not in ("none", "null", ""):
        error = str(error_raw)

    return RunDetail(
        run_id=run_id,
        created_at=row.get("created_at", ""),
        task=row.get("task", ""),
        filename=filename,
        status=row.get("status", ""),
        success=bool(row.get("success", False)),
        error=error,
        metrics=ProcessingMetrics(**metrics_dict),
        assets=assets,
        user_context=user_context,
        route_id=_resolve_route_id(task_root, _route_output_map(workspace)),
    )


@router.get("/{run_id}/report")
async def get_report(run_id: str) -> FileResponse:
    """Serve the autoclean PDF report for a run."""
    workspace = _require_workspace()
    _row, _task_root, _stem, asset_path = _get_stem_and_asset(
        run_id, workspace, "report"
    )
    return FileResponse(path=str(asset_path), media_type="application/pdf")


@router.get("/{run_id}/ica-report")
async def get_ica_report(run_id: str) -> FileResponse:
    """Serve the ICA components PDF report for a run."""
    workspace = _require_workspace()
    _row, _task_root, _stem, asset_path = _get_stem_and_asset(
        run_id, workspace, "ica_report"
    )
    return FileResponse(path=str(asset_path), media_type="application/pdf")


@router.get("/{run_id}/psd")
async def get_psd(run_id: str) -> FileResponse:
    """Serve the PSD topomap PNG for a run."""
    workspace = _require_workspace()
    _row, _task_root, _stem, asset_path = _get_stem_and_asset(run_id, workspace, "psd")
    return FileResponse(
        path=str(asset_path),
        media_type="image/png",
        filename=asset_path.name,
    )


@router.get("/{run_id}/overlay")
async def get_overlay(run_id: str) -> FileResponse:
    """Serve the raw-vs-cleaned overlay PNG for a run."""
    workspace = _require_workspace()
    _row, _task_root, _stem, asset_path = _get_stem_and_asset(
        run_id, workspace, "overlay"
    )
    return FileResponse(
        path=str(asset_path),
        media_type="image/png",
        filename=asset_path.name,
    )


@router.get("/{run_id}/metadata")
async def get_metadata(run_id: str) -> JSONResponse:
    """Return the full autoclean metadata JSON for a run."""
    workspace = _require_workspace()

    # Try JSON file on disk first
    row, task_root = _get_run_or_404(run_id, workspace)
    unprocessed = row.get("unprocessed_file") or ""
    stem = _extract_stem(Path(unprocessed).name)
    asset_path = _resolve_asset(task_root, stem, "metadata")

    if asset_path:
        try:
            data = json.loads(asset_path.read_text(encoding="utf-8"))
            return JSONResponse(content=data)
        except Exception as exc:
            logger.warning("Failed to read metadata file %s: %s", asset_path, exc)

    # Fallback: return the metadata column from the DB
    raw_meta = row.get("metadata")
    if raw_meta:
        try:
            data = json.loads(raw_meta)
            return JSONResponse(content=data)
        except Exception:
            pass

    raise HTTPException(
        status_code=404, detail=f"Metadata not available for run '{run_id}'"
    )


@router.get("/{run_id}/channels")
async def get_channels(run_id: str) -> JSONResponse:
    """Return flagged channels as a JSON array, parsed from the TSV file."""
    workspace = _require_workspace()
    row, task_root = _get_run_or_404(run_id, workspace)
    unprocessed = row.get("unprocessed_file") or ""
    stem = _extract_stem(Path(unprocessed).name)
    asset_path = _resolve_asset(task_root, stem, "channels")

    if not asset_path:
        # Fallback: build from metadata channel_removals
        raw_meta = row.get("metadata")
        if raw_meta:
            try:
                meta = json.loads(raw_meta)
                removals = meta.get("channel_removals", [])
                if isinstance(removals, list):
                    return JSONResponse(content={"channels": removals})
            except Exception:
                pass
        raise HTTPException(
            status_code=404,
            detail=f"Channels data not available for run '{run_id}'",
        )

    try:
        text = asset_path.read_text(encoding="utf-8")
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        rows = [dict(r) for r in reader]
        # Normalise to {channel, reason} shape expected by the frontend
        normalised = [
            {
                "channel": r.get("channel", r.get("label", "")),
                "reason": r.get("label", r.get("reason", "")),
            }
            for r in rows
        ]
        return JSONResponse(content={"channels": normalised})
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to parse channels file: {exc}"
        )


# ── ICA PDF endpoints ─────────────────────────────────────────────


@router.get("/{run_id}/events")
async def get_run_events(run_id: str) -> JSONResponse:
    """Return event analysis for a processed run."""
    workspace = _require_workspace()
    row, task_root = _get_run_or_404(run_id, workspace)

    # Parse metadata
    raw_meta = row.get("metadata", "{}")
    meta = json.loads(raw_meta) if isinstance(raw_meta, str) else (raw_meta or {})

    import_info = meta.get("import_eeg", {})
    has_events = import_info.get("hasEvents", False)
    event_dict = import_info.get("event_dict", {})  # {label: code}
    event_count = import_info.get("event_count", 0)
    unique_types = import_info.get("unique_event_types", [])

    # Try to read BIDS events.tsv for timeline data
    events_timeline: list[dict[str, Any]] = []
    bids_dir = task_root / "bids"
    if bids_dir.exists():
        for events_tsv in bids_dir.rglob("*_events.tsv"):
            try:
                with open(events_tsv) as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for ev_row in reader:
                        try:
                            events_timeline.append(
                                {
                                    "onset": float(ev_row.get("onset") or 0),
                                    "duration": float(ev_row.get("duration") or 0),
                                    "trial_type": ev_row.get("trial_type", "unknown"),
                                    "value": ev_row.get("value", ""),
                                }
                            )
                        except (ValueError, TypeError):
                            continue  # skip malformed rows
            except Exception:
                pass
            break  # Only read the first events.tsv found

    # Sort timeline by onset once — all downstream analysis uses this order
    events_timeline.sort(key=lambda e: e["onset"])
    onsets = [e["onset"] for e in events_timeline]

    # Per-type timing
    type_summary: list[dict[str, Any]] = []
    if event_dict:
        type_onsets: dict[str, list[float]] = {}
        for ev in events_timeline:
            type_onsets.setdefault(ev["trial_type"], []).append(ev["onset"])

        for label, code in event_dict.items():
            typed_onsets = type_onsets.get(label, type_onsets.get(str(code), []))
            count = len(typed_onsets)
            per_isis = [
                typed_onsets[i + 1] - typed_onsets[i]
                for i in range(len(typed_onsets) - 1)
            ]
            type_summary.append(
                {
                    "label": label,
                    "code": code,
                    "count": count,
                    "first_onset": round(typed_onsets[0], 3) if typed_onsets else None,
                    "last_onset": round(typed_onsets[-1], 3) if typed_onsets else None,
                    "mean_isi": (
                        round(statistics.mean(per_isis), 3) if per_isis else None
                    ),
                    "median_isi": (
                        round(statistics.median(per_isis), 3) if per_isis else None
                    ),
                }
            )

    # Global ISI, long gaps
    isi_stats: dict[str, Any] | None = None
    long_gaps: list[dict[str, Any]] = []
    if len(onsets) > 1:
        isis = [onsets[i + 1] - onsets[i] for i in range(len(onsets) - 1)]
        if isis:
            isi_stats = {
                "min": round(min(isis), 3),
                "max": round(max(isis), 3),
                "mean": round(statistics.mean(isis), 3),
                "median": round(statistics.median(isis), 3),
                "std": round(statistics.stdev(isis), 3) if len(isis) > 1 else 0,
                "count": len(isis),
            }
        long_gaps = [
            {
                "start": round(onsets[i], 3),
                "end": round(onsets[i + 1], 3),
                "duration": round(onsets[i + 1] - onsets[i], 3),
            }
            for i in range(len(onsets) - 1)
            if onsets[i + 1] - onsets[i] > 30.0
        ][:10]

    # Transitions (uses sorted timeline order)
    transitions_counter: Counter[tuple[str, str]] = Counter()
    for i in range(len(events_timeline) - 1):
        transitions_counter[
            (events_timeline[i]["trial_type"], events_timeline[i + 1]["trial_type"])
        ] += 1
    top_transitions = [
        {"from": f, "to": t, "count": c}
        for (f, t), c in transitions_counter.most_common(10)
    ]

    # Duration and rate
    duration_sec = round(onsets[-1] - onsets[0], 3) if onsets else None
    events_per_min = (
        round(len(onsets) / (duration_sec / 60.0), 2)
        if duration_sec and duration_sec > 0
        else None
    )

    return JSONResponse(
        content={
            "has_events": has_events,
            "event_count": event_count,
            "event_types": type_summary,
            "unique_type_count": len(unique_types),
            "isi_stats": isi_stats,
            "recording_type": (
                "event_related"
                if has_events and event_count > _MIN_EVENT_RELATED
                else "resting_state"
            ),
            "long_gaps": long_gaps,
            "transitions": top_transitions,
            "duration_sec": duration_sec,
            "events_per_min": events_per_min,
        }
    )


@router.get("/export/csv")
async def export_results_csv() -> Response:
    """Export all run summaries as a downloadable CSV file."""
    workspace = _require_workspace()
    all_runs = _find_all_runs(workspace)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["run_id", "created_at", "task", "filename", "status", "success"])
    for row, _task_root in all_runs:
        unprocessed = row.get("unprocessed_file") or ""
        filename = Path(unprocessed).name if unprocessed else ""
        writer.writerow(
            [
                row.get("run_id", ""),
                row.get("created_at", ""),
                row.get("task", ""),
                filename,
                row.get("status", ""),
                "Yes" if row.get("success") else "No",
            ]
        )

    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=autoclean_results.csv"},
    )


@router.get("/{run_id}/download")
async def download_run_artifacts(run_id: str) -> Response:
    """Download all available artifacts for a run as a ZIP archive."""
    import zipfile  # noqa: PLC0415

    workspace = _require_workspace()
    row, task_root = _get_run_or_404(run_id, workspace)
    unprocessed = row.get("unprocessed_file") or ""
    stem = _extract_stem(Path(unprocessed).name)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for asset_name in (
            "report",
            "ica_report",
            "psd",
            "overlay",
            "metadata",
            "channels",
        ):
            asset_path = _resolve_asset(task_root, stem, asset_name)
            if asset_path:
                zf.write(str(asset_path), asset_path.name)

    buf.seek(0)
    zip_filename = f"{stem}_artifacts.zip"
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={zip_filename}"},
    )


@router.get("/{run_id}/ica/summary")
async def get_ica_summary(run_id: str) -> JSONResponse:
    """Return structured ICA component classification data and PDF page structure."""
    workspace = _require_workspace()
    row, task_root = _get_run_or_404(run_id, workspace)
    unprocessed = row.get("unprocessed_file") or ""
    stem = _extract_stem(Path(unprocessed).name)
    pdf_path = _resolve_asset(task_root, stem, "ica_report")
    if not pdf_path:
        raise HTTPException(status_code=404, detail="ICA report not found")

    try:
        from autoclean.api.pdf_extractor import extract_ica_full  # noqa: PLC0415

        result = extract_ica_full(pdf_path)
        components = result["components"]
        structure = result["structure"]
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="PDF extraction requires PyMuPDF (pip install pymupdf)",
        )
    except Exception as exc:
        logger.exception("ICA summary extraction failed for run %s: %s", run_id, exc)
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {exc}")

    return JSONResponse(content={"components": components, "structure": structure})


@router.get("/{run_id}/ica/page/{page_num}")
async def get_ica_page(
    run_id: str,
    page_num: int,
    dpi: int = Query(default=120, ge=40, le=200),
) -> Response:
    """Render a single ICA PDF page as a PNG image."""
    workspace = _require_workspace()
    row, task_root = _get_run_or_404(run_id, workspace)
    unprocessed = row.get("unprocessed_file") or ""
    stem = _extract_stem(Path(unprocessed).name)
    pdf_path = _resolve_asset(task_root, stem, "ica_report")
    if not pdf_path:
        raise HTTPException(status_code=404, detail="ICA report not found")

    try:
        from autoclean.api.pdf_extractor import (  # noqa: PLC0415
            get_ica_page_count,
            render_ica_page,
        )

        total = get_ica_page_count(pdf_path)
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="PDF extraction requires PyMuPDF (pip install pymupdf)",
        )
    except Exception as exc:
        logger.exception("ICA page count failed for run %s: %s", run_id, exc)
        raise HTTPException(status_code=500, detail=f"PDF read failed: {exc}")

    if page_num < 0 or page_num >= total:
        raise HTTPException(
            status_code=404,
            detail=f"Page {page_num} not found (PDF has {total} pages)",
        )

    try:
        png_b64 = render_ica_page(pdf_path, page_num, dpi=dpi)
        png_bytes = base64.b64decode(png_b64)
    except Exception as exc:
        logger.exception(
            "ICA page render failed for run %s page %d: %s", run_id, page_num, exc
        )
        raise HTTPException(status_code=500, detail=f"Page render failed: {exc}")

    return Response(content=png_bytes, media_type="image/png")


# ── Decision workflow ──────────────────────────────────────────────


class DecisionInput(BaseModel):
    decision: Literal["pass", "fail", "review", "clear"]
    notes: str = ""


class DecisionRecord(BaseModel):
    run_id: str
    decision: str
    notes: str
    decided_at: str
    filename: str


class DecisionsResponse(BaseModel):
    decisions: dict[str, DecisionRecord]
    total: int


def _decisions_path(workspace: Path) -> Path:
    return workspace / "decisions.json"


def _load_decisions(workspace: Path) -> dict[str, Any]:
    try:
        return json.loads(_decisions_path(workspace).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _decisions_to_csv(decisions: dict[str, Any]) -> str:
    """Render decisions dict as a CSV string."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["run_id", "filename", "decision", "notes", "decided_at"])
    for rec in decisions.values():
        writer.writerow(
            [
                rec.get("run_id", ""),
                rec.get("filename", ""),
                rec.get("decision", ""),
                rec.get("notes", ""),
                rec.get("decided_at", ""),
            ]
        )
    return buf.getvalue()


def _save_decisions(workspace: Path, decisions: dict[str, Any]) -> None:
    path = _decisions_path(workspace)
    path.write_text(json.dumps(decisions, indent=2), encoding="utf-8")


@router.get("/decisions")
async def get_decisions() -> DecisionsResponse:
    """Return all review decisions for the current workspace."""
    workspace = _require_workspace()
    decisions = _load_decisions(workspace)
    records = {k: DecisionRecord(**v) for k, v in decisions.items()}
    return DecisionsResponse(decisions=records, total=len(records))


@router.put("/{run_id}/decision")
async def set_decision(run_id: str, body: DecisionInput) -> JSONResponse:
    """Set or update the review decision for a run."""
    from datetime import datetime, timezone  # noqa: PLC0415

    workspace = _require_workspace()

    # Verify run exists
    row, _task_root = _get_run_or_404(run_id, workspace)
    unprocessed = row.get("unprocessed_file") or ""
    filename = Path(unprocessed).name

    decisions = _load_decisions(workspace)

    if body.decision == "clear":
        decisions.pop(run_id, None)
    else:
        decisions[run_id] = {
            "run_id": run_id,
            "filename": filename,
            "decision": body.decision,
            "notes": body.notes,
            "decided_at": datetime.now(timezone.utc).isoformat(),
        }

    _save_decisions(workspace, decisions)

    return JSONResponse(
        content={"success": True, "run_id": run_id, "decision": body.decision}
    )


@router.get("/decisions/export/csv")
async def export_decisions_csv() -> Response:
    """Export decisions as a downloadable CSV file."""
    workspace = _require_workspace()
    decisions = _load_decisions(workspace)

    return Response(
        content=_decisions_to_csv(decisions),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=decisions.csv"},
    )
