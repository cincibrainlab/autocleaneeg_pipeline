"""Single-file event analyzer.

Mirrors the CLI `events discover` and `events analyze` commands.
Loads a raw EEG file with MNE, extracts events, and returns the same
EventsResponse format used by the Results Events tab.
"""

import asyncio
import logging
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from autoclean.api.auth.dependencies import require_permission
from autoclean.api.auth.models import Permission
logger = logging.getLogger(__name__)
router = APIRouter()

_SUPPORTED_EXTENSIONS = {".set", ".edf", ".bdf", ".fif", ".vhdr", ".raw", ".mff"}
_MIN_EVENT_RELATED = 10


class AnalyzeRequest(BaseModel):
    file_path: str


def _analyze_file(file_path: Path) -> dict[str, Any]:
    """Synchronous event analysis — runs in thread pool."""
    import mne  # Heavy import, keep lazy

    # Load raw file (preload=False for speed — we only need events)
    ext = file_path.suffix.lower()
    readers = {
        ".set": lambda p: mne.io.read_raw_eeglab(str(p), preload=False, verbose=False),
        ".edf": lambda p: mne.io.read_raw_edf(str(p), preload=False, verbose=False),
        ".bdf": lambda p: mne.io.read_raw_bdf(str(p), preload=False, verbose=False),
        ".fif": lambda p: mne.io.read_raw_fif(str(p), preload=False, verbose=False),
        ".vhdr": lambda p: mne.io.read_raw_brainvision(str(p), preload=False, verbose=False),
    }
    reader = readers.get(ext)
    if reader is None:
        # Try generic read_raw for other formats
        raw = mne.io.read_raw(str(file_path), preload=False, verbose=False)
    else:
        raw = reader(file_path)

    # Extract events
    events = None
    event_id: dict[str, int] = {}

    # Method 1: Try mne.find_events (for stim channels)
    try:
        events = mne.find_events(raw, verbose=False)
    except Exception:
        pass

    # Method 2: Fall back to annotations
    if events is None or len(events) == 0:
        if raw.annotations and len(raw.annotations) > 0:
            events, event_id = mne.events_from_annotations(raw, verbose=False)

    # Duration is always available
    duration_raw = round(raw.times[-1], 3) if len(raw.times) > 0 else 0

    if events is None or len(events) == 0:
        # No events found
        return {
            "has_events": False,
            "event_count": 0,
            "event_types": [],
            "unique_type_count": 0,
            "isi_stats": None,
            "recording_type": "resting_state",
            "long_gaps": [],
            "transitions": [],
            "duration_sec": None,
            "events_per_min": None,
            "file_info": {
                "filename": file_path.name,
                "n_channels": raw.info["nchan"],
                "sfreq": raw.info["sfreq"],
                "duration": duration_raw,
            },
        }

    # Build event_id if not from annotations
    if not event_id:
        unique_codes = sorted(set(int(ev[2]) for ev in events))
        event_id = {str(code): code for code in unique_codes}

    # Reverse the event_id to get code -> label
    code_to_label: dict[int, str] = {v: k for k, v in event_id.items()}

    # Build timeline from events array
    sfreq = raw.info["sfreq"]
    timeline = []
    for ev in events:
        onset = ev[0] / sfreq
        code = int(ev[2])
        label = code_to_label.get(code, str(code))
        timeline.append({"onset": onset, "trial_type": label, "code": code})

    # Sort by onset
    timeline.sort(key=lambda e: e["onset"])
    onsets = [e["onset"] for e in timeline]

    # Per-type timing
    type_onsets: dict[str, list[float]] = {}
    for ev in timeline:
        type_onsets.setdefault(ev["trial_type"], []).append(ev["onset"])

    type_summary = []
    for label, code in event_id.items():
        typed = type_onsets.get(label, [])
        count = len(typed)
        per_isis = [typed[i + 1] - typed[i] for i in range(len(typed) - 1)]
        type_summary.append({
            "label": label,
            "code": code,
            "count": count,
            "first_onset": round(typed[0], 3) if typed else None,
            "last_onset": round(typed[-1], 3) if typed else None,
            "mean_isi": round(statistics.mean(per_isis), 3) if per_isis else None,
            "median_isi": round(statistics.median(per_isis), 3) if per_isis else None,
        })

    # Global ISI
    isi_stats = None
    long_gaps: list[dict[str, float]] = []
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

    # Transitions
    tc: Counter[tuple[str, str]] = Counter()
    for i in range(len(timeline) - 1):
        tc[(timeline[i]["trial_type"], timeline[i + 1]["trial_type"])] += 1
    top_transitions = [
        {"from": f, "to": t, "count": c} for (f, t), c in tc.most_common(10)
    ]

    # Duration and rate (span between first and last event)
    duration_sec = round(onsets[-1] - onsets[0], 3) if onsets else None
    events_per_min = (
        round(len(onsets) / (duration_sec / 60.0), 2)
        if duration_sec and duration_sec > 0
        else None
    )

    return {
        "has_events": True,
        "event_count": len(events),
        "event_types": type_summary,
        "unique_type_count": len(event_id),
        "isi_stats": isi_stats,
        "recording_type": (
            "event_related" if len(events) > _MIN_EVENT_RELATED else "resting_state"
        ),
        "long_gaps": long_gaps,
        "transitions": top_transitions,
        "duration_sec": duration_sec,
        "events_per_min": events_per_min,
        "file_info": {
            "filename": file_path.name,
            "n_channels": raw.info["nchan"],
            "sfreq": raw.info["sfreq"],
            "duration": duration_raw,
        },
    }


@router.post("/analyze", dependencies=[Depends(require_permission(Permission.EVENT_ANALYZE))])
async def analyze_events(body: AnalyzeRequest) -> JSONResponse:
    """Analyze events in a raw EEG file. Heavy operation — runs in thread pool."""
    file_path = Path(body.file_path).expanduser().resolve()

    if not file_path.exists():
        raise HTTPException(404, f"File not found: {file_path}")
    if file_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported format: {file_path.suffix}. "
            f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}",
        )

    try:
        result = await asyncio.to_thread(_analyze_file, file_path)
    except Exception as exc:
        logger.exception("Event analysis failed for %s", file_path)
        raise HTTPException(500, f"Analysis failed: {exc}")

    return JSONResponse(content=result)
