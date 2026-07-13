"""EEGLAB import provenance extraction and reporting.

The schema is intentionally flat at the edges so batch tooling can consume it:
``summary["summary_row"]`` is one dataset-table row, while
``summary["documented_provenance"]`` keeps the more detailed source fields.
Missing values are represented as ``"unknown"`` when a field exists but is
blank/ambiguous, and ``"unavailable"`` when the field is absent.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio

UNKNOWN = "unknown"
UNAVAILABLE = "unavailable"
SCHEMA_VERSION = "1.0"


def extract_eeglab_provenance(file_path: Path) -> dict[str, Any]:
    """Load an EEGLAB ``.set`` file and return a provenance summary."""

    eeg = _load_eeg_struct(file_path)
    return summarize_eeglab_provenance(eeg, source_file=file_path)


def summarize_eeglab_provenance(
    eeg: Any, source_file: Path | str | None = None
) -> dict[str, Any]:
    """Summarize documented EEGLAB metadata from an EEG-like object."""

    source_name = Path(source_file).name if source_file else UNAVAILABLE
    documented = {
        "setname": _clean_value(_field(eeg, "setname")),
        "srate": _clean_value(_field(eeg, "srate")),
        "nbchan": _clean_value(_field(eeg, "nbchan")),
        "trials": _clean_value(_field(eeg, "trials")),
        "pnts": _clean_value(_field(eeg, "pnts")),
        "epoch_window": _extract_epoch_window(eeg),
        "reference": _clean_value(_field(eeg, "ref")),
        "history": _clean_value(_field(eeg, "history")),
        "comments": _clean_value(_field(eeg, "comments")),
        "etc_keys": _field_names(_field(eeg, "etc")),
        "channels": _extract_channels(_field(eeg, "chanlocs")),
        "events": _extract_events(_field(eeg, "event")),
        "ica": _extract_ica(eeg),
        "iclabel": _extract_iclabel(_field(eeg, "etc")),
        "interpolation": _extract_interpolation(_field(eeg, "etc")),
        "task_counts": _extract_task_counts(_field(eeg, "etc")),
    }
    unavailable = _collect_unavailable(documented)
    summary_row = _build_summary_row(source_name, documented)
    inferred = _infer_preprocessing(documented)

    return {
        "schema_version": SCHEMA_VERSION,
        "source_file": source_name,
        "created_at": datetime.now().isoformat(),
        "documented_provenance": documented,
        "inferred_preprocessing": inferred,
        "unavailable": unavailable,
        "summary_row": summary_row,
        "artifact_paths": {},
    }


def write_eeglab_provenance_artifacts(
    summary: dict[str, Any], output_dir: Path, stem: str
) -> dict[str, str]:
    """Write per-file machine and human-readable provenance artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}_eeglab_provenance.json"
    report_path = output_dir / f"{stem}_eeglab_provenance.md"

    artifact_paths = {
        "json": str(json_path),
        "report": str(report_path),
    }
    summary["artifact_paths"] = artifact_paths
    json_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    report_path.write_text(render_eeglab_provenance_report(summary), encoding="utf-8")
    return artifact_paths


def render_eeglab_provenance_report(summary: dict[str, Any]) -> str:
    """Render a concise Markdown provenance report."""

    documented = summary["documented_provenance"]
    row = summary["summary_row"]
    lines = [
        f"# EEGLAB Provenance: {summary['source_file']}",
        "",
        "## Documented Provenance",
        f"- Set name: {row['setname']}",
        f"- Sampling rate: {row['srate']}",
        f"- Channels: {row['nbchan']}",
        f"- Trials: {row['trials']}",
        f"- Points: {row['pnts']}",
        f"- Epoch window: {row['epoch_window']}",
        f"- Reference: {row['reference']}",
        f"- Channel labels: {row['channel_labels']}",
        f"- Channel types: {row['channel_types']}",
        f"- Event labels: {row['event_labels']}",
        f"- Event counts: {row['event_counts']}",
        f"- ICA structure: {row['ica_structure']}",
        f"- ICLabel structure: {row['iclabel_structure']}",
        f"- Interpolation metadata: {row['interpolation_metadata']}",
        f"- Task count fields: {row['task_counts']}",
        "",
        "## Inferred Preprocessing",
    ]
    inferred = summary["inferred_preprocessing"]
    if inferred["steps"]:
        lines.extend(f"- {step}" for step in inferred["steps"])
    else:
        lines.append("- unavailable")
    lines.extend(
        [
            "",
            "## Unavailable Or Unknown Metadata",
            *[f"- {item}" for item in summary["unavailable"]],
            "",
            "## Source Text",
            f"- EEG.history: {_short_text(documented['history'])}",
            f"- EEG.comments: {_short_text(documented['comments'])}",
            "",
        ]
    )
    return "\n".join(lines)


def build_eeglab_dataset_summary(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate one summary row per file and compute consistency warnings."""

    rows = [summary["summary_row"] for summary in summaries]
    return {
        "schema_version": SCHEMA_VERSION,
        "rows": rows,
        "warnings": build_eeglab_consistency_warnings(rows),
    }


def build_eeglab_consistency_warnings(rows: list[dict[str, Any]]) -> list[str]:
    """Warn when batch-level provenance fields differ across files."""

    checks = {
        "sampling rate": "srate",
        "channel labels": "channel_labels",
        "epoch window": "epoch_window",
        "reference": "reference",
        "event labels/counts": "event_counts",
        "ICA structure": "ica_structure",
        "ICLabel structure": "iclabel_structure",
    }
    warnings = []
    for label, key in checks.items():
        values = {str(row.get(key, UNAVAILABLE)) for row in rows}
        known_values = values - {UNKNOWN, UNAVAILABLE, ""}
        if len(known_values) > 1:
            warnings.append(f"Inconsistent {label}: {', '.join(sorted(known_values))}")
    return warnings


def resolve_eeglab_provenance_dir(autoclean_dict: dict[str, Any]) -> Path:
    """Resolve the artifact directory from run config with conservative fallbacks."""

    if autoclean_dict.get("reports_dir"):
        return Path(autoclean_dict["reports_dir"]) / "run_reports" / "provenance"
    if autoclean_dict.get("metadata_dir"):
        return Path(autoclean_dict["metadata_dir"]) / "provenance"
    return Path(autoclean_dict["unprocessed_file"]).parent / "provenance"


def _load_eeg_struct(file_path: Path) -> Any:
    data = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    if "EEG" not in data:
        raise ValueError("EEGLAB .set file did not contain an EEG structure")
    return data["EEG"]


def _field(obj: Any, name: str, default: Any = UNAVAILABLE) -> Any:
    if _is_missing(obj):
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _field_names(obj: Any) -> list[str]:
    if _is_missing(obj):
        return []
    if isinstance(obj, dict):
        return sorted(str(key) for key in obj)
    names = getattr(obj, "_fieldnames", None)
    if names:
        return sorted(str(name) for name in names)
    if hasattr(obj, "__dict__"):
        return sorted(name for name in vars(obj) if not name.startswith("_"))
    return []


def _clean_value(value: Any) -> Any:
    if _is_missing(value):
        return UNAVAILABLE
    value = _json_safe(value)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else UNKNOWN
    if isinstance(value, list) and not value:
        return UNAVAILABLE
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        public = {k: v for k, v in vars(value).items() if not k.startswith("_")}
        if public:
            return _json_safe(public)
    return str(value)


def _shape(value: Any) -> list[int] | str:
    if _is_missing(value):
        return UNAVAILABLE
    shape = getattr(value, "shape", None)
    if shape is None:
        arr = np.asarray(value)
        shape = arr.shape
    return [int(dim) for dim in shape]


def _iter_records(value: Any) -> list[Any]:
    if _is_missing(value):
        return []
    if isinstance(value, np.ndarray):
        return list(value.ravel())
    if isinstance(value, list):
        return value
    return [value]


def _extract_epoch_window(eeg: Any) -> dict[str, Any] | str:
    xmin = _field(eeg, "xmin")
    xmax = _field(eeg, "xmax")
    if _is_missing(xmin) and _is_missing(xmax):
        return UNAVAILABLE
    return {"xmin": _clean_value(xmin), "xmax": _clean_value(xmax)}


def _extract_channels(chanlocs: Any) -> dict[str, Any]:
    records = _iter_records(chanlocs)
    labels = []
    types = []
    for chan in records:
        label = _clean_value(_field(chan, "labels"))
        kind = _clean_value(_field(chan, "type"))
        if not _is_missing(label):
            labels.append(label)
        if not _is_missing(kind):
            types.append(kind)
    return {
        "count": len(records) if records else UNAVAILABLE,
        "labels": labels or UNAVAILABLE,
        "types": dict(Counter(str(kind) for kind in types)) if types else UNAVAILABLE,
    }


def _extract_events(events: Any) -> dict[str, Any]:
    records = _iter_records(events)
    labels = []
    codes = []
    for event in records:
        event_type = _clean_value(_field(event, "type"))
        event_code = _clean_value(_field(event, "code"))
        if not _is_missing(event_type):
            labels.append(str(event_type))
        if not _is_missing(event_code):
            codes.append(str(event_code))
    counts = dict(Counter(labels)) if labels else UNAVAILABLE
    return {
        "count": len(records) if records else UNAVAILABLE,
        "labels": sorted(set(labels)) if labels else UNAVAILABLE,
        "codes": sorted(set(codes)) if codes else UNAVAILABLE,
        "counts": counts,
    }


def _extract_ica(eeg: Any) -> dict[str, Any]:
    fields = {}
    for name in ("icaweights", "icasphere", "icawinv", "icaact"):
        value = _field(eeg, name)
        fields[name] = {
            "present": not _is_missing(value),
            "shape": _shape(value) if not _is_missing(value) else UNAVAILABLE,
        }
    return fields


def _extract_iclabel(etc: Any) -> dict[str, Any]:
    ic_classification = _field(etc, "ic_classification")
    iclabel = _field(ic_classification, "ICLabel")
    classes = _clean_value(_field(iclabel, "classes"))
    classifications = _field(iclabel, "classifications")
    return {
        "present": not _is_missing(iclabel),
        "classes": classes,
        "probability_matrix_shape": (
            _shape(classifications) if not _is_missing(classifications) else UNAVAILABLE
        ),
    }


def _extract_interpolation(etc: Any) -> dict[str, Any] | str:
    fields = {}
    for name in _field_names(etc):
        if "interp" in name.lower() or "interpolat" in name.lower():
            fields[name] = _clean_value(_field(etc, name))
    return fields or UNAVAILABLE


def _extract_task_counts(etc: Any) -> dict[str, Any] | str:
    fields = {}
    for name in _field_names(etc):
        lowered = name.lower()
        if "task" in lowered and ("count" in lowered or "trial" in lowered):
            fields[name] = _clean_value(_field(etc, name))
    return fields or UNAVAILABLE


def _infer_preprocessing(documented: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(
        str(value)
        for value in (documented.get("history"), documented.get("comments"))
        if not _is_missing(value) and value != UNKNOWN
    ).lower()
    patterns = {
        "filtering": ("filter", "pop_eegfilt", "eegfilt"),
        "resampling": ("resample", "pop_resample"),
        "rereferencing": ("reref", "reference", "pop_reref"),
        "ica": ("runica", "ica", "pop_runica"),
        "channel interpolation": ("interpol", "pop_interp"),
        "epoching": ("epoch", "pop_epoch"),
    }
    steps = [
        step
        for step, needles in patterns.items()
        if any(needle in text for needle in needles)
    ]
    return {
        "source": "EEG.history/EEG.comments keyword scan" if text else UNAVAILABLE,
        "steps": steps,
    }


def _collect_unavailable(documented: dict[str, Any]) -> list[str]:
    paths = []

    def walk(prefix: str, value: Any) -> None:
        if _is_unavailable_path_value(value):
            paths.append(prefix)
        elif isinstance(value, dict):
            for key, child in value.items():
                walk(f"{prefix}.{key}", child)

    for key, value in documented.items():
        walk(key, value)
    return sorted(set(paths))


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value == UNAVAILABLE)


def _is_unavailable_path_value(value: Any) -> bool:
    if _is_missing(value) or value == UNKNOWN:
        return True
    if isinstance(value, (list, tuple, dict)) and not value:
        return True
    return False


def _build_summary_row(source_name: str, documented: dict[str, Any]) -> dict[str, Any]:
    channels = documented["channels"]
    events = documented["events"]
    return {
        "source_file": source_name,
        "setname": documented["setname"],
        "srate": documented["srate"],
        "nbchan": documented["nbchan"],
        "trials": documented["trials"],
        "pnts": documented["pnts"],
        "epoch_window": _compact(documented["epoch_window"]),
        "reference": documented["reference"],
        "channel_labels": _compact(channels["labels"]),
        "channel_types": _compact(channels["types"]),
        "event_labels": _compact(events["labels"]),
        "event_codes": _compact(events["codes"]),
        "event_counts": _compact(events["counts"]),
        "ica_structure": _compact(documented["ica"]),
        "iclabel_structure": _compact(documented["iclabel"]),
        "interpolation_metadata": _compact(documented["interpolation"]),
        "task_counts": _compact(documented["task_counts"]),
    }


def _compact(value: Any) -> str:
    if _is_missing(value) or value == UNKNOWN:
        return value
    return json.dumps(_json_safe(value), sort_keys=True)


def _short_text(value: Any, limit: int = 500) -> str:
    if _is_missing(value) or value == UNKNOWN:
        return value
    text = str(value).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."
