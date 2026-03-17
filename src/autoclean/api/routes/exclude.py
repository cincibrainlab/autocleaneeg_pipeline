"""Browser-native AutoClean Exclude endpoints."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import mne
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from autoclean.api.pdf_extractor import extract_ica_full
from autoclean.api.state import api_state

router = APIRouter()

_SUFFIXES = ["_comp_epo", "_comp", "_epo", "_postedit", "_preproc", "_raw", "_clean"]
_REPROCESS_JOBS: dict[str, dict[str, Any]] = {}
_PREPROCESSING_LOG_NAME = "preprocessing_log.csv"
_PREPROCESSING_KEY_COLUMN = "subj_basename"


def _strip_suffixes(stem: str) -> str:
    for suffix in sorted(_SUFFIXES, key=len, reverse=True):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _resolve_exports_root() -> Path:
    workspace = api_state.workspace_dir
    if workspace is None:
        raise HTTPException(status_code=500, detail="Workspace not configured")

    direct = workspace / "exports"
    if direct.exists() and direct.is_dir():
        return direct.resolve()

    candidates = sorted(
        [p.resolve() for p in workspace.rglob("exports") if p.is_dir()],
        key=lambda p: len(p.parts),
    )
    if candidates:
        return candidates[0]

    raise HTTPException(status_code=404, detail="No exports directory found in workspace")


def _decisions_paths(root: Path) -> tuple[Path, Path]:
    return (
        root / "autoclean_exclusion_decisions.json",
        root / "autoclean_exclusion_decisions.csv",
    )


def _default_record(relative_path: str) -> dict[str, Any]:
    return {
        "status": "UNSET",
        "notes": "",
        "relative_path": relative_path,
        "last_updated": "",
        "epochs_reviewed": False,
        "bad_epochs_count": 0,
        "bad_epoch_indices": "",
        "bad_epoch_times": "",
        "bad_epoch_events": "",
        "total_epochs": 0,
        "epoch_rejection_rate": 0.0,
        "manual_bad_channels": [],
        "manual_rejected_ica": [],
        "qa_export_hash": "",
        "qa_export_timestamp": "",
        "qa_export_path": "",
        "reprocess_modified": False,
        "reprocess_fix_type": "",
        "reprocess_timestamp": "",
        "modified_source": "web",
        "revision": 0,
    }


def _load_decisions(root: Path) -> dict[str, dict[str, Any]]:
    json_path, _ = _decisions_paths(root)
    if not json_path.exists():
        return {}
    try:
        data = json.loads(json_path.read_text())
        if isinstance(data, dict):
            return {str(k): v for k, v in data.items() if isinstance(v, dict)}
    except Exception:
        pass
    return {}


def _save_decisions(root: Path, decisions: dict[str, dict[str, Any]]) -> None:
    json_path, csv_path = _decisions_paths(root)
    json_path.write_text(json.dumps(decisions, indent=2, sort_keys=True))

    fieldnames = [
        "entry",
        "status",
        "notes",
        "relative_path",
        "last_updated",
        "epochs_reviewed",
        "bad_epochs_count",
        "bad_epoch_indices",
        "bad_epoch_times",
        "bad_epoch_events",
        "total_epochs",
        "epoch_rejection_rate",
        "manual_bad_channels",
        "manual_rejected_ica",
        "qa_export_hash",
        "qa_export_timestamp",
        "qa_export_path",
        "reprocess_modified",
        "reprocess_fix_type",
        "reprocess_timestamp",
        "modified_source",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for key, record in sorted(decisions.items()):
            writer.writerow(
                {
                    "entry": key,
                    "status": record.get("status", "UNSET"),
                    "notes": record.get("notes", ""),
                    "relative_path": record.get("relative_path", ""),
                    "last_updated": record.get("last_updated", ""),
                    "epochs_reviewed": record.get("epochs_reviewed", False),
                    "bad_epochs_count": record.get("bad_epochs_count", 0),
                    "bad_epoch_indices": record.get("bad_epoch_indices", ""),
                    "bad_epoch_times": record.get("bad_epoch_times", ""),
                    "bad_epoch_events": record.get("bad_epoch_events", ""),
                    "total_epochs": record.get("total_epochs", 0),
                    "epoch_rejection_rate": record.get("epoch_rejection_rate", 0.0),
                    "manual_bad_channels": ",".join(record.get("manual_bad_channels", [])),
                    "manual_rejected_ica": ",".join(
                        str(v) for v in record.get("manual_rejected_ica", [])
                    ),
                    "qa_export_hash": record.get("qa_export_hash", ""),
                    "qa_export_timestamp": record.get("qa_export_timestamp", ""),
                    "qa_export_path": record.get("qa_export_path", ""),
                    "reprocess_modified": record.get("reprocess_modified", False),
                    "reprocess_fix_type": record.get("reprocess_fix_type", ""),
                    "reprocess_timestamp": record.get("reprocess_timestamp", ""),
                    "modified_source": record.get("modified_source", "web"),
                }
            )


def _resolve_file(root: Path, file_key: str) -> tuple[Path, str]:
    rel_no_ext = Path(file_key)
    if rel_no_ext.is_absolute():
        raise HTTPException(status_code=400, detail="file_key must be relative")

    matches = sorted(root.rglob(rel_no_ext.name + ".set"))
    for path in matches:
        try:
            rel = path.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        candidate_key = str(rel.with_suffix(""))
        if candidate_key == file_key:
            return path.resolve(), str(rel)

    direct = (root / rel_no_ext).with_suffix(".set")
    if direct.exists():
        rel = direct.resolve().relative_to(root.resolve())
        return direct.resolve(), str(rel)
    raise HTTPException(status_code=404, detail=f"Exclude file not found: {file_key}")


def _related_paths(root: Path, file_path: Path) -> dict[str, Optional[Path]]:
    stem = _strip_suffixes(file_path.stem)
    task_root = root.parent
    candidates = {
        "run_report": task_root / "reports" / "run_reports" / f"{stem}_autoclean_report.pdf",
        "ica_report": task_root / "reports" / "ica_components" / f"{stem}_ica_components_all.pdf",
        "psd": task_root / "reports" / "psd_topo" / f"{stem}_psd_topo_figure.png",
        "metadata": task_root / "reports" / "run_reports" / f"{stem}_autoclean_metadata.json",
        "processing_log": task_root / "exports" / f"{stem}_processing_log.csv",
        "postedit": _postedit_path(task_root, file_path),
    }
    return {k: v if v.exists() else None for k, v in candidates.items()}


def _parse_metadata(path: Optional[Path]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "bad_channels": [],
        "channel_removals": [],
        "rejected_ica": [],
        "valid_channels": [],
        "max_components": 0,
    }
    if path is None:
        return result
    try:
        data = json.loads(path.read_text())
        metadata = data.get("metadata", {})
        removals = metadata.get("channel_removals", [])
        if isinstance(removals, list):
            result["channel_removals"] = removals
            result["bad_channels"] = [
                r.get("channel", "") for r in removals if isinstance(r, dict) and r.get("channel")
            ]
        legacy = metadata.get("step_clean_bad_channels", {}).get("bads", [])
        if not result["bad_channels"] and isinstance(legacy, list):
            result["bad_channels"] = legacy

        rejected = (
            metadata.get("step_apply_ica_component_rejection", {})
            .get("ica", {})
            .get("final_excluded_indices", [])
        )
        if isinstance(rejected, list):
            result["rejected_ica"] = [int(v) for v in rejected if isinstance(v, (int, float))]

        valid_channels = metadata.get("import_eeg", {}).get("originalChannelNames", [])
        if isinstance(valid_channels, list):
            result["valid_channels"] = [str(v) for v in valid_channels]

        ica_components = (
            metadata.get("step_run_ica", {}).get("ica", {}).get("ica_components")
        )
        if ica_components is not None:
            try:
                result["max_components"] = int(ica_components)
            except Exception:
                pass
    except Exception:
        return result
    return result


def _resolve_override_validation_context(
    *,
    file_path: Path,
    related: dict[str, Optional[Path]],
    metadata: dict[str, Any],
) -> tuple[list[str], int]:
    valid_channels = [str(v) for v in metadata.get("valid_channels", []) if str(v).strip()]
    max_components = int(metadata.get("max_components", 0) or 0)

    if not valid_channels:
        try:
            valid_channels = list(_load_epochs(file_path).ch_names)
        except Exception:
            valid_channels = []

    if max_components <= 0:
        ica_path = related.get("ica_report")
        if ica_path is not None and ica_path.exists():
            try:
                extracted = extract_ica_full(ica_path)
                components = extracted.get("components", [])
                if isinstance(components, list) and components:
                    max_components = len(components)
            except Exception:
                max_components = 0

    return valid_channels, max_components


def _read_processing_metrics(path: Optional[Path]) -> dict[str, Any]:
    import csv

    if path is None or not path.exists():
        return {}
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        return {}
    if not rows:
        return {}
    row = rows[-1]
    return {
        "data_retained": row.get("proc_xmax_post"),
        "channels_retained": row.get("net_nbchan_post"),
        "channels_original": row.get("net_nbchan_orig"),
        "epochs_kept": row.get("epoch_trials"),
        "epochs_rejected": row.get("epoch_badtrials"),
    }


def _record_for(decisions: dict[str, dict[str, Any]], key: str, relative_path: str) -> dict[str, Any]:
    record = decisions.setdefault(key, _default_record(relative_path))
    record["relative_path"] = relative_path
    if "modified_source" not in record:
        record["modified_source"] = "web"
    if "revision" not in record:
        record["revision"] = 0
    return record


def _touch_record(record: dict[str, Any], *, bump_revision: bool = True) -> None:
    record["last_updated"] = _human_timestamp()
    if bump_revision:
        record["revision"] = int(record.get("revision", 0) or 0) + 1


def _human_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _find_task_file(task_root: Path) -> Optional[Path]:
    status_dir = task_root / "status"
    if not status_dir.exists():
        return None
    py_files = sorted(status_dir.glob("*.py"))
    return py_files[0] if py_files else None


def _build_manual_fix_payload(
    *,
    file_key: str,
    root: Path,
    file_path: Path,
    metadata_path: Optional[Path],
    record: Optional[dict[str, Any]],
    manual_bad_channels: list[str],
    manual_rejected_ica: list[int],
) -> dict[str, Any]:
    task_root = root.parent
    stem = _strip_suffixes(file_path.stem)
    task_file = _find_task_file(task_root)
    task_file_hash = None
    task_file_relative = None
    if task_file and task_file.exists():
        task_file_relative = f"status/{task_file.name}"
        task_file_hash = __import__("hashlib").sha256(task_file.read_bytes()).hexdigest()

    ica_file_path = task_root / "ica" / f"{stem}-ica.fif"
    ica_file_relative = f"ica/{stem}-ica.fif" if ica_file_path.exists() else None

    fix_type = str((record or {}).get("reprocess_fix_type", "") or "none")

    payload = {
        "file_key": file_key,
        "file_stem": stem,
        "fix_type": fix_type,
        "timestamp": datetime.now().isoformat(),
        "modifications": {
            "epoch_review": {
                "count": int((record or {}).get("bad_epochs_count", 0) or 0),
                "indices": [
                    int(v)
                    for v in str((record or {}).get("bad_epoch_indices", "")).split(",")
                    if v.strip()
                ],
                "times": [
                    v
                    for v in str((record or {}).get("bad_epoch_times", "")).split(",")
                    if v
                ],
                "events": [
                    v
                    for v in str((record or {}).get("bad_epoch_events", "")).split(",")
                    if v
                ],
            },
            "bad_channels": {
                "original": [],
                "modified": manual_bad_channels,
                "added": manual_bad_channels,
                "removed": [],
            },
            "rejected_ica": {
                "original": [],
                "modified": manual_rejected_ica,
                "added": manual_rejected_ica,
                "removed": [],
            },
        },
        "ica_file_path": ica_file_relative,
        "metadata_source": (
            f"reports/run_reports/{metadata_path.name}" if metadata_path else None
        ),
        "task_file_path": task_file_relative,
        "task_file_hash": task_file_hash,
        "validation": {
            "can_apply_ica_fix": bool(manual_rejected_ica) and not bool(manual_bad_channels) and ica_file_path.exists(),
            "requires_full_reprocess": bool(manual_bad_channels),
            "ica_file_exists": ica_file_path.exists(),
            "task_file_exists": bool(task_file and task_file.exists()),
        },
    }
    return payload


def _manual_bad_epoch_indices(record: dict[str, Any]) -> list[int]:
    return [
        int(v)
        for v in str(record.get("bad_epoch_indices", "")).split(",")
        if str(v).strip()
    ]


def _normalize_manual_bad_channels(values: list[str]) -> list[str]:
    normalized: set[str] = set()
    for value in values:
        text = str(value).strip().upper()
        if not text:
            continue
        if text.isdigit():
            text = f"E{text}"
        normalized.add(text)
    return sorted(normalized)


def _resolve_override_fix_type(
    existing_bad_channels: list[str],
    existing_rejected_ica: list[int],
    next_bad_channels: list[str],
    next_rejected_ica: list[int],
) -> str:
    channel_changed = existing_bad_channels != next_bad_channels
    ica_changed = existing_rejected_ica != next_rejected_ica
    if channel_changed and ica_changed:
        raise HTTPException(
            status_code=400,
            detail="Channel and ICA overrides cannot both be changed in the same run.",
        )
    if channel_changed:
        return "channel"
    if ica_changed:
        return "ica"
    return ""


def _validate_override_values(
    *,
    valid_channels: list[str],
    max_components: int,
    manual_bad_channels: list[str],
    manual_rejected_ica: list[int],
) -> None:
    valid_channel_set = {str(value).upper() for value in valid_channels}
    invalid_channels = [channel for channel in manual_bad_channels if channel.upper() not in valid_channel_set]
    if invalid_channels:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid bad channel override(s): {', '.join(invalid_channels)}",
        )

    if max_components > 0:
        invalid_ica = [component for component in manual_rejected_ica if component < 0 or component >= max_components]
        if invalid_ica:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid ICA override(s): "
                    + ", ".join(f"IC {component}" for component in invalid_ica)
                    + f". Valid range is 0-{max_components - 1}."
                ),
            )


def _backup_existing_file(path: Path, backups_dir: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    backups_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backups_dir / path.name)


def _backup_and_remove_matching_files(base_dir: Path, patterns: list[str], backups_dir: Path) -> None:
    if not base_dir.exists():
        return
    seen: set[Path] = set()
    for pattern in patterns:
        for path in base_dir.glob(pattern):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            _backup_existing_file(path, backups_dir)
            path.unlink(missing_ok=True)


def _replace_existing_reprocess_artifacts(task_root: Path, stem: str, backups_dir: Path) -> None:
    _backup_and_remove_matching_files(
        task_root / "exports",
        [f"{stem}*.set", f"{stem}*.fdt", f"{stem}*_processing_log.csv"],
        backups_dir,
    )
    _backup_and_remove_matching_files(
        task_root / "reports" / "run_reports",
        [f"{stem}_autoclean_report.pdf", f"{stem}_autoclean_metadata.json"],
        backups_dir,
    )
    _backup_and_remove_matching_files(
        task_root / "reports" / "ica_components",
        [f"{stem}_ica_components_all.pdf"],
        backups_dir,
    )
    _backup_and_remove_matching_files(
        task_root / "reports" / "psd_topo",
        [f"{stem}_psd_topo_figure.png"],
        backups_dir,
    )
    _backup_and_remove_matching_files(
        task_root / "ica",
        [f"{stem}*"],
        backups_dir,
    )


def _drop_bad_epochs(epochs: mne.BaseEpochs, bad_indices: list[int]) -> None:
    if not bad_indices:
        return
    selection = epochs.selection.tolist() if hasattr(epochs.selection, "tolist") else list(epochs.selection)
    bad_selection_indices: list[int] = []
    for bad_num in bad_indices:
        if bad_num in selection:
            bad_selection_indices.append(selection.index(bad_num))
    if bad_selection_indices:
        epochs.drop(bad_selection_indices, reason="USER", verbose=False)


def _calculate_export_hash(file_key: str, record: dict[str, Any]) -> str:
    metadata = {
        "file_key": file_key,
        "bad_epoch_indices": record.get("bad_epoch_indices", ""),
        "total_epochs": record.get("total_epochs", 0),
    }
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()


def _preprocessing_log_path(task_root: Path) -> Path:
    return task_root / _PREPROCESSING_LOG_NAME


def _load_preprocessing_log_rows(task_root: Path) -> list[dict[str, str]]:
    path = _preprocessing_log_path(task_root)
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except Exception:
        return []


def _create_qa_preprocessing_log(task_root: Path, decisions: dict[str, dict[str, Any]]) -> Optional[Path]:
    rows = _load_preprocessing_log_rows(task_root)
    qa_dir = task_root / "qa"
    if not rows or not qa_dir.exists():
        return None

    manual_data: dict[str, dict[str, Any]] = {}
    for entry, record in decisions.items():
        if not isinstance(record, dict):
            continue
        subj_basename = _strip_suffixes(entry)
        manual_data[subj_basename] = {
            "qa_status": record.get("status", ""),
            "manual_bad_epochs": int(record.get("bad_epochs_count", 0) or 0),
            "manual_bad_epoch_indices": record.get("bad_epoch_indices", ""),
            "manual_review_timestamp": record.get("last_updated", ""),
            "manual_review_notes": record.get("notes", ""),
            "qa_exported": record.get("qa_export_timestamp", ""),
        }

    merged_rows: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get(_PREPROCESSING_KEY_COLUMN, ""))
        qa = manual_data.get(key, {})
        merged = dict(row)
        merged["qa_status"] = qa.get("qa_status", "")
        merged["manual_bad_epochs"] = qa.get("manual_bad_epochs", 0)
        merged["manual_bad_epoch_indices"] = qa.get("manual_bad_epoch_indices", "")
        merged["manual_review_timestamp"] = qa.get("manual_review_timestamp", "")
        merged["manual_review_notes"] = qa.get("manual_review_notes", "")
        merged["qa_exported"] = qa.get("qa_exported", "")

        if "epoch_badtrials" in merged:
            try:
                epoch_badtrials = float(merged.get("epoch_badtrials", 0) or 0)
                epoch_trials = float(merged.get("epoch_trials", 0) or 0)
                epoch_badtrials += float(qa.get("manual_bad_epochs", 0) or 0)
                merged["epoch_badtrials"] = epoch_badtrials
                if epoch_trials > 0:
                    merged["epoch_percent"] = (epoch_trials - epoch_badtrials) / epoch_trials
            except Exception:
                pass
        merged_rows.append(merged)

    fieldnames: list[str] = []
    for row in merged_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    qa_log_path = qa_dir / "qa_preprocessing_log.csv"
    with qa_log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)
    return qa_log_path


def _export_file_to_qa(
    *,
    root: Path,
    task_root: Path,
    file_key: str,
    file_path: Path,
    record: dict[str, Any],
) -> dict[str, Any]:
    qa_dir = task_root / "qa"
    qa_dir.mkdir(exist_ok=True)
    current_hash = _calculate_export_hash(file_key, record)
    existing_hash = str(record.get("qa_export_hash", ""))
    if existing_hash == current_hash and str(record.get("qa_export_path", "")):
        return {"exported": False, "skipped": True, "path": record.get("qa_export_path", "")}

    epochs = _load_epochs(file_path)
    _drop_bad_epochs(epochs, _manual_bad_epoch_indices(record))

    dest_file = qa_dir / file_path.name
    epochs.export(str(dest_file), fmt="eeglab", overwrite=True)
    record["qa_export_hash"] = current_hash
    record["qa_export_timestamp"] = datetime.now().isoformat()
    record["qa_export_path"] = f"qa/{file_path.name}"
    return {"exported": True, "skipped": False, "path": record["qa_export_path"]}


def _postedit_path(task_root: Path, file_path: Path) -> Path:
    stem = _strip_suffixes(file_path.stem)
    return task_root / "postedit" / f"{stem}_postedit.set"


def _sync_postedit_export(task_root: Path, file_path: Path, record: dict[str, Any]) -> Optional[str]:
    postedit_path = _postedit_path(task_root, file_path)
    bad_indices = _manual_bad_epoch_indices(record)
    if not bad_indices:
        if postedit_path.exists():
            postedit_path.unlink()
        return None

    epochs = _load_epochs(file_path)
    _drop_bad_epochs(epochs, bad_indices)
    postedit_path.parent.mkdir(parents=True, exist_ok=True)
    epochs.export(str(postedit_path), fmt="eeglab", overwrite=True)
    return str(postedit_path)


def _inject_reprocess_metadata(task_root: Path, stem: str, timestamp: str, payload: dict[str, Any]) -> None:
    metadata_json_path = task_root / "reports" / "run_reports" / f"{stem}_autoclean_metadata.json"
    if metadata_json_path.exists():
        metadata = json.loads(metadata_json_path.read_text(encoding="utf-8"))
        metadata["reprocessed"] = True
        metadata["reprocessed_timestamp"] = datetime.now().isoformat()
        metadata["reprocess_reason"] = f"manual_override_{payload.get('fix_type', 'unknown')}"
        metadata["manual_overrides"] = payload.get("modifications", {})
        metadata["original_backup"] = f"exports/backups/{stem}_{timestamp}/"
        metadata_json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    processing_log_path = task_root / "exports" / f"{stem}_processing_log.csv"
    if not processing_log_path.exists():
        return

    with processing_log_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        return

    if "reprocessed_flag" not in fieldnames:
        fieldnames.append("reprocessed_flag")
    rows[-1]["reprocessed_flag"] = f"REPROCESSED_{timestamp}"
    with processing_log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_manual_fix_payload(task_root: Path, stem: str, payload: dict[str, Any]) -> Path:
    payload_dir = task_root / "qa" / "manual_fixes"
    payload_dir.mkdir(parents=True, exist_ok=True)
    payload_path = payload_dir / f"{stem}_manual_fix.json"
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload_path


def _generate_reprocess_task_from_original(
    original_task_path: Path,
    payload: dict[str, Any],
    new_class_name: str,
    timestamp: str,
) -> str:
    original_source = original_task_path.read_text(encoding="utf-8")
    tree = ast.parse(original_source)
    bad_channels = payload["modifications"]["bad_channels"]["modified"]
    rejected_ica = payload["modifications"]["rejected_ica"]["modified"]
    file_stem = payload.get("file_stem", "unknown")
    dataset_name = _reprocess_dataset_name(file_stem, timestamp)

    class ConfigModifier(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign):  # type: ignore[override]
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "config"
                and isinstance(node.value, ast.Dict)
            ):
                node.value.keys.append(ast.Constant(value="dataset_name"))
                node.value.values.append(ast.Constant(value=dataset_name))
            return node

    class ClassRenamer(ast.NodeTransformer):
        def visit_ClassDef(self, node: ast.ClassDef):  # type: ignore[override]
            node.name = new_class_name
            return node

    class MethodModifier(ast.NodeTransformer):
        def __init__(self) -> None:
            self.in_run = False

        def visit_FunctionDef(self, node: ast.FunctionDef):  # type: ignore[override]
            if node.name == "run":
                self.in_run = True
                node.body = [self.visit(stmt) for stmt in node.body]  # type: ignore[assignment]
                self.in_run = False
            return node

        def visit_Call(self, node: ast.Call):  # type: ignore[override]
            if not self.in_run:
                return node
            if isinstance(node.func, ast.Attribute) and node.func.attr == "clean_bad_channels" and bad_channels:
                node.keywords.append(
                    ast.keyword(
                        arg="manual_bad_channels",
                        value=ast.List(elts=[ast.Constant(value=ch) for ch in bad_channels], ctx=ast.Load()),
                    )
                )
            if isinstance(node.func, ast.Attribute) and node.func.attr == "apply_ica_component_rejection" and rejected_ica:
                found = False
                for kw in node.keywords:
                    if kw.arg == "manual_rejected_components":
                        kw.value = ast.List(elts=[ast.Constant(value=v) for v in rejected_ica], ctx=ast.Load())
                        found = True
                        break
                if not found:
                    node.keywords.append(
                        ast.keyword(
                            arg="manual_rejected_components",
                            value=ast.List(elts=[ast.Constant(value=v) for v in rejected_ica], ctx=ast.Load()),
                        )
                    )
            return node

    tree = ConfigModifier().visit(tree)
    tree = ClassRenamer().visit(tree)
    tree = MethodModifier().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def _reprocess_dataset_name(file_stem: str, timestamp: str) -> str:
    return f"{file_stem}_{timestamp}"


def _finalize_reprocess_job(job_id: str) -> None:
    job = _REPROCESS_JOBS.get(job_id)
    if not job or job.get("finalized"):
        return
    proc: subprocess.Popen[str] = job["process"]
    exit_code = proc.poll()
    if exit_code is None:
        return

    job["finalized"] = True
    job["exit_code"] = exit_code
    if exit_code != 0:
        job["status"] = "failed"
        return

    try:
        reprocess_folder = Path(job["reprocess_folder"])
        task_root = Path(job["task_root"])
        root = Path(job["exports_root"])
        exports_dir = task_root / "exports"
        reports_dir = task_root / "reports"
        ica_dir = task_root / "ica"
        backups_dir = exports_dir / "backups" / f"{job['stem']}_{job['timestamp']}"
        backups_dir.mkdir(parents=True, exist_ok=True)
        _replace_existing_reprocess_artifacts(task_root, str(job["stem"]), backups_dir)

        reprocess_exports = reprocess_folder / "exports"
        if reprocess_exports.exists():
            for file_path in reprocess_exports.glob("*"):
                if file_path.is_file():
                    dest = exports_dir / file_path.name
                    shutil.copy2(file_path, dest)

        reprocess_reports = reprocess_folder / "reports"
        if reprocess_reports.exists():
            for item in reprocess_reports.rglob("*"):
                if item.is_file():
                    rel = item.relative_to(reprocess_reports)
                    dest = reports_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest)

        reprocess_ica = reprocess_folder / "ica"
        if reprocess_ica.exists():
            ica_dir.mkdir(parents=True, exist_ok=True)
            for item in reprocess_ica.glob("*"):
                if item.is_file():
                    dest = ica_dir / item.name
                    shutil.copy2(item, dest)

        decisions = _load_decisions(root)
        record = _record_for(decisions, str(job["file_key"]), str(job["relative_path"]))
        record["reprocess_modified"] = False
        record["reprocess_timestamp"] = _human_timestamp()
        payload: dict[str, Any] = {}
        payload_path = Path(str(job["payload_path"]))
        if payload_path.exists():
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        _inject_reprocess_metadata(task_root, str(job["stem"]), str(job["timestamp"]), payload)
        try:
            file_path, _relative_path = _resolve_file(root, str(job["file_key"]))
            _sync_postedit_export(task_root, file_path, record)
        except Exception:
            pass
        _touch_record(record)
        _save_decisions(root, decisions)
        job["status"] = "completed"
        job["message"] = "Reprocess completed and outputs copied into the original task folder."
    except Exception as exc:
        job["status"] = "failed"
        job["message"] = f"Reprocess completed but finalization failed: {exc}"


class ExcludeRootResponse(BaseModel):
    exports_root: str


class ExcludeFileSummary(BaseModel):
    file_key: str
    relative_path: str
    name: str
    notes_present: bool = False
    epochs_reviewed: bool = False
    bad_epochs_count: int = 0
    has_overrides: bool = False
    status: str = "UNSET"


class ExcludeFilesResponse(BaseModel):
    exports_root: str
    files: list[ExcludeFileSummary]


class ExcludeFileDetail(BaseModel):
    file_key: str
    relative_path: str
    name: str
    exports_root: str
    status: str
    notes: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    baseline_bad_channels: list[str] = Field(default_factory=list)
    baseline_rejected_ica: list[int] = Field(default_factory=list)
    valid_channels: list[str] = Field(default_factory=list)
    max_components: int = 0
    manual_bad_channels: list[str] = Field(default_factory=list)
    manual_rejected_ica: list[int] = Field(default_factory=list)
    epoch_review: dict[str, Any] = Field(default_factory=dict)
    qa_export: dict[str, Any] = Field(default_factory=dict)
    reprocess: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Optional[str]] = Field(default_factory=dict)


class EpochManifest(BaseModel):
    file_key: str
    relative_path: str
    mode: str = "epochs"
    sampling_rate: float
    channel_names: list[str]
    n_channels: int
    n_epochs: int
    epoch_length_samples: int
    epoch_duration_seconds: float
    existing_bad_epoch_indices: list[int]
    default_scaling_uv: float = 25.0
    visible_epoch_count: int = 10


class EpochWindowResponse(BaseModel):
    file_key: str
    start_epoch: int
    count: int
    channel_names: list[str]
    sampling_rate: float
    epoch_duration_seconds: float
    epochs: list[dict[str, Any]]


class ReprocessRequest(BaseModel):
    manual_bad_channels: list[str] = Field(default_factory=list)
    manual_rejected_ica: list[int] = Field(default_factory=list)


class ReprocessResponse(BaseModel):
    job_id: str
    status: str
    message: str


class QaExportRequest(BaseModel):
    file_keys: list[str] = Field(default_factory=list)


class EpochReviewUpdate(BaseModel):
    bad_epoch_indices: list[int]


class NotesUpdate(BaseModel):
    notes: str
    status: Optional[str] = None


class OverridesUpdate(BaseModel):
    manual_bad_channels: list[str] = Field(default_factory=list)
    manual_rejected_ica: list[int] = Field(default_factory=list)


@router.get("/root", response_model=ExcludeRootResponse)
async def get_exclude_root() -> ExcludeRootResponse:
    root = _resolve_exports_root()
    return ExcludeRootResponse(exports_root=str(root))


@router.get("/files", response_model=ExcludeFilesResponse)
async def list_exclude_files() -> ExcludeFilesResponse:
    root = _resolve_exports_root()
    decisions = _load_decisions(root)
    files: list[ExcludeFileSummary] = []
    for path in sorted(root.rglob("*.set")):
        rel = str(path.resolve().relative_to(root.resolve()))
        key = str(Path(rel).with_suffix(""))
        record = decisions.get(key, {})
        files.append(
            ExcludeFileSummary(
                file_key=key,
                relative_path=rel,
                name=path.name,
                notes_present=bool(record.get("notes")),
                epochs_reviewed=bool(record.get("epochs_reviewed")),
                bad_epochs_count=int(record.get("bad_epochs_count", 0) or 0),
                has_overrides=bool(record.get("manual_bad_channels") or record.get("manual_rejected_ica")),
                status=str(record.get("status", "UNSET")),
            )
        )
    return ExcludeFilesResponse(exports_root=str(root), files=files)


@router.get("/files/{file_key:path}/artifacts/{asset_name}")
async def get_exclude_artifact(file_key: str, asset_name: str):
    root = _resolve_exports_root()
    file_path, _relative_path = _resolve_file(root, file_key)
    related = _related_paths(root, file_path)
    path = related.get(asset_name)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {asset_name}")
    return FileResponse(path)


@router.get("/files/{file_key:path}/ica-summary")
async def get_exclude_ica_summary(file_key: str) -> dict[str, Any]:
    root = _resolve_exports_root()
    file_path, _relative_path = _resolve_file(root, file_key)
    related = _related_paths(root, file_path)
    ica_path = related.get("ica_report")
    if ica_path is None or not ica_path.exists():
        raise HTTPException(status_code=404, detail="ICA report not found")
    extracted = extract_ica_full(ica_path)
    return extracted


def _load_epochs(file_path: Path) -> mne.BaseEpochs:
    try:
        return mne.read_epochs_eeglab(str(file_path), verbose="ERROR")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not load epochs: {exc}")


@router.get("/files/{file_key:path}/eeg/manifest", response_model=EpochManifest)
async def get_eeg_manifest(file_key: str) -> EpochManifest:
    root = _resolve_exports_root()
    file_path, relative_path = _resolve_file(root, file_key)
    decisions = _load_decisions(root)
    record = _record_for(decisions, file_key, relative_path)
    epochs = _load_epochs(file_path)
    existing_bad = [int(v) for v in str(record.get("bad_epoch_indices", "")).split(",") if v.strip()]
    return EpochManifest(
        file_key=file_key,
        relative_path=relative_path,
        sampling_rate=float(epochs.info["sfreq"]),
        channel_names=list(epochs.ch_names),
        n_channels=len(epochs.ch_names),
        n_epochs=len(epochs),
        epoch_length_samples=int(len(epochs.times)),
        epoch_duration_seconds=float(epochs.times[-1] - epochs.times[0]) if len(epochs.times) > 1 else 0.0,
        existing_bad_epoch_indices=existing_bad,
    )


@router.get("/files/{file_key:path}/eeg/epochs", response_model=EpochWindowResponse)
async def get_eeg_epochs(
    file_key: str,
    start: int = Query(default=0, ge=0),
    count: int = Query(default=10, ge=1),
    channels: Optional[str] = None,
) -> EpochWindowResponse:
    root = _resolve_exports_root()
    file_path, relative_path = _resolve_file(root, file_key)
    decisions = _load_decisions(root)
    record = _record_for(decisions, file_key, relative_path)
    epochs = _load_epochs(file_path)
    selected_channels = list(epochs.ch_names)
    if channels:
        requested = [c.strip() for c in channels.split(",") if c.strip()]
        selected_channels = [c for c in requested if c in epochs.ch_names]
        if not selected_channels:
            selected_channels = list(epochs.ch_names[: min(8, len(epochs.ch_names))])
    bad_epoch_indices = {
        int(v) for v in str(record.get("bad_epoch_indices", "")).split(",") if v.strip()
    }
    data = epochs.get_data(picks=selected_channels)
    sfreq = float(epochs.info["sfreq"])
    end = min(start + count, len(epochs))
    payload: list[dict[str, Any]] = []
    for idx in range(start, end):
        traces_uv = {}
        for ch_idx, ch_name in enumerate(selected_channels):
            samples = data[idx, ch_idx, :]
            step = max(1, len(samples) // 250)
            traces_uv[ch_name] = [round(float(v) * 1e6, 3) for v in samples[::step]]
        event_code: Optional[str] = None
        start_time = 0.0
        if hasattr(epochs, "events") and idx < len(epochs.events):
            start_time = float(epochs.events[idx, 0] / sfreq)
            event_code = str(epochs.events[idx, 2])
        payload.append(
            {
                "epoch_index": idx,
                "event_code": event_code,
                "start_time_seconds": round(start_time, 3),
                "is_bad": idx in bad_epoch_indices,
                "traces_uv": traces_uv,
            }
        )
    return EpochWindowResponse(
        file_key=file_key,
        start_epoch=start,
        count=len(payload),
        channel_names=selected_channels,
        sampling_rate=sfreq,
        epoch_duration_seconds=float(epochs.times[-1] - epochs.times[0]) if len(epochs.times) > 1 else 0.0,
        epochs=payload,
    )


@router.get("/files/{file_key:path}/epoch-review")
async def get_epoch_review(file_key: str) -> dict[str, Any]:
    root = _resolve_exports_root()
    _, relative_path = _resolve_file(root, file_key)
    decisions = _load_decisions(root)
    record = _record_for(decisions, file_key, relative_path)
    return {
        "bad_epoch_indices": [int(v) for v in str(record.get("bad_epoch_indices", "")).split(",") if v.strip()],
        "bad_epochs_count": int(record.get("bad_epochs_count", 0) or 0),
        "total_epochs": int(record.get("total_epochs", 0) or 0),
        "epoch_rejection_rate": float(record.get("epoch_rejection_rate", 0.0) or 0.0),
        "last_updated": record.get("last_updated", ""),
        "revision": int(record.get("revision", 0) or 0),
    }


@router.put("/files/{file_key:path}/epoch-review")
async def save_epoch_review(file_key: str, body: EpochReviewUpdate) -> dict[str, Any]:
    root = _resolve_exports_root()
    file_path, relative_path = _resolve_file(root, file_key)
    task_root = root.parent
    decisions = _load_decisions(root)
    record = _record_for(decisions, file_key, relative_path)
    epochs = _load_epochs(file_path)

    bad_epochs = sorted(set(int(v) for v in body.bad_epoch_indices if 0 <= int(v) < len(epochs)))
    epoch_times: list[str] = []
    epoch_events: list[str] = []
    if hasattr(epochs, "events"):
        sfreq = float(epochs.info["sfreq"])
        for idx in bad_epochs:
            if idx < len(epochs.events):
                epoch_times.append(f"{epochs.events[idx, 0] / sfreq:.3f}")
                epoch_events.append(str(epochs.events[idx, 2]))

    total_epochs = len(epochs)
    record["epochs_reviewed"] = True
    record["bad_epochs_count"] = len(bad_epochs)
    record["bad_epoch_indices"] = ",".join(str(v) for v in bad_epochs)
    record["bad_epoch_times"] = ",".join(epoch_times)
    record["bad_epoch_events"] = ",".join(epoch_events)
    record["total_epochs"] = total_epochs
    record["epoch_rejection_rate"] = (len(bad_epochs) / total_epochs * 100.0) if total_epochs else 0.0
    _touch_record(record)
    _save_decisions(root, decisions)
    warning: Optional[str] = None
    try:
        _sync_postedit_export(task_root, file_path, record)
    except Exception as exc:
        warning = f"Saved review state but could not refresh postedit export: {exc}"
    return {
        "saved": True,
        "bad_epochs_count": record["bad_epochs_count"],
        "total_epochs": record["total_epochs"],
        "epoch_rejection_rate": record["epoch_rejection_rate"],
        "last_updated": record["last_updated"],
        "server_revision": record["revision"],
        "warning": warning,
    }


@router.get("/files/{file_key:path}", response_model=ExcludeFileDetail)
async def get_exclude_file_detail(file_key: str) -> ExcludeFileDetail:
    root = _resolve_exports_root()
    file_path, relative_path = _resolve_file(root, file_key)
    decisions = _load_decisions(root)
    record = _record_for(decisions, file_key, relative_path)
    related = _related_paths(root, file_path)
    metadata = _parse_metadata(related["metadata"])
    metrics = _read_processing_metrics(related["processing_log"])
    return ExcludeFileDetail(
        file_key=file_key,
        relative_path=relative_path,
        name=file_path.name,
        exports_root=str(root),
        status=str(record.get("status", "UNSET")),
        notes=str(record.get("notes", "")),
        metrics=metrics,
        baseline_bad_channels=[str(v) for v in metadata.get("bad_channels", [])],
        baseline_rejected_ica=[int(v) for v in metadata.get("rejected_ica", [])],
        valid_channels=[str(v) for v in metadata.get("valid_channels", [])],
        max_components=int(metadata.get("max_components", 0) or 0),
        manual_bad_channels=[str(v) for v in record.get("manual_bad_channels", [])],
        manual_rejected_ica=[int(v) for v in record.get("manual_rejected_ica", [])],
        epoch_review={
            "epochs_reviewed": bool(record.get("epochs_reviewed")),
            "bad_epochs_count": int(record.get("bad_epochs_count", 0) or 0),
            "bad_epoch_indices": [
                int(v) for v in str(record.get("bad_epoch_indices", "")).split(",") if v.strip()
            ],
            "bad_epoch_times": [v for v in str(record.get("bad_epoch_times", "")).split(",") if v],
            "bad_epoch_events": [v for v in str(record.get("bad_epoch_events", "")).split(",") if v],
            "total_epochs": int(record.get("total_epochs", 0) or 0),
            "epoch_rejection_rate": float(record.get("epoch_rejection_rate", 0.0) or 0.0),
        },
        qa_export={
            "hash": str(record.get("qa_export_hash", "")),
            "timestamp": str(record.get("qa_export_timestamp", "")),
            "path": str(record.get("qa_export_path", "")),
        },
        reprocess={
            "modified": bool(record.get("reprocess_modified", False)),
            "fix_type": str(record.get("reprocess_fix_type", "")),
            "timestamp": str(record.get("reprocess_timestamp", "")),
        },
        artifacts={
            "run_report": f"/api/exclude/files/{file_key}/artifacts/run_report" if related["run_report"] else None,
            "ica_report": f"/api/exclude/files/{file_key}/artifacts/ica_report" if related["ica_report"] else None,
            "psd": f"/api/exclude/files/{file_key}/artifacts/psd" if related["psd"] else None,
            "metadata": f"/api/exclude/files/{file_key}/artifacts/metadata" if related["metadata"] else None,
            "postedit": f"/api/exclude/files/{file_key}/artifacts/postedit" if related["postedit"] else None,
        },
    )


@router.put("/files/{file_key:path}/notes")
async def save_notes(file_key: str, body: NotesUpdate) -> dict[str, Any]:
    from datetime import datetime

    root = _resolve_exports_root()
    _, relative_path = _resolve_file(root, file_key)
    decisions = _load_decisions(root)
    record = _record_for(decisions, file_key, relative_path)
    record["notes"] = body.notes.strip()
    if body.status:
        record["status"] = body.status
    _touch_record(record)
    _save_decisions(root, decisions)
    return {"saved": True, "last_updated": record["last_updated"], "server_revision": record["revision"]}


@router.put("/files/{file_key:path}/overrides")
async def save_overrides(file_key: str, body: OverridesUpdate) -> dict[str, Any]:
    from datetime import datetime

    root = _resolve_exports_root()
    file_path, relative_path = _resolve_file(root, file_key)
    decisions = _load_decisions(root)
    record = _record_for(decisions, file_key, relative_path)
    related = _related_paths(root, file_path)
    metadata = _parse_metadata(related["metadata"])
    valid_channels, max_components = _resolve_override_validation_context(
        file_path=file_path,
        related=related,
        metadata=metadata,
    )
    existing_bad_channels = [str(v) for v in record.get("manual_bad_channels", [])]
    existing_rejected_ica = [int(v) for v in record.get("manual_rejected_ica", [])]
    next_bad_channels = _normalize_manual_bad_channels(body.manual_bad_channels)
    next_rejected_ica = sorted({int(v) for v in body.manual_rejected_ica})
    _validate_override_values(
        valid_channels=valid_channels,
        max_components=max_components,
        manual_bad_channels=next_bad_channels,
        manual_rejected_ica=next_rejected_ica,
    )
    fix_type = _resolve_override_fix_type(
        existing_bad_channels,
        existing_rejected_ica,
        next_bad_channels,
        next_rejected_ica,
    )
    record["manual_bad_channels"] = next_bad_channels
    record["manual_rejected_ica"] = next_rejected_ica
    record["reprocess_modified"] = bool(fix_type)
    record["reprocess_fix_type"] = fix_type
    record["reprocess_timestamp"] = _human_timestamp() if fix_type else ""
    record["modified_source"] = "web"
    _touch_record(record)
    _save_decisions(root, decisions)
    return {
        "saved": True,
        "last_updated": record["last_updated"],
        "reprocess_modified": record["reprocess_modified"],
        "reprocess_fix_type": record["reprocess_fix_type"],
        "reprocess_timestamp": record["reprocess_timestamp"],
        "server_revision": record["revision"],
    }


@router.post("/files/{file_key:path}/reprocess", response_model=ReprocessResponse)
async def start_reprocess(file_key: str, body: ReprocessRequest) -> ReprocessResponse:
    root = _resolve_exports_root()
    file_path, relative_path = _resolve_file(root, file_key)
    task_root = root.parent
    if "reprocess" in task_root.parts:
        raise HTTPException(status_code=400, detail="Cannot reprocess from inside a reprocess folder")

    related = _related_paths(root, file_path)
    metadata_path = related.get("metadata")
    if metadata_path is None or not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Metadata file not found for selected export")

    metadata_json = json.loads(metadata_path.read_text())
    raw_file = metadata_json.get("unprocessed_file")
    if not raw_file:
        raise HTTPException(status_code=400, detail="Original raw file path missing from metadata")
    raw_path = Path(raw_file)
    if not raw_path.exists():
        raise HTTPException(status_code=404, detail=f"Original raw file not found: {raw_path}")

    task_file = _find_task_file(task_root)
    if task_file is None or not task_file.exists():
        raise HTTPException(status_code=404, detail="Original task file not found in status/")

    metadata = _parse_metadata(metadata_path)
    valid_channels, max_components = _resolve_override_validation_context(
        file_path=file_path,
        related=related,
        metadata=metadata,
    )
    manual_bad_channels = _normalize_manual_bad_channels(body.manual_bad_channels)
    manual_rejected_ica = sorted({int(v) for v in body.manual_rejected_ica})
    _validate_override_values(
        valid_channels=valid_channels,
        max_components=max_components,
        manual_bad_channels=manual_bad_channels,
        manual_rejected_ica=manual_rejected_ica,
    )
    decisions = _load_decisions(root)
    record = _record_for(decisions, file_key, relative_path)
    existing_bad_channels = [str(v) for v in record.get("manual_bad_channels", [])]
    existing_rejected_ica = [int(v) for v in record.get("manual_rejected_ica", [])]
    payload_fix_type = _resolve_override_fix_type(
        existing_bad_channels,
        existing_rejected_ica,
        manual_bad_channels,
        manual_rejected_ica,
    )
    record["manual_bad_channels"] = manual_bad_channels
    record["manual_rejected_ica"] = manual_rejected_ica
    record["reprocess_modified"] = True
    record["reprocess_fix_type"] = payload_fix_type
    record["reprocess_timestamp"] = _human_timestamp() if payload_fix_type else ""
    record["modified_source"] = "web"
    payload = _build_manual_fix_payload(
        file_key=file_key,
        root=root,
        file_path=file_path,
        metadata_path=metadata_path,
        record=record,
        manual_bad_channels=manual_bad_channels,
        manual_rejected_ica=manual_rejected_ica,
    )
    stem = _strip_suffixes(file_path.stem)
    payload_path = _save_manual_fix_payload(task_root, stem, payload)
    _touch_record(record)
    _save_decisions(root, decisions)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized = stem.replace("-", "_").replace(" ", "_")
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized)
    class_name = f"Task_{sanitized}_Reprocess" if sanitized and sanitized[0].isdigit() else f"{sanitized}_Reprocess"
    task_output_path = task_root / "status" / f"{stem}_Reprocess.py"
    rendered_task = _generate_reprocess_task_from_original(task_file, payload, class_name, timestamp)
    task_output_path.write_text(rendered_task, encoding="utf-8")

    reprocess_dir = task_root / "reprocess"
    reprocess_dir.mkdir(exist_ok=True)
    reprocess_folder = reprocess_dir / _reprocess_dataset_name(stem, timestamp)
    cmd = [
        sys.executable,
        "-m",
        "autoclean",
        "process",
        "--task-file",
        str(task_output_path),
        "--file",
        str(raw_path),
        "--output",
        str(reprocess_dir),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    job_id = uuid.uuid4().hex
    _REPROCESS_JOBS[job_id] = {
        "job_id": job_id,
        "status": "running",
        "message": f"Started reprocess for {stem}",
        "process": proc,
        "task_root": str(task_root),
        "reprocess_folder": str(reprocess_folder),
        "timestamp": timestamp,
        "stem": stem,
        "payload_path": str(payload_path),
        "task_output_path": str(task_output_path),
        "file_key": file_key,
        "relative_path": relative_path,
        "exports_root": str(root),
    }
    return ReprocessResponse(job_id=job_id, status="running", message=f"Started reprocess for {stem}")


@router.get("/reprocess/{job_id}")
async def get_reprocess_status(job_id: str) -> dict[str, Any]:
    if job_id not in _REPROCESS_JOBS:
        raise HTTPException(status_code=404, detail="Unknown reprocess job")
    _finalize_reprocess_job(job_id)
    job = _REPROCESS_JOBS[job_id]
    proc: subprocess.Popen[str] = job["process"]
    return {
        "job_id": job_id,
        "status": job["status"],
        "message": job.get("message", ""),
        "running": proc.poll() is None,
        "exit_code": job.get("exit_code"),
        "file_key": job.get("file_key"),
    }


@router.post("/qa/export")
async def export_to_qa(body: QaExportRequest) -> dict[str, Any]:
    root = _resolve_exports_root()
    task_root = root.parent
    decisions = _load_decisions(root)

    selected_keys = body.file_keys or [
        key
        for key, record in decisions.items()
        if isinstance(record, dict) and int(record.get("bad_epochs_count", 0) or 0) > 0
    ]

    exported = 0
    skipped = 0
    errors: list[dict[str, str]] = []
    for file_key in selected_keys:
        try:
            file_path, relative_path = _resolve_file(root, file_key)
            record = _record_for(decisions, file_key, relative_path)
            result = _export_file_to_qa(
                root=root,
                task_root=task_root,
                file_key=file_key,
                file_path=file_path,
                record=record,
            )
            if result["skipped"]:
                skipped += 1
            else:
                exported += 1
        except Exception as exc:
            errors.append({"file_key": file_key, "error": str(exc)})

    _save_decisions(root, decisions)
    qa_log_path = _create_qa_preprocessing_log(task_root, decisions)
    return {
        "exported": exported,
        "skipped": skipped,
        "errors": errors,
        "qa_log_path": str(qa_log_path) if qa_log_path else None,
    }


@router.get("/qa/preprocessing-log")
async def get_qa_preprocessing_log():
    root = _resolve_exports_root()
    qa_log_path = root.parent / "qa" / "qa_preprocessing_log.csv"
    if not qa_log_path.exists():
        raise HTTPException(status_code=404, detail="QA preprocessing log not found")
    return FileResponse(qa_log_path)
