"""Montage preflight scanning, planning, and apply helpers."""

from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import mne

from autoclean.io.import_ import discover_plugins, get_format_from_extension
from autoclean.utils.task_montage import (
    read_task_montage,
    replace_task_class_name,
    update_task_montage_source,
)

SUPPORTED_HYDROCEL_MONTAGES = {"GSN-HydroCel-128", "GSN-HydroCel-129"}


@dataclass(frozen=True)
class MontagePreflightFileResult:
    """Preflight result for one input path."""

    path: str
    relative_path: str
    format_id: str | None
    expected_montage: str | None
    detected_montage: str | None
    status: str
    eeg_channel_count: int | None = None
    e129_present: bool = False
    reason: str = ""
    size_bytes: int = 0

    @property
    def is_actionable(self) -> bool:
        """Return True when this file can be routed automatically."""

        return self.detected_montage in SUPPORTED_HYDROCEL_MONTAGES


@dataclass(frozen=True)
class MontagePreflightGroup:
    """Grouped preflight results by status and detected montage."""

    detected_montage: str
    status: str
    file_count: int
    total_size_bytes: int
    examples: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MontageBatchPlan:
    """Dry-run batch plan emitted by montage preflight."""

    input_path: str
    task_path: str
    expected_montage: str | None
    output_dir: str
    groups: list[MontagePreflightGroup]
    files: list[MontagePreflightFileResult]
    unknown_files: list[str]
    actionable_files: list[str]

    def to_json_dict(self) -> dict:
        """Return a JSON-serializable representation."""

        return {
            "input_path": self.input_path,
            "task_path": self.task_path,
            "expected_montage": self.expected_montage,
            "output_dir": self.output_dir,
            "groups": [asdict(group) for group in self.groups],
            "files": [asdict(result) for result in self.files],
            "unknown_files": self.unknown_files,
            "actionable_files": self.actionable_files,
        }


@dataclass(frozen=True)
class MontageCopyResult:
    """Summary of a copy-originals apply step."""

    split_output_root: str
    copied_files: list[dict]
    skipped_files: list[str]
    required_bytes: int
    free_bytes_before: int
    free_bytes_after_estimate: int
    completed: bool = True
    errors: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class MontageCopyEstimate:
    """Pre-confirmation size and free-space estimate for copy-originals."""

    split_output_root: str
    actionable_file_count: int
    skipped_file_count: int
    required_bytes: int
    free_bytes_before: int
    free_bytes_after_estimate: int


class MontageCopyError(RuntimeError):
    """Raised when copy-originals fails after recording partial progress."""

    def __init__(self, message: str, partial_result: MontageCopyResult) -> None:
        super().__init__(message)
        self.partial_result = partial_result


@dataclass(frozen=True)
class MontageMoveResult:
    """Summary of a move-originals apply step."""

    split_output_root: str
    planned_manifest: str
    moved_files: list[dict]
    skipped_files: list[str]
    required_bytes: int
    free_bytes_before: int
    free_bytes_after_estimate: int
    same_volume: bool
    source_volume: str
    destination_volume: str
    completed: bool = True
    errors: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class MontageMoveEstimate:
    """Pre-confirmation size, volume, and free-space estimate for move-originals."""

    source_path: str
    split_output_root: str
    actionable_file_count: int
    unknown_file_count: int
    required_bytes: int
    free_bytes_before: int
    free_bytes_after_estimate: int
    same_volume: bool
    source_volume: str
    destination_volume: str


class MontageMoveError(RuntimeError):
    """Raised when move-originals fails after recording partial progress."""

    def __init__(self, message: str, partial_result: MontageMoveResult) -> None:
        super().__init__(message)
        self.partial_result = partial_result


@dataclass(frozen=True)
class MontageTaskCloneResult:
    """Summary of one cloned task file."""

    source_task: str
    cloned_task: str
    source_montage: str | None
    cloned_montage: str
    class_name: str


def discover_eeg_inputs(input_path: Path) -> list[Path]:
    """Return supported file inputs under a file or directory."""

    if input_path.is_file():
        return [input_path]

    if input_path.is_dir() and input_path.suffix.lower() == ".mff":
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    inputs: list[Path] = []
    pending = [input_path]
    while pending:
        current = pending.pop()
        children = sorted(current.iterdir())
        for child in children:
            if child.is_dir() and child.suffix.lower() == ".mff":
                inputs.append(child)
            elif child.is_dir():
                pending.append(child)
            elif child.is_file() and get_format_from_extension(child.suffix):
                inputs.append(child)

    return sorted(inputs)


def detect_hydrocel_montage(raw) -> tuple[str | None, int | None, bool, str]:
    """Classify HydroCel 128/129 layouts without changing channels."""

    ch_names = list(getattr(raw, "ch_names", []))
    try:
        ch_types = raw.get_channel_types()
    except Exception:
        ch_types = [
            "eeg" if re.fullmatch(r"E\d+", name) else "misc" for name in ch_names
        ]

    eeg_names = [
        name
        for name, ch_type in zip(ch_names, ch_types, strict=False)
        if ch_type == "eeg"
    ]
    if not eeg_names and ch_names:
        eeg_names = [name for name in ch_names if re.fullmatch(r"E\d+", name)]

    eeg_count = len(eeg_names)
    e129_present = "E129" in eeg_names

    if eeg_count == 128 and not e129_present:
        return "GSN-HydroCel-128", eeg_count, e129_present, ""
    if eeg_count == 129 and e129_present:
        return "GSN-HydroCel-129", eeg_count, e129_present, ""

    return (
        None,
        eeg_count,
        e129_present,
        "Unsupported or ambiguous HydroCel channel layout",
    )


def scan_file(
    file_path: Path,
    *,
    input_root: Path,
    expected_montage: str | None,
    raw_loader: Callable[[Path], object] | None = None,
) -> MontagePreflightFileResult:
    """Scan one EEG file with header-only loading when possible."""

    format_id = get_format_from_extension(file_path.suffix)
    relative_path = _relative_to_input(file_path, input_root)
    size_bytes = _path_size(file_path)

    if format_id is None:
        return MontagePreflightFileResult(
            path=str(file_path),
            relative_path=relative_path,
            format_id=None,
            expected_montage=expected_montage,
            detected_montage=None,
            status="unsupported",
            reason="Unsupported file extension",
            size_bytes=size_bytes,
        )

    loader = raw_loader or _read_raw_header
    try:
        raw = loader(file_path)
        detected, eeg_count, e129_present, reason = detect_hydrocel_montage(raw)
    except Exception as exc:
        return MontagePreflightFileResult(
            path=str(file_path),
            relative_path=relative_path,
            format_id=format_id,
            expected_montage=expected_montage,
            detected_montage=None,
            status="unknown",
            reason=f"Could not read EEG header: {exc}",
            size_bytes=size_bytes,
        )

    if detected is None:
        status = "unknown"
    elif detected == expected_montage:
        status = "match"
    else:
        status = "mismatch"

    return MontagePreflightFileResult(
        path=str(file_path),
        relative_path=relative_path,
        format_id=format_id,
        expected_montage=expected_montage,
        detected_montage=detected,
        status=status,
        eeg_channel_count=eeg_count,
        e129_present=e129_present,
        reason=reason,
        size_bytes=size_bytes,
    )


def build_batch_plan(
    *,
    input_path: Path,
    task_path: Path,
    output_dir: Path,
    raw_loader: Callable[[Path], object] | None = None,
) -> MontageBatchPlan:
    """Scan inputs and build a dry-run montage batch plan."""

    discover_plugins()
    expected_montage = read_task_montage(task_path)
    if input_path.is_dir() and input_path.suffix.lower() == ".mff":
        input_root = input_path.parent
    else:
        input_root = input_path if input_path.is_dir() else input_path.parent
    files = [
        scan_file(
            path,
            input_root=input_root,
            expected_montage=expected_montage,
            raw_loader=raw_loader,
        )
        for path in discover_eeg_inputs(input_path)
    ]

    groups = _group_results(files)
    unknown_files = [
        result.path
        for result in files
        if result.status in {"unknown", "unsupported"} or not result.is_actionable
    ]
    actionable_files = [result.path for result in files if result.is_actionable]

    return MontageBatchPlan(
        input_path=str(input_path),
        task_path=str(task_path),
        expected_montage=expected_montage,
        output_dir=str(output_dir),
        groups=groups,
        files=files,
        unknown_files=unknown_files,
        actionable_files=actionable_files,
    )


def write_scan_csv(plan: MontageBatchPlan, output_dir: Path) -> Path:
    """Write autoclean_montage_scan.csv."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "autoclean_montage_scan.csv"
    fields = [
        "path",
        "relative_path",
        "format_id",
        "expected_montage",
        "detected_montage",
        "status",
        "eeg_channel_count",
        "e129_present",
        "reason",
        "size_bytes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in plan.files:
            writer.writerow(asdict(result))
    return csv_path


def write_batch_plan_json(plan: MontageBatchPlan, output_dir: Path) -> Path:
    """Write autoclean_montage_batch_plan.json."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "autoclean_montage_batch_plan.json"
    path.write_text(json.dumps(plan.to_json_dict(), indent=2), encoding="utf-8")
    return path


def copy_originals_for_plan(
    plan: MontageBatchPlan,
    *,
    split_output_root: Path,
    overwrite: bool = False,
) -> MontageCopyResult:
    """Copy actionable original inputs into montage-specific folders."""

    estimate = estimate_copy_originals_for_plan(
        plan,
        split_output_root=split_output_root,
    )
    actionable = [result for result in plan.files if result.is_actionable]
    required_bytes = estimate.required_bytes
    free_before = estimate.free_bytes_before
    free_after = estimate.free_bytes_after_estimate
    if required_bytes > free_before:
        raise RuntimeError(
            "Insufficient free space for montage preflight copy: "
            f"need {required_bytes} bytes, available {free_before} bytes"
        )

    copy_targets = [
        (
            result,
            split_output_root / str(result.detected_montage) / result.relative_path,
        )
        for result in actionable
    ]
    for _result, destination in copy_targets:
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing destination: {destination}"
            )

    copied: list[dict] = []
    skipped = list(plan.unknown_files)
    for result, destination in copy_targets:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = Path(result.path)
        try:
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=overwrite)
            else:
                shutil.copy2(source, destination)
        except Exception as exc:
            partial_result = MontageCopyResult(
                split_output_root=str(split_output_root),
                copied_files=copied,
                skipped_files=skipped,
                required_bytes=required_bytes,
                free_bytes_before=free_before,
                free_bytes_after_estimate=free_after,
                completed=False,
                errors=[
                    {
                        "source": result.path,
                        "destination": str(destination),
                        "error": str(exc),
                    }
                ],
            )
            raise MontageCopyError(
                "Copy failed after "
                f"{len(copied)} file(s): {result.path} -> {destination}: {exc}",
                partial_result,
            ) from exc

        copied.append(
            {
                "source": result.path,
                "destination": str(destination),
                "detected_montage": result.detected_montage,
                "size_bytes": result.size_bytes,
            }
        )

    return MontageCopyResult(
        split_output_root=str(split_output_root),
        copied_files=copied,
        skipped_files=skipped,
        required_bytes=required_bytes,
        free_bytes_before=free_before,
        free_bytes_after_estimate=free_after,
    )


def estimate_copy_originals_for_plan(
    plan: MontageBatchPlan,
    *,
    split_output_root: Path,
) -> MontageCopyEstimate:
    """Estimate copy-originals size and destination free space without copying."""

    actionable = [result for result in plan.files if result.is_actionable]
    required_bytes = sum(result.size_bytes for result in actionable)
    disk_usage_path = _nearest_existing_path(split_output_root.parent)
    free_before = shutil.disk_usage(disk_usage_path).free
    return MontageCopyEstimate(
        split_output_root=str(split_output_root),
        actionable_file_count=len(actionable),
        skipped_file_count=len(plan.unknown_files),
        required_bytes=required_bytes,
        free_bytes_before=free_before,
        free_bytes_after_estimate=free_before - required_bytes,
    )


def estimate_move_originals_for_plan(
    plan: MontageBatchPlan,
    *,
    split_output_root: Path,
) -> MontageMoveEstimate:
    """Estimate move-originals size, volume, and temporary free-space needs."""

    actionable = [result for result in plan.files if result.is_actionable]
    required_bytes = sum(result.size_bytes for result in actionable)
    source_root = _nearest_existing_path(Path(plan.input_path))
    destination_root = _nearest_existing_path(split_output_root.parent)
    free_before = shutil.disk_usage(destination_root).free
    source_volume = _volume_identity(source_root)
    destination_volume = _volume_identity(destination_root)
    return MontageMoveEstimate(
        source_path=plan.input_path,
        split_output_root=str(split_output_root),
        actionable_file_count=len(actionable),
        unknown_file_count=len(plan.unknown_files),
        required_bytes=required_bytes,
        free_bytes_before=free_before,
        free_bytes_after_estimate=free_before - required_bytes,
        same_volume=source_volume == destination_volume,
        source_volume=source_volume,
        destination_volume=destination_volume,
    )


def write_planned_move_manifest(
    plan: MontageBatchPlan,
    *,
    output_dir: Path,
    split_output_root: Path,
    estimate: MontageMoveEstimate | None = None,
) -> Path:
    """Write the machine-readable planned move manifest before moving files."""

    estimate = estimate or estimate_move_originals_for_plan(
        plan,
        split_output_root=split_output_root,
    )
    targets = _move_targets_for_plan(plan, split_output_root=split_output_root)
    _validate_move_plan(plan, targets=targets, estimate=estimate)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "autoclean_montage_move_manifest.json"
    payload = {
        "input_path": plan.input_path,
        "task_path": plan.task_path,
        "split_output_root": str(split_output_root),
        "required_bytes": estimate.required_bytes,
        "free_bytes_before": estimate.free_bytes_before,
        "free_bytes_after_estimate": estimate.free_bytes_after_estimate,
        "same_volume": estimate.same_volume,
        "source_volume": estimate.source_volume,
        "destination_volume": estimate.destination_volume,
        "unknown_files": plan.unknown_files,
        "moves": [
            {
                "source": result.path,
                "destination": str(destination),
                "detected_montage": result.detected_montage,
                "size_bytes": result.size_bytes,
            }
            for result, destination in targets
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def move_originals_for_plan(
    plan: MontageBatchPlan,
    *,
    split_output_root: Path,
    planned_manifest: Path,
    estimate: MontageMoveEstimate | None = None,
) -> MontageMoveResult:
    """Copy, verify, then delete actionable originals into montage folders."""

    estimate = estimate or estimate_move_originals_for_plan(
        plan,
        split_output_root=split_output_root,
    )
    if not planned_manifest.is_file():
        raise FileNotFoundError(f"Planned move manifest not found: {planned_manifest}")
    targets = _move_targets_for_plan(plan, split_output_root=split_output_root)
    _validate_move_plan(plan, targets=targets, estimate=estimate)

    moved: list[dict] = []
    skipped: list[str] = []
    base_result = {
        "split_output_root": str(split_output_root),
        "planned_manifest": str(planned_manifest),
        "skipped_files": skipped,
        "required_bytes": estimate.required_bytes,
        "free_bytes_before": estimate.free_bytes_before,
        "free_bytes_after_estimate": estimate.free_bytes_after_estimate,
        "same_volume": estimate.same_volume,
        "source_volume": estimate.source_volume,
        "destination_volume": estimate.destination_volume,
    }

    for result, destination in targets:
        source = Path(result.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            _copy_path(source, destination)
            _verify_copied_path(destination, result.size_bytes)
            _delete_path(source)
        except Exception as exc:
            partial_result = MontageMoveResult(
                moved_files=moved,
                completed=False,
                errors=[
                    {
                        "source": result.path,
                        "destination": str(destination),
                        "error": str(exc),
                    }
                ],
                **base_result,
            )
            raise MontageMoveError(
                "Move failed after "
                f"{len(moved)} file(s): {result.path} -> {destination}: {exc}",
                partial_result,
            ) from exc

        moved.append(
            {
                "source": result.path,
                "destination": str(destination),
                "detected_montage": result.detected_montage,
                "size_bytes": result.size_bytes,
                "verified": True,
                "deleted_source": True,
            }
        )

    return MontageMoveResult(moved_files=moved, **base_result)


def write_apply_summary(
    *,
    output_dir: Path,
    copy_result: MontageCopyResult | None = None,
    move_result: MontageMoveResult | None = None,
    cloned_tasks: list[MontageTaskCloneResult] | None = None,
) -> Path:
    """Write autoclean_montage_apply_summary.json."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "autoclean_montage_apply_summary.json"
    operation_fields = _apply_summary_operation_fields(
        copy_result=copy_result,
        move_result=move_result,
    )
    summary = {
        **operation_fields,
        "copy_result": asdict(copy_result) if copy_result else None,
        "move_result": asdict(move_result) if move_result else None,
        "cloned_tasks": [asdict(task) for task in (cloned_tasks or [])],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def _apply_summary_operation_fields(
    *,
    copy_result: MontageCopyResult | None,
    move_result: MontageMoveResult | None,
) -> dict:
    """Build top-level audit fields for apply-mode file operations."""

    if move_result:
        return {
            "operation": "move_originals",
            "move_originals": True,
            "copied_originals": False,
            "delete_after_verified_copy": True,
            "requires_user_confirmation": True,
            "same_volume": move_result.same_volume,
            "temporary_space_required_bytes": move_result.required_bytes,
            "skipped_unknown_files": move_result.skipped_files,
            "file_operations": _move_file_operations(move_result),
        }
    if copy_result:
        return {
            "operation": "copy_originals",
            "move_originals": False,
            "copied_originals": True,
            "delete_after_verified_copy": False,
            "requires_user_confirmation": True,
            "same_volume": None,
            "temporary_space_required_bytes": copy_result.required_bytes,
            "skipped_unknown_files": copy_result.skipped_files,
            "file_operations": _copy_file_operations(copy_result),
        }
    return {
        "operation": None,
        "move_originals": False,
        "copied_originals": False,
        "delete_after_verified_copy": False,
        "requires_user_confirmation": False,
        "same_volume": None,
        "temporary_space_required_bytes": 0,
        "skipped_unknown_files": [],
        "file_operations": [],
    }


def _move_file_operations(move_result: MontageMoveResult) -> list[dict]:
    operations = [
        {
            "source": item["source"],
            "destination": item["destination"],
            "operation": "move",
            "bytes": item["size_bytes"],
            "copy_status": "completed",
            "verification_status": "completed" if item.get("verified") else "unknown",
            "delete_source_status": (
                "completed" if item.get("deleted_source") else "unknown"
            ),
            "status": "completed",
        }
        for item in move_result.moved_files
    ]
    operations.extend(
        {
            "source": item.get("source"),
            "destination": item.get("destination"),
            "operation": "move",
            "bytes": item.get("size_bytes"),
            "copy_status": "unknown",
            "verification_status": "failed",
            "delete_source_status": "not_started",
            "status": "failed",
            "error": item.get("error"),
        }
        for item in move_result.errors
    )
    return operations


def _copy_file_operations(copy_result: MontageCopyResult) -> list[dict]:
    operations = [
        {
            "source": item["source"],
            "destination": item["destination"],
            "operation": "copy",
            "bytes": item["size_bytes"],
            "copy_status": "completed",
            "verification_status": "not_applicable",
            "delete_source_status": "not_applicable",
            "status": "completed",
        }
        for item in copy_result.copied_files
    ]
    operations.extend(
        {
            "source": item.get("source"),
            "destination": item.get("destination"),
            "operation": "copy",
            "bytes": item.get("size_bytes"),
            "copy_status": "failed",
            "verification_status": "not_applicable",
            "delete_source_status": "not_applicable",
            "status": "failed",
            "error": item.get("error"),
        }
        for item in copy_result.errors
    )
    return operations


def clone_task_for_montage(
    *,
    source_task_path: Path,
    task_output_dir: Path,
    target_montage: str,
    class_name: str | None = None,
    file_stem: str | None = None,
    overwrite: bool = False,
) -> MontageTaskCloneResult:
    """Clone a task, changing only its montage value and class/file identity."""

    source = source_task_path.read_text(encoding="utf-8")
    source_montage = read_task_montage(source_task_path)
    old_class = _first_task_class_name(source)
    class_name = class_name or f"{old_class}_{_safe_identifier_suffix(target_montage)}"
    file_stem = file_stem or class_name
    destination = task_output_dir / f"{file_stem}.py"

    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing task clone: {destination}"
        )

    updated = update_task_montage_source(source, target_montage)
    updated = replace_task_class_name(updated, old_class, class_name)
    updated = _add_clone_provenance_comment(
        updated,
        source_task_path.name,
        source_montage,
        target_montage,
    )

    task_output_dir.mkdir(parents=True, exist_ok=True)
    destination.write_text(updated, encoding="utf-8")

    return MontageTaskCloneResult(
        source_task=str(source_task_path),
        cloned_task=str(destination),
        source_montage=source_montage,
        cloned_montage=target_montage,
        class_name=class_name,
    )


def clone_tasks_for_mismatches(
    *,
    plan: MontageBatchPlan,
    task_output_dir: Path,
    overwrite: bool = False,
) -> list[MontageTaskCloneResult]:
    """Clone one task per detected montage that differs from the source task."""

    targets = sorted(
        {
            result.detected_montage
            for result in plan.files
            if result.detected_montage
            and result.detected_montage != plan.expected_montage
            and result.is_actionable
        }
    )
    return [
        clone_task_for_montage(
            source_task_path=Path(plan.task_path),
            task_output_dir=task_output_dir,
            target_montage=target,
            overwrite=overwrite,
        )
        for target in targets
    ]


def _read_raw_header(file_path: Path):
    suffix = file_path.suffix.lower()
    if suffix == ".set":
        return mne.io.read_raw_eeglab(str(file_path), preload=False, verbose=False)
    if suffix == ".edf":
        return mne.io.read_raw_edf(str(file_path), preload=False, verbose=False)
    if suffix == ".bdf":
        return mne.io.read_raw_bdf(str(file_path), preload=False, verbose=False)
    if suffix == ".fif":
        return mne.io.read_raw_fif(str(file_path), preload=False, verbose=False)
    if suffix == ".vhdr":
        return mne.io.read_raw_brainvision(str(file_path), preload=False, verbose=False)
    if suffix == ".cnt":
        return mne.io.read_raw_cnt(str(file_path), preload=False, verbose=False)
    if suffix in {".mff", ".raw"}:
        return mne.io.read_raw_egi(str(file_path), preload=False, verbose=False)
    raise ValueError(f"Unsupported file format: {file_path.suffix}")


def _group_results(
    results: list[MontagePreflightFileResult],
) -> list[MontagePreflightGroup]:
    grouped: dict[tuple[str, str], list[MontagePreflightFileResult]] = {}
    for result in results:
        detected = result.detected_montage or "unknown"
        grouped.setdefault((detected, result.status), []).append(result)

    groups = []
    for (detected, status), items in sorted(grouped.items()):
        groups.append(
            MontagePreflightGroup(
                detected_montage=detected,
                status=status,
                file_count=len(items),
                total_size_bytes=sum(item.size_bytes for item in items),
                examples=[item.relative_path for item in items[:5]],
            )
        )
    return groups


def _relative_to_input(file_path: Path, input_root: Path) -> str:
    try:
        return str(file_path.relative_to(input_root))
    except ValueError:
        return file_path.name


def _path_size(path: Path) -> int:
    if path.is_dir():
        return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
    return path.stat().st_size if path.exists() else 0


def _nearest_existing_path(path: Path) -> Path:
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise FileNotFoundError(
                f"No existing parent directory for copy destination: {path}"
            )
        current = parent
    return current


def _volume_identity(path: Path) -> str:
    stat_result = path.stat()
    return f"device:{stat_result.st_dev}"


def _move_targets_for_plan(
    plan: MontageBatchPlan,
    *,
    split_output_root: Path,
) -> list[tuple[MontagePreflightFileResult, Path]]:
    return [
        (
            result,
            split_output_root / str(result.detected_montage) / result.relative_path,
        )
        for result in plan.files
        if result.is_actionable
    ]


def _validate_move_plan(
    plan: MontageBatchPlan,
    *,
    targets: list[tuple[MontagePreflightFileResult, Path]],
    estimate: MontageMoveEstimate,
) -> None:
    if plan.unknown_files:
        raise ValueError(
            "Refusing to move originals while unknown or unsupported files remain: "
            + ", ".join(plan.unknown_files)
        )

    if estimate.required_bytes > estimate.free_bytes_before:
        raise RuntimeError(
            "Insufficient temporary free space for montage preflight move: "
            f"need {estimate.required_bytes} bytes, "
            f"available {estimate.free_bytes_before} bytes"
        )

    for _result, destination in targets:
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing move destination: {destination}"
            )


def _copy_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def _verify_copied_path(destination: Path, expected_size_bytes: int) -> None:
    if not destination.exists():
        raise FileNotFoundError(f"Move destination was not created: {destination}")
    actual_size = _path_size(destination)
    if actual_size != expected_size_bytes:
        raise RuntimeError(
            "Move destination size mismatch after copy: "
            f"{destination} expected {expected_size_bytes} bytes, "
            f"found {actual_size} bytes"
        )


def _delete_path(source: Path) -> None:
    if source.is_dir():
        shutil.rmtree(source)
    else:
        source.unlink()


def _first_task_class_name(source: str) -> str:
    match = re.search(
        r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", source, re.MULTILINE
    )
    if not match:
        raise ValueError("Could not find a Python task class declaration")
    return match.group(1)


def _safe_identifier_suffix(value: str) -> str:
    suffix = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_")
    if suffix and suffix[0].isdigit():
        suffix = f"Montage_{suffix}"
    return suffix or "Montage"


def _add_clone_provenance_comment(
    source: str,
    source_task_name: str,
    source_montage: str | None,
    target_montage: str,
) -> str:
    comment = (
        "# Montage preflight clone note:\n"
        f"# This task was derived from {source_task_name}, which expected {source_montage}.\n"
        "# At creation, the only intended behavioral change was updating the "
        f'montage value to "{target_montage}".\n'
        "# Review this task before processing if any other settings should differ.\n\n"
    )
    return comment + source
