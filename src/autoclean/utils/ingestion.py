"""Ingestion utilities for automation readiness and provenance."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Optional, Sequence

import yaml

DEFAULT_RECEIPT_VERSION = "1.0"
DEFAULT_STATUS = "pending"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_bytes(payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def compute_file_hash(path: Path) -> str:
    """Return SHA256 hash for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_provenance_hash(relative_path: Path, metadata: dict[str, Any]) -> str:
    """Compute deterministic hash for a provenance subfolder."""
    payload = {
        "relative_path": relative_path.as_posix(),
        "metadata": {key: str(metadata[key]) for key in sorted(metadata)},
    }
    return _hash_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))


def resolve_provenance_folder(
    root: Path, relative_path: Path, metadata: dict[str, Any]
) -> tuple[Path, str]:
    """Resolve deterministic provenance folder and hash value."""
    hash_value = compute_provenance_hash(relative_path, metadata)
    return root / hash_value, hash_value


def receipt_path(folder: Path) -> Path:
    """Return receipt JSON sidecar path for a folder."""
    return folder / f"{folder.name}.json"


def _file_entry(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "hash": compute_file_hash(path),
    }


def build_receipt(
    *,
    folder: Path,
    relative_path: Path,
    metadata: dict[str, Any],
    files: Iterable[Path],
    status: str = DEFAULT_STATUS,
    receipt_version: str = DEFAULT_RECEIPT_VERSION,
    route_id: Optional[str] = None,
) -> dict[str, Any]:
    """Create receipt payload without writing to disk."""
    file_entries = [_file_entry(path) for path in files]
    hash_value = compute_provenance_hash(relative_path, metadata)
    receipt = {
        "receipt_version": receipt_version,
        "hash_inputs": {
            "relative_path": relative_path.as_posix(),
            "metadata": {key: str(metadata[key]) for key in sorted(metadata)},
        },
        "hash_value": hash_value,
        "files": file_entries,
        "status": status,
        "timestamps": {
            "created_at": _timestamp(),
            "updated_at": _timestamp(),
        },
        "revisions": [
            {
                "revision": 1,
                "status": status,
                "timestamp": _timestamp(),
                "note": "initial receipt",
            }
        ],
    }
    if route_id:
        receipt["route_id"] = route_id
    return receipt


def write_receipt(folder: Path, receipt: dict[str, Any]) -> Path:
    """Write receipt JSON sidecar to disk (atomic)."""
    path = receipt_path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def load_receipt(folder: Path) -> Optional[dict[str, Any]]:
    path = receipt_path(folder)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def append_receipt_revision(
    folder: Path, *, status: str, note: str, extra: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Append a receipt revision entry and persist updates."""
    receipt = load_receipt(folder)
    if receipt is None:
        raise FileNotFoundError(f"Receipt not found for {folder}")
    revisions = receipt.get("revisions", [])
    revision_number = len(revisions) + 1
    entry = {
        "revision": revision_number,
        "status": status,
        "timestamp": _timestamp(),
        "note": note,
    }
    if extra:
        entry.update(extra)
    revisions.append(entry)
    receipt["revisions"] = revisions
    receipt["status"] = status
    receipt.setdefault("timestamps", {})["updated_at"] = _timestamp()
    write_receipt(folder, receipt)
    return receipt


def stage_provenance_receipt(
    *,
    root: Path,
    relative_path: Path,
    metadata: dict[str, Any],
    files: Iterable[Path],
    status: str = DEFAULT_STATUS,
    ledger: Optional["IngestionLedger"] = None,
    route_id: Optional[str] = None,
) -> dict[str, Any]:
    """Create provenance folder, write receipt, and optionally record ledger."""
    folder, hash_value = resolve_provenance_folder(root, relative_path, metadata)
    receipt = build_receipt(
        folder=folder,
        relative_path=relative_path,
        metadata=metadata,
        files=files,
        status=status,
        route_id=route_id,
    )
    write_receipt(folder, receipt)

    duplicate = False
    if ledger is not None:
        duplicate = ledger.is_duplicate(hash_value, route_id=route_id)
        if not duplicate:
            ledger.record(
                hash_value,
                {
                    "relative_path": relative_path.as_posix(),
                    "folder": str(folder),
                },
                route_id=route_id,
            )

    return {
        "folder": folder,
        "hash": hash_value,
        "receipt": receipt,
        "duplicate": duplicate,
    }


def load_serve_config(config_path: Path) -> dict[str, Any]:
    """Load serve YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Serve config must be a mapping")
    return data


class ServeConfigError(ValueError):
    """Collect validation errors for serve configuration."""

    def __init__(self, errors: Sequence[str], warnings: Sequence[str]) -> None:
        message = "\n".join(errors)
        super().__init__(message)
        self.errors = list(errors)
        self.warnings = list(warnings)


@dataclass
class ServeRoute:
    id: str
    enabled: bool
    priority: int
    taskfile: str
    montage: str
    version: Optional[str]
    ingestion_folders: list[Path]
    ingestion_excludes: list[Path]
    file_globs: list[str]
    recursive: bool
    sentinel_ext: str
    automation_root: Path
    workspace_name: str


@dataclass
class ServeConfig:
    mode: str
    runtime_path: Path
    routes: list[ServeRoute]
    legacy: bool = False


_TOP_LEVEL_KEYS = {
    "mode",
    "runtime",
    "runtime_package",
    "automation_mode",
    "defaults",
    "automations",
    "automation_root",
    "workspace_name",
    "taskfile",
    "montage",
    "version",
    "ingestion_folders",
    "ingestion_excludes",
    "file_glob",
    "file_globs",
    "sentinel_ext",
    "recursive",
    "priority",
    "enabled",
}
_DEFAULT_KEYS = {
    "automation_root",
    "workspace_name",
    "file_glob",
    "file_globs",
    "sentinel_ext",
    "recursive",
    "ingestion_excludes",
}
_ROUTE_KEYS = {
    "id",
    "enabled",
    "priority",
    "taskfile",
    "montage",
    "version",
    "ingestion_folders",
    "ingestion_excludes",
    "file_glob",
    "file_globs",
    "sentinel_ext",
    "recursive",
    "automation_root",
    "workspace_name",
}


def _taskfile_label(taskfile: str) -> str:
    path = Path(taskfile)
    if path.suffix == ".py":
        return path.stem
    if path.name != taskfile:
        return path.name
    return taskfile


def _normalize_route_id(taskfile: str, montage: str, version: Optional[str]) -> str:
    return build_workspace_name(
        "taskfile-montage-version",
        taskfile=_taskfile_label(taskfile),
        montage=montage,
        version=version,
    )


def _normalize_file_globs(value: Any, errors: list[str], label: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    errors.append(f"{label} must be a string or list of strings")
    return []


def _coerce_bool(value: Any, *, default: bool, errors: list[str], label: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    errors.append(f"{label} must be a boolean")
    return default


def _normalize_path_list(
    value: Any,
    workspace_dir: Path,
    errors: list[str],
    label: str,
    *,
    required: bool,
) -> list[Path]:
    if value is None:
        if required:
            errors.append(f"{label} must be a non-empty list of paths")
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        items = list(value)
    else:
        errors.append(f"{label} must be a list of strings")
        return []
    if required and not items:
        errors.append(f"{label} must be a non-empty list of paths")
    paths: list[Path] = []
    for entry in items:
        path = Path(entry)
        if not path.is_absolute():
            path = (workspace_dir / path).resolve()
        paths.append(path)
    return paths


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_known_keys(
    mapping: dict[str, Any], allowed: set[str], errors: list[str], label: str
) -> None:
    for key in mapping:
        if key not in allowed:
            errors.append(f"Unknown {label} key: {key}")


def parse_serve_config(
    config: dict[str, Any],
    workspace_dir: Path,
    *,
    strict: bool = True,
) -> tuple[ServeConfig, list[str]]:
    """Normalize and validate serve configuration for multi-route ingestion."""
    errors: list[str] = []
    warnings: list[str] = []

    _validate_known_keys(config, _TOP_LEVEL_KEYS, errors, "config")

    mode = config.get("mode")
    if not mode:
        errors.append("Missing required key: mode")
    runtime_value = config.get("runtime")
    if not runtime_value:
        errors.append("Missing required key: runtime")

    runtime_path = None
    if runtime_value:
        runtime_path = _resolve_relative_path(workspace_dir, str(runtime_value))
        if not runtime_path.exists():
            errors.append(f"Runtime path not found: {runtime_path}")

    if config.get("automation_mode") is not True:
        warnings.append("automation_mode is not true")

    defaults_raw = config.get("defaults", {})
    if defaults_raw is None:
        defaults_raw = {}
    if not isinstance(defaults_raw, dict):
        errors.append("defaults must be a mapping")
        defaults_raw = {}

    if isinstance(defaults_raw, dict):
        _validate_known_keys(defaults_raw, _DEFAULT_KEYS, errors, "defaults")

    def _default_value(key: str) -> Any:
        if key in defaults_raw:
            return defaults_raw.get(key)
        return config.get(key)

    default_file_globs = _normalize_file_globs(
        defaults_raw.get("file_globs")
        if "file_globs" in defaults_raw
        else defaults_raw.get("file_glob")
        if "file_glob" in defaults_raw
        else config.get("file_globs", config.get("file_glob")),
        errors,
        "file_globs",
    )
    default_sentinel_ext = _default_value("sentinel_ext") or ".ready"
    if not isinstance(default_sentinel_ext, str):
        errors.append("sentinel_ext must be a string")
        default_sentinel_ext = ".ready"
    default_recursive = _coerce_bool(
        _default_value("recursive"), default=True, errors=errors, label="recursive"
    )
    default_automation_root = _default_value("automation_root")
    default_workspace_name = _default_value("workspace_name")
    default_excludes = _normalize_path_list(
        _default_value("ingestion_excludes"),
        workspace_dir,
        errors,
        "ingestion_excludes",
        required=False,
    )

    automations = config.get("automations")
    legacy = automations is None
    if automations is None:
        legacy_route = {key: config.get(key) for key in _ROUTE_KEYS if key in config}
        automations = [legacy_route]
    if legacy and not default_file_globs:
        default_file_globs = ["*"]
    if not isinstance(automations, list) or not automations:
        errors.append("automations must be a non-empty list")
        automations = []

    routes: list[ServeRoute] = []
    for idx, route_data in enumerate(automations):
        if not isinstance(route_data, dict):
            errors.append(f"automations[{idx}] must be a mapping")
            continue
        _validate_known_keys(route_data, _ROUTE_KEYS, errors, f"automations[{idx}]")

        enabled_value = route_data.get("enabled", True)
        if not isinstance(enabled_value, bool):
            errors.append(f"automations[{idx}].enabled must be a boolean")
            enabled_value = True
        priority_value = route_data.get("priority", 0)
        if isinstance(priority_value, bool) or not isinstance(priority_value, int):
            errors.append(f"automations[{idx}].priority must be an integer")
            priority_value = 0

        taskfile_value = route_data.get("taskfile")
        montage_value = route_data.get("montage")
        if not taskfile_value:
            if strict:
                errors.append(f"automations[{idx}].taskfile is required")
            else:
                warnings.append(f"automations[{idx}].taskfile is empty")
            taskfile_value = ""
        if not montage_value:
            if strict:
                errors.append(f"automations[{idx}].montage is required")
            else:
                warnings.append(f"automations[{idx}].montage is empty")
            montage_value = ""

        version_value = route_data.get("version")
        if version_value is not None:
            version_value = str(version_value)

        route_file_globs = _normalize_file_globs(
            route_data.get("file_globs")
            if "file_globs" in route_data
            else route_data.get("file_glob")
            if "file_glob" in route_data
            else default_file_globs,
            errors,
            f"automations[{idx}].file_globs",
        )
        if not route_file_globs:
            errors.append(f"automations[{idx}].file_globs must be set")

        recursive_value = _coerce_bool(
            route_data.get("recursive", default_recursive),
            default=default_recursive,
            errors=errors,
            label=f"automations[{idx}].recursive",
        )

        sentinel_value = route_data.get("sentinel_ext", default_sentinel_ext)
        if not isinstance(sentinel_value, str):
            errors.append(f"automations[{idx}].sentinel_ext must be a string")
            sentinel_value = default_sentinel_ext

        ingestion_folders = _normalize_path_list(
            route_data.get("ingestion_folders"),
            workspace_dir,
            errors,
            f"automations[{idx}].ingestion_folders",
            required=strict,
        )
        if not ingestion_folders and not strict:
            warnings.append(f"automations[{idx}].ingestion_folders is empty")
        if strict and ingestion_folders:
            for root in ingestion_folders:
                if not root.exists():
                    errors.append(
                        f"automations[{idx}].ingestion_folders path not found: {root}"
                    )
        ingestion_excludes = _normalize_path_list(
            route_data.get("ingestion_excludes", default_excludes),
            workspace_dir,
            errors,
            f"automations[{idx}].ingestion_excludes",
            required=False,
        )
        if ingestion_excludes and ingestion_folders:
            for exclude in ingestion_excludes:
                if not any(_is_relative_to(exclude, root) for root in ingestion_folders):
                    errors.append(
                        f"{exclude} is not under any ingestion_folders for automations[{idx}]"
                    )

        automation_root_value = route_data.get("automation_root", default_automation_root)
        if not automation_root_value:
            errors.append(f"automations[{idx}].automation_root is required")
            automation_root_value = ""
        automation_root_path = (
            _resolve_relative_path(workspace_dir, str(automation_root_value))
            if automation_root_value
            else workspace_dir
        )
        if automation_root_value and not automation_root_path.exists():
            errors.append(f"Automation root not found: {automation_root_path}")

        workspace_template = route_data.get("workspace_name", default_workspace_name)
        if not workspace_template:
            errors.append(f"automations[{idx}].workspace_name is required")
            workspace_template = "taskfile-montage-version"
        taskfile_label = _taskfile_label(str(taskfile_value))
        workspace_name = build_workspace_name(
            str(workspace_template),
            taskfile=taskfile_label,
            montage=str(montage_value),
            version=version_value,
        )

        route_id = route_data.get("id")
        if not route_id:
            route_id = _normalize_route_id(str(taskfile_value), str(montage_value), version_value)
        route_id = _normalize_workspace_name(str(route_id))

        routes.append(
            ServeRoute(
                id=route_id,
                enabled=enabled_value,
                priority=priority_value,
                taskfile=str(taskfile_value),
                montage=str(montage_value),
                version=version_value,
                ingestion_folders=ingestion_folders,
                ingestion_excludes=ingestion_excludes,
                file_globs=route_file_globs,
                recursive=recursive_value,
                sentinel_ext=sentinel_value,
                automation_root=automation_root_path,
                workspace_name=workspace_name,
            )
        )

    if routes:
        seen_ids: set[str] = set()
        for route in routes:
            if route.id in seen_ids:
                errors.append(f"Duplicate route id: {route.id}")
            seen_ids.add(route.id)

    for idx, route in enumerate(routes):
        if not route.enabled:
            continue
        for other in routes[idx + 1 :]:
            if not other.enabled:
                continue
            if route.priority != other.priority:
                continue
            overlap = False
            for root in route.ingestion_folders:
                for other_root in other.ingestion_folders:
                    if _is_relative_to(root, other_root) or _is_relative_to(
                        other_root, root
                    ):
                        overlap = True
                        break
                if overlap:
                    break
            if overlap:
                errors.append(
                    f"Routes '{route.id}' and '{other.id}' overlap ingestion roots "
                    f"with the same priority ({route.priority})"
                )

    if errors:
        raise ServeConfigError(errors, warnings)

    if runtime_path is None:
        runtime_path = workspace_dir

    return ServeConfig(
        mode=str(mode),
        runtime_path=runtime_path,
        routes=routes,
        legacy=legacy,
    ), warnings


def resolve_ingestion_roots(
    config: ServeConfig | dict[str, Any], workspace_dir: Optional[Path] = None
) -> list[Path]:
    """Resolve ingestion roots from serve config."""
    if isinstance(config, ServeConfig):
        roots: list[Path] = []
        for route in config.routes:
            roots.extend(route.ingestion_folders)
        return sorted({path.resolve() for path in roots})
    if workspace_dir is None:
        raise ValueError("workspace_dir is required for raw config")
    roots: list[Path] = []
    for entry in config.get("ingestion_folders", []):
        if not isinstance(entry, str):
            continue
        path = Path(entry)
        if not path.is_absolute():
            path = (workspace_dir / path).resolve()
        roots.append(path)
    return roots


def _resolve_relative_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _normalize_workspace_name(name: str) -> str:
    cleaned = name.strip().replace(" ", "-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-_")


def build_workspace_name(
    template: str,
    *,
    taskfile: str,
    montage: str,
    version: Optional[str] = None,
) -> str:
    """Build workspace name from a template."""
    if "{" in template:
        name = template.format(
            taskfile=taskfile,
            montage=montage,
            version=version or "",
        )
        return _normalize_workspace_name(name)

    segments = template.split("-")
    replacements = {"taskfile": taskfile, "montage": montage, "version": version}
    rendered: list[str] = []
    for segment in segments:
        if segment in replacements:
            value = replacements[segment]
            if value:
                rendered.append(value)
        else:
            rendered.append(segment)
    return _normalize_workspace_name("-".join(rendered))


@dataclass
class DispatchPlan:
    mode: str
    taskfile: str
    montage: str
    runtime_path: Path
    automation_root: Path
    workspace_name: str
    workspace_dir: Path
    files: list[Path]


def build_dispatch_plan(
    *,
    config: dict[str, Any],
    workspace_dir: Path,
    files: Iterable[Path],
    version: Optional[str] = None,
) -> DispatchPlan:
    """Build a dispatch plan from serve config and files."""
    required = [
        "mode",
        "taskfile",
        "montage",
        "runtime",
        "automation_root",
        "workspace_name",
        "ingestion_folders",
    ]
    for key in required:
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")

    mode = str(config["mode"])
    taskfile = str(config["taskfile"])
    montage = str(config["montage"])
    runtime_path = _resolve_relative_path(workspace_dir, str(config["runtime"]))
    automation_root = _resolve_relative_path(
        workspace_dir, str(config["automation_root"])
    )
    workspace_name = build_workspace_name(
        str(config["workspace_name"]),
        taskfile=taskfile,
        montage=montage,
        version=version,
    )
    workspace_path = automation_root / workspace_name
    return DispatchPlan(
        mode=mode,
        taskfile=taskfile,
        montage=montage,
        runtime_path=runtime_path,
        automation_root=automation_root,
        workspace_name=workspace_name,
        workspace_dir=workspace_path,
        files=list(files),
    )


def build_dispatch_plan_for_route(
    *,
    config: ServeConfig,
    route: ServeRoute,
    files: Iterable[Path],
) -> DispatchPlan:
    """Build a dispatch plan from a normalized route."""
    workspace_path = route.automation_root / route.workspace_name
    return DispatchPlan(
        mode=config.mode,
        taskfile=route.taskfile,
        montage=route.montage,
        runtime_path=config.runtime_path,
        automation_root=route.automation_root,
        workspace_name=route.workspace_name,
        workspace_dir=workspace_path,
        files=list(files),
    )


def _runtime_cli_name() -> str:
    return (
        "autocleaneeg-pipeline.exe"
        if sys.platform.startswith("win")
        else "autocleaneeg-pipeline"
    )


def resolve_runtime_cli(runtime_path: Path) -> Path:
    """Resolve CLI binary within a runtime directory."""
    if runtime_path.is_file():
        return runtime_path

    candidates: list[Path] = []
    venv_dir = runtime_path / ".venv"
    if sys.platform.startswith("win"):
        candidates.extend(
            [
                venv_dir / "Scripts" / _runtime_cli_name(),
                runtime_path / "Scripts" / _runtime_cli_name(),
            ]
        )
    else:
        candidates.extend(
            [
                venv_dir / "bin" / _runtime_cli_name(),
                runtime_path / "bin" / _runtime_cli_name(),
            ]
        )

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Runtime CLI not found under {runtime_path}")


def resolve_taskfile_path(taskfile: str, workspace_dir: Path) -> Optional[Path]:
    """Resolve taskfile path if a Python file is specified."""
    candidate = Path(taskfile)
    if candidate.suffix != ".py":
        return None
    if candidate.is_absolute():
        if not candidate.exists():
            raise FileNotFoundError(f"Task file not found: {candidate}")
        return candidate

    for base in [workspace_dir, workspace_dir.parent]:
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"Task file not found: {candidate}")


def build_process_command(
    *,
    plan: DispatchPlan,
    file_path: Path,
    automation: bool = True,
    yes: bool = True,
    runtime_cli: Optional[Path] = None,
) -> list[str]:
    """Build autocleaneeg-pipeline process command for a file."""
    cli_path = runtime_cli or resolve_runtime_cli(plan.runtime_path)
    cmd = [str(cli_path), "process"]
    taskfile_path = resolve_taskfile_path(plan.taskfile, plan.workspace_dir)
    if taskfile_path:
        cmd.extend(["--task-file", str(taskfile_path)])
    elif plan.taskfile:
        cmd.extend(["--task", plan.taskfile])
    else:
        raise ValueError("Taskfile or task name is required")

    cmd.extend(["--file", str(file_path), "--output", str(plan.workspace_dir)])
    if automation:
        cmd.append("--automation")
    if yes:
        cmd.append("--yes")
    return cmd


@dataclass
class DispatchResult:
    processed: list[Path]
    failed: dict[Path, str]
    attempts: int


def execute_dispatch_plan(
    plan: DispatchPlan,
    *,
    processor: Callable[[Path, DispatchPlan], None],
    max_attempts: int = 1,
) -> DispatchResult:
    """Execute a dispatch plan using the provided processor."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    pending = list(plan.files)
    processed: list[Path] = []
    failed: dict[Path, str] = {}
    attempts = 0
    while pending and attempts < max_attempts:
        attempts += 1
        next_pending = []
        for path in pending:
            try:
                processor(path, plan)
                processed.append(path)
                failed.pop(path, None)
            except Exception as exc:  # pragma: no cover - errors expected in tests
                failed[path] = str(exc)
                next_pending.append(path)
        pending = next_pending

    return DispatchResult(processed=processed, failed=failed, attempts=attempts)


def run_dispatch_plan(
    plan: DispatchPlan,
    *,
    automation: bool = True,
    yes: bool = True,
    max_attempts: int = 1,
    runner: Optional[Callable[[list[str]], None]] = None,
) -> DispatchResult:
    """Execute a dispatch plan using the runtime CLI."""
    cli_path = resolve_runtime_cli(plan.runtime_path)

    def _processor(file_path: Path, dispatch_plan: DispatchPlan) -> None:
        cmd = build_process_command(
            plan=dispatch_plan,
            file_path=file_path,
            automation=automation,
            yes=yes,
            runtime_cli=cli_path,
        )
        if runner is None:
            subprocess.run(cmd, check=True)
        else:
            runner(cmd)

    return execute_dispatch_plan(plan, processor=_processor, max_attempts=max_attempts)


@dataclass
class IngestionDispatchResult:
    route_id: str
    ingestion_roots: list[Path]
    ready: "ReadyScanResult"
    plan: Optional[DispatchPlan]
    result: Optional[DispatchResult]


@dataclass(frozen=True)
class RouteMatch:
    route: ServeRoute
    ingestion_root: Path
    specificity: tuple[int, int, int]


def _glob_specificity(pattern: str) -> tuple[int, int, int]:
    wildcard_count = sum(1 for char in pattern if char in "*?")
    literal_count = sum(1 for char in pattern if char not in "*?")
    double_star = pattern.count("**")
    return (literal_count, -double_star, -wildcard_count)


def _matches_route(route: ServeRoute, file_path: Path) -> Optional[RouteMatch]:
    best: Optional[RouteMatch] = None
    best_spec: Optional[tuple[int, int, int]] = None
    best_depth = -1
    for root in route.ingestion_folders:
        if not _is_relative_to(file_path, root):
            continue
        if any(_is_relative_to(file_path, exclude) for exclude in route.ingestion_excludes):
            continue
        rel_path = file_path.relative_to(root)
        if not route.recursive and len(rel_path.parts) > 1:
            continue
        rel_posix = PurePosixPath(rel_path.as_posix())
        for pattern in route.file_globs:
            patterns = [pattern]
            if route.recursive and "/" not in pattern and "\\" not in pattern and "**" not in pattern:
                patterns.append(f"**/{pattern}")
            for match_pattern in patterns:
                if rel_posix.match(match_pattern):
                    spec = _glob_specificity(match_pattern)
                    depth = len(root.parts)
                    if best_spec is None or spec > best_spec or (
                        spec == best_spec and depth > best_depth
                    ):
                        best_spec = spec
                        best_depth = depth
                        best = RouteMatch(
                            route=route, ingestion_root=root, specificity=spec
                        )
    return best


def _select_route_for_file(
    file_path: Path, routes: Sequence[ServeRoute]
) -> Optional[RouteMatch]:
    matches: list[RouteMatch] = []
    for route in routes:
        match = _matches_route(route, file_path)
        if match is not None:
            matches.append(match)
    if not matches:
        return None
    max_priority = max(match.route.priority for match in matches)
    priority_matches = [match for match in matches if match.route.priority == max_priority]
    priority_matches.sort(key=lambda match: match.specificity, reverse=True)
    best = priority_matches[0]
    tied = [
        match
        for match in priority_matches
        if match.specificity == best.specificity
    ]
    if len(tied) > 1:
        tied_ids = ", ".join(match.route.id for match in tied)
        raise ValueError(
            f"Routing tie for {file_path} between routes with priority "
            f"{max_priority}: {tied_ids}"
        )
    return best


def _strip_sentinel(path: Path, sentinel_ext: str) -> Path:
    name = path.name
    if name.endswith(sentinel_ext):
        return path.with_name(name[: -len(sentinel_ext)])
    return path


def dispatch_ready_ingestion(
    *,
    config_path: Path,
    workspace_dir: Path,
    ingestion_root: Optional[Path] = None,
    file_glob: Optional[str] = None,
    sentinel_ext: Optional[str] = None,
    require_sentinel: bool = True,
    stability_window_seconds: int = 0,
    use_watchfiles: bool = True,
    max_events: int = 1,
    automation: bool = True,
    yes: bool = True,
    max_attempts: int = 1,
    runner: Optional[Callable[[list[str]], None]] = None,
    config: Optional[ServeConfig | dict[str, Any]] = None,
    queue: Optional["IngestionQueue"] = None,
) -> list[IngestionDispatchResult]:
    """Dispatch ready ingestion files using serve configuration."""
    serve_config: Optional[ServeConfig]
    if config is None:
        raw_config = load_serve_config(config_path)
        serve_config, _ = parse_serve_config(raw_config, workspace_dir, strict=True)
    elif isinstance(config, ServeConfig):
        serve_config = config
    else:
        serve_config, _ = parse_serve_config(config, workspace_dir, strict=True)

    routes = [route for route in serve_config.routes if route.enabled]
    if not routes:
        return []

    if queue is not None:
        unassigned = queue.pending_without_route()
        if unassigned:
            raise ValueError(
                "Queue has pending entries without route_id; migrate queue data before running."
            )

    if serve_config.legacy:
        for route in routes:
            if file_glob is not None:
                route.file_globs = [file_glob]
            if sentinel_ext is not None:
                route.sentinel_ext = sentinel_ext

    roots_filter = None
    if ingestion_root is not None:
        roots_filter = {ingestion_root.resolve()}
        known_roots = resolve_ingestion_roots(serve_config)
        if ingestion_root.resolve() not in known_roots:
            raise ValueError("ingestion_root is not listed in ingestion_folders")

    ready_by_route: dict[str, ReadyScanResult] = {}
    for route in routes:
        ready_result = ReadyScanResult()
        for root in route.ingestion_folders:
            if roots_filter and root.resolve() not in roots_filter:
                continue
            ready = watch_ready_files(
                root,
                file_glob=route.file_globs,
                sentinel_ext=route.sentinel_ext,
                require_sentinel=require_sentinel,
                stability_window_seconds=stability_window_seconds,
                recursive=route.recursive,
                max_events=max_events,
                use_watchfiles=use_watchfiles,
            )
            ready_result.ready_files.extend(
                [
                    path
                    for path in ready.ready_files
                    if _matches_route(route, path) is not None
                ]
            )
            ready_result.pending_files.extend(
                [
                    path
                    for path in ready.pending_files
                    if _matches_route(route, path) is not None
                ]
            )
            ready_result.missing_sentinels.extend(
                [
                    path
                    for path in ready.missing_sentinels
                    if _matches_route(route, _strip_sentinel(path, route.sentinel_ext))
                    is not None
                ]
            )
            ready_result.unstable_files.extend(
                [
                    path
                    for path in ready.unstable_files
                    if _matches_route(route, path) is not None
                ]
            )
        ready_by_route[route.id] = ready_result

    file_ready_routes: dict[Path, list[ServeRoute]] = {}
    for route in routes:
        ready_files = ready_by_route.get(route.id, ReadyScanResult()).ready_files
        for path in ready_files:
            file_ready_routes.setdefault(path, []).append(route)

    assigned_files: dict[str, list[Path]] = {route.id: [] for route in routes}
    assigned_root: dict[Path, Path] = {}
    for path, candidate_routes in file_ready_routes.items():
        match = _select_route_for_file(path, candidate_routes)
        if match is None:
            continue
        assigned_files[match.route.id].append(path)
        assigned_root[path] = match.ingestion_root

    dispatch_results: list[IngestionDispatchResult] = []
    for route in routes:
        ready_result = ready_by_route.get(route.id, ReadyScanResult())
        ready_result.ready_files = assigned_files.get(route.id, [])

        if queue is not None and ready_result.ready_files:
            entries = [
                QueueEntry(
                    path=path,
                    route_id=route.id,
                    ingestion_root=assigned_root.get(path),
                )
                for path in ready_result.ready_files
            ]
            queue.enqueue_entries(entries)

        pending_entries = (
            queue.pending_entries(route_id=route.id)
            if queue is not None
            else [
                QueueEntry(
                    path=path,
                    route_id=route.id,
                    ingestion_root=assigned_root.get(path),
                )
                for path in ready_result.ready_files
            ]
        )
        pending_files = [entry.path for entry in pending_entries]
        plan = None
        result = None
        if pending_files:
            plan = build_dispatch_plan_for_route(
                config=serve_config, route=route, files=pending_files
            )
            result = run_dispatch_plan(
                plan,
                automation=automation,
                yes=yes,
                max_attempts=max_attempts,
                runner=runner,
            )

        if queue is not None and result is not None:
            for path in result.processed:
                queue.mark_processed(path)
            for path, error in result.failed.items():
                queue.mark_failed(path, error)

        dispatch_results.append(
            IngestionDispatchResult(
                route_id=route.id,
                ingestion_roots=route.ingestion_folders,
                ready=ready_result,
                plan=plan,
                result=result,
            )
        )

    return dispatch_results


@dataclass
class IngestionLoopResult:
    iterations: int
    dispatch_results: list[IngestionDispatchResult]
    pending_roots: list[Path]
    pending_routes: list[str] = field(default_factory=list)


def run_ingestion_loop(
    *,
    config_path: Path,
    workspace_dir: Path,
    max_cycles: int = 1,
    file_glob: Optional[str] = None,
    sentinel_ext: Optional[str] = None,
    require_sentinel: bool = True,
    stability_window_seconds: int = 0,
    use_watchfiles: bool = True,
    max_events: int = 1,
    automation: bool = True,
    yes: bool = True,
    max_attempts: int = 1,
    runner: Optional[Callable[[list[str]], None]] = None,
    queue: Optional["IngestionQueue"] = None,
    sleep_fn: Optional[Callable[[float], None]] = None,
    sleep_seconds: float = 1.0,
) -> IngestionLoopResult:
    """Run readiness/dispatch loop over all ingestion roots."""
    if max_cycles < 1:
        raise ValueError("max_cycles must be >= 1")
    raw_config = load_serve_config(config_path)
    config, _ = parse_serve_config(raw_config, workspace_dir, strict=True)
    roots = resolve_ingestion_roots(config)
    if not roots:
        raise ValueError("No ingestion_folders configured")
    dispatch_results: list[IngestionDispatchResult] = []
    sleep = sleep_fn or time.sleep
    pending_roots = list(roots)

    for cycle in range(max_cycles):
        ready_roots: list[Path] = []
        results = dispatch_ready_ingestion(
            config_path=config_path,
            workspace_dir=workspace_dir,
            file_glob=file_glob,
            sentinel_ext=sentinel_ext,
            require_sentinel=require_sentinel,
            stability_window_seconds=stability_window_seconds,
            use_watchfiles=use_watchfiles,
            max_events=max_events,
            automation=automation,
            yes=yes,
            max_attempts=max_attempts,
            runner=runner,
            config=config,
            queue=queue,
        )
        dispatch_results.extend(results)
        ready_routes: list[str] = []
        for result in results:
            processed = bool(
                result.result and (result.result.processed or result.result.failed)
            )
            if result.ready.ready or processed:
                ready_routes.append(result.route_id)
                ready_roots.extend(result.ingestion_roots)
        pending_roots = [root for root in roots if root not in ready_roots]
        if not ready_routes:
            return IngestionLoopResult(
                iterations=cycle + 1,
                dispatch_results=dispatch_results,
                pending_roots=pending_roots,
                pending_routes=[
                    route.id
                    for route in config.routes
                    if route.enabled and route.id not in ready_routes
                ],
            )
        if cycle < max_cycles - 1:
            sleep(sleep_seconds)

    return IngestionLoopResult(
        iterations=max_cycles,
        dispatch_results=dispatch_results,
        pending_roots=pending_roots,
        pending_routes=[
            route.id
            for route in config.routes
            if route.enabled and route.id not in ready_routes
        ],
    )


@dataclass
class IngestionServiceResult:
    cycles: int
    idle_cycles: int
    loop_results: list[IngestionLoopResult]


def run_ingestion_service(
    *,
    config_path: Path,
    workspace_dir: Path,
    max_cycles: int = 1,
    idle_limit: int = 1,
    file_glob: Optional[str] = None,
    sentinel_ext: Optional[str] = None,
    require_sentinel: bool = True,
    stability_window_seconds: int = 0,
    use_watchfiles: bool = True,
    max_events: int = 1,
    automation: bool = True,
    yes: bool = True,
    max_attempts: int = 1,
    runner: Optional[Callable[[list[str]], None]] = None,
    queue: Optional["IngestionQueue"] = None,
    queue_path: Optional[Path] = None,
    sleep_fn: Optional[Callable[[float], None]] = None,
    sleep_seconds: float = 1.0,
) -> IngestionServiceResult:
    """Run repeated ingestion loops until idle or cycle limit reached."""
    if max_cycles < 1:
        raise ValueError("max_cycles must be >= 1")
    if idle_limit < 1:
        raise ValueError("idle_limit must be >= 1")

    if queue is None and queue_path is not None:
        queue = IngestionQueue(queue_path)

    loop_results: list[IngestionLoopResult] = []
    idle_cycles = 0
    sleep = sleep_fn or time.sleep

    for cycle in range(max_cycles):
        loop_result = run_ingestion_loop(
            config_path=config_path,
            workspace_dir=workspace_dir,
            max_cycles=1,
            file_glob=file_glob,
            sentinel_ext=sentinel_ext,
            require_sentinel=require_sentinel,
            stability_window_seconds=stability_window_seconds,
            use_watchfiles=use_watchfiles,
            max_events=max_events,
            automation=automation,
            yes=yes,
            max_attempts=max_attempts,
            runner=runner,
            queue=queue,
            sleep_fn=lambda _: None,
        )
        loop_results.append(loop_result)
        any_ready = False
        for result in loop_result.dispatch_results:
            processed = bool(
                result.result and (result.result.processed or result.result.failed)
            )
            if result.ready.ready or processed:
                any_ready = True
                break
        if any_ready:
            idle_cycles = 0
        else:
            idle_cycles += 1
            if idle_cycles >= idle_limit:
                return IngestionServiceResult(
                    cycles=cycle + 1,
                    idle_cycles=idle_cycles,
                    loop_results=loop_results,
                )
        if cycle < max_cycles - 1:
            sleep(sleep_seconds)

    return IngestionServiceResult(
        cycles=max_cycles,
        idle_cycles=idle_cycles,
        loop_results=loop_results,
    )


@dataclass
class ReadinessResult:
    ready: bool
    reasons: list[str] = field(default_factory=list)
    missing_sentinels: list[Path] = field(default_factory=list)
    unstable_files: list[Path] = field(default_factory=list)


@dataclass
class ReadyScanResult:
    ready_files: list[Path] = field(default_factory=list)
    pending_files: list[Path] = field(default_factory=list)
    missing_sentinels: list[Path] = field(default_factory=list)
    unstable_files: list[Path] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return bool(self.ready_files)


def _sentinel_path(file_path: Path, sentinel_ext: str) -> Path:
    return file_path.with_name(f"{file_path.name}{sentinel_ext}")


def _check_stability(files: Iterable[Path], window_seconds: int) -> list[Path]:
    if window_seconds <= 0:
        return []
    sizes_before = {path: path.stat().st_size for path in files}
    time.sleep(window_seconds)
    unstable = []
    for path, size in sizes_before.items():
        if path.stat().st_size != size:
            unstable.append(path)
    return unstable


def evaluate_readiness(
    files: Iterable[Path],
    *,
    sentinel_ext: str = ".ready",
    require_sentinel: bool = True,
    stability_window_seconds: int = 0,
) -> ReadinessResult:
    """Evaluate readiness based on sentinels and stability window."""
    file_list = list(files)
    missing = []
    if require_sentinel:
        for path in file_list:
            sentinel = _sentinel_path(path, sentinel_ext)
            if not sentinel.exists():
                missing.append(sentinel)

    unstable = _check_stability(file_list, stability_window_seconds)
    reasons = []
    if missing:
        reasons.append("missing_sentinels")
    if unstable:
        reasons.append("files_not_stable")
    return ReadinessResult(
        ready=not missing and not unstable,
        reasons=reasons,
        missing_sentinels=missing,
        unstable_files=unstable,
    )


def list_ingestion_files(
    root: Path,
    *,
    file_glob: str | Sequence[str],
    sentinel_ext: str,
    recursive: bool = True,
) -> list[Path]:
    """Return candidate ingestion files under root."""
    patterns = [file_glob] if isinstance(file_glob, str) else list(file_glob)
    files: list[Path] = []
    for pattern in patterns:
        iterator = root.rglob(pattern) if recursive else root.glob(pattern)
        for path in iterator:
            if not path.is_file():
                continue
            if path.name.endswith(sentinel_ext):
                continue
            files.append(path)
    return sorted(set(files))


def scan_ready_files(
    files: Iterable[Path],
    *,
    sentinel_ext: str = ".ready",
    require_sentinel: bool = True,
    stability_window_seconds: int = 0,
) -> ReadyScanResult:
    """Scan files and separate ready vs pending items."""
    ready = []
    pending = []
    missing: list[Path] = []
    unstable: list[Path] = []
    for path in files:
        result = evaluate_readiness(
            [path],
            sentinel_ext=sentinel_ext,
            require_sentinel=require_sentinel,
            stability_window_seconds=stability_window_seconds,
        )
        if result.ready:
            ready.append(path)
        else:
            pending.append(path)
        missing.extend(result.missing_sentinels)
        unstable.extend(result.unstable_files)
    return ReadyScanResult(
        ready_files=ready,
        pending_files=pending,
        missing_sentinels=missing,
        unstable_files=unstable,
    )


def poll_ready_files(
    root: Path,
    *,
    file_glob: str | Sequence[str] = "*",
    sentinel_ext: str = ".ready",
    require_sentinel: bool = True,
    stability_window_seconds: int = 0,
    recursive: bool = True,
    poll_interval_seconds: float = 1.0,
    max_loops: int = 1,
    sleep_fn: Optional[Callable[[float], None]] = None,
) -> ReadyScanResult:
    """Poll ingestion root until ready files are detected or loops exhausted."""
    sleep = sleep_fn or time.sleep
    result = ReadyScanResult()
    loops = max(1, max_loops)
    for _ in range(loops):
        files = list_ingestion_files(
            root,
            file_glob=file_glob,
            sentinel_ext=sentinel_ext,
            recursive=recursive,
        )
        result = scan_ready_files(
            files,
            sentinel_ext=sentinel_ext,
            require_sentinel=require_sentinel,
            stability_window_seconds=stability_window_seconds,
        )
        if result.ready:
            return result
        sleep(poll_interval_seconds)
    return result


def watch_ready_files(
    root: Path,
    *,
    file_glob: str | Sequence[str] = "*",
    sentinel_ext: str = ".ready",
    require_sentinel: bool = True,
    stability_window_seconds: int = 0,
    poll_interval_seconds: float = 1.0,
    recursive: bool = True,
    max_events: int = 25,
    use_watchfiles: bool = True,
) -> ReadyScanResult:
    """Watch for ingestion events and return first ready scan result."""
    if not use_watchfiles:
        return poll_ready_files(
            root,
            file_glob=file_glob,
            sentinel_ext=sentinel_ext,
            require_sentinel=require_sentinel,
            stability_window_seconds=stability_window_seconds,
            recursive=recursive,
            poll_interval_seconds=poll_interval_seconds,
            max_loops=max_events,
        )

    try:
        from watchfiles import watch
    except ImportError:
        return poll_ready_files(
            root,
            file_glob=file_glob,
            sentinel_ext=sentinel_ext,
            require_sentinel=require_sentinel,
            stability_window_seconds=stability_window_seconds,
            recursive=recursive,
            poll_interval_seconds=poll_interval_seconds,
            max_loops=max_events,
        )

    result = ReadyScanResult()
    for idx, _ in enumerate(watch(root)):
        files = list_ingestion_files(
            root,
            file_glob=file_glob,
            sentinel_ext=sentinel_ext,
            recursive=recursive,
        )
        result = scan_ready_files(
            files,
            sentinel_ext=sentinel_ext,
            require_sentinel=require_sentinel,
            stability_window_seconds=stability_window_seconds,
        )
        if result.ready:
            return result
        if idx + 1 >= max_events:
            break
    return result


class IngestionLedger:
    """Track seen hashes to prevent duplicate ingestion."""

    def __init__(self, ledger_path: Path) -> None:
        self.path = ledger_path
        self.data: dict[str, Any] = {"entries": {}}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self.data = {"entries": {}}

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def _key(self, hash_value: str, route_id: Optional[str]) -> str:
        return f"{route_id}:{hash_value}" if route_id else hash_value

    def is_duplicate(self, hash_value: str, *, route_id: Optional[str] = None) -> bool:
        entries = self.data.get("entries", {})
        return self._key(hash_value, route_id) in entries

    def record(
        self, hash_value: str, info: dict[str, Any], *, route_id: Optional[str] = None
    ) -> None:
        entries = self.data.setdefault("entries", {})
        entries[self._key(hash_value, route_id)] = {
            "info": info,
            "recorded_at": _timestamp(),
            "route_id": route_id,
        }
        self.save()


@dataclass(frozen=True)
class QueueEntry:
    path: Path
    route_id: Optional[str] = None
    ingestion_root: Optional[Path] = None


class IngestionQueue:
    """Persistent queue for ingestion dispatch."""

    def __init__(self, queue_path: Path) -> None:
        self.path = queue_path
        self.data: dict[str, Any] = {"entries": {}}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self.data = {"entries": {}}

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def entries(self) -> dict[str, Any]:
        return self.data.setdefault("entries", {})

    def enqueue(
        self,
        paths: Iterable[Path],
        *,
        route_id: Optional[str] = None,
        ingestion_root: Optional[Path] = None,
    ) -> None:
        entries = self.entries()
        for path in paths:
            key = str(path)
            if key in entries:
                entry = entries[key]
                if route_id and not entry.get("route_id"):
                    entry["route_id"] = route_id
                if ingestion_root and not entry.get("ingestion_root"):
                    entry["ingestion_root"] = str(ingestion_root)
                continue
            entries[key] = {
                "status": "pending",
                "added_at": _timestamp(),
                "last_error": None,
                "route_id": route_id,
                "ingestion_root": str(ingestion_root) if ingestion_root else None,
            }
        self.save()

    def enqueue_entries(self, entries: Iterable[QueueEntry]) -> None:
        for entry in entries:
            self.enqueue(
                [entry.path],
                route_id=entry.route_id,
                ingestion_root=entry.ingestion_root,
            )

    def pending_without_route(self) -> list[Path]:
        return [
            Path(path)
            for path, data in self.entries().items()
            if data.get("status") == "pending" and not data.get("route_id")
        ]

    def pending_entries(
        self, *, route_id: Optional[str] = None, include_unassigned: bool = False
    ) -> list[QueueEntry]:
        pending: list[QueueEntry] = []
        for path_str, data in self.entries().items():
            if data.get("status") != "pending":
                continue
            entry_route = data.get("route_id")
            if route_id is not None:
                if entry_route is None and not include_unassigned:
                    continue
                if entry_route is not None and entry_route != route_id:
                    continue
            ingestion_root = data.get("ingestion_root")
            pending.append(
                QueueEntry(
                    path=Path(path_str),
                    route_id=entry_route,
                    ingestion_root=Path(ingestion_root)
                    if isinstance(ingestion_root, str)
                    else None,
                )
            )
        return pending

    def pending(self) -> list[Path]:
        return [entry.path for entry in self.pending_entries()]

    def mark_processed(self, path: Path) -> None:
        key = str(path)
        entry = self.entries().setdefault(key, {})
        entry["status"] = "processed"
        entry["processed_at"] = _timestamp()
        entry.pop("last_error", None)
        self.save()

    def mark_failed(self, path: Path, error: str) -> None:
        key = str(path)
        entry = self.entries().setdefault(key, {})
        entry["status"] = "failed"
        entry["last_error"] = error
        entry["failed_at"] = _timestamp()
        self.save()
