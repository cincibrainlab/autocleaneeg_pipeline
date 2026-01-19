"""Ingestion utilities for automation readiness and provenance."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

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
) -> dict[str, Any]:
    """Create receipt payload without writing to disk."""
    file_entries = [_file_entry(path) for path in files]
    hash_value = compute_provenance_hash(relative_path, metadata)
    return {
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
) -> dict[str, Any]:
    """Create provenance folder, write receipt, and optionally record ledger."""
    folder, hash_value = resolve_provenance_folder(root, relative_path, metadata)
    receipt = build_receipt(
        folder=folder,
        relative_path=relative_path,
        metadata=metadata,
        files=files,
        status=status,
    )
    write_receipt(folder, receipt)

    duplicate = False
    if ledger is not None:
        duplicate = ledger.is_duplicate(hash_value)
        if not duplicate:
            ledger.record(
                hash_value,
                {
                    "relative_path": relative_path.as_posix(),
                    "folder": str(folder),
                },
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


def resolve_ingestion_roots(config: dict[str, Any], workspace_dir: Path) -> list[Path]:
    """Resolve ingestion roots from serve config."""
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
    ingestion_root: Path
    ready: "ReadyScanResult"
    plan: Optional[DispatchPlan]
    result: Optional[DispatchResult]


def dispatch_ready_ingestion(
    *,
    config_path: Path,
    workspace_dir: Path,
    ingestion_root: Path,
    file_glob: str = "*",
    sentinel_ext: str = ".ready",
    require_sentinel: bool = True,
    stability_window_seconds: int = 0,
    use_watchfiles: bool = True,
    max_events: int = 1,
    automation: bool = True,
    yes: bool = True,
    max_attempts: int = 1,
    runner: Optional[Callable[[list[str]], None]] = None,
    config: Optional[dict[str, Any]] = None,
) -> IngestionDispatchResult:
    """Dispatch ready ingestion files using serve configuration."""
    config_data = config or load_serve_config(config_path)
    ingestion_paths = resolve_ingestion_roots(config_data, workspace_dir)

    if ingestion_paths and ingestion_root.resolve() not in ingestion_paths:
        raise ValueError("ingestion_root is not listed in ingestion_folders")

    ready = watch_ready_files(
        ingestion_root,
        file_glob=file_glob,
        sentinel_ext=sentinel_ext,
        require_sentinel=require_sentinel,
        stability_window_seconds=stability_window_seconds,
        max_events=max_events,
        use_watchfiles=use_watchfiles,
    )
    if not ready.ready:
        return IngestionDispatchResult(
            ingestion_root=ingestion_root, ready=ready, plan=None, result=None
        )

    plan = build_dispatch_plan(
        config=config_data,
        workspace_dir=workspace_dir,
        files=ready.ready_files,
    )
    result = run_dispatch_plan(
        plan,
        automation=automation,
        yes=yes,
        max_attempts=max_attempts,
        runner=runner,
    )
    return IngestionDispatchResult(
        ingestion_root=ingestion_root, ready=ready, plan=plan, result=result
    )


@dataclass
class IngestionLoopResult:
    iterations: int
    dispatch_results: list[IngestionDispatchResult]
    pending_roots: list[Path]


def run_ingestion_loop(
    *,
    config_path: Path,
    workspace_dir: Path,
    max_cycles: int = 1,
    file_glob: str = "*",
    sentinel_ext: str = ".ready",
    require_sentinel: bool = True,
    stability_window_seconds: int = 0,
    use_watchfiles: bool = True,
    max_events: int = 1,
    automation: bool = True,
    yes: bool = True,
    max_attempts: int = 1,
    runner: Optional[Callable[[list[str]], None]] = None,
    sleep_fn: Optional[Callable[[float], None]] = None,
    sleep_seconds: float = 1.0,
) -> IngestionLoopResult:
    """Run readiness/dispatch loop over all ingestion roots."""
    if max_cycles < 1:
        raise ValueError("max_cycles must be >= 1")
    config = load_serve_config(config_path)
    roots = resolve_ingestion_roots(config, workspace_dir)
    if not roots:
        raise ValueError("No ingestion_folders configured")
    dispatch_results: list[IngestionDispatchResult] = []
    sleep = sleep_fn or time.sleep
    pending_roots = list(roots)

    for cycle in range(max_cycles):
        ready_roots: list[Path] = []
        for root in roots:
            result = dispatch_ready_ingestion(
                config_path=config_path,
                workspace_dir=workspace_dir,
                ingestion_root=root,
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
            )
            dispatch_results.append(result)
            if result.ready.ready:
                ready_roots.append(root)
        pending_roots = [root for root in roots if root not in ready_roots]
        if not ready_roots:
            return IngestionLoopResult(
                iterations=cycle + 1,
                dispatch_results=dispatch_results,
                pending_roots=pending_roots,
            )
        if cycle < max_cycles - 1:
            sleep(sleep_seconds)

    return IngestionLoopResult(
        iterations=max_cycles,
        dispatch_results=dispatch_results,
        pending_roots=pending_roots,
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
    root: Path, *, file_glob: str, sentinel_ext: str
) -> list[Path]:
    """Return candidate ingestion files under root."""
    files: list[Path] = []
    for path in root.rglob(file_glob):
        if not path.is_file():
            continue
        if path.name.endswith(sentinel_ext):
            continue
        files.append(path)
    return sorted(files)


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
    file_glob: str = "*",
    sentinel_ext: str = ".ready",
    require_sentinel: bool = True,
    stability_window_seconds: int = 0,
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
            root, file_glob=file_glob, sentinel_ext=sentinel_ext
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
    file_glob: str = "*",
    sentinel_ext: str = ".ready",
    require_sentinel: bool = True,
    stability_window_seconds: int = 0,
    poll_interval_seconds: float = 1.0,
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
            poll_interval_seconds=poll_interval_seconds,
            max_loops=max_events,
        )

    result = ReadyScanResult()
    for idx, _ in enumerate(watch(root)):
        files = list_ingestion_files(
            root, file_glob=file_glob, sentinel_ext=sentinel_ext
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

    def is_duplicate(self, hash_value: str) -> bool:
        entries = self.data.get("entries", {})
        return hash_value in entries

    def record(self, hash_value: str, info: dict[str, Any]) -> None:
        entries = self.data.setdefault("entries", {})
        entries[hash_value] = {"info": info, "recorded_at": _timestamp()}
        self.save()
