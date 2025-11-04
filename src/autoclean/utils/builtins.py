"""Helpers for working with bundled and remote built-in tasks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:  # Python 3.10+ provides importlib.resources.files
    from importlib.resources import files
except ImportError:  # pragma: no cover - fallback for very old Python
    from importlib_resources import files  # type: ignore

RAW_BASE = (
    "https://raw.githubusercontent.com/cincibrainlab/autocleaneeg-task-registry/main"
)
CACHE_ROOT = Path.home() / ".config" / "autocleaneeg" / ".builtin_cache"

MANIFEST_NAME = "manifest.json"
DEFAULT_TIMEOUT = 5  # seconds
TIMEOUT_ENV = "AUTOCLEANEEG_TASK_REGISTRY_TIMEOUT"


def _timestamp() -> str:
    """Return current UTC timestamp in ISO format."""

    return datetime.now(timezone.utc).isoformat()


def _hash_file(path: Path) -> Optional[str]:
    """Return SHA256 hash of a file, or None if the path is missing."""

    if not path.exists():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CacheManifest:
    """Simple helper to persist cache metadata for the built-in registry."""

    def __init__(self, manifest_path: Path) -> None:
        self.path = manifest_path
        self.data: Dict[str, object] = {
            "registry_commit": None,
            "synced_at": None,
            "last_error": None,
            "tasks": {},
        }
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # Corrupt manifest; keep defaults but rename the bad file so we can recover.
            with suppress(OSError):
                self.path.rename(self.path.with_suffix(".broken"))

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    # Convenience mutation helpers -------------------------------------------------
    def record_success(self, commit: str) -> None:
        self.data.update(
            {
                "registry_commit": commit,
                "synced_at": _timestamp(),
                "last_error": None,
            }
        )
        self.save()

    def record_error(self, message: str) -> None:
        self.data["last_error"] = {
            "message": message,
            "timestamp": _timestamp(),
        }
        self.save()

    def update_task(
        self,
        task_name: str,
        *,
        path: str,
        cache_path: Path,
        source: str = "cache",
    ) -> None:
        tasks: Dict[str, object] = self.data.setdefault("tasks", {})  # type: ignore[assignment]
        tasks[task_name] = {
            "path": path,
            "hash": _hash_file(cache_path),
            "fetched_at": _timestamp(),
            "source": source,
        }
        self.save()

    def update_task_remote(
        self, task_name: str, *, path: str, remote_hash: Optional[str]
    ) -> None:
        tasks: Dict[str, object] = self.data.setdefault("tasks", {})  # type: ignore[assignment]
        rec: Dict[str, object] = tasks.get(task_name, {})  # type: ignore[assignment]
        rec.update(
            {
                "path": path,
                "remote_hash": remote_hash,
                "last_seen_commit": self.data.get("registry_commit"),
            }
        )
        tasks[task_name] = rec
        self.save()

    def task_hash(self, task_name: str) -> Optional[str]:
        tasks: Dict[str, Dict[str, object]] = self.data.get("tasks", {})  # type: ignore[assignment]
        record = tasks.get(task_name)
        if not record:
            return None
        return record.get("hash")  # type: ignore[return-value]

    def task_record(self, task_name: str) -> Optional[Dict[str, object]]:
        tasks: Dict[str, Dict[str, object]] = self.data.get("tasks", {})  # type: ignore[assignment]
        return tasks.get(task_name)

    # Read helpers -----------------------------------------------------------------
    @property
    def registry_commit(self) -> Optional[str]:
        commit = self.data.get("registry_commit")
        return commit if isinstance(commit, str) else None

    @property
    def synced_at(self) -> Optional[str]:
        synced_at = self.data.get("synced_at")
        return synced_at if isinstance(synced_at, str) else None

    @property
    def last_error(self) -> Optional[Dict[str, str]]:
        error = self.data.get("last_error")
        return error if isinstance(error, dict) else None


@dataclass(frozen=True)
class BuiltinTask:
    """Description of a built-in task available to the CLI."""

    name: str
    path: str


class BuiltinRegistry:
    """Access tasks hosted in the public registry with offline fallback."""

    def __init__(
        self,
        raw_base: str = RAW_BASE,
        cache_root: Path = CACHE_ROOT,
        *,
        timeout: Optional[float] = None,
    ) -> None:
        self.raw_base = raw_base.rstrip("/")
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.manifest = CacheManifest(self.cache_root / MANIFEST_NAME)
        env_timeout = os.environ.get(TIMEOUT_ENV)
        try:
            self.timeout = float(
                timeout or (float(env_timeout) if env_timeout else DEFAULT_TIMEOUT)
            )
        except (TypeError, ValueError):
            self.timeout = DEFAULT_TIMEOUT
        # Last update diff snapshot for CLI rendering
        self._last_update_summary: Dict[str, List[str]] = {
            "new": [],
            "updated": [],
            "removed": [],
        }

    # ---------------- Remote fetch helpers -----------------
    def _fetch_bytes(self, url: str) -> bytes:
        req = Request(url, headers={"User-Agent": "autocleaneeg-task-registry/1.0"})
        try:
            with urlopen(req, timeout=self.timeout) as resp:  # type: ignore[call-arg]
                return resp.read()
        except socket.timeout as exc:  # pragma: no cover - network dependent
            raise URLError(f"Request timed out after {self.timeout}s") from exc

    def _download_to(self, url: str, dest: Path) -> None:
        cache_buster = _timestamp().replace(":", "-")
        separator = "&" if "?" in url else "?"
        data = self._fetch_bytes(f"{url}{separator}cb={cache_buster}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest.with_suffix(dest.suffix + ".tmp")
        tmp_path.write_bytes(data)
        tmp_path.replace(dest)

    # ---------------- Index handling -----------------
    def _cache_index_path(self) -> Path:
        return self.cache_root / "registry.json"

    def _pkg_root(self):
        return files("autoclean.data.builtins")

    def _pkg_index_text(self) -> str:
        resource = self._pkg_root().joinpath("registry.json")
        with resource.open("r", encoding="utf-8") as fh:
            return fh.read()

    def _pkg_task_resource(self, rel_path: str):
        return self._pkg_root().joinpath(rel_path)

    def update_cache(self, *, allow_network: bool = True) -> str:
        """Fetch the registry index from GitHub into the local cache."""

        if not allow_network:
            commit = self.manifest.registry_commit or "not yet synced"
            return (
                "Skipped online check (offline mode). "
                f"Using cached Task Library version {commit}."
            )

        url = f"{self.raw_base}/registry.json"
        try:
            # Keep a snapshot of known tasks before update
            before_records: Dict[str, Dict[str, object]] = (
                self.manifest.data.get("tasks", {})  # type: ignore[assignment]
                if isinstance(self.manifest.data.get("tasks"), dict)
                else {}
            )

            self._download_to(url, self._cache_index_path())
            index = json.loads(self._cache_index_path().read_text(encoding="utf-8"))
            commit = index.get("commit", "unknown")
            # Compute diff of names
            names_now = [e.get("name") for e in index.get("tasks", [])]
            names_now = [n for n in names_now if isinstance(n, str)]
            names_before = list(before_records.keys())
            new = sorted([n for n in names_now if n not in names_before])
            removed = sorted([n for n in names_before if n not in names_now])

            # For each task in index, compute remote hash (fast enough at current scale)
            updated: List[str] = []
            for entry in index.get("tasks", []):
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                path = entry.get("path")
                if not isinstance(name, str) or not isinstance(path, str):
                    continue
                remote_url = f"{self.raw_base}/{path}"
                remote_hash: Optional[str]
                try:
                    data = self._fetch_bytes(remote_url)
                    remote_hash = hashlib.sha256(data).hexdigest()
                except (
                    URLError,
                    HTTPError,
                    socket.timeout,
                ):  # pragma: no cover - network
                    remote_hash = None
                # Compare with prior remote hash if available
                before = before_records.get(name) or {}
                before_hash = (
                    before.get("remote_hash") if isinstance(before, dict) else None
                )
                if remote_hash and before_hash and isinstance(before_hash, str):
                    if remote_hash != before_hash:
                        updated.append(name)
                # Persist the latest remote hash in the manifest
                self.manifest.update_task_remote(
                    name, path=path, remote_hash=remote_hash
                )

            # Record success and summarize
            self.manifest.record_success(commit)
            self._last_update_summary = {
                "new": new,
                "updated": sorted(updated),
                "removed": removed,
            }
            label = commit if commit not in ("unknown", "") else "latest"
            return f"Task Library refreshed (version {label})."
        except (URLError, HTTPError) as exc:
            message = str(exc)
            self.manifest.record_error(message)
            return (
                "Could not reach the Task Library online. "
                f"Using bundled templates ({message})."
            )
        except Exception as exc:  # pragma: no cover - defensive
            message = str(exc)
            self.manifest.record_error(message)
            raise

    def _load_index(self) -> Dict[str, object]:
        cache_index = self._cache_index_path()
        if cache_index.exists():
            return json.loads(cache_index.read_text(encoding="utf-8"))
        return json.loads(self._pkg_index_text())

    # ---------------- Query helpers -----------------
    def list_tasks(self) -> List[BuiltinTask]:
        index = self._load_index()
        return [
            BuiltinTask(entry["name"], entry["path"])
            for entry in index.get("tasks", [])  # type: ignore[arg-type]
        ]

    def get_task(self, task_name: str) -> Optional[BuiltinTask]:
        return next(
            (task for task in self.list_tasks() if task.name == task_name), None
        )

    def _cache_path_for(self, rel_path: str) -> Path:
        return self.cache_root / rel_path

    def _workspace_task_path(
        self, task_name: str, workspace_dir: Optional[Path]
    ) -> Optional[Path]:
        if workspace_dir is None:
            return None
        candidate = workspace_dir / f"{task_name}.py"
        return candidate if candidate.exists() else None

    def task_sync_status(
        self, task_name: str, workspace_dir: Optional[Path] = None
    ) -> Dict[str, Optional[str]]:
        """Return sync metadata for a task.

        The returned dictionary contains keys: ``status`` (synced/modified/not_installed/
        unknown), ``cache_hash``, ``workspace_hash``, ``source``, and ``workspace_path``.
        """

        task = self.get_task(task_name)
        if not task:
            return {
                "status": "missing",
                "cache_hash": None,
                "workspace_hash": None,
                "source": None,
                "workspace_path": None,
            }

        record = self.manifest.task_record(task.name) or {}
        cache_hash = record.get("hash") if isinstance(record, dict) else None
        if cache_hash is not None and not isinstance(cache_hash, str):
            cache_hash = None

        source = record.get("source") if isinstance(record, dict) else None
        if source is not None and not isinstance(source, str):
            source = None

        cache_path = self._cache_path_for(task.path)
        if cache_hash is None and cache_path.exists():
            cache_hash = _hash_file(cache_path)
            if not source:
                source = "cache"

        workspace_path = self._workspace_task_path(task.name, workspace_dir)
        workspace_hash = _hash_file(workspace_path) if workspace_path else None

        status: str
        if workspace_path is None:
            status = "not_installed"
        elif cache_hash and workspace_hash == cache_hash:
            status = "synced"
        elif cache_hash and workspace_hash and workspace_hash != cache_hash:
            status = "modified"
        elif cache_hash is None:
            status = "unknown"
        else:
            status = "unknown"

        return {
            "status": status,
            "cache_hash": cache_hash,
            "workspace_hash": workspace_hash,
            "source": source or self.task_source(task.name),
            "workspace_path": str(workspace_path) if workspace_path else None,
        }

    def task_source(self, task_name: str) -> str:
        task = self.get_task(task_name)
        if not task:
            return "missing"

        record = self.manifest.task_record(task.name)
        if record:
            source = record.get("source")
            if isinstance(source, str):
                return source

        cached = self._cache_path_for(task.path)
        if cached.exists():
            return "cache"

        resource = self._pkg_task_resource(task.path)
        if resource.is_file():
            return "package"
        return "missing"

    # ---------------- Materialization -----------------
    def materialize_task_to(self, task_name: str, dest_dir: Path) -> Path:
        task = self.get_task(task_name)
        if not task:
            raise ValueError(f"Built-in task '{task_name}' not found in registry.")

        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{task.name}.py"

        remote_url = f"{self.raw_base}/{task.path}"
        cache_target = self._cache_path_for(task.path)

        try:
            self._download_to(remote_url, cache_target)
            self.manifest.update_task(
                task.name, path=task.path, cache_path=cache_target, source="cache"
            )
            shutil.copy2(cache_target, dest_path)
            return dest_path
        except (URLError, HTTPError):
            pass

        if cache_target.exists():
            cache_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cache_target, dest_path)
            self.manifest.update_task(
                task.name, path=task.path, cache_path=cache_target, source="cache"
            )
            return dest_path

        resource = self._pkg_task_resource(task.path)
        if not resource.is_file():
            raise FileNotFoundError(
                f"Packaged fallback for '{task.name}' not found at {task.path}"
            )

        cache_target.parent.mkdir(parents=True, exist_ok=True)
        with resource.open("rb") as src:
            data = src.read()
        cache_target.write_bytes(data)
        dest_path.write_bytes(data)
        self.manifest.update_task(
            task.name, path=task.path, cache_path=cache_target, source="package"
        )
        return dest_path

    # Convenience helpers for CLI output ---------------------------------
    def registry_status(self) -> Dict[str, Optional[str]]:
        """Return high-level registry sync information."""

        return {
            "commit": self.manifest.registry_commit,
            "synced_at": self.manifest.synced_at,
            "last_error": self.manifest.last_error,
        }

    def last_update_summary(self) -> Dict[str, List[str]]:
        """Return the last update summary (new/updated/removed names)."""
        return self._last_update_summary

    def iter_sources(self, tasks: Iterable[BuiltinTask]) -> List[tuple[str, str]]:
        """Return (task_name, source) pairs for display purposes."""
        return [(task.name, self.task_source(task.name)) for task in tasks]
