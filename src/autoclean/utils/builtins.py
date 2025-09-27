"""Helpers for working with bundled and remote built-in tasks."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:  # Python 3.10+ provides importlib.resources.files
    from importlib.resources import files
except ImportError:  # pragma: no cover - fallback for very old Python
    from importlib_resources import files  # type: ignore

RAW_BASE = "https://raw.githubusercontent.com/cincibrainlab/autoclean-builtins/main"
CACHE_ROOT = Path.home() / ".config" / "autocleaneeg" / ".builtin_cache"


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
    ) -> None:
        self.raw_base = raw_base.rstrip("/")
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)

    # ---------------- Remote fetch helpers -----------------
    def _fetch_bytes(self, url: str) -> bytes:
        req = Request(url, headers={"User-Agent": "autocleaneeg-builtins/1.0"})
        with urlopen(req) as resp:  # type: ignore[call-arg]
            return resp.read()

    def _download_to(self, url: str, dest: Path) -> None:
        data = self._fetch_bytes(url)
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

    def update_cache(self) -> str:
        """Fetch the registry index from GitHub into the local cache."""
        url = f"{self.raw_base}/registry.json"
        try:
            self._download_to(url, self._cache_index_path())
            index = json.loads(self._cache_index_path().read_text(encoding="utf-8"))
            commit = index.get("commit", "unknown")
            return f"Updated built-ins index (commit={commit})"
        except (URLError, HTTPError) as exc:
            return (
                "Could not update built-ins index from GitHub. "
                f"Using packaged fallback ({exc})."
            )

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
        return next((task for task in self.list_tasks() if task.name == task_name), None)

    def _cache_path_for(self, rel_path: str) -> Path:
        return self.cache_root / rel_path

    def task_source(self, task_name: str) -> str:
        task = self.get_task(task_name)
        if not task:
            return "missing"

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
            shutil.copy2(cache_target, dest_path)
            return dest_path
        except (URLError, HTTPError):
            pass

        if cache_target.exists():
            cache_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cache_target, dest_path)
            return dest_path

        resource = self._pkg_task_resource(task.path)
        if not resource.is_file():
            raise FileNotFoundError(
                f"Packaged fallback for '{task.name}' not found at {task.path}"
            )

        with resource.open("rb") as src, dest_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return dest_path

    # Convenience helpers for CLI output ---------------------------------
    def iter_sources(self, tasks: Iterable[BuiltinTask]) -> List[tuple[str, str]]:
        """Return (task_name, source) pairs for display purposes."""
        return [(task.name, self.task_source(task.name)) for task in tasks]
