"""Helpers for working with bundled and remote processing blocks."""

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

RAW_BASE = "https://raw.githubusercontent.com/cincibrainlab/autocleaneeg-task-registry/main"
CACHE_ROOT = Path.home() / ".config" / "autocleaneeg" / ".block_cache"

MANIFEST_NAME = "manifest.json"
DEFAULT_TIMEOUT = 5  # seconds
TIMEOUT_ENV = "AUTOCLEANEEG_BLOCK_REGISTRY_TIMEOUT"


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


def _hash_directory(directory: Path) -> Optional[Dict[str, str]]:
    """Return SHA256 hashes for all files in a directory structure."""
    if not directory.exists() or not directory.is_dir():
        return None

    hashes = {}
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file() and not file_path.name.startswith("."):
            relative_path = file_path.relative_to(directory)
            hashes[str(relative_path)] = _hash_file(file_path) or ""
    return hashes


class CacheManifest:
    """Simple helper to persist cache metadata for the block registry."""

    def __init__(self, manifest_path: Path) -> None:
        self.path = manifest_path
        self.data: Dict[str, object] = {
            "registry_commit": None,
            "synced_at": None,
            "last_error": None,
            "blocks": {},
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

    def update_block(
        self,
        block_name: str,
        *,
        category: str,
        path: str,
        cache_path: Path,
        source: str = "cache",
        source_commit: Optional[str] = None,
    ) -> None:
        blocks: Dict[str, object] = self.data.setdefault("blocks", {})  # type: ignore[assignment]
        block_data: Dict[str, object] = {
            "category": category,
            "path": path,
            "hashes": _hash_directory(cache_path),
            "fetched_at": _timestamp(),
            "source": source,
        }

        # Store the commit hash this block came from (for reproducibility)
        if source_commit:
            block_data["source_commit"] = source_commit
        elif source == "cache":
            # Use registry commit if available
            block_data["source_commit"] = self.data.get("registry_commit")

        blocks[block_name] = block_data
        self.save()

    def update_block_remote(
        self, block_name: str, *, category: str, path: str, remote_hashes: Optional[Dict[str, str]]
    ) -> None:
        blocks: Dict[str, object] = self.data.setdefault("blocks", {})  # type: ignore[assignment]
        rec: Dict[str, object] = blocks.get(block_name, {})  # type: ignore[assignment]
        rec.update({
            "category": category,
            "path": path,
            "remote_hashes": remote_hashes,
            "last_seen_commit": self.data.get("registry_commit"),
        })
        blocks[block_name] = rec
        self.save()

    def block_hashes(self, block_name: str) -> Optional[Dict[str, str]]:
        blocks: Dict[str, Dict[str, object]] = self.data.get("blocks", {})  # type: ignore[assignment]
        record = blocks.get(block_name)
        if not record:
            return None
        return record.get("hashes")  # type: ignore[return-value]

    def block_record(self, block_name: str) -> Optional[Dict[str, object]]:
        blocks: Dict[str, Dict[str, object]] = self.data.get("blocks", {})  # type: ignore[assignment]
        return blocks.get(block_name)

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
class ProcessingBlock:
    """Description of a processing block available to the pipeline."""

    name: str
    category: str
    path: str
    version: str
    description: str


class BlockRegistry:
    """Access blocks hosted in the public registry with offline fallback."""

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
            self.timeout = float(timeout or (float(env_timeout) if env_timeout else DEFAULT_TIMEOUT))
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
        req = Request(url, headers={"User-Agent": "autocleaneeg-block-registry/1.0"})
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
        return self.cache_root / "blocks_registry.json"

    def _pkg_root(self):
        return files("autoclean.blocks")

    def _pkg_index_text(self) -> str:
        resource = self._pkg_root().joinpath("registry.json")
        with resource.open("r", encoding="utf-8") as fh:
            return fh.read()

    def _pkg_registry_commit(self) -> str:
        """Get the commit hash from the bundled registry."""
        try:
            index = json.loads(self._pkg_index_text())
            return index.get("commit", "bundled-unknown")
        except Exception:
            return "bundled-unknown"

    def update_cache(self, *, allow_network: bool = True) -> str:
        """Fetch the registry index from GitHub into the local cache."""
        if not allow_network:
            commit = self.manifest.registry_commit or "not yet synced"
            return (
                "Skipped online check (offline mode). "
                f"Using cached Block Library version {commit}."
            )

        url = f"{self.raw_base}/blocks_registry.json"
        try:
            # Keep a snapshot of known blocks before update
            before_records: Dict[str, Dict[str, object]] = (
                self.manifest.data.get("blocks", {})  # type: ignore[assignment]
                if isinstance(self.manifest.data.get("blocks"), dict)
                else {}
            )

            self._download_to(url, self._cache_index_path())
            index = json.loads(self._cache_index_path().read_text(encoding="utf-8"))
            commit = index.get("commit", "unknown")

            # Compute diff of names
            names_now = [e.get("name") for e in index.get("blocks", [])]
            names_now = [n for n in names_now if isinstance(n, str)]
            names_before = list(before_records.keys())
            new = sorted([n for n in names_now if n not in names_before])
            removed = sorted([n for n in names_before if n not in names_now])

            # For each block in index, compute remote hashes for all files
            updated: List[str] = []
            for entry in index.get("blocks", []):
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                category = entry.get("category")
                path = entry.get("path")
                if not isinstance(name, str) or not isinstance(path, str):
                    continue
                if not isinstance(category, str):
                    category = "unknown"

                # Download all files in block directory to compute hashes
                remote_hashes: Dict[str, str] = {}
                block_files = ["mixin.py", "manifest.json", "README.md"]
                # Check if algorithm.py exists (some blocks have it)
                try:
                    test_url = f"{self.raw_base}/{path}/algorithm.py"
                    self._fetch_bytes(test_url)
                    block_files.append("algorithm.py")
                except (URLError, HTTPError):
                    pass  # algorithm.py doesn't exist for this block

                for filename in block_files:
                    try:
                        file_url = f"{self.raw_base}/{path}/{filename}"
                        data = self._fetch_bytes(file_url)
                        remote_hashes[filename] = hashlib.sha256(data).hexdigest()
                    except (URLError, HTTPError, socket.timeout):  # pragma: no cover - network
                        remote_hashes[filename] = ""

                # Compare with prior remote hashes if available
                before = before_records.get(name) or {}
                before_hashes = before.get("remote_hashes") if isinstance(before, dict) else None
                if remote_hashes and before_hashes and isinstance(before_hashes, dict):
                    # Check if any file changed
                    if remote_hashes != before_hashes:
                        updated.append(name)

                # Persist the latest remote hashes in the manifest
                self.manifest.update_block_remote(
                    name, category=category, path=path, remote_hashes=remote_hashes
                )

            # Record success and summarize
            self.manifest.record_success(commit)
            self._last_update_summary = {
                "new": new,
                "updated": sorted(updated),
                "removed": removed,
            }
            label = commit if commit not in ("unknown", "") else "latest"
            return f"Block Library refreshed (version {label})."
        except (URLError, HTTPError) as exc:
            message = str(exc)
            self.manifest.record_error(message)
            return (
                "Could not reach the Block Library online. "
                f"Using bundled blocks ({message})."
            )
        except Exception as exc:  # pragma: no cover - defensive
            message = str(exc)
            self.manifest.record_error(message)
            raise

    def _load_index(self) -> Dict[str, object]:
        """Load registry by merging bundled and cache (cache overlays bundled)."""
        # Always start with bundled blocks (guaranteed to have all shipped blocks)
        bundled_index = json.loads(self._pkg_index_text())
        bundled_blocks = {b["name"]: b for b in bundled_index.get("blocks", [])}

        # Overlay cache blocks if available (may have updates from remote)
        cache_index_path = self._cache_index_path()
        if cache_index_path.exists():
            try:
                cache_index = json.loads(cache_index_path.read_text(encoding="utf-8"))
                cache_blocks = {b["name"]: b for b in cache_index.get("blocks", [])}
                # Merge: bundled + cache (cache wins for same name)
                bundled_blocks.update(cache_blocks)
                # Use cache metadata (commit, version)
                return {
                    "version": cache_index.get("version", bundled_index.get("version")),
                    "commit": cache_index.get("commit", bundled_index.get("commit")),
                    "blocks": list(bundled_blocks.values()),
                }
            except (json.JSONDecodeError, OSError):
                # Corrupt cache - use bundled
                pass

        return bundled_index

    # ---------------- Query helpers -----------------
    def _get_actual_block_version(self, block_name: str, category: str, block_path: str) -> str:
        """Get the version from the actual manifest.json file that will be used at runtime.

        Checks in priority order: cache first, then bundled (matching mixin loading priority).

        Args:
            block_name: Block name (e.g., "fooof_periodic")
            category: Block category (e.g., "analysis")
            block_path: Relative path to bundled block (e.g., "blocks/analysis/fooof_analysis")
        """
        # Check cache first (highest priority, like mixin loading)
        # Cache uses category/name structure
        cache_manifest = self.cache_root / category / block_name / "manifest.json"
        if cache_manifest.exists():
            try:
                with cache_manifest.open("r", encoding="utf-8") as f:
                    manifest = json.load(f)
                    return manifest.get("version", "unknown")
            except (json.JSONDecodeError, OSError):
                pass

        # Fallback to bundled
        # Bundled uses path from registry (includes "blocks/" prefix)
        bundled_manifest = Path(__file__).parent.parent / block_path / "manifest.json"
        if bundled_manifest.exists():
            try:
                with bundled_manifest.open("r", encoding="utf-8") as f:
                    manifest = json.load(f)
                    return manifest.get("version", "unknown")
            except (json.JSONDecodeError, OSError):
                pass

        return "unknown"

    def list_blocks(self) -> List[ProcessingBlock]:
        index = self._load_index()
        blocks = []
        for entry in index.get("blocks", []):  # type: ignore[arg-type]
            name = entry["name"]
            category = entry.get("category", "unknown")
            path = entry["path"]

            # Get actual version from the file that will be used at runtime
            actual_version = self._get_actual_block_version(name, category, path)

            blocks.append(ProcessingBlock(
                name=name,
                category=category,
                path=path,
                version=actual_version,  # Use actual version, not registry metadata
                description=entry.get("description", ""),
            ))
        return blocks

    def get_block(self, block_name: str) -> Optional[ProcessingBlock]:
        return next((block for block in self.list_blocks() if block.name == block_name), None)

    def _cache_path_for(self, rel_path: str) -> Path:
        return self.cache_root / rel_path

    def block_sync_status(self, block_name: str) -> Dict[str, Optional[str]]:
        """Return sync metadata for a block.

        The returned dictionary contains keys: ``status`` (synced/outdated/cache_only/
        bundled_only/missing), ``cache_hashes``, ``bundled_hashes``, ``source``.
        """
        block = self.get_block(block_name)
        if not block:
            return {
                "status": "missing",
                "cache_hashes": None,
                "bundled_hashes": None,
                "source": None,
            }

        record = self.manifest.block_record(block.name) or {}
        cache_hashes = record.get("hashes") if isinstance(record, dict) else None
        if cache_hashes is not None and not isinstance(cache_hashes, dict):
            cache_hashes = None

        source = record.get("source") if isinstance(record, dict) else None
        if source is not None and not isinstance(source, str):
            source = None

        cache_path = self._cache_path_for(block.path)
        if cache_hashes is None and cache_path.exists():
            cache_hashes = _hash_directory(cache_path)
            if not source:
                source = "cache"

        # Check bundled blocks
        bundled_path = Path(__file__).parent.parent / "blocks" / block.category / block.name
        bundled_hashes = _hash_directory(bundled_path) if bundled_path.exists() else None

        status: str
        if cache_hashes and bundled_hashes:
            if cache_hashes == bundled_hashes:
                status = "synced"
            else:
                status = "outdated"
        elif cache_hashes and not bundled_hashes:
            status = "cache_only"
        elif bundled_hashes and not cache_hashes:
            status = "bundled_only"
        else:
            status = "missing"

        return {
            "status": status,
            "cache_hashes": cache_hashes,
            "bundled_hashes": bundled_hashes,
            "source": source or self.block_source(block.name),
        }

    def block_source(self, block_name: str) -> str:
        """Determine which source will be used at runtime (matches mixin loading priority).

        Returns: "cache", "bundled", or "missing"
        """
        block = self.get_block(block_name)
        if not block:
            return "missing"

        # Check cache first (highest priority, like mixin loading)
        # Cache uses category/name structure
        cache_dir = self.cache_root / block.category / block.name
        if cache_dir.exists() and cache_dir.is_dir():
            # Verify it has a mixin.py file (actually loadable)
            if (cache_dir / "mixin.py").exists():
                return "cache"

        # Check bundled second (fallback)
        # Bundled uses path from registry
        bundled_dir = Path(__file__).parent.parent / block.path
        if bundled_dir.exists() and bundled_dir.is_dir():
            # Verify it has a mixin.py file (actually loadable)
            if (bundled_dir / "mixin.py").exists():
                return "bundled"

        return "missing"

    # ---------------- Materialization -----------------
    def materialize_block_to(self, block_name: str, dest_dir: Path, commit: Optional[str] = None) -> Path:
        """Download and cache a block from the remote registry.

        Parameters
        ----------
        block_name : str
            Name of the block to download
        dest_dir : Path
            Destination directory for the block
        commit : str, optional
            Specific git commit hash to fetch from. If provided, overrides
            the default (registry HEAD). For reproducibility: browse GitHub
            to find the commit you want, then specify it here.
        """
        block = self.get_block(block_name)
        if not block:
            raise ValueError(f"Processing block '{block_name}' not found in registry.")

        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / block.category / block.name

        # Determine which commit to use
        fetch_commit = commit if commit else self.manifest.registry_commit

        # Build base URL with commit (if fetching from GitHub)
        # Format: https://raw.githubusercontent.com/org/repo/COMMIT_HASH/path
        if commit:
            # User specified exact commit - construct URL with that commit
            base_parts = self.raw_base.split("/")
            # raw_base format: https://raw.githubusercontent.com/org/repo/branch
            # We want: https://raw.githubusercontent.com/org/repo/COMMIT_HASH
            if len(base_parts) >= 5:
                base_url = "/".join(base_parts[:5]) + f"/{commit}"
            else:
                base_url = f"{self.raw_base}/{commit}"
        else:
            # Use default (main/HEAD)
            base_url = self.raw_base

        # Try to download from remote
        block_files = ["mixin.py", "manifest.json", "README.md"]
        # Check for algorithm.py
        try:
            test_url = f"{base_url}/{block.path}/algorithm.py"
            self._fetch_bytes(test_url)
            block_files.append("algorithm.py")
        except (URLError, HTTPError):
            pass

        # Additional files that might exist
        optional_files = ["reporting.py", "__init__.py"]

        try:
            dest_path.mkdir(parents=True, exist_ok=True)
            for filename in block_files:
                file_url = f"{base_url}/{block.path}/{filename}"
                self._download_to(file_url, dest_path / filename)

            # Try optional files
            for filename in optional_files:
                try:
                    file_url = f"{base_url}/{block.path}/{filename}"
                    self._download_to(file_url, dest_path / filename)
                except (URLError, HTTPError):
                    pass  # Optional file doesn't exist

            self.manifest.update_block(
                block.name,
                category=block.category,
                path=block.path,
                cache_path=dest_path,
                source="cache",
                source_commit=fetch_commit,
            )
            return dest_path
        except (URLError, HTTPError):
            pass  # Fall back to bundled

        # Try bundled blocks
        bundled_path = Path(__file__).parent.parent / "blocks" / block.category / block.name
        if bundled_path.exists() and bundled_path.is_dir():
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(bundled_path, dest_path, dirs_exist_ok=True)
            # Bundled blocks don't have a specific commit - they're versioned with the package
            bundled_commit = self._pkg_registry_commit()
            self.manifest.update_block(
                block.name,
                category=block.category,
                path=block.path,
                cache_path=dest_path,
                source="bundled",
                source_commit=bundled_commit,
            )
            return dest_path

        raise FileNotFoundError(
            f"Block '{block.name}' not found in remote registry or bundled blocks"
        )

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

    def iter_sources(self, blocks: Iterable[ProcessingBlock]) -> List[tuple[str, str]]:
        """Return (block_name, source) pairs for display purposes."""
        return [(block.name, self.block_source(block.name)) for block in blocks]
