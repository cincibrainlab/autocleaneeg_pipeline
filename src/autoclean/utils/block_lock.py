"""Lock file management for reproducible block versioning.

This module provides functionality to create and use lock files that capture
the exact state of all processing blocks, enabling perfect reproducibility.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from autoclean.utils.block_registry import BlockRegistry, CACHE_ROOT


class BlockLockFile:
    """Manages lock files for processing blocks."""

    def __init__(self, lock_file_path: Path | str = "blocks.lock"):
        """Initialize lock file manager.

        Parameters
        ----------
        lock_file_path : Path or str
            Path to the lock file (default: "blocks.lock" in current directory)
        """
        self.lock_file_path = Path(lock_file_path)

    def generate(self, registry: Optional[BlockRegistry] = None) -> Dict:
        """Generate lock file from current cache state.

        Parameters
        ----------
        registry : BlockRegistry, optional
            Registry instance. If not provided, creates new one.

        Returns
        -------
        dict
            Lock file contents

        Examples
        --------
        >>> lock = BlockLockFile()
        >>> lock_data = lock.generate()
        >>> lock.save(lock_data)
        """
        if registry is None:
            registry = BlockRegistry(cache_root=CACHE_ROOT)

        # Get current registry commit
        registry_commit = registry.manifest.registry_commit or "unknown"

        # Get all blocks and their info from cache manifest
        blocks_data = {}
        for block in registry.list_blocks():
            record = registry.manifest.block_record(block.name)
            if record:
                blocks_data[block.name] = {
                    "commit": record.get("source_commit") or registry_commit,
                    "hash": self._compute_block_hash(record.get("hashes", {})),
                    "source": record.get("source", "unknown"),
                }

        lock_data = {
            "version": 1,
            "locked_at": datetime.now(timezone.utc).isoformat(),
            "registry_commit": registry_commit,
            "blocks": blocks_data,
        }

        return lock_data

    def _compute_block_hash(self, hashes: Dict[str, str]) -> str:
        """Compute combined hash from file hashes."""
        if not hashes:
            return "unknown"

        import hashlib

        # Create deterministic combined hash
        combined = "|".join(f"{k}:{v}" for k, v in sorted(hashes.items()))
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def save(self, lock_data: Dict) -> None:
        """Save lock file to disk.

        Parameters
        ----------
        lock_data : dict
            Lock file contents from generate()
        """
        self.lock_file_path.write_text(json.dumps(lock_data, indent=2) + "\n", encoding="utf-8")

    def load(self) -> Dict:
        """Load lock file from disk.

        Returns
        -------
        dict
            Lock file contents

        Raises
        ------
        FileNotFoundError
            If lock file doesn't exist
        ValueError
            If lock file is invalid JSON
        """
        if not self.lock_file_path.exists():
            raise FileNotFoundError(f"Lock file not found: {self.lock_file_path}")

        content = self.lock_file_path.read_text(encoding="utf-8")
        return json.loads(content)

    def install_from_lock(self, registry: Optional[BlockRegistry] = None) -> List[str]:
        """Install all blocks from lock file.

        Parameters
        ----------
        registry : BlockRegistry, optional
            Registry instance. If not provided, creates new one.

        Returns
        -------
        list of str
            List of installed block names

        Raises
        ------
        FileNotFoundError
            If lock file doesn't exist
        ValueError
            If lock file is invalid
        """
        lock_data = self.load()

        if lock_data.get("version") != 1:
            raise ValueError(f"Unsupported lock file version: {lock_data.get('version')}")

        if registry is None:
            registry = BlockRegistry(cache_root=CACHE_ROOT)

        installed = []
        blocks_data = lock_data.get("blocks", {})

        for block_name, block_info in blocks_data.items():
            commit = block_info.get("commit")

            try:
                # Install from specific commit
                registry.materialize_block_to(block_name, registry.cache_root, commit=commit)
                installed.append(block_name)
            except Exception as exc:
                # Continue with other blocks even if one fails
                print(f"Warning: Failed to install {block_name}: {exc}")
                continue

        return installed

    def verify(self, registry: Optional[BlockRegistry] = None) -> Dict[str, str]:
        """Verify current state matches lock file.

        Parameters
        ----------
        registry : BlockRegistry, optional
            Registry instance. If not provided, creates new one.

        Returns
        -------
        dict
            Mapping of block names to status: "ok", "missing", "mismatch"
        """
        lock_data = self.load()

        if registry is None:
            registry = BlockRegistry(cache_root=CACHE_ROOT)

        results = {}
        blocks_data = lock_data.get("blocks", {})

        for block_name, expected in blocks_data.items():
            record = registry.manifest.block_record(block_name)

            if not record:
                results[block_name] = "missing"
                continue

            expected_commit = expected.get("commit")
            actual_commit = record.get("source_commit")

            if expected_commit == actual_commit:
                # Also verify hash if available
                expected_hash = expected.get("hash")
                actual_hash = self._compute_block_hash(record.get("hashes", {}))

                if expected_hash == actual_hash:
                    results[block_name] = "ok"
                else:
                    results[block_name] = "mismatch"
            else:
                results[block_name] = "mismatch"

        return results
