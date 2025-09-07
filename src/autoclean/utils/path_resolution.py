"""Utilities for resolving moved directories after auto-backups.

Phase 1 implementation: derive relocations from recent runs' metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from autoclean.utils.database import manage_database_conditionally


def resolve_moved_path(stored_path: str | Path, max_records: int = 200) -> Path:
    """Resolve a possibly outdated path by applying recent directory_backup moves.

    - If the path exists, returns it unchanged.
    - Otherwise, scans a bounded window of recent runs' metadata for
      directory_backup entries and applies reverse-chronological prefix rewrites.
    - Returns the first existing candidate; falls back to the original path.
    """
    p = Path(stored_path)
    if p.exists():
        return p

    try:
        records = manage_database_conditionally("get_collection") or []
    except Exception:
        records = []

    # Sort most recent first by created_at and id fallback
    def _key(r):
        return (r.get("created_at") or "", r.get("id") or 0)

    for rec in sorted(records, key=_key, reverse=True)[:max_records]:
        meta = rec.get("metadata") or {}
        backup = meta.get("directory_backup")
        if not backup:
            continue
        moved_from = backup.get("moved_from")
        moved_to = backup.get("moved_to")
        if not moved_from or not moved_to:
            continue
        s = str(stored_path)
        if s.startswith(moved_from):
            candidate = Path(moved_to + s[len(moved_from) :])
            if candidate.exists():
                return candidate

    return p

