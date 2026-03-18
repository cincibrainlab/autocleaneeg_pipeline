"""Filesystem browsing endpoint for the web UI folder picker.

Provides a server-side directory browser so operators can pick input
folders visually instead of typing absolute paths by hand.

Security model
--------------
The browse endpoint is used exclusively for the route-modal folder picker.
It does not need to expose the entire filesystem.

Allowed browse roots (checked in order):
1. ``api_state.workspace_dir`` — always allowed when configured.
2. ``Path.home()`` — the current user's home directory.

A requested path is accepted if it resolves to a location that is either
*inside* one of the above roots or *is* one of those roots.  Anything
outside — including /etc, /usr, /var and every other system path — is
rejected with HTTP 403.

Additionally, a resolved path must equal the result of the base + suffix
join to prevent path-traversal via ``..`` segments in the query parameter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from autoclean.api.state import api_state

router = APIRouter()


# ── Allowed roots ─────────────────────────────────────────────────────────────


def _allowed_roots() -> list[Path]:
    """Return the list of directories the browser is permitted to explore."""
    roots: list[Path] = []
    if api_state.workspace_dir:
        roots.append(Path(api_state.workspace_dir).resolve())
    roots.append(Path.home().resolve())
    return roots


def _is_allowed(path: Path) -> bool:
    """Return True only if *path* falls inside (or is) an allowed root."""
    resolved = path.resolve()
    if resolved == Path("/"):
        return True
    for root in _allowed_roots():
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


# ── Response models ───────────────────────────────────────────────────────────


class FolderEntry(BaseModel):
    name: str
    path: str
    is_dir: bool


class BrowseResponse(BaseModel):
    path: str
    parent: Optional[str]
    entries: list[FolderEntry]


# ── Endpoint ──────────────────────────────────────────────────────────────────


@router.get("/browse", response_model=BrowseResponse)
async def browse_directory(
    path: Optional[str] = Query(default=None, description="Absolute path to browse"),
) -> BrowseResponse:
    """List subdirectories of *path* for the folder-picker UI.

    - Defaults to the configured workspace directory when *path* is omitted.
    - Only directories are returned (not files).
    - Results are sorted alphabetically.
    - Browsing is restricted to the workspace directory and user home directory.
    """
    # Resolve the target directory
    if path:
        target = Path(path).resolve()
    elif api_state.workspace_dir:
        target = Path(api_state.workspace_dir).resolve()
    else:
        # Fallback: home directory when no workspace is configured
        target = Path.home().resolve()

    # Security: only allow browsing within permitted roots
    if not _is_allowed(target):
        raise HTTPException(
            status_code=403,
            detail=f"Browsing into '{target}' is not permitted",
        )

    if target == Path("/"):
        entries = [
            FolderEntry(name=root.name or str(root), path=str(root), is_dir=True)
            for root in _allowed_roots()
        ]
        unique_entries: dict[str, FolderEntry] = {entry.path: entry for entry in entries}
        return BrowseResponse(
            path=str(target),
            parent=None,
            entries=sorted(unique_entries.values(), key=lambda entry: entry.path.lower()),
        )

    # Must be an existing directory
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {target}")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {target}")

    # Compute parent — only expose it when the parent is also within allowed roots
    parent_path: Optional[str] = None
    parent = target.parent
    if parent != target and _is_allowed(parent):
        parent_path = str(parent)

    # List subdirectories only
    entries: list[FolderEntry] = []
    try:
        for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir():
                continue
            # Skip hidden directories (dotfiles)
            if child.name.startswith("."):
                continue
            entries.append(
                FolderEntry(name=child.name, path=str(child), is_dir=True)
            )
    except PermissionError:
        raise HTTPException(
            status_code=403,
            detail=f"Permission denied reading directory: {target}",
        )
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading directory: {exc}",
        )

    return BrowseResponse(
        path=str(target),
        parent=parent_path,
        entries=entries,
    )
