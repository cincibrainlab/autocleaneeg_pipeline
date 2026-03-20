"""Tests for Serve auth-related API state paths."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException
import pytest

from autoclean.api.state import APIState


def test_get_serve_state_dir_requires_workspace() -> None:
    """Serve state dir lookup should fail when workspace is unset."""
    state = APIState()

    with pytest.raises(HTTPException):
        state.get_serve_state_dir()


def test_get_serve_state_dir_can_create_directory(tmp_path: Path) -> None:
    """Serve state dir helper should create the .serve directory on demand."""
    state = APIState()
    state.configure(tmp_path)

    state_dir = state.get_serve_state_dir(create=True)

    assert state_dir == tmp_path / ".serve"
    assert state_dir.exists()
    assert state_dir.is_dir()


def test_get_auth_paths_resolve_under_workspace(tmp_path: Path) -> None:
    """Auth config and DB paths should resolve under the active workspace."""
    state = APIState()
    state.configure(tmp_path, mode="live")

    assert state.get_auth_config_path() == tmp_path / "serve-auth.json"
    assert state.get_auth_db_path() == tmp_path / ".serve" / "serve_state.db"


def test_get_auth_db_path_can_create_parent_directory(tmp_path: Path) -> None:
    """Auth DB helper should create the parent Serve state directory when asked."""
    state = APIState()
    state.configure(tmp_path)

    db_path = state.get_auth_db_path(create_parent=True)

    assert db_path == tmp_path / ".serve" / "serve_state.db"
    assert db_path.parent.exists()
