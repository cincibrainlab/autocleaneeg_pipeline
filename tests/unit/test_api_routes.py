"""Tests for API route modules: tunnel, results, filesystem, service.

Tests the core logic of each module without requiring a running server,
MNE, or real EEG data. Mocks are used for subprocess, file I/O, and
external dependencies.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ── Tunnel state management tests ─────────────────────────────────────


class TestTunnelState:
    """Tests for tunnel.py module-level state and config persistence."""

    def _reset_tunnel_state(self):
        """Reset module-level tunnel globals between tests."""
        import autoclean.api.routes.tunnel as t
        with t._tunnel_lock:
            t._tunnel_process = None
            t._tunnel_url = None
            t._tunnel_password = None
            t._tunnel_mode = None
            t._tunnel_start_time = None

    def test_get_tunnel_state_inactive(self):
        """Tunnel state is inactive when no process is running."""
        import autoclean.api.routes.tunnel as t
        self._reset_tunnel_state()

        state = t.get_tunnel_state()
        assert state["active"] is False
        assert state["url"] is None
        assert state["password"] is None
        assert state["mode"] is None

    def test_get_tunnel_state_active(self):
        """Tunnel state reflects active process."""
        import autoclean.api.routes.tunnel as t
        self._reset_tunnel_state()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # Still running

        with t._tunnel_lock:
            t._tunnel_process = mock_proc
            t._tunnel_url = "https://test.trycloudflare.com"
            t._tunnel_password = "abc123"
            t._tunnel_mode = "quick"

        state = t.get_tunnel_state()
        assert state["active"] is True
        assert state["url"] == "https://test.trycloudflare.com"
        assert state["password"] == "abc123"
        assert state["mode"] == "quick"

    def test_get_tunnel_state_dead_process_cleared(self):
        """Dead process is automatically cleared from state."""
        import autoclean.api.routes.tunnel as t
        self._reset_tunnel_state()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # Exited

        with t._tunnel_lock:
            t._tunnel_process = mock_proc
            t._tunnel_url = "https://old.trycloudflare.com"
            t._tunnel_password = "old"

        state = t.get_tunnel_state()
        assert state["active"] is False
        assert state["url"] is None

    def test_config_save_and_load(self, tmp_path: Path):
        """Tunnel config round-trips through JSON."""
        import autoclean.api.routes.tunnel as t

        with patch.object(t, "_config_path", return_value=tmp_path / "tunnel_config.json"):
            t._save_config({"token": "eyJ...", "url": "https://lab.example.com"})
            config = t._load_config()

        assert config["token"] == "eyJ..."
        assert config["url"] == "https://lab.example.com"

    def test_config_load_missing_file(self, tmp_path: Path):
        """Missing config file returns empty dict."""
        import autoclean.api.routes.tunnel as t

        with patch.object(t, "_config_path", return_value=tmp_path / "nonexistent.json"):
            config = t._load_config()

        assert config == {}

    def test_config_load_corrupt_file(self, tmp_path: Path):
        """Corrupt config file returns empty dict (no crash)."""
        import autoclean.api.routes.tunnel as t

        bad_file = tmp_path / "tunnel_config.json"
        bad_file.write_text("not json {{{", encoding="utf-8")

        with patch.object(t, "_config_path", return_value=bad_file):
            config = t._load_config()

        assert config == {}

    def test_clear_tunnel_state(self):
        """_clear_tunnel_state resets all globals."""
        import autoclean.api.routes.tunnel as t

        with t._tunnel_lock:
            t._tunnel_process = MagicMock()
            t._tunnel_url = "https://something"
            t._tunnel_password = "pw"
            t._tunnel_mode = "named"

        t._clear_tunnel_state()

        with t._tunnel_lock:
            assert t._tunnel_process is None
            assert t._tunnel_url is None
            assert t._tunnel_password is None
            assert t._tunnel_mode is None


# ── Results cache locking tests ───────────────────────────────────────


class TestResultsCache:
    """Tests for the thread-safe results cache in results.py."""

    def test_cache_returns_empty_when_no_automations(self, tmp_path: Path):
        """Cache returns empty list when automations/ doesn't exist."""
        import autoclean.api.routes.results as r

        # Reset cache
        with r._runs_cache_lock:
            r._runs_cache = []
            r._runs_cache_time = 0.0

        result = r._find_all_runs(tmp_path)
        assert result == []

    def test_cache_ttl_prevents_rescan(self, tmp_path: Path):
        """Cache TTL prevents rescanning within 5 seconds."""
        import autoclean.api.routes.results as r

        # Seed the cache
        with r._runs_cache_lock:
            r._runs_cache = [
                ({"run_id": "test123", "created_at": "2026-01-01"}, tmp_path)
            ]
            r._runs_cache_time = time.time()

        # Should return cached result without scanning
        result = r._find_all_runs(tmp_path)
        assert len(result) == 1
        assert result[0][0]["run_id"] == "test123"

    def test_cache_expired_rescans(self, tmp_path: Path):
        """Expired cache triggers a rescan."""
        import autoclean.api.routes.results as r

        # Seed with expired cache
        with r._runs_cache_lock:
            r._runs_cache = [
                ({"run_id": "stale"}, tmp_path)
            ]
            r._runs_cache_time = time.time() - 10  # 10s ago, > 5s TTL

        # No automations/ dir exists, so rescan returns empty
        result = r._find_all_runs(tmp_path)
        assert result == []

    def test_extract_stem_strips_suffixes(self):
        """_extract_stem strips known pipeline suffixes."""
        import autoclean.api.routes.results as r

        assert r._extract_stem("201001_D1BL_EC_comp_epo.set") == "201001_D1BL_EC"
        assert r._extract_stem("201001_D1BL_EC_comp.set") == "201001_D1BL_EC"
        assert r._extract_stem("201001_D1BL_EC_raw.set") == "201001_D1BL_EC"
        assert r._extract_stem("201001_D1BL_EC.set") == "201001_D1BL_EC"

    def test_decisions_round_trip(self, tmp_path: Path):
        """Decisions save/load round-trip."""
        import autoclean.api.routes.results as r

        decisions = {
            "run1": {
                "run_id": "run1",
                "filename": "test.set",
                "decision": "pass",
                "notes": "looks good",
                "decided_at": "2026-03-17T12:00:00Z",
            }
        }

        with patch.object(r, "_decisions_path", return_value=tmp_path / "decisions.json"):
            r._save_decisions(tmp_path, decisions)
            loaded = r._load_decisions(tmp_path)

        assert loaded["run1"]["decision"] == "pass"
        assert loaded["run1"]["notes"] == "looks good"

    def test_decisions_csv_output(self, tmp_path: Path):
        """_decisions_to_csv generates valid CSV."""
        import autoclean.api.routes.results as r

        decisions = {
            "run1": {
                "run_id": "run1",
                "filename": "test.set",
                "decision": "fail",
                "notes": "bad channels",
                "decided_at": "2026-03-17",
            }
        }

        csv_str = r._decisions_to_csv(decisions)
        lines = [line.strip() for line in csv_str.strip().split("\n")]
        assert lines[0] == "run_id,filename,decision,notes,decided_at"
        assert "run1" in lines[1]
        assert "fail" in lines[1]


# ── Filesystem path traversal tests ───────────────────────────────────


class TestFilesystemSecurity:
    """Tests for filesystem.py path traversal protection."""

    def test_is_allowed_within_workspace(self, tmp_path: Path):
        """Paths within workspace are allowed."""
        import autoclean.api.routes.filesystem as fs

        with patch.object(fs.api_state, "workspace_dir", tmp_path):
            subdir = tmp_path / "data"
            subdir.mkdir()
            assert fs._is_allowed(subdir) is True

    def test_is_allowed_blocks_traversal(self, tmp_path: Path):
        """Path traversal attempts are blocked."""
        import autoclean.api.routes.filesystem as fs

        with patch.object(fs.api_state, "workspace_dir", tmp_path):
            # Trying to escape workspace via ..
            escaped = tmp_path / ".." / ".." / "etc"
            assert fs._is_allowed(escaped) is False

    def test_is_allowed_blocks_absolute_outside(self, tmp_path: Path):
        """Absolute paths outside workspace are blocked."""
        import autoclean.api.routes.filesystem as fs

        with patch.object(fs.api_state, "workspace_dir", tmp_path):
            assert fs._is_allowed(Path("/etc/passwd")) is False

    def test_is_allowed_home_directory(self, tmp_path: Path):
        """Home directory is allowed as a secondary root."""
        import autoclean.api.routes.filesystem as fs

        with patch.object(fs.api_state, "workspace_dir", tmp_path):
            home = Path.home()
            assert fs._is_allowed(home) is True

    def test_is_allowed_workspace_parent_directory(self, tmp_path: Path):
        """Parent of the current workspace is allowed for workspace reselection."""
        import autoclean.api.routes.filesystem as fs

        workspace = tmp_path / "Autoclean_Serve_Workspaces" / "project-a"
        workspace.mkdir(parents=True)

        with patch.object(fs.api_state, "workspace_dir", workspace):
            assert fs._is_allowed(workspace.parent) is True

    def test_is_allowed_root_directory(self, tmp_path: Path):
        """Root directory is allowed only as a top-level chooser for allowed roots."""
        import autoclean.api.routes.filesystem as fs

        with patch.object(fs.api_state, "workspace_dir", tmp_path):
            assert fs._is_allowed(Path("/")) is True

    def test_browse_root_returns_allowed_roots(self, tmp_path: Path):
        """Browsing / returns the configured allowed roots rather than a full filesystem listing."""
        import asyncio
        import autoclean.api.routes.filesystem as fs

        with patch.object(fs.api_state, "workspace_dir", tmp_path):
            response = asyncio.run(fs.browse_directory(path="/"))
            returned_paths = {entry.path for entry in response.entries}
            assert str(tmp_path.resolve()) in returned_paths
            assert str(Path.home().resolve()) in returned_paths

    def test_browse_workspace_parent_directory_is_permitted(self, tmp_path: Path):
        """Browsing up from a workspace into its parent directory is allowed."""
        import asyncio
        import autoclean.api.routes.filesystem as fs

        workspace = tmp_path / "Autoclean_Serve_Workspaces" / "project-a"
        sibling = workspace.parent / "project-b"
        workspace.mkdir(parents=True)
        sibling.mkdir()

        with patch.object(fs.api_state, "workspace_dir", workspace):
            response = asyncio.run(fs.browse_directory(path=str(workspace.parent)))
            returned_paths = {entry.path for entry in response.entries}
            assert str(workspace) in returned_paths
            assert str(sibling) in returned_paths


# ── Service stop_service threading tests ──────────────────────────────


class TestServiceStop:
    """Tests for the refactored _stop_service_blocking helper."""

    def test_stop_when_not_running(self):
        """Stopping when no process is running returns (False, 0)."""
        import autoclean.api.routes.service as svc

        with svc._service_lock:
            svc._process = None

        was_running, pid = svc._stop_service_blocking()
        assert was_running is False
        assert pid == 0

    def test_stop_graceful(self):
        """Graceful stop sends SIGTERM and waits."""
        import autoclean.api.routes.service as svc

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.wait.return_value = 0

        with svc._service_lock:
            svc._process = mock_proc
            svc._start_time = time.time()

        was_running, pid = svc._stop_service_blocking()

        assert was_running is True
        assert pid == 12345
        mock_proc.send_signal.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=10)

        # State should be cleared
        with svc._service_lock:
            assert svc._process is None
            assert svc._start_time is None

    def test_stop_timeout_kills(self):
        """Timeout on SIGTERM falls back to kill."""
        import subprocess
        import autoclean.api.routes.service as svc

        mock_proc = MagicMock()
        mock_proc.pid = 99999
        mock_proc.wait.side_effect = subprocess.TimeoutExpired("cloudflared", 10)

        with svc._service_lock:
            svc._process = mock_proc
            svc._start_time = time.time()

        was_running, pid = svc._stop_service_blocking()

        assert was_running is True
        mock_proc.kill.assert_called_once()


# ── Results require_workspace status code tests ───────────────────────


class TestRequireWorkspace:
    """Tests that _require_workspace returns 409, not 500."""

    def test_returns_409_when_unconfigured(self):
        """_require_workspace raises 409 when workspace is not set."""
        from fastapi import HTTPException
        import autoclean.api.routes.results as r

        with patch.object(r.api_state, "workspace_dir", None):
            with pytest.raises(HTTPException) as exc_info:
                r._require_workspace()
            assert exc_info.value.status_code == 409

    def test_returns_path_when_configured(self, tmp_path: Path):
        """_require_workspace returns Path when workspace is set."""
        import autoclean.api.routes.results as r

        with patch.object(r.api_state, "workspace_dir", tmp_path):
            result = r._require_workspace()
            assert result == tmp_path
