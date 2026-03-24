"""Managed background sessions for long-running CLI commands."""

from __future__ import annotations

import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Mapping, Sequence

from autoclean_mcp.models import SessionStatus, utc_now_iso


class _ManagedProcess:
    def __init__(
        self,
        session_id: str,
        command: Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.session_id = session_id
        self.command = list(command)
        self.cwd = str(Path(cwd).resolve()) if cwd is not None else str(Path.cwd())
        self.started_at = utc_now_iso()
        self.finished_at: str | None = None
        self.exit_code: int | None = None
        self.stdout_chunks: deque[str] = deque(maxlen=1000)
        self.stderr_chunks: deque[str] = deque(maxlen=1000)
        self.process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=dict(env) if env is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stdout_thread = threading.Thread(
            target=self._stream_reader,
            args=(self.process.stdout, self.stdout_chunks),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._stream_reader,
            args=(self.process.stderr, self.stderr_chunks),
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    @staticmethod
    def _stream_reader(stream, sink: deque[str]) -> None:
        if stream is None:
            return
        try:
            for line in stream:
                sink.append(line)
        finally:
            stream.close()

    def poll(self) -> int | None:
        code = self.process.poll()
        if code is not None and self.finished_at is None:
            self.finished_at = utc_now_iso()
            self.exit_code = code
        return code

    def terminate(self) -> None:
        if self.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.poll()

    def status(self) -> SessionStatus:
        state = "running" if self.poll() is None else "exited"
        return SessionStatus(
            session_id=self.session_id,
            command=self.command,
            cwd=self.cwd,
            state=state,
            pid=self.process.pid,
            started_at=self.started_at,
            finished_at=self.finished_at,
            exit_code=self.exit_code,
            stdout="".join(self.stdout_chunks),
            stderr="".join(self.stderr_chunks),
        )


class SessionManager:
    """In-memory manager for long-running CLI subprocesses."""

    def __init__(self) -> None:
        self._sessions: dict[str, _ManagedProcess] = {}
        self._lock = threading.Lock()
        self._startup_report: dict[str, object] = {
            "initialized": False,
            "persistence_supported": False,
            "reattached_sessions": 0,
            "pruned_exited_sessions": 0,
            "notes": [
                "Session state is in-memory only in the current implementation.",
                "Managed sessions are not reattached across MCP server restart.",
            ],
        }

    def initialize_startup(self) -> dict[str, object]:
        """Initialize startup policy and prune any exited in-memory sessions."""
        pruned = self.prune_exited_sessions()
        self._startup_report = {
            "initialized": True,
            "persistence_supported": False,
            "reattached_sessions": 0,
            "pruned_exited_sessions": pruned,
            "notes": [
                "Session state is in-memory only in the current implementation.",
                "Managed sessions are not reattached across MCP server restart.",
                "Exited in-memory sessions are pruned during MCP server startup.",
            ],
        }
        return dict(self._startup_report)

    def startup_report(self) -> dict[str, object]:
        """Return the current startup recovery policy report."""
        return dict(self._startup_report)

    def start(
        self,
        command: Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> SessionStatus:
        session_id = secrets.token_hex(8)
        managed = _ManagedProcess(session_id, command, cwd=cwd, env=env)
        with self._lock:
            self._sessions[session_id] = managed
        return managed.status()

    def get(self, session_id: str) -> SessionStatus | None:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            return None
        return session.status()

    def stop(self, session_id: str) -> SessionStatus | None:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            return None
        session.terminate()
        return session.status()

    def list(self) -> list[SessionStatus]:
        with self._lock:
            sessions = list(self._sessions.values())
        return [session.status() for session in sessions]

    def prune_exited_sessions(self) -> int:
        """Remove exited sessions from the in-memory registry."""
        removed = 0
        with self._lock:
            session_ids = list(self._sessions.keys())
            for session_id in session_ids:
                session = self._sessions[session_id]
                if session.poll() is not None:
                    del self._sessions[session_id]
                    removed += 1
        return removed


SESSION_MANAGER = SessionManager()
