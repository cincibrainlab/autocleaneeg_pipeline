from __future__ import annotations

import sys
import time

from autoclean_mcp.session_manager import SessionManager


def test_session_manager_can_start_and_stop_process() -> None:
    manager = SessionManager()
    status = manager.start(
        [
            sys.executable,
            "-c",
            "import time; print('started', flush=True); time.sleep(30)",
        ]
    )

    assert status.state == "running"
    assert status.pid is not None

    time.sleep(0.2)
    running = manager.get(status.session_id)
    assert running is not None
    assert "started" in running.stdout

    stopped = manager.stop(status.session_id)
    assert stopped is not None
    assert stopped.state == "exited"
    assert stopped.exit_code is not None


def test_session_manager_startup_initialization_prunes_exited_sessions() -> None:
    manager = SessionManager()
    status = manager.start([sys.executable, "-c", "print('done')"])
    time.sleep(0.2)

    report = manager.initialize_startup()

    assert report["initialized"] is True
    assert report["persistence_supported"] is False
    assert report["reattached_sessions"] == 0
    assert report["pruned_exited_sessions"] == 1
    assert manager.get(status.session_id) is None
