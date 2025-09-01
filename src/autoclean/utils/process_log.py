"""Utilities for tracking CLI actions in the process log."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

MAX_LOG_SIZE = 5 * 1024 * 1024  # 5MB


def _rotate_log(log_path: Path) -> None:
    """Rotate log file when it exceeds the maximum size."""
    try:
        for i in range(4, 0, -1):
            old_path = log_path.with_suffix(f".{i}.txt")
            new_path = log_path.with_suffix(f".{i + 1}.txt")
            if old_path.exists():
                old_path.rename(new_path)
        if log_path.exists():
            rotated_path = log_path.with_suffix(".1.txt")
            log_path.rename(rotated_path)
    except Exception:
        try:
            with log_path.open("w", encoding="utf-8") as f:
                f.write(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Log rotated due to size limit\n"
                )
        except Exception:
            pass


def _get_workspace_dir() -> Optional[Path]:
    """Return current workspace directory if available."""
    try:
        from autoclean.utils.user_config import user_config

        return user_config.config_dir
    except Exception:
        return None


def log_cli_action(action: str) -> None:
    """Append a CLI action entry to the process log."""
    try:
        workspace_dir = _get_workspace_dir()
        if not workspace_dir or not workspace_dir.exists():
            return

        log_path = workspace_dir / "process_log.txt"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if log_path.exists() and log_path.stat().st_size > MAX_LOG_SIZE:
            _rotate_log(log_path)

        log_entry = f"[{timestamp}] {action}\n"
        temp_path = log_path.with_suffix('.tmp')
        try:
            existing = ""
            if log_path.exists():
                with log_path.open('r', encoding='utf-8') as f:
                    existing = f.read()
            with temp_path.open('w', encoding='utf-8') as f:
                f.write(existing + log_entry)
            temp_path.replace(log_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise
    except Exception as e:
        print(f"Warning: Failed to log CLI action: {e}", file=sys.stderr)
