"""Utility functions for CLI process logging.

These helpers were split from the monolithic ``cli.py`` module so that
logging responsibilities are isolated and easier to maintain."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from autoclean.utils.user_config import user_config

# Maximum log file size (5MB)
MAX_LOG_SIZE = 5 * 1024 * 1024


def strip_wrapping_quotes(text: Optional[str]) -> Optional[str]:
    """Remove a single or nested pair of matching wrapping quotes from ``text``.

    Handles common copy/paste cases like ``"'/Users/me/My Folder'"`` or ``'"/path"'``.
    Returns ``None`` unchanged and leaves interior quotes untouched.
    """
    if text is None:
        return None
    s = text.strip()
    # Remove up to two layers of matching quotes (single or double)
    for _ in range(2):
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            s = s[1:-1].strip()
        else:
            break
    return s


def sanitize_arguments(args: List[str]) -> List[str]:
    """Sanitize command-line arguments to remove sensitive information."""
    sanitized: List[str] = []

    # Patterns for sensitive information
    sensitive_patterns = [
        # File paths with potentially sensitive directory names
        r"(/[Uu]sers?/[^/]+/[Dd]esktop|/[Uu]sers?/[^/]+/[Dd]ocuments)",
        r"(/home/[^/]+/[Dd]esktop|/home/[^/]+/[Dd]ocuments)",
        # API tokens and keys
        r"(--?(?:token|key|password|pass|secret)(?:=|\s+)\S+)",
        # Email addresses
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    ]

    for arg in args:
        sanitized_arg = arg
        for pattern in sensitive_patterns:
            if re.search(pattern, arg, re.IGNORECASE):
                # Replace file paths with just the filename
                if "/" in arg or "\\" in arg:
                    path_obj = Path(arg)
                    sanitized_arg = (
                        f"[REDACTED]/{path_obj.name}" if path_obj.name else "[REDACTED]"
                    )
                else:
                    # For tokens/keys, show only the parameter name
                    if "=" in arg:
                        param_name = arg.split("=")[0]
                        sanitized_arg = f"{param_name}=[REDACTED]"
                    else:
                        sanitized_arg = "[REDACTED]"
                break

        sanitized.append(sanitized_arg)

    return sanitized


def rotate_log(log_path: Path) -> None:
    """Rotate log file when it gets too large."""
    try:
        # Keep last 5 rotated logs
        for i in range(4, 0, -1):
            old_path = log_path.with_suffix(f".{i}.txt")
            new_path = log_path.with_suffix(f".{i + 1}.txt")
            if old_path.exists():
                old_path.rename(new_path)

        # Move current log to .1
        if log_path.exists():
            rotated_path = log_path.with_suffix(".1.txt")
            log_path.rename(rotated_path)
    except Exception:
        # If rotation fails, truncate the log
        try:
            with log_path.open("w", encoding="utf-8") as f:
                f.write(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Log rotated due to size limit\n"
                )
        except Exception:
            pass


def log_cli_execution(args: argparse.Namespace) -> None:
    """Log CLI execution to workspace process log with security and error handling."""
    try:
        # Only log if workspace exists to avoid setup errors
        workspace_dir = user_config.config_dir
        if not workspace_dir.exists():
            return

        log_path = workspace_dir / "process_log.txt"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Get original command arguments, excluding the script name
        original_args = (
            sys.argv[1:] if hasattr(sys, "argv") and len(sys.argv) > 1 else []
        )

        # Skip logging if no meaningful command (just bare invocation)
        if not original_args or not args.command:
            return

        # Sanitize arguments for security
        safe_args = sanitize_arguments(original_args)
        command_str = f"autocleaneeg-pipeline {' '.join(safe_args)}"

        # Check file size and rotate if necessary
        if log_path.exists() and log_path.stat().st_size > MAX_LOG_SIZE:
            rotate_log(log_path)

        # Atomic write to prevent corruption
        log_entry = f"[{timestamp}] {command_str}\n"

        # Write to temporary file first, then move
        temp_path = log_path.with_suffix(".tmp")
        try:
            # Read existing content if file exists
            existing_content = ""
            if log_path.exists():
                with log_path.open("r", encoding="utf-8") as f:
                    existing_content = f.read()

            # Write to temp file
            with temp_path.open("w", encoding="utf-8") as f:
                f.write(existing_content + log_entry)

            # Atomic move
            temp_path.replace(log_path)

        except Exception:
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            raise

    except Exception as exc:
        # Log to stderr but don't break CLI functionality
        print(f"Warning: Failed to log command execution: {exc}", file=sys.stderr)
