# src/autoclean/utils/audit.py
"""Audit utilities for enhanced tracking and compliance."""

import getpass
import os
import socket
from datetime import datetime
from typing import Any, Dict


def get_user_context() -> Dict[str, Any]:
    """Get current user context for audit trail.

    Captures basic system and user information for tracking who
    performed operations without requiring authentication.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - username: Current system username
        - hostname: Machine hostname
        - pid: Process ID of the pipeline
        - session_start: When this context was created

    Examples
    --------
    >>> context = get_user_context()
    >>> print(context['username'])
    'researcher1'
    """
    try:
        username = getpass.getuser()
    except Exception:
        username = "unknown"

    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = "unknown"

    return {
        "username": username,
        "hostname": hostname,
        "pid": os.getpid(),
        "session_start": datetime.now().isoformat(),
    }
