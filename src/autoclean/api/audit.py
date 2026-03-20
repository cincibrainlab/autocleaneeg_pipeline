"""Audit logging helpers for Serve admin actions."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel, Field

from autoclean.api.auth.service import utc_now_iso


class AuditEntry(BaseModel):
    """Stored Serve audit record."""

    id: int
    actor_user_id: str | None = None
    actor_login: str | None = None
    action: str
    resource_type: str
    resource_id: str | None = None
    details: dict[str, object] = Field(default_factory=dict)
    ip_address: str | None = None
    created_at: str


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_audit_schema(db_path: Path) -> None:
    """Ensure audit log storage exists in the Serve state DB."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS serve_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor_user_id TEXT,
                actor_login TEXT,
                action TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                resource_id TEXT,
                details_json TEXT NOT NULL,
                ip_address TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def log_audit_event(
    db_path: Path,
    *,
    action: str,
    resource_type: str,
    resource_id: str | None = None,
    actor_user_id: str | None = None,
    actor_login: str | None = None,
    details_json: str = "{}",
    ip_address: str | None = None,
) -> None:
    """Persist a new Serve audit entry."""
    ensure_audit_schema(db_path)
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO serve_audit_log(
                actor_user_id,
                actor_login,
                action,
                resource_type,
                resource_id,
                details_json,
                ip_address,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                actor_user_id,
                actor_login,
                action,
                resource_type,
                resource_id,
                details_json,
                ip_address,
                utc_now_iso(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def list_audit_events(db_path: Path, *, limit: int = 100) -> list[AuditEntry]:
    """Return recent audit records newest-first."""
    ensure_audit_schema(db_path)
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT id, actor_user_id, actor_login, action, resource_type, resource_id,
                   details_json, ip_address, created_at
            FROM serve_audit_log
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    import json

    return [
        AuditEntry(
            id=row["id"],
            actor_user_id=row["actor_user_id"],
            actor_login=row["actor_login"],
            action=row["action"],
            resource_type=row["resource_type"],
            resource_id=row["resource_id"],
            details=json.loads(row["details_json"]) if row["details_json"] else {},
            ip_address=row["ip_address"],
            created_at=row["created_at"],
        )
        for row in rows
    ]
