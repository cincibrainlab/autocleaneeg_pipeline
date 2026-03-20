"""SQLite-backed storage for Serve auth state."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from autoclean.api.auth.models import (
    AuthSession,
    AuthUser,
    Permission,
    ROLE_PERMISSIONS,
    Role,
)

_SCHEMA_LOCK = threading.Lock()


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_auth_schema(db_path: Path) -> None:
    """Create the Serve auth schema when it does not yet exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _SCHEMA_LOCK:
        conn = _connect(db_path)
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS serve_users (
                    id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    login TEXT NOT NULL,
                    email TEXT,
                    display_name TEXT,
                    avatar_url TEXT,
                    created_at TEXT,
                    last_login_at TEXT,
                    disabled INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(provider, subject)
                );

                CREATE TABLE IF NOT EXISTS serve_roles (
                    name TEXT PRIMARY KEY
                );

                CREATE TABLE IF NOT EXISTS serve_user_roles (
                    user_id TEXT NOT NULL,
                    role_name TEXT NOT NULL,
                    granted_by TEXT,
                    granted_at TEXT,
                    PRIMARY KEY (user_id, role_name),
                    FOREIGN KEY(user_id) REFERENCES serve_users(id),
                    FOREIGN KEY(role_name) REFERENCES serve_roles(name)
                );

                CREATE TABLE IF NOT EXISTS serve_sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    csrf_token TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT,
                    last_seen_at TEXT,
                    revoked_at TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    FOREIGN KEY(user_id) REFERENCES serve_users(id)
                );
                """
            )
            conn.executemany(
                "INSERT OR IGNORE INTO serve_roles(name) VALUES (?)",
                [(role.value,) for role in Role],
            )
            conn.commit()
        finally:
            conn.close()


def _row_to_user(row: sqlite3.Row | None, roles: list[Role]) -> AuthUser | None:
    if row is None:
        return None
    return AuthUser(
        id=row["id"],
        provider=row["provider"],
        subject=row["subject"],
        login=row["login"],
        email=row["email"],
        display_name=row["display_name"],
        avatar_url=row["avatar_url"],
        created_at=row["created_at"],
        last_login_at=row["last_login_at"],
        disabled=bool(row["disabled"]),
        roles=roles,
    )


def _row_to_session(row: sqlite3.Row | None) -> AuthSession | None:
    if row is None:
        return None
    return AuthSession(
        id=row["id"],
        user_id=row["user_id"],
        csrf_token=row["csrf_token"],
        expires_at=row["expires_at"],
        created_at=row["created_at"],
        last_seen_at=row["last_seen_at"],
        revoked_at=row["revoked_at"],
        ip_address=row["ip_address"],
        user_agent=row["user_agent"],
    )


def list_roles_for_user(db_path: Path, user_id: str) -> list[Role]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT role_name FROM serve_user_roles WHERE user_id = ? ORDER BY role_name",
            (user_id,),
        ).fetchall()
        return [Role(row["role_name"]) for row in rows]
    finally:
        conn.close()


def list_permissions_for_roles(roles: Iterable[Role]) -> list[Permission]:
    permissions: set[Permission] = set()
    for role in roles:
        permissions.update(ROLE_PERMISSIONS.get(role, ()))
    return sorted(permissions, key=lambda permission: permission.value)


def get_user_by_id(db_path: Path, user_id: str) -> AuthUser | None:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM serve_users WHERE id = ?",
            (user_id,),
        ).fetchone()
        roles = list_roles_for_user(db_path, user_id) if row is not None else []
        return _row_to_user(row, roles)
    finally:
        conn.close()


def get_user_by_identity(db_path: Path, provider: str, subject: str) -> AuthUser | None:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM serve_users WHERE provider = ? AND subject = ?",
            (provider, subject),
        ).fetchone()
        roles = list_roles_for_user(db_path, row["id"]) if row is not None else []
        return _row_to_user(row, roles)
    finally:
        conn.close()


def upsert_user(
    db_path: Path,
    *,
    user_id: str,
    provider: str,
    subject: str,
    login: str,
    email: str | None,
    display_name: str | None,
    avatar_url: str | None,
    now_iso: str,
) -> AuthUser:
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO serve_users (
                id, provider, subject, login, email, display_name, avatar_url,
                created_at, last_login_at, disabled
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(provider, subject) DO UPDATE SET
                login = excluded.login,
                email = excluded.email,
                display_name = excluded.display_name,
                avatar_url = excluded.avatar_url,
                last_login_at = excluded.last_login_at
            """,
            (
                user_id,
                provider,
                subject,
                login,
                email,
                display_name,
                avatar_url,
                now_iso,
                now_iso,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    user = get_user_by_identity(db_path, provider, subject)
    if user is None:
        raise RuntimeError("Failed to persist Serve auth user")
    return user


def set_user_roles(
    db_path: Path,
    user_id: str,
    roles: Iterable[Role],
    *,
    granted_by: str | None,
    granted_at: str,
) -> None:
    role_values = {role.value for role in roles}
    conn = _connect(db_path)
    try:
        conn.execute("DELETE FROM serve_user_roles WHERE user_id = ?", (user_id,))
        conn.executemany(
            """
            INSERT INTO serve_user_roles(user_id, role_name, granted_by, granted_at)
            VALUES (?, ?, ?, ?)
            """,
            [(user_id, role_name, granted_by, granted_at) for role_name in sorted(role_values)],
        )
        conn.commit()
    finally:
        conn.close()


def create_session(
    db_path: Path,
    *,
    session_id: str,
    user_id: str,
    csrf_token: str,
    expires_at: str,
    now_iso: str,
    ip_address: str | None,
    user_agent: str | None,
) -> AuthSession:
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO serve_sessions(
                id, user_id, csrf_token, expires_at, created_at, last_seen_at,
                revoked_at, ip_address, user_agent
            ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            (
                session_id,
                user_id,
                csrf_token,
                expires_at,
                now_iso,
                now_iso,
                ip_address,
                user_agent,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    session = get_session(db_path, session_id)
    if session is None:
        raise RuntimeError("Failed to persist Serve auth session")
    return session


def get_session(db_path: Path, session_id: str) -> AuthSession | None:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM serve_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        return _row_to_session(row)
    finally:
        conn.close()


def revoke_session(db_path: Path, session_id: str, *, revoked_at: str) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            "UPDATE serve_sessions SET revoked_at = ? WHERE id = ?",
            (revoked_at, session_id),
        )
        conn.commit()
    finally:
        conn.close()


def touch_session(db_path: Path, session_id: str, *, last_seen_at: str) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            "UPDATE serve_sessions SET last_seen_at = ? WHERE id = ?",
            (last_seen_at, session_id),
        )
        conn.commit()
    finally:
        conn.close()


def list_users(db_path: Path) -> list[AuthUser]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM serve_users ORDER BY COALESCE(last_login_at, created_at, '') DESC, login ASC"
        ).fetchall()
    finally:
        conn.close()
    return [
        _row_to_user(row, list_roles_for_user(db_path, row["id"]))  # type: ignore[arg-type]
        for row in rows
    ]
