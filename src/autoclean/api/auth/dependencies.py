"""FastAPI auth dependencies for Serve."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import Depends, HTTPException, Request

from autoclean.api.auth.models import Permission
from autoclean.api.auth.service import (
    auth_is_enforced,
    ensure_store_ready,
    get_current_user_permissions,
    get_user_for_session_user_id,
    utc_now_iso,
)
from autoclean.api.auth.store import get_session, revoke_session, touch_session
from autoclean.api.state import api_state


def _now() -> datetime:
    return datetime.now(timezone.utc)


def get_current_user(request: Request):
    """Resolve the current authenticated user from the session cookie."""
    if not auth_is_enforced():
        return None

    ensure_store_ready()
    from autoclean.api.auth.service import load_current_auth_config

    config = load_current_auth_config()
    session_id = request.cookies.get(config.session.cookie_name)
    if not session_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    session = get_session(api_state.get_auth_db_path(create_parent=True), session_id)
    if session is None or session.revoked_at:
        raise HTTPException(status_code=401, detail="Session is invalid")

    expires_at = datetime.fromisoformat(session.expires_at)
    if expires_at <= _now():
        revoke_session(
            api_state.get_auth_db_path(create_parent=True),
            session.id,
            revoked_at=utc_now_iso(),
        )
        raise HTTPException(status_code=401, detail="Session has expired")

    touch_session(
        api_state.get_auth_db_path(create_parent=True),
        session.id,
        last_seen_at=utc_now_iso(),
    )
    user = get_user_for_session_user_id(session.user_id)
    if user is None or user.disabled:
        raise HTTPException(status_code=401, detail="User is unavailable")
    request.state.auth_session = session
    request.state.auth_user = user
    return user


def require_permission(permission: Permission):
    """Return a dependency that enforces a specific permission."""

    def _dependency(request: Request, user=Depends(get_current_user)):
        if not auth_is_enforced():
            return None
        if user is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        if request.method.upper() not in {"GET", "HEAD", "OPTIONS"}:
            session = getattr(request.state, "auth_session", None)
            csrf_token = request.headers.get("x-csrf-token")
            if session is None or not csrf_token or csrf_token != session.csrf_token:
                raise HTTPException(status_code=403, detail="CSRF token required")
        permissions = get_current_user_permissions(user.id)
        if permission.value not in permissions:
            raise HTTPException(
                status_code=403,
                detail=f"Permission required: {permission.value}",
            )
        return user

    return _dependency
