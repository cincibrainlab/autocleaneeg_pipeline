"""Session cookie helpers for Serve auth."""

from __future__ import annotations

from fastapi import Request, Response

from autoclean.api.auth.service import load_current_auth_config

SESSION_STATE_COOKIE = "autoclean_oauth_state"


def _cookie_secure(request: Request) -> bool:
    config = load_current_auth_config()
    if config.session.secure is not None:
        return bool(config.session.secure)
    forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",", 1)[0].strip().lower()
    if forwarded_proto:
        return forwarded_proto == "https"
    forwarded = request.headers.get("forwarded", "")
    if "proto=https" in forwarded.lower():
        return True
    return request.url.scheme == "https"


def set_session_cookie(response: Response, session_id: str, request: Request) -> None:
    """Set the session cookie on the response."""
    config = load_current_auth_config()
    response.set_cookie(
        key=config.session.cookie_name,
        value=session_id,
        httponly=True,
        secure=_cookie_secure(request),
        samesite="lax",
        max_age=config.session.ttl_hours * 3600,
        path="/",
    )


def clear_session_cookie(response: Response, request: Request) -> None:
    """Remove the session cookie from the response."""
    config = load_current_auth_config()
    response.delete_cookie(
        config.session.cookie_name,
        path="/",
        secure=_cookie_secure(request),
        samesite="lax",
    )


def set_oauth_state_cookie(response: Response, state: str, request: Request) -> None:
    """Persist the pending OAuth state between login and callback."""
    response.set_cookie(
        key=SESSION_STATE_COOKIE,
        value=state,
        httponly=True,
        secure=_cookie_secure(request),
        samesite="lax",
        max_age=600,
        path="/",
    )


def clear_oauth_state_cookie(response: Response, request: Request) -> None:
    """Remove the pending OAuth state cookie."""
    response.delete_cookie(
        SESSION_STATE_COOKIE,
        path="/",
        secure=_cookie_secure(request),
        samesite="lax",
    )
