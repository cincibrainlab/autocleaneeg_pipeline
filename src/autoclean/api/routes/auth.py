"""Serve auth endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from autoclean.api.auth.dependencies import get_current_user
from autoclean.api.auth.github import GitHubAuthError
from autoclean.api.auth.oidc import OIDCAuthError
from autoclean.api.auth.service import (
    auth_is_enforced,
    auth_provider_health,
    build_login_url,
    complete_login,
    get_current_user_permissions,
    get_provider,
    load_current_auth_config,
    utc_now_iso,
)
from autoclean.api.auth.session import (
    SESSION_STATE_COOKIE,
    clear_oauth_state_cookie,
    clear_session_cookie,
    set_oauth_state_cookie,
    set_session_cookie,
)
from autoclean.api.auth.store import ensure_auth_schema, revoke_session
from autoclean.api.state import api_state

router = APIRouter()


class LoginRequest(BaseModel):
    provider: str = "github"


def _bootstrap_allowed(request: Request, configured: bool) -> bool:
    client_host = request.client.host if request.client else "127.0.0.1"
    is_local = client_host in {"127.0.0.1", "::1", "testclient"}
    config_exists = api_state.get_auth_config_path().exists()
    return is_local and (not config_exists or not auth_is_enforced()) and not configured


@router.get("/status")
async def auth_status(request: Request) -> dict[str, object]:
    """Return current auth status for the active workspace."""
    if api_state.workspace_dir is None:
        return {
            "enabled": False,
            "mode": "disabled",
            "provider": None,
            "configured": False,
            "authenticated": False,
        }

    config = load_current_auth_config()
    provider_health = auth_provider_health()
    configured_providers = [name for name, status in provider_health.items() if status.get("configured")]
    configured = bool(configured_providers)
    user = None
    try:
        user = await _resolve_optional_user(request)
    except HTTPException:
        user = None
    return {
        "enabled": auth_is_enforced(),
        "mode": config.mode if isinstance(config.mode, str) else config.mode.value,
        "provider": config.provider,
        "configured": configured,
        "selected_provider_configured": bool(provider_health.get(config.provider, {}).get("configured")),
        "authenticated": user is not None,
        "providers": provider_health,
        "configured_providers": configured_providers,
        "bootstrap_allowed": _bootstrap_allowed(request, configured),
    }


async def _resolve_optional_user(request: Request):
    if not auth_is_enforced():
        return None
    return get_current_user(request)


@router.post("/login")
async def login(body: LoginRequest, request: Request, response: Response) -> dict[str, str]:
    """Start an OAuth login flow."""
    if api_state.workspace_dir is None:
        raise HTTPException(status_code=409, detail="Workspace not configured")
    try:
        login_url, state = build_login_url(provider=body.provider)
    except (ValueError, GitHubAuthError, OIDCAuthError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    set_oauth_state_cookie(response, state, request)
    return {"provider": body.provider, "login_url": login_url}


@router.get("/callback/{provider}")
async def auth_callback(
    provider: str,
    code: str,
    state: str,
    request: Request,
    response: Response,
) -> Response:
    """Complete an OAuth callback."""
    expected_state = request.cookies.get(SESSION_STATE_COOKIE)
    if not expected_state or expected_state != state:
        raise HTTPException(status_code=400, detail="Invalid OAuth state")

    try:
        session_id, csrf_token, user_id = complete_login(
            provider=provider,
            code=code,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
    except (ValueError, GitHubAuthError, OIDCAuthError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    redirect = RedirectResponse(url="/", status_code=303)
    clear_oauth_state_cookie(redirect, request)
    set_session_cookie(redirect, session_id, request)
    return redirect


@router.post("/logout")
async def logout(request: Request, response: Response) -> dict[str, bool]:
    """End the current Serve session."""
    if api_state.workspace_dir is None:
        return {"success": True}
    config = load_current_auth_config()
    session_id = request.cookies.get(config.session.cookie_name)
    if session_id:
        ensure_auth_schema(api_state.get_auth_db_path(create_parent=True))
        revoke_session(
            api_state.get_auth_db_path(create_parent=True),
            session_id,
            revoked_at=utc_now_iso(),
        )
    clear_session_cookie(response, request)
    clear_oauth_state_cookie(response, request)
    return {"success": True}


@router.get("/me")
async def me(request: Request) -> dict[str, object]:
    """Return the current authenticated user."""
    user = await _resolve_optional_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    session = getattr(request.state, "auth_session", None)
    return {
        "user": user.model_dump(mode="json"),
        "permissions": get_current_user_permissions(user.id),
        "csrf_token": session.csrf_token if session is not None else None,
    }
