"""Admin endpoints for Serve auth configuration."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request

from autoclean.api.audit import log_audit_event
from autoclean.api.auth.config import save_auth_config
from autoclean.api.auth.dependencies import get_current_user
from autoclean.api.auth.models import AuthConfig, Permission
from autoclean.api.auth.service import auth_is_enforced, auth_provider_health, get_current_user_permissions, load_current_auth_config
from autoclean.api.state import api_state

router = APIRouter()


def _allow_bootstrap_or_admin(request: Request):
    client_host = request.client.host if request.client else "127.0.0.1"
    if client_host in {"127.0.0.1", "::1", "testclient"}:
        config_path = api_state.get_auth_config_path()
        config = load_current_auth_config()
        provider_health = auth_provider_health()
        any_configured = any(bool(status.get("configured")) for status in provider_health.values())
        if not auth_is_enforced() or (not config_path.exists() and not any_configured):
            return None
    user = get_current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    if Permission.AUTH_ADMIN.value not in get_current_user_permissions(user.id):
        raise HTTPException(status_code=403, detail="Permission required: auth.admin")
    if request.method.upper() not in {"GET", "HEAD", "OPTIONS"}:
        session = getattr(request.state, "auth_session", None)
        csrf_token = request.headers.get("x-csrf-token")
        if session is None or not csrf_token or csrf_token != session.csrf_token:
            raise HTTPException(status_code=403, detail="CSRF token required")
    return user


def _configured_provider_names(config: AuthConfig) -> list[str]:
    providers = {
        "github": bool(config.github.client_id.strip() and config.github.client_secret.strip()),
        "oidc": bool(config.oidc.issuer_url.strip() and config.oidc.client_id.strip() and config.oidc.client_secret.strip()),
    }
    return [name for name, configured in providers.items() if configured]


@router.get("/auth/config")
async def get_auth_config(request: Request):
    """Return the current workspace auth configuration."""
    if api_state.workspace_dir is None:
        raise HTTPException(status_code=409, detail="Workspace not configured")
    _allow_bootstrap_or_admin(request)
    config = load_current_auth_config()
    payload = config.model_dump(mode="json")
    if payload.get("github", {}).get("client_secret"):
        payload["github"]["client_secret"] = "***"
    if payload.get("oidc", {}).get("client_secret"):
        payload["oidc"]["client_secret"] = "***"
    payload["providers"] = auth_provider_health()
    return payload


@router.put("/auth/config")
async def put_auth_config(body: AuthConfig, request: Request):
    """Persist auth configuration for the current workspace."""
    if api_state.workspace_dir is None:
        raise HTTPException(status_code=409, detail="Workspace not configured")
    user = _allow_bootstrap_or_admin(request)
    current = load_current_auth_config()
    if body.provider not in {"github", "oidc"}:
        raise HTTPException(status_code=400, detail=f"Unsupported auth provider: {body.provider}")
    if body.allow_disable_auth is False and body.mode != "oauth":
        raise HTTPException(status_code=400, detail="Auth cannot be disabled while allow_disable_auth is false")
    if current.allow_disable_auth is False and body.mode != "oauth":
        raise HTTPException(status_code=400, detail="This workspace requires auth to remain enabled")
    if body.mode == "oauth":
        configured_providers = _configured_provider_names(body)
        if not configured_providers:
            raise HTTPException(status_code=400, detail="OAuth mode requires at least one configured auth provider")
    save_auth_config(api_state.get_auth_config_path(), body)
    log_audit_event(
        api_state.get_auth_db_path(create_parent=True),
        action="update",
        resource_type="auth_config",
        resource_id="workspace",
        actor_user_id=getattr(user, "id", None),
        actor_login=getattr(user, "login", None),
        details_json=json.dumps(
            {
                "mode": body.mode,
                "provider": body.provider,
                "allow_disable_auth": body.allow_disable_auth,
                "github_client_id_present": bool(body.github.client_id),
                "github_client_secret_present": bool(body.github.client_secret),
                "allowed_orgs": body.github.allowed_orgs,
                "allowed_users": body.github.allowed_users,
                "bootstrap_admins": body.bootstrap_admins,
            }
        ),
        ip_address=request.client.host if request.client else None,
    )
    return {"success": True}
