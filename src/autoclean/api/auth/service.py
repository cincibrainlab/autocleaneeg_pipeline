"""High-level auth helpers for Serve."""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone

from autoclean.api.auth.config import load_auth_config
from autoclean.api.auth.base import AuthProvider
from autoclean.api.auth.github import GitHubAuthError, GitHubAuthProvider
from autoclean.api.auth.oidc import OIDCAuthError, OIDCAuthProvider
from autoclean.api.auth.models import AuthConfig, AuthMode, ProviderIdentity, Role
from autoclean.api.auth.store import (
    create_session,
    ensure_auth_schema,
    get_user_by_identity,
    get_user_by_id,
    list_permissions_for_roles,
    list_roles_for_user,
    set_user_roles,
    upsert_user,
)
from autoclean.api.state import api_state


def utc_now() -> datetime:
    """Return timezone-aware current UTC time."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Return current UTC time as ISO string."""
    return utc_now().isoformat()


def load_current_auth_config() -> AuthConfig:
    """Load auth config for the current workspace."""
    return load_auth_config(api_state.get_auth_config_path())


def auth_is_enforced() -> bool:
    """Return True when Serve auth should be enforced for the active workspace."""
    if api_state.workspace_dir is None:
        return False
    return load_current_auth_config().mode == AuthMode.OAUTH


def _provider_registry(config: AuthConfig) -> dict[str, AuthProvider]:
    """Return the configured auth provider registry."""
    return {
        "github": GitHubAuthProvider(config.github),
        "oidc": OIDCAuthProvider(config.oidc),
    }


def get_provider(name: str) -> AuthProvider:
    """Return the configured auth provider."""
    config = load_current_auth_config()
    registry = _provider_registry(config)
    if name not in registry:
        raise ValueError(f"Unsupported auth provider: {name}")
    return registry[name]


def ensure_store_ready() -> None:
    """Create the auth store schema if required."""
    ensure_auth_schema(api_state.get_auth_db_path(create_parent=True))


def build_login_url(*, provider: str) -> tuple[str, str]:
    """Build login URL and return (url, state_token)."""
    auth_provider = get_provider(provider)
    state = secrets.token_urlsafe(24)
    return auth_provider.build_login_url(state=state), state


def _roles_for_identity(identity: ProviderIdentity, config: AuthConfig) -> list[Role]:
    if identity.login in config.bootstrap_admins:
        return [Role.ADMIN]
    existing = get_user_by_identity(
        api_state.get_auth_db_path(create_parent=True),
        identity.provider,
        identity.subject,
    )
    if existing is not None and existing.roles:
        return existing.roles
    return [Role.VIEWER]


def complete_login(
    *,
    provider: str,
    code: str,
    ip_address: str | None,
    user_agent: str | None,
) -> tuple[str, str, str]:
    """Complete provider login and return session details.

    Returns (session_id, csrf_token, user_id).
    """
    ensure_store_ready()
    config = load_current_auth_config()
    auth_provider = get_provider(provider)
    identity = auth_provider.exchange_code(code=code)

    if provider == "github":
        allowed_users = {user for user in config.github.allowed_users if user}
        if allowed_users and identity.login not in allowed_users:
            raise GitHubAuthError(f"GitHub user '{identity.login}' is not allowed")
        allowed_orgs = {org for org in config.github.allowed_orgs if org}
        if allowed_orgs and not (allowed_orgs & set(identity.groups)):
            raise GitHubAuthError(
                f"GitHub user '{identity.login}' is not a member of an allowed organization"
            )
    elif provider == "oidc":
        allowed_users = {user for user in config.oidc.allowed_users if user}
        if allowed_users and identity.login not in allowed_users:
            raise OIDCAuthError(f"OIDC user '{identity.login}' is not allowed")
        allowed_groups = {group for group in config.oidc.allowed_groups if group}
        if allowed_groups and not (allowed_groups & set(identity.groups)):
            raise OIDCAuthError(
                f"OIDC user '{identity.login}' is not a member of an allowed group"
            )

    user_id = f"{identity.provider}:{identity.subject}"
    now_iso = utc_now_iso()
    user = upsert_user(
        api_state.get_auth_db_path(create_parent=True),
        user_id=user_id,
        provider=identity.provider,
        subject=identity.subject,
        login=identity.login,
        email=identity.email,
        display_name=identity.display_name,
        avatar_url=identity.avatar_url,
        now_iso=now_iso,
    )
    roles = _roles_for_identity(identity, config)
    set_user_roles(
        api_state.get_auth_db_path(create_parent=True),
        user.id,
        roles,
        granted_by="system",
        granted_at=now_iso,
    )

    session_id = secrets.token_urlsafe(32)
    csrf_token = secrets.token_urlsafe(24)
    expires_at = (utc_now() + timedelta(hours=config.session.ttl_hours)).isoformat()
    create_session(
        api_state.get_auth_db_path(create_parent=True),
        session_id=session_id,
        user_id=user.id,
        csrf_token=csrf_token,
        expires_at=expires_at,
        now_iso=now_iso,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    return session_id, csrf_token, user.id


def get_current_user_permissions(user_id: str) -> list[str]:
    """Return sorted permission ids for a user."""
    roles = list_roles_for_user(api_state.get_auth_db_path(create_parent=True), user_id)
    return [permission.value for permission in list_permissions_for_roles(roles)]


def auth_provider_health() -> dict[str, dict[str, object]]:
    """Return provider-specific configured/healthy information."""
    config = load_current_auth_config()
    registry = _provider_registry(config)
    health: dict[str, dict[str, object]] = {}
    for name, provider in registry.items():
        configured = getattr(provider, "is_configured", lambda: False)()
        health[name] = {
            "configured": bool(configured),
            "selected": config.provider == name,
        }
    return health


def get_user_for_session_user_id(user_id: str):
    """Return the persisted user model for a session-owned user id."""
    return get_user_by_id(api_state.get_auth_db_path(create_parent=True), user_id)
