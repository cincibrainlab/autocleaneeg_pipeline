"""Models for Serve authentication and authorization."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AuthMode(str, Enum):
    """Authentication mode for the Serve web app."""

    DISABLED = "disabled"
    OAUTH = "oauth"


class Role(str, Enum):
    """Built-in roles for Serve."""

    VIEWER = "viewer"
    OPERATOR = "operator"
    EDITOR = "editor"
    ADMIN = "admin"


class Permission(str, Enum):
    """Centralized permission ids for the first Serve auth release."""

    DASHBOARD_READ = "dashboard.read"
    ROUTES_READ = "routes.read"
    ROUTES_WRITE = "routes.write"
    CONFIG_READ = "config.read"
    CONFIG_DEPLOY = "config.deploy"
    RESULTS_READ = "results.read"
    RESULTS_WRITE = "results.write"
    EXCLUDE_READ = "exclude.read"
    EXCLUDE_WRITE = "exclude.write"
    TASKS_READ = "tasks.read"
    TASKS_WRITE = "tasks.write"
    MONTAGES_READ = "montages.read"
    FILESYSTEM_BROWSE = "filesystem.browse"
    WORKER_READ = "worker.read"
    WORKER_CONTROL = "worker.control"
    TUTORIAL_WRITE = "tutorial.write"
    EVENT_ANALYZE = "event.analyze"
    SERVICE_CONTROL = "service.control"
    QUEUE_CONTROL = "queue.control"
    TUNNEL_ADMIN = "tunnel.admin"
    AUTH_ADMIN = "auth.admin"
    USERS_ADMIN = "users.admin"
    EVENTS_READ = "events.read"


ROLE_PERMISSIONS: dict[Role, tuple[Permission, ...]] = {
    Role.VIEWER: (
        Permission.DASHBOARD_READ,
        Permission.ROUTES_READ,
        Permission.CONFIG_READ,
        Permission.RESULTS_READ,
        Permission.EXCLUDE_READ,
        Permission.TASKS_READ,
        Permission.MONTAGES_READ,
        Permission.FILESYSTEM_BROWSE,
        Permission.EVENT_ANALYZE,
        Permission.EVENTS_READ,
    ),
    Role.OPERATOR: (
        Permission.DASHBOARD_READ,
        Permission.ROUTES_READ,
        Permission.CONFIG_READ,
        Permission.RESULTS_READ,
        Permission.RESULTS_WRITE,
        Permission.EXCLUDE_READ,
        Permission.EXCLUDE_WRITE,
        Permission.TASKS_READ,
        Permission.MONTAGES_READ,
        Permission.FILESYSTEM_BROWSE,
        Permission.WORKER_READ,
        Permission.WORKER_CONTROL,
        Permission.EVENT_ANALYZE,
        Permission.EVENTS_READ,
        Permission.SERVICE_CONTROL,
        Permission.QUEUE_CONTROL,
    ),
    Role.EDITOR: (
        Permission.DASHBOARD_READ,
        Permission.ROUTES_READ,
        Permission.ROUTES_WRITE,
        Permission.CONFIG_READ,
        Permission.CONFIG_DEPLOY,
        Permission.RESULTS_READ,
        Permission.RESULTS_WRITE,
        Permission.EXCLUDE_READ,
        Permission.EXCLUDE_WRITE,
        Permission.TASKS_READ,
        Permission.TASKS_WRITE,
        Permission.MONTAGES_READ,
        Permission.FILESYSTEM_BROWSE,
        Permission.WORKER_READ,
        Permission.WORKER_CONTROL,
        Permission.TUTORIAL_WRITE,
        Permission.EVENT_ANALYZE,
        Permission.EVENTS_READ,
        Permission.SERVICE_CONTROL,
        Permission.QUEUE_CONTROL,
    ),
    Role.ADMIN: tuple(permission for permission in Permission),
}


class SessionCookieConfig(BaseModel):
    """Cookie settings for Serve sessions."""

    cookie_name: str = Field(default="autoclean_session", min_length=1)
    ttl_hours: int = Field(default=12, ge=1, le=24 * 30)
    secure: bool | None = None


class GitHubAuthConfig(BaseModel):
    """GitHub OAuth configuration."""

    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str = "http://localhost:8000/api/auth/callback/github"
    allowed_orgs: list[str] = Field(default_factory=list)
    allowed_users: list[str] = Field(default_factory=list)


class OIDCAuthConfig(BaseModel):
    """Generic OIDC configuration."""

    issuer_url: str = ""
    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str = "http://localhost:8000/api/auth/callback/oidc"
    scopes: list[str] = Field(default_factory=lambda: ["openid", "profile", "email"])
    allowed_groups: list[str] = Field(default_factory=list)
    allowed_users: list[str] = Field(default_factory=list)
    username_claim: str = "preferred_username"
    groups_claim: str = "groups"


class AuthConfig(BaseModel):
    """Workspace-scoped Serve auth configuration."""

    model_config = ConfigDict(use_enum_values=True)

    mode: AuthMode = AuthMode.OAUTH
    provider: str = "github"
    allow_disable_auth: bool = True
    session: SessionCookieConfig = Field(default_factory=SessionCookieConfig)
    github: GitHubAuthConfig = Field(default_factory=GitHubAuthConfig)
    oidc: OIDCAuthConfig = Field(default_factory=OIDCAuthConfig)
    bootstrap_admins: list[str] = Field(default_factory=list)


class ProviderIdentity(BaseModel):
    """Normalized identity returned by an auth provider."""

    provider: str
    subject: str
    login: str
    email: str | None = None
    display_name: str | None = None
    avatar_url: str | None = None
    groups: list[str] = Field(default_factory=list)
    raw_claims: dict[str, Any] = Field(default_factory=dict)


class AuthUser(BaseModel):
    """Persisted Serve user model."""

    id: str
    provider: str
    subject: str
    login: str
    email: str | None = None
    display_name: str | None = None
    avatar_url: str | None = None
    created_at: str | None = None
    last_login_at: str | None = None
    disabled: bool = False
    roles: list[Role] = Field(default_factory=list)


class AuthSession(BaseModel):
    """Persisted Serve session model."""

    id: str
    user_id: str
    csrf_token: str
    expires_at: str
    created_at: str | None = None
    last_seen_at: str | None = None
    revoked_at: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None
