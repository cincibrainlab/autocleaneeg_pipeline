"""Tests for Serve auth config primitives."""

from __future__ import annotations

import json

import pytest

from autoclean.api.auth.config import load_auth_config, save_auth_config
from autoclean.api.auth.models import AuthConfig, AuthMode, Permission, ROLE_PERMISSIONS, Role


def test_load_auth_config_returns_defaults_when_missing(tmp_path) -> None:
    """Missing config should return default auth settings."""
    path = tmp_path / "serve-auth.json"

    config = load_auth_config(path)

    assert config.mode == AuthMode.OAUTH
    assert config.provider == "github"
    assert config.session.cookie_name == "autoclean_session"
    assert config.session.secure is None


def test_save_then_load_auth_config_round_trips(tmp_path) -> None:
    """Saved auth config should deserialize back to the same values."""
    path = tmp_path / "serve-auth.json"
    config = AuthConfig(
        mode=AuthMode.DISABLED,
        bootstrap_admins=["lab-admin", "pi-user"],
    )
    config.github.allowed_orgs = ["example-lab"]
    config.github.allowed_users = ["analyst1"]
    config.oidc.issuer_url = "https://issuer.example.com"
    config.oidc.client_id = "oidc-client"
    config.oidc.client_secret = "oidc-secret"

    save_auth_config(path, config)
    loaded = load_auth_config(path)

    assert path.exists()
    assert loaded.mode == AuthMode.DISABLED
    assert loaded.bootstrap_admins == ["lab-admin", "pi-user"]
    assert loaded.github.allowed_orgs == ["example-lab"]
    assert loaded.github.allowed_users == ["analyst1"]
    assert loaded.oidc.issuer_url == "https://issuer.example.com"
    assert loaded.oidc.client_secret == "oidc-secret"


def test_load_auth_config_rejects_invalid_json(tmp_path) -> None:
    """Invalid JSON should raise a clear error."""
    path = tmp_path / "serve-auth.json"
    path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid auth config JSON"):
        load_auth_config(path)


def test_saved_auth_config_uses_json_enum_values(tmp_path) -> None:
    """Enum-backed config should serialize to plain JSON values."""
    path = tmp_path / "serve-auth.json"

    save_auth_config(path, AuthConfig(mode=AuthMode.DISABLED))

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["mode"] == "disabled"


def test_auth_secrets_are_stored_by_reference(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOCLEAN_SECRET_BACKEND", "file")
    path = tmp_path / "serve-auth.json"
    config = AuthConfig()
    config.github.client_secret = "github-secret"
    config.oidc.client_secret = "oidc-secret"

    save_auth_config(path, config)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["github"]["client_secret"].startswith("secret://")
    assert payload["oidc"]["client_secret"].startswith("secret://")
    secret_store = json.loads((tmp_path / ".serve" / "secret_store.json").read_text(encoding="utf-8"))
    assert secret_store["auth/github/client_secret"] == "github-secret"
    assert secret_store["auth/oidc/client_secret"] == "oidc-secret"


def test_role_permissions_cover_all_builtin_roles() -> None:
    """Every built-in role should have a permission mapping."""
    assert set(ROLE_PERMISSIONS) == set(Role)
    assert Permission.DASHBOARD_READ in ROLE_PERMISSIONS[Role.VIEWER]
    assert Permission.AUTH_ADMIN in ROLE_PERMISSIONS[Role.ADMIN]
