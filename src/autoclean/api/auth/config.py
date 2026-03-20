"""Helpers for Serve auth configuration files."""

from __future__ import annotations

import json
from pathlib import Path

from autoclean.api.auth.models import AuthConfig
from autoclean.api.secrets import is_secret_ref, resolve_secret, store_secret


def load_auth_config(path: Path) -> AuthConfig:
    """Load auth config from disk or return defaults when missing."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return AuthConfig()
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid auth config JSON at {path}: {exc}") from exc

    try:
        config = AuthConfig.model_validate(raw)
    except Exception as exc:
        raise ValueError(f"Invalid auth config at {path}: {exc}") from exc
    config.github.client_secret = resolve_secret(path, config.github.client_secret)
    config.oidc.client_secret = resolve_secret(path, config.oidc.client_secret)
    return config


def save_auth_config(path: Path, config: AuthConfig) -> None:
    """Persist auth config to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json")
    github_secret = payload.get("github", {}).get("client_secret", "")
    if github_secret and not is_secret_ref(github_secret) and github_secret != "***":
        payload["github"]["client_secret"] = store_secret(path, "auth/github/client_secret", github_secret)
    oidc_secret = payload.get("oidc", {}).get("client_secret", "")
    if oidc_secret and not is_secret_ref(oidc_secret) and oidc_secret != "***":
        payload["oidc"]["client_secret"] = store_secret(path, "auth/oidc/client_secret", oidc_secret)
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
