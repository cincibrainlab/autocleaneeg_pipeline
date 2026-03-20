"""Workspace-scoped notification config persistence."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from autoclean.api.notifications.models import NotificationConfig
from autoclean.api.secrets import is_secret_ref, resolve_secret, store_secret


def load_notification_config(path: Path) -> NotificationConfig:
    """Load the Serve notification config from disk."""
    if not path.exists():
        return NotificationConfig()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid notification config JSON: {exc}") from exc
    try:
        config = NotificationConfig.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid notification config payload: {exc}") from exc
    config.resend.api_key = resolve_secret(path, config.resend.api_key)
    return config


def save_notification_config(path: Path, config: NotificationConfig) -> None:
    """Persist the Serve notification config to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json")
    api_key = payload.get("resend", {}).get("api_key", "")
    if api_key and not is_secret_ref(api_key) and api_key != "***":
        payload["resend"]["api_key"] = store_secret(path, "notifications/resend/api_key", api_key)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
