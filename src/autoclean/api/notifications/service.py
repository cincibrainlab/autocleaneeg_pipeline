"""High-level Serve notification helpers."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone
from html import escape

from autoclean.api.notifications.base import NotificationProvider
from autoclean.api.notifications.config import load_notification_config, save_notification_config
from autoclean.api.notifications.models import (
    DeliveryResult,
    EmailMessage,
    NotificationConfig,
    NotificationEventKey,
)
from autoclean.api.notifications.resend import ResendProvider
from autoclean.api.state import api_state

MASKED_SECRET = "***"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def load_current_notification_config() -> NotificationConfig:
    """Load workspace-scoped notification config."""
    return load_notification_config(api_state.get_notifications_config_path())


def masked_notification_config(config: NotificationConfig) -> dict[str, object]:
    """Return a client-safe notification config payload."""
    payload = config.model_dump(mode="json")
    if payload.get("resend", {}).get("api_key"):
        payload["resend"]["api_key"] = MASKED_SECRET
        payload["resend"]["has_api_key"] = True
    else:
        payload["resend"]["has_api_key"] = False
    return payload


def save_current_notification_config(config: NotificationConfig) -> NotificationConfig:
    """Persist notification config, preserving masked secrets."""
    path = api_state.get_notifications_config_path()
    existing = load_notification_config(path)
    if config.resend.api_key == MASKED_SECRET:
        config.resend.api_key = existing.resend.api_key
    validate_notification_config(config)
    save_notification_config(path, config)
    return load_notification_config(path)


def get_provider(config: NotificationConfig) -> NotificationProvider:
    """Return the configured notification provider."""
    registry: dict[str, NotificationProvider] = {
        "resend": ResendProvider(config.resend),
    }
    if config.provider in registry:
        return registry[config.provider]
    raise ValueError(f"Unsupported notification provider: {config.provider}")


def validate_notification_config(config: NotificationConfig) -> None:
    """Raise ValueError when notification config is obviously incomplete."""
    if not config.enabled:
        return
    if config.provider != "resend":
        raise ValueError(f"Unsupported notification provider: {config.provider}")
    if not config.resend.api_key.strip():
        raise ValueError("Resend API key is required when notifications are enabled")
    if not config.resend.sender_email.strip():
        raise ValueError("Sender email is required when notifications are enabled")


def notification_provider_health() -> dict[str, dict[str, object]]:
    """Return provider configured/healthy information for notifications."""
    config = load_current_notification_config()
    providers: dict[str, NotificationProvider] = {
        "resend": ResendProvider(config.resend),
    }
    return {
        name: {
            "configured": provider.is_configured(),
            "selected": config.provider == name,
        }
        for name, provider in providers.items()
    }


def _connect_state_db() -> sqlite3.Connection:
    db_path = api_state.get_auth_db_path(create_parent=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_notification_state() -> None:
    """Bootstrap notification cooldown persistence."""
    conn = _connect_state_db()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS serve_notification_history (
                dedupe_key TEXT PRIMARY KEY,
                event_key TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                last_sent_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _get_recipients(config: NotificationConfig, event_key: NotificationEventKey) -> list[str]:
    recipients = getattr(config.recipients, event_key.value, [])
    return sorted({email.strip() for email in recipients if email.strip()})


def _with_route_recipients(config: NotificationConfig, recipients: list[str], route_id: str | None) -> list[str]:
    route_specific = config.route_recipients.get(route_id or "", []) if route_id else []
    return sorted({email.strip() for email in [*recipients, *route_specific] if email.strip()})


def _get_cooldown_minutes(config: NotificationConfig, event_key: NotificationEventKey) -> int:
    return getattr(config.cooldown_minutes, event_key.value)


def _body_hash(subject: str, text: str) -> str:
    return hashlib.sha256(f"{subject}\n{text}".encode("utf-8")).hexdigest()


def should_send_notification(
    *,
    event_key: NotificationEventKey,
    dedupe_key: str,
    subject: str,
    text: str,
    config: NotificationConfig,
) -> bool:
    """Return True when cooldown rules permit sending this notification."""
    ensure_notification_state()
    conn = _connect_state_db()
    try:
        row = conn.execute(
            """
            SELECT content_hash, last_sent_at
            FROM serve_notification_history
            WHERE dedupe_key = ?
            """,
            (dedupe_key,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return True
    cooldown_minutes = _get_cooldown_minutes(config, event_key)
    if cooldown_minutes <= 0:
        return True
    last_sent = datetime.fromisoformat(row["last_sent_at"])
    content_hash = _body_hash(subject, text)
    within_cooldown = utc_now() < last_sent + timedelta(minutes=cooldown_minutes)
    return not (within_cooldown and row["content_hash"] == content_hash)


def record_notification_send(
    *,
    event_key: NotificationEventKey,
    dedupe_key: str,
    subject: str,
    text: str,
) -> None:
    """Persist notification send time for cooldown and dedupe."""
    ensure_notification_state()
    conn = _connect_state_db()
    try:
        conn.execute(
            """
            INSERT INTO serve_notification_history(dedupe_key, event_key, content_hash, last_sent_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(dedupe_key) DO UPDATE SET
                event_key = excluded.event_key,
                content_hash = excluded.content_hash,
                last_sent_at = excluded.last_sent_at
            """,
            (dedupe_key, event_key.value, _body_hash(subject, text), utc_now_iso()),
        )
        conn.commit()
    finally:
        conn.close()


def send_email(
    *,
    event_key: NotificationEventKey,
    subject: str,
    text: str,
    dedupe_key: str,
    recipients: list[str] | None = None,
    route_id: str | None = None,
    force: bool = False,
) -> DeliveryResult:
    """Send a notification if the workspace config allows it."""
    config = load_current_notification_config()
    if not config.enabled:
        return DeliveryResult(ok=False, provider=config.provider, error="Notifications are disabled")
    provider = get_provider(config)
    to = recipients or _with_route_recipients(config, _get_recipients(config, event_key), route_id)
    if not to:
        return DeliveryResult(ok=False, provider=config.provider, error="No recipients configured")
    if not provider.is_configured():
        return DeliveryResult(ok=False, provider=config.provider, error="Notification provider is not configured")
    if not force and not should_send_notification(
        event_key=event_key,
        dedupe_key=dedupe_key,
        subject=subject,
        text=text,
        config=config,
    ):
        return DeliveryResult(ok=True, provider=config.provider, message_id="cooldown-suppressed")
    app_link = config.app_base_url.rstrip("/") if config.app_base_url else ""
    html_body = f"<div style='font-family:Arial,sans-serif'><h2>{escape(subject)}</h2><pre style='white-space:pre-wrap'>{escape(text)}</pre>"
    if app_link:
        html_body += f"<p><a href='{escape(app_link)}'>Open AutoClean Serve</a></p>"
    html_body += "</div>"
    message = EmailMessage(
        subject=subject,
        text=text,
        html=html_body,
        to=to,
        reply_to=config.resend.reply_to or None,
    )
    result = provider.send(message)
    if result.ok:
        record_notification_send(
            event_key=event_key,
            dedupe_key=dedupe_key,
            subject=subject,
            text=text,
        )
    return result


def try_send_email(**kwargs) -> DeliveryResult:
    """Best-effort wrapper for runtime paths that must not fail closed."""
    try:
        return send_email(**kwargs)
    except Exception as exc:
        return DeliveryResult(ok=False, provider="internal", error=str(exc))


def send_daily_digest() -> DeliveryResult:
    """Send a simple daily digest email summarizing recent audit and notification activity."""
    config = load_current_notification_config()
    if not config.daily_digest_recipients:
        return DeliveryResult(ok=False, provider=config.provider, error="No daily digest recipients configured")
    ensure_notification_state()
    conn = _connect_state_db()
    try:
        audit_rows = conn.execute(
            """
            SELECT resource_type, action, COUNT(*) AS count
            FROM serve_audit_log
            WHERE created_at >= ?
            GROUP BY resource_type, action
            ORDER BY count DESC
            """,
            ((utc_now() - timedelta(days=1)).isoformat(),),
        ).fetchall()
        notification_rows = conn.execute(
            """
            SELECT event_key, COUNT(*) AS count
            FROM serve_notification_history
            WHERE last_sent_at >= ?
            GROUP BY event_key
            ORDER BY count DESC
            """,
            ((utc_now() - timedelta(days=1)).isoformat(),),
        ).fetchall()
    finally:
        conn.close()
    audit_lines = [f"- {row['resource_type']} / {row['action']}: {row['count']}" for row in audit_rows]
    notification_lines = [f"- {row['event_key']}: {row['count']}" for row in notification_rows]
    body = "Serve activity in the last 24 hours\n\nAudit events:\n"
    body += "\n".join(audit_lines) if audit_lines else "- none"
    body += "\n\nNotification sends:\n"
    body += "\n".join(notification_lines) if notification_lines else "- none"
    return send_email(
        event_key=NotificationEventKey.SERVICE,
        subject="AutoClean Serve daily digest",
        text=body,
        dedupe_key=f"daily-digest:{utc_now().date().isoformat()}",
        recipients=config.daily_digest_recipients,
        force=True,
    )
