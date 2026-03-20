"""Admin endpoints for Serve notification configuration."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from autoclean.api.audit import log_audit_event
from autoclean.api.auth.dependencies import require_permission
from autoclean.api.auth.models import Permission
from autoclean.api.notifications.models import NotificationConfig, NotificationEventKey
from autoclean.api.notifications.service import (
    load_current_notification_config,
    masked_notification_config,
    notification_provider_health,
    save_current_notification_config,
    send_daily_digest,
    try_send_email,
)
from autoclean.api.state import api_state

router = APIRouter()


class TestEmailRequest(BaseModel):
    to: list[str] = Field(default_factory=list)
    subject: str = "AutoClean Serve test email"
    message: str = "This is a test email from AutoClean Serve."


def _require_workspace() -> None:
    if api_state.workspace_dir is None:
        raise HTTPException(status_code=409, detail="Workspace not configured")


@router.get(
    "/notifications/config",
    dependencies=[Depends(require_permission(Permission.AUTH_ADMIN))],
)
async def get_notifications_config():
    """Return workspace notification config with masked secrets."""
    _require_workspace()
    payload = masked_notification_config(load_current_notification_config())
    payload["providers"] = notification_provider_health()
    return payload


@router.put(
    "/notifications/config",
    dependencies=[Depends(require_permission(Permission.AUTH_ADMIN))],
)
async def put_notifications_config(body: NotificationConfig, request: Request):
    """Persist workspace notification config."""
    _require_workspace()
    try:
        saved = save_current_notification_config(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    admin_user = getattr(request.state, "auth_user", None)
    log_audit_event(
        api_state.get_auth_db_path(create_parent=True),
        action="update",
        resource_type="notification_config",
        resource_id="workspace",
        actor_user_id=getattr(admin_user, "id", None),
        actor_login=getattr(admin_user, "login", None),
        details_json=json.dumps(
            {
                "enabled": saved.enabled,
                "provider": saved.provider,
                "recipients": saved.recipients.model_dump(mode="json"),
                "cooldown_minutes": saved.cooldown_minutes.model_dump(mode="json"),
                "resend_sender_email": saved.resend.sender_email,
                "resend_sender_name": saved.resend.sender_name,
                "resend_reply_to": saved.resend.reply_to,
                "resend_api_key_present": bool(saved.resend.api_key),
            }
        ),
        ip_address=request.client.host if request.client else None,
    )
    return {"success": True, "config": masked_notification_config(saved)}


@router.post(
    "/notifications/test-email",
    dependencies=[Depends(require_permission(Permission.AUTH_ADMIN))],
)
async def send_test_email(body: TestEmailRequest, request: Request):
    """Send a test notification using the current provider settings."""
    _require_workspace()
    result = try_send_email(
        event_key=NotificationEventKey.SERVICE,
        subject=body.subject,
        text=body.message,
        dedupe_key="test-email",
        recipients=[str(address) for address in body.to],
        force=True,
    )
    if not result.ok:
        raise HTTPException(status_code=400, detail=result.error or "Test email failed")
    admin_user = getattr(request.state, "auth_user", None)
    log_audit_event(
        api_state.get_auth_db_path(create_parent=True),
        action="send_test_email",
        resource_type="notification_config",
        resource_id="workspace",
        actor_user_id=getattr(admin_user, "id", None),
        actor_login=getattr(admin_user, "login", None),
        details_json=json.dumps({"recipient_count": len(body.to), "subject": body.subject}),
        ip_address=request.client.host if request.client else None,
    )
    return {"success": True, "message_id": result.message_id}


@router.post(
    "/notifications/daily-digest",
    dependencies=[Depends(require_permission(Permission.AUTH_ADMIN))],
)
async def run_daily_digest(request: Request):
    """Trigger a daily digest email immediately."""
    _require_workspace()
    result = send_daily_digest()
    if not result.ok:
        raise HTTPException(status_code=400, detail=result.error or "Daily digest failed")
    admin_user = getattr(request.state, "auth_user", None)
    log_audit_event(
        api_state.get_auth_db_path(create_parent=True),
        action="send_daily_digest",
        resource_type="notification_config",
        resource_id="workspace",
        actor_user_id=getattr(admin_user, "id", None),
        actor_login=getattr(admin_user, "login", None),
        details_json=json.dumps({"recipient_count": len(load_current_notification_config().daily_digest_recipients)}),
        ip_address=request.client.host if request.client else None,
    )
    return {"success": True, "message_id": result.message_id}
