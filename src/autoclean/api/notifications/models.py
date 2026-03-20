"""Models for Serve email notifications."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class NotificationProvider(str, Enum):
    """Supported notification providers."""

    RESEND = "resend"


class NotificationEventKey(str, Enum):
    """Notification event categories supported by Serve."""

    SERVICE = "service"
    CONFIG_FAILURE = "config_failure"
    QUEUE_FAILURE = "queue_failure"
    JOB_FAILURE = "job_failure"


class ResendConfig(BaseModel):
    """Resend provider settings."""

    api_key: str = ""
    sender_email: str = ""
    sender_name: str = "AutoClean Serve"
    reply_to: str = ""


class NotificationRecipients(BaseModel):
    """Recipients grouped by event category."""

    service: list[str] = Field(default_factory=list)
    config_failure: list[str] = Field(default_factory=list)
    queue_failure: list[str] = Field(default_factory=list)
    job_failure: list[str] = Field(default_factory=list)


class NotificationCooldowns(BaseModel):
    """Cooldown windows in minutes by event category."""

    service: int = Field(default=15, ge=0, le=24 * 60)
    config_failure: int = Field(default=30, ge=0, le=24 * 60)
    queue_failure: int = Field(default=60, ge=0, le=24 * 60)
    job_failure: int = Field(default=60, ge=0, le=24 * 60)


class NotificationConfig(BaseModel):
    """Workspace-scoped Serve notification configuration."""

    model_config = ConfigDict(use_enum_values=True)

    enabled: bool = False
    provider: NotificationProvider = NotificationProvider.RESEND
    app_base_url: str = ""
    resend: ResendConfig = Field(default_factory=ResendConfig)
    recipients: NotificationRecipients = Field(default_factory=NotificationRecipients)
    cooldown_minutes: NotificationCooldowns = Field(default_factory=NotificationCooldowns)
    route_recipients: dict[str, list[str]] = Field(default_factory=dict)
    daily_digest_recipients: list[str] = Field(default_factory=list)


class EmailMessage(BaseModel):
    """Normalized outbound email message."""

    subject: str
    html: str
    text: str
    to: list[str] = Field(default_factory=list)
    reply_to: str | None = None


class DeliveryResult(BaseModel):
    """Provider delivery result."""

    ok: bool
    provider: str
    message_id: str | None = None
    error: str | None = None
