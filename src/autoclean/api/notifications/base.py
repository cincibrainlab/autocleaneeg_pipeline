"""Notification provider interfaces."""

from __future__ import annotations

from typing import Protocol

from autoclean.api.notifications.models import DeliveryResult, EmailMessage


class NotificationProvider(Protocol):
    """Provider contract for outbound notification delivery."""

    def is_configured(self) -> bool:
        """Return True when the provider can send email."""

    def send(self, message: EmailMessage) -> DeliveryResult:
        """Send an email message."""
