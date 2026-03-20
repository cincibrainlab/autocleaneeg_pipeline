"""Resend-backed email delivery."""

from __future__ import annotations

import requests

from autoclean.api.notifications.models import DeliveryResult, EmailMessage, ResendConfig


class ResendProvider:
    """Simple Resend email sender."""

    API_URL = "https://api.resend.com/emails"

    def __init__(self, config: ResendConfig) -> None:
        self._config = config

    def is_configured(self) -> bool:
        return bool(self._config.api_key and self._config.sender_email)

    def send(self, message: EmailMessage) -> DeliveryResult:
        if not self.is_configured():
            return DeliveryResult(ok=False, provider="resend", error="Resend is not configured")
        sender = self._config.sender_email
        if self._config.sender_name:
            sender = f"{self._config.sender_name} <{self._config.sender_email}>"
        payload = {
            "from": sender,
            "to": message.to,
            "subject": message.subject,
            "html": message.html,
            "text": message.text,
        }
        if message.reply_to:
            payload["reply_to"] = message.reply_to
        response = requests.post(
            self.API_URL,
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=15,
        )
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            detail = None
            try:
                detail = response.text
            except Exception:
                detail = None
            error = detail or str(exc)
            return DeliveryResult(ok=False, provider="resend", error=error)
        data = response.json() if response.content else {}
        return DeliveryResult(
            ok=True,
            provider="resend",
            message_id=data.get("id") if isinstance(data, dict) else None,
        )
