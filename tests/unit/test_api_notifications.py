"""Tests for Serve notification config and delivery."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import requests

from autoclean.api.auth.config import save_auth_config
from autoclean.api.auth.models import AuthConfig, AuthMode, Role
from autoclean.api.auth.store import create_session, ensure_auth_schema, set_user_roles, upsert_user
from autoclean.api.notifications.config import load_notification_config
from autoclean.api.notifications.models import NotificationConfig, NotificationEventKey
from autoclean.api.notifications.service import send_email
from autoclean.api.server import create_app


class _MockResponse:
    def __init__(self, payload: dict[str, object] | None = None, status_code: int = 200) -> None:
        self._payload = payload or {"id": "email_123"}
        self.status_code = status_code
        self.content = b'{"id":"email_123"}'
        self.text = '{"id":"email_123"}'

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(self.text)

    def json(self) -> dict[str, object]:
        return self._payload


def _auth_client(tmp_path: Path, role: Role = Role.ADMIN) -> TestClient:
    config = AuthConfig(mode=AuthMode.OAUTH)
    save_auth_config(tmp_path / "serve-auth.json", config)
    ensure_auth_schema(tmp_path / ".serve" / "serve_state.db")
    user = upsert_user(
        tmp_path / ".serve" / "serve_state.db",
        user_id="github:admin",
        provider="github",
        subject="admin",
        login="admin-user",
        email="admin@example.edu",
        display_name="Admin User",
        avatar_url=None,
        now_iso="2026-03-20T00:00:00+00:00",
    )
    set_user_roles(
        tmp_path / ".serve" / "serve_state.db",
        user.id,
        [role],
        granted_by="system",
        granted_at="2026-03-20T00:00:00+00:00",
    )
    create_session(
        tmp_path / ".serve" / "serve_state.db",
        session_id="session-id",
        user_id=user.id,
        csrf_token="csrf-token",
        expires_at="2099-03-20T00:00:00+00:00",
        now_iso="2026-03-20T00:00:00+00:00",
        ip_address="127.0.0.1",
        user_agent="pytest",
    )
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)
    client.cookies.set(config.session.cookie_name, "session-id")
    return client


def test_notifications_config_round_trip_masks_api_key(tmp_path: Path) -> None:
    client = _auth_client(tmp_path)

    response = client.put(
        "/api/admin/notifications/config",
        json={
            "enabled": True,
            "provider": "resend",
            "resend": {
                "api_key": "re_test_123",
                "sender_email": "lab@example.edu",
                "sender_name": "Lab Alerts",
                "reply_to": "reply@example.edu",
            },
            "recipients": {
                "service": ["ops@example.edu"],
                "config_failure": ["ops@example.edu"],
                "queue_failure": [],
                "job_failure": [],
            },
            "cooldown_minutes": {
                "service": 15,
                "config_failure": 30,
                "queue_failure": 60,
                "job_failure": 60,
            },
        },
        headers={"x-csrf-token": "csrf-token"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["config"]["resend"]["api_key"] == "***"
    assert payload["config"]["resend"]["has_api_key"] is True

    stored = load_notification_config(tmp_path / "notifications.json")
    assert stored.resend.api_key == "re_test_123"
    raw_payload = (tmp_path / "notifications.json").read_text(encoding="utf-8")
    assert "re_test_123" not in raw_payload


def test_notifications_config_preserves_existing_secret_on_masked_save(tmp_path: Path) -> None:
    client = _auth_client(tmp_path)
    initial = {
        "enabled": True,
        "provider": "resend",
        "resend": {
            "api_key": "re_test_123",
            "sender_email": "lab@example.edu",
            "sender_name": "Lab Alerts",
            "reply_to": "",
        },
        "recipients": {
            "service": ["ops@example.edu"],
            "config_failure": [],
            "queue_failure": [],
            "job_failure": [],
        },
        "cooldown_minutes": {
            "service": 15,
            "config_failure": 30,
            "queue_failure": 60,
            "job_failure": 60,
        },
    }
    client.put("/api/admin/notifications/config", json=initial, headers={"x-csrf-token": "csrf-token"})

    response = client.put(
        "/api/admin/notifications/config",
        json={
            **initial,
            "resend": {
                **initial["resend"],
                "api_key": "***",
                "sender_name": "Updated Name",
            },
        },
        headers={"x-csrf-token": "csrf-token"},
    )

    assert response.status_code == 200
    stored = load_notification_config(tmp_path / "notifications.json")
    assert stored.resend.api_key == "re_test_123"
    assert stored.resend.sender_name == "Updated Name"


def test_test_email_endpoint_sends_with_mocked_resend(tmp_path: Path, monkeypatch) -> None:
    client = _auth_client(tmp_path)
    captured: dict[str, object] = {}

    def fake_post(url: str, headers: dict[str, str], json: dict[str, object], timeout: int):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _MockResponse()

    monkeypatch.setattr("autoclean.api.notifications.resend.requests.post", fake_post)
    client.put(
        "/api/admin/notifications/config",
        json={
            "enabled": True,
            "provider": "resend",
            "resend": {
                "api_key": "re_test_123",
                "sender_email": "lab@example.edu",
                "sender_name": "Lab Alerts",
                "reply_to": "reply@example.edu",
            },
            "recipients": {
                "service": ["ops@example.edu"],
                "config_failure": [],
                "queue_failure": [],
                "job_failure": [],
            },
            "cooldown_minutes": {
                "service": 15,
                "config_failure": 30,
                "queue_failure": 60,
                "job_failure": 60,
            },
        },
        headers={"x-csrf-token": "csrf-token"},
    )

    response = client.post(
        "/api/admin/notifications/test-email",
        json={
            "to": ["scientist@example.edu"],
            "subject": "Test",
            "message": "Hello",
        },
        headers={"x-csrf-token": "csrf-token"},
    )

    assert response.status_code == 200
    assert captured["url"] == "https://api.resend.com/emails"
    assert captured["json"]["to"] == ["scientist@example.edu"]  # type: ignore[index]
    assert captured["json"]["subject"] == "Test"  # type: ignore[index]


def test_notification_cooldown_suppresses_duplicate_send(tmp_path: Path, monkeypatch) -> None:
    app = create_app(workspace_dir=tmp_path)
    sent: list[dict[str, object]] = []

    def fake_post(url: str, headers: dict[str, str], json: dict[str, object], timeout: int):
        sent.append(json)
        return _MockResponse()

    monkeypatch.setattr("autoclean.api.notifications.resend.requests.post", fake_post)
    path = tmp_path / "notifications.json"
    from autoclean.api.notifications.config import save_notification_config

    save_notification_config(
        path,
        NotificationConfig.model_validate(
            {
                "enabled": True,
                "provider": "resend",
                "resend": {
                    "api_key": "re_test_123",
                    "sender_email": "lab@example.edu",
                    "sender_name": "Lab Alerts",
                    "reply_to": "",
                },
                "recipients": {
                    "service": ["ops@example.edu"],
                    "config_failure": [],
                    "queue_failure": [],
                    "job_failure": [],
                },
                "cooldown_minutes": {
                    "service": 60,
                    "config_failure": 30,
                    "queue_failure": 60,
                    "job_failure": 60,
                },
            }
        ),
    )

    first = send_email(
        event_key=NotificationEventKey.SERVICE,
        subject="Service started",
        text="Service started",
        dedupe_key="service:test:started",
    )
    second = send_email(
        event_key=NotificationEventKey.SERVICE,
        subject="Service started",
        text="Service started",
        dedupe_key="service:test:started",
    )

    assert first.ok is True
    assert second.ok is True
    assert len(sent) == 1


def test_notifications_endpoints_require_admin_permission(tmp_path: Path) -> None:
    client = _auth_client(tmp_path, role=Role.VIEWER)

    response = client.get("/api/admin/notifications/config")

    assert response.status_code == 403


def test_notification_admin_changes_are_audited(tmp_path: Path, monkeypatch) -> None:
    client = _auth_client(tmp_path)

    def fake_post(url: str, headers: dict[str, str], json: dict[str, object], timeout: int):
        return _MockResponse()

    monkeypatch.setattr("autoclean.api.notifications.resend.requests.post", fake_post)
    save_response = client.put(
        "/api/admin/notifications/config",
        json={
            "enabled": True,
            "provider": "resend",
            "resend": {
                "api_key": "re_test_123",
                "sender_email": "lab@example.edu",
                "sender_name": "Lab Alerts",
                "reply_to": "",
            },
            "recipients": {
                "service": ["ops@example.edu"],
                "config_failure": [],
                "queue_failure": [],
                "job_failure": [],
            },
            "cooldown_minutes": {
                "service": 15,
                "config_failure": 30,
                "queue_failure": 60,
                "job_failure": 60,
            },
        },
        headers={"x-csrf-token": "csrf-token"},
    )
    assert save_response.status_code == 200

    test_response = client.post(
        "/api/admin/notifications/test-email",
        json={"to": ["scientist@example.edu"], "subject": "Test", "message": "Hello"},
        headers={"x-csrf-token": "csrf-token"},
    )
    assert test_response.status_code == 200

    audit_response = client.get("/api/admin/audit")
    assert audit_response.status_code == 200
    events = audit_response.json()["events"]
    assert any(event["resource_type"] == "notification_config" and event["action"] == "update" for event in events)
    assert any(event["resource_type"] == "notification_config" and event["action"] == "send_test_email" for event in events)


def test_daily_digest_endpoint_sends_digest(tmp_path: Path, monkeypatch) -> None:
    client = _auth_client(tmp_path)

    def fake_post(url: str, headers: dict[str, str], json: dict[str, object], timeout: int):
        return _MockResponse()

    monkeypatch.setattr("autoclean.api.notifications.resend.requests.post", fake_post)
    client.put(
        "/api/admin/notifications/config",
        json={
            "enabled": True,
            "provider": "resend",
            "app_base_url": "http://localhost:8000",
            "resend": {
                "api_key": "re_test_123",
                "sender_email": "lab@example.edu",
                "sender_name": "Lab Alerts",
                "reply_to": "",
            },
            "recipients": {
                "service": ["ops@example.edu"],
                "config_failure": [],
                "queue_failure": [],
                "job_failure": [],
            },
            "route_recipients": {"route-1": ["owner@example.edu"]},
            "daily_digest_recipients": ["digest@example.edu"],
            "cooldown_minutes": {
                "service": 15,
                "config_failure": 30,
                "queue_failure": 60,
                "job_failure": 60,
            },
        },
        headers={"x-csrf-token": "csrf-token"},
    )

    response = client.post(
        "/api/admin/notifications/daily-digest",
        headers={"x-csrf-token": "csrf-token"},
    )

    assert response.status_code == 200


def test_end_to_end_oidc_auth_and_notifications_workflow(tmp_path: Path, monkeypatch) -> None:
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)

    class MockResponse:
        def __init__(self, payload):
            self._payload = payload
            self.content = b"ok"
            self.status_code = 200
            self.text = "ok"

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self._payload

    def fake_oidc_get(url: str, headers=None, timeout=10):
        if url.endswith("/.well-known/openid-configuration"):
            return MockResponse(
                {
                    "authorization_endpoint": "https://issuer.example.com/auth",
                    "token_endpoint": "https://issuer.example.com/token",
                    "userinfo_endpoint": "https://issuer.example.com/userinfo",
                }
            )
        if url.endswith("/userinfo"):
            return MockResponse(
                {
                    "sub": "oidc-admin",
                    "preferred_username": "oidc-admin",
                    "email": "admin@example.edu",
                    "name": "OIDC Admin",
                    "groups": [],
                }
            )
        raise AssertionError(f"Unexpected URL: {url}")

    def fake_post(url: str, data=None, headers=None, json=None, timeout=10):
        if url == "https://issuer.example.com/token":
            return MockResponse({"access_token": "oidc-token"})
        if url == "https://api.resend.com/emails":
            return _MockResponse()
        raise AssertionError(f"Unexpected POST URL: {url}")

    monkeypatch.setattr("autoclean.api.auth.oidc.requests.get", fake_oidc_get)
    monkeypatch.setattr("autoclean.api.auth.oidc.requests.post", fake_post)
    monkeypatch.setattr("autoclean.api.notifications.resend.requests.post", fake_post)

    bootstrap = client.put(
        "/api/admin/auth/config",
        json={
            "mode": "oauth",
            "provider": "oidc",
            "allow_disable_auth": True,
            "session": {"cookie_name": "autoclean_session", "ttl_hours": 12, "secure": True},
            "github": {
                "client_id": "",
                "client_secret": "",
                "redirect_uri": "http://localhost:8000/api/auth/callback/github",
                "allowed_orgs": [],
                "allowed_users": [],
            },
            "oidc": {
                "issuer_url": "https://issuer.example.com",
                "client_id": "oidc-client",
                "client_secret": "oidc-secret",
                "redirect_uri": "http://localhost:8000/api/auth/callback/oidc",
                "scopes": ["openid", "profile", "email"],
                "allowed_groups": [],
                "allowed_users": [],
                "username_claim": "preferred_username",
                "groups_claim": "groups",
            },
            "bootstrap_admins": ["oidc-admin"],
        },
    )
    assert bootstrap.status_code == 200

    login_response = client.post("/api/auth/login", json={"provider": "oidc"})
    state = login_response.cookies.get("autoclean_oauth_state")
    client.cookies.set("autoclean_oauth_state", state)
    callback = client.get(f"/api/auth/callback/oidc?code=test-code&state={state}", follow_redirects=False)
    assert callback.status_code == 303
    session_cookie = callback.cookies.get("autoclean_session")
    client.cookies.set("autoclean_session", session_cookie)

    me_response = client.get("/api/auth/me")
    assert me_response.status_code == 200
    csrf_token = me_response.json()["csrf_token"]

    notify_response = client.put(
        "/api/admin/notifications/config",
        json={
            "enabled": True,
            "provider": "resend",
            "app_base_url": "http://localhost:8000",
            "resend": {
                "api_key": "re_test_123",
                "sender_email": "lab@example.edu",
                "sender_name": "Lab Alerts",
                "reply_to": "",
            },
            "recipients": {
                "service": ["ops@example.edu"],
                "config_failure": [],
                "queue_failure": [],
                "job_failure": [],
            },
            "route_recipients": {},
            "daily_digest_recipients": ["digest@example.edu"],
            "cooldown_minutes": {
                "service": 15,
                "config_failure": 30,
                "queue_failure": 60,
                "job_failure": 60,
            },
        },
        headers={"x-csrf-token": csrf_token},
    )
    assert notify_response.status_code == 200

    digest_response = client.post("/api/admin/notifications/daily-digest", headers={"x-csrf-token": csrf_token})
    assert digest_response.status_code == 200


def test_test_email_reports_provider_errors(tmp_path: Path, monkeypatch) -> None:
    client = _auth_client(tmp_path)

    def fake_post(url: str, headers: dict[str, str], json: dict[str, object], timeout: int):
        return _MockResponse({"message": "bad credentials"}, status_code=401)

    monkeypatch.setattr("autoclean.api.notifications.resend.requests.post", fake_post)
    client.put(
        "/api/admin/notifications/config",
        json={
            "enabled": True,
            "provider": "resend",
            "resend": {
                "api_key": "re_bad",
                "sender_email": "lab@example.edu",
                "sender_name": "Lab Alerts",
                "reply_to": "",
            },
            "recipients": {
                "service": ["ops@example.edu"],
                "config_failure": [],
                "queue_failure": [],
                "job_failure": [],
            },
            "cooldown_minutes": {
                "service": 15,
                "config_failure": 30,
                "queue_failure": 60,
                "job_failure": 60,
            },
        },
        headers={"x-csrf-token": "csrf-token"},
    )

    response = client.post(
        "/api/admin/notifications/test-email",
        json={"to": ["scientist@example.edu"], "subject": "Test", "message": "Hello"},
        headers={"x-csrf-token": "csrf-token"},
    )

    assert response.status_code == 400


def test_notifications_config_requires_csrf_under_auth(tmp_path: Path) -> None:
    client = _auth_client(tmp_path)

    response = client.put(
        "/api/admin/notifications/config",
        json={
            "enabled": False,
            "provider": "resend",
            "resend": {"api_key": "", "sender_email": "", "sender_name": "AutoClean Serve", "reply_to": ""},
            "recipients": {"service": [], "config_failure": [], "queue_failure": [], "job_failure": []},
            "cooldown_minutes": {"service": 15, "config_failure": 30, "queue_failure": 60, "job_failure": 60},
        },
    )

    assert response.status_code == 403
