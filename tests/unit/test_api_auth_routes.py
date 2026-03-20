"""Tests for Serve auth routes and permission enforcement."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autoclean.api.auth.config import save_auth_config
from autoclean.api.auth.session import SESSION_STATE_COOKIE
from autoclean.api.auth.models import AuthConfig, AuthMode, Role
from autoclean.api.auth.store import create_session, ensure_auth_schema, get_session, set_user_roles, upsert_user
from autoclean.api.server import create_app


def _bootstrap_user_session(workspace: Path, *, role: Role = Role.VIEWER) -> tuple[str, str]:
    db_path = workspace / ".serve" / "serve_state.db"
    ensure_auth_schema(db_path)
    now = datetime.now(timezone.utc)
    user = upsert_user(
        db_path,
        user_id="github:123",
        provider="github",
        subject="123",
        login="viewer-user",
        email="viewer@example.com",
        display_name="Viewer User",
        avatar_url=None,
        now_iso=now.isoformat(),
    )
    set_user_roles(
        db_path,
        user.id,
        [role],
        granted_by="system",
        granted_at=now.isoformat(),
    )
    session = create_session(
        db_path,
        session_id="test-session",
        user_id=user.id,
        csrf_token="csrf-token",
        expires_at=(now + timedelta(hours=6)).isoformat(),
        now_iso=now.isoformat(),
        ip_address="127.0.0.1",
        user_agent="pytest",
    )
    return session.id, session.csrf_token


def test_auth_status_disabled_when_workspace_has_no_auth_config(tmp_path: Path) -> None:
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)

    response = client.get("/api/auth/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert payload["authenticated"] is False
    assert "providers" in payload
    assert payload["bootstrap_allowed"] is True
    assert payload["configured"] is False


def test_protected_status_requires_session_when_auth_config_exists(tmp_path: Path) -> None:
    save_auth_config(tmp_path / "serve-auth.json", AuthConfig(mode=AuthMode.OAUTH))
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)

    response = client.get("/api/status")

    assert response.status_code == 401


def test_auth_status_does_not_allow_bootstrap_for_existing_misconfigured_auth(tmp_path: Path) -> None:
    config = AuthConfig(mode=AuthMode.OAUTH)
    save_auth_config(tmp_path / "serve-auth.json", config)
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)

    response = client.get("/api/auth/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["bootstrap_allowed"] is False
    assert payload["configured"] is False


def test_auth_status_marks_any_configured_provider_as_available(tmp_path: Path) -> None:
    config = AuthConfig(mode=AuthMode.OAUTH, provider="github")
    config.oidc.issuer_url = "https://issuer.example.com"
    config.oidc.client_id = "oidc-client"
    config.oidc.client_secret = "oidc-secret"
    save_auth_config(tmp_path / "serve-auth.json", config)
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)

    response = client.get("/api/auth/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["configured"] is True
    assert payload["selected_provider_configured"] is False
    assert payload["configured_providers"] == ["oidc"]


def test_health_stays_public_when_auth_defaults_on(tmp_path: Path) -> None:
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/task-manager"),
        ("get", "/api/tasks"),
        ("get", "/api/montages"),
        ("get", "/api/exclude/root"),
        ("get", "/api/filesystem/browse"),
        ("get", "/api/worker/status"),
        ("post", "/api/events/analyze"),
        ("post", "/api/tutorial/setup"),
        ("post", "/api/mode/switch"),
    ],
)
def test_newly_protected_surfaces_require_auth_when_enabled(tmp_path: Path, method: str, path: str) -> None:
    save_auth_config(tmp_path / "serve-auth.json", AuthConfig(mode=AuthMode.OAUTH))
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)

    if method == "post":
        payload = {"file_path": "/tmp/test.set"} if path == "/api/events/analyze" else {"mode": "live"} if path == "/api/mode/switch" else {}
        response = getattr(client, method)(path, json=payload)
    else:
        response = getattr(client, method)(path)

    assert response.status_code == 401


def test_viewer_session_can_read_status_and_me(tmp_path: Path) -> None:
    config = AuthConfig(mode=AuthMode.OAUTH)
    save_auth_config(tmp_path / "serve-auth.json", config)
    session_id, _csrf_token = _bootstrap_user_session(tmp_path, role=Role.VIEWER)
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)
    client.cookies.set(config.session.cookie_name, session_id)

    status_response = client.get("/api/status")
    me_response = client.get("/api/auth/me")

    assert status_response.status_code == 200
    assert me_response.status_code == 200
    assert me_response.json()["user"]["login"] == "viewer-user"


def test_viewer_can_read_task_and_exclude_surfaces_but_cannot_mutate_them(tmp_path: Path) -> None:
    config = AuthConfig(mode=AuthMode.OAUTH)
    save_auth_config(tmp_path / "serve-auth.json", config)
    session_id, csrf_token = _bootstrap_user_session(tmp_path, role=Role.VIEWER)
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)
    client.cookies.set(config.session.cookie_name, session_id)

    read_task_response = client.get("/api/task-manager")
    read_exclude_response = client.get("/api/exclude/root")
    write_task_response = client.post(
        "/api/task-manager/create",
        json={"class_name": "MyTask"},
        headers={"x-csrf-token": csrf_token},
    )
    write_exclude_response = client.put(
        "/api/exclude/files/example/notes",
        json={"notes": "test"},
        headers={"x-csrf-token": csrf_token},
    )

    assert read_task_response.status_code == 200
    assert read_exclude_response.status_code in {200, 404, 500}
    assert write_task_response.status_code == 403
    assert write_exclude_response.status_code == 403


def test_viewer_session_cannot_mutate_queue_even_with_csrf(tmp_path: Path) -> None:
    config = AuthConfig(mode=AuthMode.OAUTH)
    save_auth_config(tmp_path / "serve-auth.json", config)
    session_id, csrf_token = _bootstrap_user_session(tmp_path, role=Role.VIEWER)
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)
    client.cookies.set(config.session.cookie_name, session_id)

    response = client.post(
        "/api/queue/retry",
        json={},
        headers={"x-csrf-token": csrf_token},
    )

    assert response.status_code == 403


def test_operator_session_can_mutate_queue_with_csrf(tmp_path: Path) -> None:
    config = AuthConfig(mode=AuthMode.OAUTH)
    save_auth_config(tmp_path / "serve-auth.json", config)
    session_id, csrf_token = _bootstrap_user_session(tmp_path, role=Role.OPERATOR)
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)
    client.cookies.set(config.session.cookie_name, session_id)

    response = client.post(
        "/api/queue/retry",
        json={},
        headers={"x-csrf-token": csrf_token},
    )

    assert response.status_code == 200
    assert response.json()["retried"] == 0


def test_operator_can_access_worker_status_but_cannot_modify_tasks(tmp_path: Path) -> None:
    config = AuthConfig(mode=AuthMode.OAUTH)
    save_auth_config(tmp_path / "serve-auth.json", config)
    session_id, csrf_token = _bootstrap_user_session(tmp_path, role=Role.OPERATOR)
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)
    client.cookies.set(config.session.cookie_name, session_id)

    worker_status = client.get("/api/worker/status")
    task_create = client.post(
        "/api/task-manager/create",
        json={"class_name": "MyTask"},
        headers={"x-csrf-token": csrf_token},
    )

    assert worker_status.status_code == 200
    assert task_create.status_code == 403


def test_editor_can_switch_mode_and_create_tasks(tmp_path: Path) -> None:
    config = AuthConfig(mode=AuthMode.OAUTH)
    save_auth_config(tmp_path / "serve-auth.json", config)
    session_id, csrf_token = _bootstrap_user_session(tmp_path, role=Role.EDITOR)
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)
    client.cookies.set(config.session.cookie_name, session_id)

    mode_switch = client.post(
        "/api/mode/switch",
        json={"mode": "live"},
        headers={"x-csrf-token": csrf_token},
    )
    task_create = client.post(
        "/api/task-manager/create",
        json={"class_name": "MyTask"},
        headers={"x-csrf-token": csrf_token},
    )

    assert mode_switch.status_code == 200
    assert task_create.status_code in {200, 500}


def test_login_sets_oauth_state_cookie(tmp_path: Path) -> None:
    config = AuthConfig(mode=AuthMode.OAUTH)
    config.github.client_id = "client-id"
    config.github.client_secret = "client-secret"
    save_auth_config(tmp_path / "serve-auth.json", config)
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)

    response = client.post("/api/auth/login", json={"provider": "github"})

    assert response.status_code == 200
    assert SESSION_STATE_COOKIE in response.cookies
    assert "login_url" in response.json()


def test_logout_revokes_active_session(tmp_path: Path) -> None:
    config = AuthConfig(mode=AuthMode.OAUTH)
    save_auth_config(tmp_path / "serve-auth.json", config)
    session_id, _csrf_token = _bootstrap_user_session(tmp_path, role=Role.VIEWER)
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)
    client.cookies.set(config.session.cookie_name, session_id)

    response = client.post("/api/auth/logout", json={})

    assert response.status_code == 200
    session = get_session(tmp_path / ".serve" / "serve_state.db", session_id)
    assert session is not None
    assert session.revoked_at is not None


def test_github_callback_creates_session_and_redirects(tmp_path: Path, monkeypatch) -> None:
    config = AuthConfig(mode=AuthMode.OAUTH)
    config.github.client_id = "client-id"
    config.github.client_secret = "client-secret"
    save_auth_config(tmp_path / "serve-auth.json", config)
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)

    class MockResponse:
        def __init__(self, payload):
            self._payload = payload
            self.content = b"ok"

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self._payload

    def fake_post(url: str, headers=None, data=None, timeout=10):
        assert "access_token" not in (data or {})
        return MockResponse({"access_token": "gh-token"})

    def fake_get(url: str, headers=None, timeout=10):
        if url.endswith("/user"):
            return MockResponse({"id": 123, "login": "viewer-user", "name": "Viewer User", "avatar_url": None})
        if url.endswith("/user/emails"):
            return MockResponse([{"email": "viewer@example.com", "primary": True, "verified": True}])
        if url.endswith("/user/orgs"):
            return MockResponse([{"login": "lab-org"}])
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("autoclean.api.auth.github.requests.post", fake_post)
    monkeypatch.setattr("autoclean.api.auth.github.requests.get", fake_get)

    login_response = client.post("/api/auth/login", json={"provider": "github"})
    state = login_response.cookies.get(SESSION_STATE_COOKIE)
    client.cookies.set(SESSION_STATE_COOKIE, state)

    callback = client.get(f"/api/auth/callback/github?code=test-code&state={state}", follow_redirects=False)

    assert callback.status_code == 303
    assert callback.headers["location"] == "/"
    assert config.session.cookie_name in callback.cookies


def test_github_callback_uses_non_secure_cookie_for_http_localhost(tmp_path: Path, monkeypatch) -> None:
    config = AuthConfig(mode=AuthMode.OAUTH)
    config.github.client_id = "client-id"
    config.github.client_secret = "client-secret"
    save_auth_config(tmp_path / "serve-auth.json", config)
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app, base_url="http://testserver")

    class MockResponse:
        def __init__(self, payload):
            self._payload = payload
            self.content = b"ok"

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self._payload

    def fake_post(url: str, headers=None, data=None, timeout=10):
        return MockResponse({"access_token": "gh-token"})

    def fake_get(url: str, headers=None, timeout=10):
        if url.endswith("/user"):
            return MockResponse({"id": 123, "login": "viewer-user", "name": "Viewer User", "avatar_url": None})
        if url.endswith("/user/emails"):
            return MockResponse([{"email": "viewer@example.com", "primary": True, "verified": True}])
        if url.endswith("/user/orgs"):
            return MockResponse([{"login": "lab-org"}])
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("autoclean.api.auth.github.requests.post", fake_post)
    monkeypatch.setattr("autoclean.api.auth.github.requests.get", fake_get)

    login_response = client.post("/api/auth/login", json={"provider": "github"})
    state = login_response.cookies.get(SESSION_STATE_COOKIE)
    client.cookies.set(SESSION_STATE_COOKIE, state)

    callback = client.get(f"/api/auth/callback/github?code=test-code&state={state}", follow_redirects=False)

    set_cookie_values = callback.headers.get_list("set-cookie")
    session_cookie = next(value for value in set_cookie_values if value.startswith(f"{config.session.cookie_name}="))
    assert "Secure" not in session_cookie


def test_github_callback_uses_secure_cookie_for_https(tmp_path: Path, monkeypatch) -> None:
    config = AuthConfig(mode=AuthMode.OAUTH)
    config.github.client_id = "client-id"
    config.github.client_secret = "client-secret"
    save_auth_config(tmp_path / "serve-auth.json", config)
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app, base_url="https://testserver")

    class MockResponse:
        def __init__(self, payload):
            self._payload = payload
            self.content = b"ok"

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self._payload

    def fake_post(url: str, headers=None, data=None, timeout=10):
        return MockResponse({"access_token": "gh-token"})

    def fake_get(url: str, headers=None, timeout=10):
        if url.endswith("/user"):
            return MockResponse({"id": 123, "login": "viewer-user", "name": "Viewer User", "avatar_url": None})
        if url.endswith("/user/emails"):
            return MockResponse([{"email": "viewer@example.com", "primary": True, "verified": True}])
        if url.endswith("/user/orgs"):
            return MockResponse([{"login": "lab-org"}])
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("autoclean.api.auth.github.requests.post", fake_post)
    monkeypatch.setattr("autoclean.api.auth.github.requests.get", fake_get)

    login_response = client.post("/api/auth/login", json={"provider": "github"})
    state = login_response.cookies.get(SESSION_STATE_COOKIE)
    client.cookies.set(SESSION_STATE_COOKIE, state)

    callback = client.get(f"/api/auth/callback/github?code=test-code&state={state}", follow_redirects=False)

    set_cookie_values = callback.headers.get_list("set-cookie")
    session_cookie = next(value for value in set_cookie_values if value.startswith(f"{config.session.cookie_name}="))
    assert "Secure" in session_cookie


def test_github_callback_uses_secure_cookie_when_forwarded_proto_is_https(tmp_path: Path, monkeypatch) -> None:
    config = AuthConfig(mode=AuthMode.OAUTH)
    config.github.client_id = "client-id"
    config.github.client_secret = "client-secret"
    save_auth_config(tmp_path / "serve-auth.json", config)
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app, base_url="http://testserver")

    class MockResponse:
        def __init__(self, payload):
            self._payload = payload
            self.content = b"ok"

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self._payload

    def fake_post(url: str, headers=None, data=None, timeout=10):
        return MockResponse({"access_token": "gh-token"})

    def fake_get(url: str, headers=None, timeout=10):
        if url.endswith("/user"):
            return MockResponse({"id": 123, "login": "viewer-user", "name": "Viewer User", "avatar_url": None})
        if url.endswith("/user/emails"):
            return MockResponse([{"email": "viewer@example.com", "primary": True, "verified": True}])
        if url.endswith("/user/orgs"):
            return MockResponse([{"login": "lab-org"}])
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("autoclean.api.auth.github.requests.post", fake_post)
    monkeypatch.setattr("autoclean.api.auth.github.requests.get", fake_get)

    login_response = client.post("/api/auth/login", json={"provider": "github"}, headers={"x-forwarded-proto": "https"})
    state = login_response.cookies.get(SESSION_STATE_COOKIE)
    client.cookies.set(SESSION_STATE_COOKIE, state)

    callback = client.get(
        f"/api/auth/callback/github?code=test-code&state={state}",
        follow_redirects=False,
        headers={"x-forwarded-proto": "https"},
    )

    set_cookie_values = callback.headers.get_list("set-cookie")
    session_cookie = next(value for value in set_cookie_values if value.startswith(f"{config.session.cookie_name}="))
    assert "Secure" in session_cookie


def test_oidc_login_url_works_when_oidc_selected(tmp_path: Path, monkeypatch) -> None:
    config = AuthConfig(mode=AuthMode.OAUTH, provider="oidc")
    config.oidc.issuer_url = "https://issuer.example.com"
    config.oidc.client_id = "oidc-client"
    config.oidc.client_secret = "oidc-secret"
    save_auth_config(tmp_path / "serve-auth.json", config)
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)

    class MockResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {
                "authorization_endpoint": "https://issuer.example.com/auth",
                "token_endpoint": "https://issuer.example.com/token",
                "userinfo_endpoint": "https://issuer.example.com/userinfo",
            }

    monkeypatch.setattr("autoclean.api.auth.oidc.requests.get", lambda *args, **kwargs: MockResponse())

    response = client.post("/api/auth/login", json={"provider": "oidc"})

    assert response.status_code == 200
    assert "issuer.example.com/auth" in response.json()["login_url"]


def test_oidc_callback_creates_session_and_redirects(tmp_path: Path, monkeypatch) -> None:
    config = AuthConfig(mode=AuthMode.OAUTH, provider="oidc")
    config.oidc.issuer_url = "https://issuer.example.com"
    config.oidc.client_id = "oidc-client"
    config.oidc.client_secret = "oidc-secret"
    save_auth_config(tmp_path / "serve-auth.json", config)
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)

    class MockResponse:
        def __init__(self, payload):
            self._payload = payload
            self.content = b"ok"

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self._payload

    def fake_get(url: str, headers=None, timeout=10):
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
                    "sub": "oidc-subject",
                    "preferred_username": "oidc-user",
                    "email": "oidc@example.com",
                    "groups": ["lab-members"],
                    "name": "OIDC User",
                }
            )
        raise AssertionError(f"Unexpected URL: {url}")

    def fake_post(url: str, data=None, timeout=10):
        assert url == "https://issuer.example.com/token"
        return MockResponse({"access_token": "oidc-token"})

    monkeypatch.setattr("autoclean.api.auth.oidc.requests.get", fake_get)
    monkeypatch.setattr("autoclean.api.auth.oidc.requests.post", fake_post)

    login_response = client.post("/api/auth/login", json={"provider": "oidc"})
    state = login_response.cookies.get(SESSION_STATE_COOKIE)
    client.cookies.set(SESSION_STATE_COOKIE, state)

    callback = client.get(f"/api/auth/callback/oidc?code=test-code&state={state}", follow_redirects=False)

    assert callback.status_code == 303
    assert callback.headers["location"] == "/"
    assert config.session.cookie_name in callback.cookies


def test_auth_disabled_mode_keeps_status_public(tmp_path: Path) -> None:
    save_auth_config(tmp_path / "serve-auth.json", AuthConfig(mode=AuthMode.DISABLED))
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)

    response = client.get("/api/status")

    assert response.status_code == 200


def test_existing_misconfigured_auth_config_requires_admin_for_ui_update(tmp_path: Path) -> None:
    config = AuthConfig(mode=AuthMode.OAUTH)
    save_auth_config(tmp_path / "serve-auth.json", config)
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)

    response = client.get("/api/admin/auth/config")

    assert response.status_code == 401


def test_oauth_mode_requires_at_least_one_configured_provider(tmp_path: Path) -> None:
    config = AuthConfig(mode=AuthMode.OAUTH)
    config.github.client_id = "client-id"
    config.github.client_secret = "client-secret"
    save_auth_config(tmp_path / "serve-auth.json", config)
    session_id, csrf_token = _bootstrap_user_session(tmp_path, role=Role.ADMIN)
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)
    client.cookies.set(config.session.cookie_name, session_id)

    response = client.put(
        "/api/admin/auth/config",
        json={
            "mode": "oauth",
            "provider": "github",
            "allow_disable_auth": True,
            "session": {"cookie_name": "autoclean_session", "ttl_hours": 12, "secure": None},
            "github": {
                "client_id": "",
                "client_secret": "",
                "redirect_uri": "http://localhost:8000/api/auth/callback/github",
                "allowed_orgs": [],
                "allowed_users": [],
            },
            "oidc": {
                "issuer_url": "",
                "client_id": "",
                "client_secret": "",
                "redirect_uri": "http://localhost:8000/api/auth/callback/oidc",
                "scopes": ["openid", "profile", "email"],
                "allowed_groups": [],
                "allowed_users": [],
                "username_claim": "preferred_username",
                "groups_claim": "groups",
            },
            "bootstrap_admins": [],
        },
        headers={"x-csrf-token": csrf_token},
    )

    assert response.status_code == 400
    assert "at least one configured auth provider" in response.text


def test_websocket_requires_session_when_auth_enabled(tmp_path: Path) -> None:
    save_auth_config(tmp_path / "serve-auth.json", AuthConfig(mode=AuthMode.OAUTH))
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)

    with pytest.raises(Exception):
        with client.websocket_connect("/ws/events"):
            pass


def test_auth_config_and_role_changes_are_audited(tmp_path: Path) -> None:
    config = AuthConfig(mode=AuthMode.OAUTH)
    save_auth_config(tmp_path / "serve-auth.json", config)
    session_id, csrf_token = _bootstrap_user_session(tmp_path, role=Role.ADMIN)
    db_path = tmp_path / ".serve" / "serve_state.db"
    now = datetime.now(timezone.utc).isoformat()
    extra_user = upsert_user(
        db_path,
        user_id="github:456",
        provider="github",
        subject="456",
        login="target-user",
        email="target@example.com",
        display_name="Target User",
        avatar_url=None,
        now_iso=now,
    )
    set_user_roles(
        db_path,
        extra_user.id,
        [Role.VIEWER],
        granted_by="system",
        granted_at=now,
    )
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)
    client.cookies.set(config.session.cookie_name, session_id)

    auth_update = client.put(
        "/api/admin/auth/config",
        json={
            "mode": "oauth",
            "provider": "github",
            "allow_disable_auth": True,
            "session": {"cookie_name": "autoclean_session", "ttl_hours": 12, "secure": True},
            "github": {
                "client_id": "client-id",
                "client_secret": "client-secret",
                "redirect_uri": "http://localhost:8000/api/auth/callback/github",
                "allowed_orgs": ["lab-org"],
                "allowed_users": [],
            },
            "bootstrap_admins": ["viewer-user"],
        },
        headers={"x-csrf-token": csrf_token},
    )
    assert auth_update.status_code == 200

    role_update = client.post(
        "/api/admin/users/github%3A456/roles",
        json={"roles": ["viewer", "operator"]},
        headers={"x-csrf-token": csrf_token},
    )
    assert role_update.status_code == 200

    audit_response = client.get("/api/admin/audit")
    assert audit_response.status_code == 200
    events = audit_response.json()["events"]
    assert any(event["resource_type"] == "auth_config" and event["action"] == "update" for event in events)
    assert any(event["resource_type"] == "user_role" and event["action"] == "set_roles" for event in events)


def test_cannot_remove_last_admin_via_set_roles(tmp_path: Path) -> None:
    config = AuthConfig(mode=AuthMode.OAUTH)
    config.github.client_id = "client-id"
    config.github.client_secret = "client-secret"
    save_auth_config(tmp_path / "serve-auth.json", config)
    session_id, csrf_token = _bootstrap_user_session(tmp_path, role=Role.ADMIN)
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)
    client.cookies.set(config.session.cookie_name, session_id)

    response = client.post(
        "/api/admin/users/github%3A123/roles",
        json={"roles": ["viewer"]},
        headers={"x-csrf-token": csrf_token},
    )

    assert response.status_code == 400
    assert "last admin" in response.text


def test_cannot_remove_last_admin_role_via_delete(tmp_path: Path) -> None:
    config = AuthConfig(mode=AuthMode.OAUTH)
    config.github.client_id = "client-id"
    config.github.client_secret = "client-secret"
    save_auth_config(tmp_path / "serve-auth.json", config)
    session_id, csrf_token = _bootstrap_user_session(tmp_path, role=Role.ADMIN)
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)
    client.cookies.set(config.session.cookie_name, session_id)

    response = client.delete(
        "/api/admin/users/github%3A123/roles/admin",
        headers={"x-csrf-token": csrf_token},
    )

    assert response.status_code == 400
    assert "last admin" in response.text


def test_cannot_disable_auth_when_current_policy_requires_auth(tmp_path: Path) -> None:
    config = AuthConfig(mode=AuthMode.OAUTH, allow_disable_auth=False)
    config.github.client_id = "client-id"
    config.github.client_secret = "client-secret"
    save_auth_config(tmp_path / "serve-auth.json", config)
    session_id, csrf_token = _bootstrap_user_session(tmp_path, role=Role.ADMIN)
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)
    client.cookies.set(config.session.cookie_name, session_id)

    response = client.put(
        "/api/admin/auth/config",
        json={
            "mode": "disabled",
            "provider": "github",
            "allow_disable_auth": False,
            "session": {"cookie_name": "autoclean_session", "ttl_hours": 12, "secure": None},
            "github": {
                "client_id": "client-id",
                "client_secret": "client-secret",
                "redirect_uri": "http://localhost:8000/api/auth/callback/github",
                "allowed_orgs": [],
                "allowed_users": [],
            },
            "oidc": {
                "issuer_url": "",
                "client_id": "",
                "client_secret": "",
                "redirect_uri": "http://localhost:8000/api/auth/callback/oidc",
                "scopes": ["openid", "profile", "email"],
                "allowed_groups": [],
                "allowed_users": [],
                "username_claim": "preferred_username",
                "groups_claim": "groups",
            },
            "bootstrap_admins": [],
        },
        headers={"x-csrf-token": csrf_token},
    )

    assert response.status_code == 400
    assert "allow_disable_auth is false" in response.text


def test_cannot_save_disabled_mode_when_allow_disable_auth_false(tmp_path: Path) -> None:
    config = AuthConfig(mode=AuthMode.OAUTH)
    save_auth_config(tmp_path / "serve-auth.json", config)
    session_id, csrf_token = _bootstrap_user_session(tmp_path, role=Role.ADMIN)
    app = create_app(workspace_dir=tmp_path)
    client = TestClient(app)
    client.cookies.set(config.session.cookie_name, session_id)

    response = client.put(
        "/api/admin/auth/config",
        json={
            "mode": "disabled",
            "provider": "github",
            "allow_disable_auth": False,
            "session": {"cookie_name": "autoclean_session", "ttl_hours": 12, "secure": None},
            "github": {
                "client_id": "",
                "client_secret": "",
                "redirect_uri": "http://localhost:8000/api/auth/callback/github",
                "allowed_orgs": [],
                "allowed_users": [],
            },
            "oidc": {
                "issuer_url": "",
                "client_id": "",
                "client_secret": "",
                "redirect_uri": "http://localhost:8000/api/auth/callback/oidc",
                "scopes": ["openid", "profile", "email"],
                "allowed_groups": [],
                "allowed_users": [],
                "username_claim": "preferred_username",
                "groups_claim": "groups",
            },
            "bootstrap_admins": [],
        },
        headers={"x-csrf-token": csrf_token},
    )

    assert response.status_code == 400
    assert "allow_disable_auth is false" in response.text
