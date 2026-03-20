# Serve Auth And Resend Implementation Plan

**Date**: 2026-03-20
**Status**: Proposed
**Scope**: `serve` web/API functionality

## Summary

AutoClean Serve should support authentication by default, while still allowing labs to swap providers or deliberately disable auth for fully local deployments. The first provider should be GitHub OAuth because it is easy for most labs to stand up, but the system should be built around a provider interface so labs can later replace GitHub with Google, Auth0, institutional OIDC, SAML-backed gateways, or a custom reverse-proxy identity layer.

Serve should also support outbound email notifications through Resend. Any scientist or lab admin should be able to paste in a Resend API key, configure sender defaults, and enable event-based emails without code changes.

## Operational Notes

### GitHub OAuth app requirements

- Register a GitHub OAuth app with callback URL matching the Serve workspace setting, for example `http://localhost:8000/api/auth/callback/github`.
- The GitHub app must allow scopes used by Serve: `read:user`, `user:email`, and `read:org`.
- If a lab uses `allowed_orgs`, the authenticating user must be visible to the GitHub org membership API used by Serve.

### Resend sender expectations

- The configured `sender_email` should use a domain verified in Resend for production/lab use.
- A sandbox or unverified sender may block delivery or restrict recipients depending on the Resend account state.
- `reply_to` is optional, but if set it should be a monitored lab mailbox rather than an unowned noreply address.

### Migration behavior for existing workspaces

- Existing workspaces now default to auth-enabled Serve behavior once a workspace is configured.
- Auth/session/audit/notification runtime state is stored in `<workspace>/.serve/serve_state.db`; existing `pipeline.db` data is not modified.
- Notification settings are stored in `<workspace>/notifications.json` and can be added incrementally without changing route or processing files.

### Production readiness findings resolved

- Session cookies now use automatic `Secure` behavior: HTTPS requests receive `Secure` cookies, while local `http://localhost` development remains functional.
- `allow_disable_auth` is now a real backend-enforced policy, not just a recorded config field.
- Secret storage now prefers the OS secure store on supported machines and falls back to the workspace sidecar file only when needed.
- Tests now cover HTTP vs HTTPS cookie behavior and auth-disable policy enforcement.

### Architecture review follow-up

- The first auth rollout left several route families outside the permission model: task management, task browsing, montage browsing, exclude review, filesystem browsing, worker control, tutorial setup, event analysis, and mode switching.
- Those gaps meant auth-enabled Serve still had meaningful unauthenticated or under-protected surfaces.
- The implementation now treats full-route-family permission coverage as part of the core auth contract rather than follow-up polish.
- Regression tests now verify that those surfaces reject unauthenticated access and that write paths require the correct elevated roles.
- Multi-provider support is now exposed in the login gate itself: all configured providers can be offered to the user instead of treating the selected provider as the only usable path.
- Local unauthenticated bootstrap is now limited to first-run or auth-disabled recovery scenarios. A merely misconfigured provider no longer re-opens the auth config UI without an authenticated admin.
- Auth config now refuses `oauth` mode unless at least one provider is actually configured, preventing admin-created login dead-ends.
- User-role management now prevents removal of the last admin so a workspace cannot be orphaned through the UI.
- Automatic cookie security now honors forwarded proxy headers so HTTPS deployments behind a tunnel or reverse proxy still receive `Secure` cookies by default.

The permissions model should be simple enough for undergrads and rotating lab members:

- `viewer`: can inspect status, queue, results, and logs
- `operator`: can run service actions and queue actions
- `editor`: can create and modify routes/config
- `admin`: can manage auth, email, users, and dangerous operations

Auth should be enabled by default for non-local sharing scenarios. A local-only install may explicitly switch auth off, but that should be a conscious setting and clearly flagged in the UI.

## Senior Engineering Review

The plan is directionally strong, but several implementation details needed correction after reviewing the current Serve codebase.

### What is solid

- GitHub-first with a provider abstraction is a good starting point
- four coarse roles are enough for the first release
- Resend is a reasonable first notification provider
- route and service permissions are the right place to start

### What needed correction

#### 1. Auth config shape was inconsistent

The earlier draft mixed:

- `enabled: true|false`
- `auth.mode = disabled`

That creates contradictory states. The plan should use one shape consistently. This document now treats auth state as:

- `mode: "oauth"` when auth is enabled
- `mode: "disabled"` when auth is intentionally off

#### 2. Session storage should be SQLite from day one

Serve already has concurrency points:

- background threads for service log capture
- background threads for tunnel lifecycle
- concurrent HTTP requests
- WebSocket event streaming

A JSON-backed session store would be fragile here. Session, user, and role data should live in a dedicated SQLite DB.

#### 3. Serve auth state should stay separate from `pipeline.db`

The existing `pipeline.db` is tied to processing runs and reporting. Serve auth/session/admin state has different access patterns and lifecycle, so it should use a separate file such as:

- `<workspace>/.serve/serve_state.db`

#### 4. WebSocket auth must be part of phase 1

`/ws/events` is currently open. Protecting only REST endpoints would still leak live queue and service activity through WebSocket connections. The plan therefore needs explicit session validation for WebSockets.

#### 5. Tunnel Basic Auth needs a migration path

The current tunnel implementation already enforces generated Basic Auth in middleware. Replacing it abruptly would be risky. The safer path is:

- keep tunnel Basic Auth as the outer gate during initial rollout
- require Serve login behind it
- decide later whether the outer gate stays permanently

#### 6. CSRF requires a concrete frontend change

The current web client uses a shared `fetch()` helper with no CSRF handling. Cookie sessions require a matching CSRF plan for write actions, not just backend middleware.

#### 7. First-admin bootstrap must not rely on "who logs in first"

The plan should require either:

- an explicit GitHub login allowlist
- or a one-time setup secret

Any arbitrary first successful login becoming admin would be a real security mistake.

#### 8. Docs and OpenAPI exposure should be intentional

The server currently exposes `/docs`, `/redoc`, and `/openapi.json`. The plan should decide whether those remain public, become protected, or are disabled when auth is enabled.

Decision:

- Keep `/docs`, `/redoc`, and `/openapi.json` accessible without auth.
- Do not hide or disable them when Serve auth is enabled.
- Treat these as intentionally public operator/developer surfaces.

#### 9. Auth-disabled mode should stay local-only

If auth is disabled, the operating model should be local-only. Tunnel/public sharing must remain blocked and any future non-loopback exposure should require deliberate opt-in.

#### 10. Notifications need throttling

Failure emails without cooldown or dedupe logic will quickly create alert fatigue. The plan should include suppression rules before shipping notifications broadly.

### Review conclusion

The overall direction makes sense and is worth implementing. The highest-risk items that must be locked down before coding are:

- dedicated SQLite storage for Serve auth state
- WebSocket authentication
- CSRF handling in the frontend API client
- controlled bootstrap-admin assignment
- tunnel migration behavior

## Goals

- Add pluggable auth to Serve without coupling the app to one provider
- Ship GitHub OAuth first as the default reference provider
- Allow auth to be disabled explicitly for trusted local-only labs
- Add role-based permissions for route creation, route modification, service control, and admin settings
- Add Resend-backed notification delivery with workspace-scoped configuration
- Keep the design compatible with existing FastAPI route organization and current Serve settings pages

## Non-Goals

- Full enterprise IAM on day one
- Per-field or per-route ACL complexity
- HIPAA-style messaging or PHI-safe email content generation
- Reusing the current `utils/auth.py` Auth0 compliance flow directly for Serve

## Product Decisions

### 1. Auth should exist by default

Serve is increasingly multi-user. Once labs have PIs, postdocs, staff, and undergrads using the same route workspace, unauthenticated route editing and service control becomes the wrong default. The default should therefore be:

- auth enabled
- first admin established during setup
- role-gated write operations

### 2. Auth can still be disabled

Some labs will run Serve only on `localhost` or behind a trusted VPN. For those cases:

- `auth.mode = disabled` is allowed
- UI should show a persistent banner when auth is disabled
- tunnel/public-share actions should be blocked unless auth is enabled
- auth-disabled mode should remain local-only

### 3. GitHub is the starter provider, not the permanent architecture

GitHub gets the project moving quickly, but the architecture must not hard-code GitHub-specific assumptions into the rest of Serve. The provider layer should own:

- login URL generation
- callback handling
- identity normalization
- token exchange / refresh
- logout behavior

The rest of the app should only consume normalized user/session objects.

## Architecture

## Auth configuration

Add a workspace-scoped auth settings file, for example:

`<workspace>/serve-auth.json`

Suggested shape:

```json
{
  "mode": "oauth",
  "provider": "github",
  "allow_disable_auth": true,
  "session": {
    "cookie_name": "autoclean_session",
    "ttl_hours": 12,
    "secure": null
  },
  "github": {
    "client_id": "",
    "client_secret": "",
    "redirect_uri": "http://localhost:8000/api/auth/callback/github",
    "allowed_orgs": [],
    "allowed_users": []
  },
  "bootstrap_admins": [
    "lab-admin"
  ]
}
```

Notes:

- `mode` controls whether auth is enforced
- `provider` selects the active backend
- provider blocks are namespaced so future providers can coexist
- `session.secure = null` means "auto": use secure cookies on HTTPS and non-secure cookies on local HTTP development
- `allowed_orgs` and `allowed_users` provide a simple first-pass admission filter for GitHub
- `bootstrap_admins` seeds the first admin role assignment

## Auth provider interface

Add a new Serve-specific auth package, for example:

- `src/autoclean/api/auth/base.py`
- `src/autoclean/api/auth/github.py`
- `src/autoclean/api/auth/session.py`
- `src/autoclean/api/auth/service.py`

Core interface:

```python
class AuthProvider(Protocol):
    name: str

    def build_login_redirect(self, state: str) -> str: ...
    async def exchange_code(self, code: str) -> ProviderIdentity: ...
    async def refresh_identity(self, refresh_token: str | None) -> ProviderIdentity | None: ...
    async def revoke(self, session: AuthSession) -> None: ...
```

Normalized identity:

```python
class ProviderIdentity(BaseModel):
    provider: str
    subject: str
    login: str
    email: str | None = None
    display_name: str | None = None
    avatar_url: str | None = None
    groups: list[str] = []
    raw_claims: dict[str, Any] = {}
```

This keeps GitHub-specific payload details out of route handlers.

## Session model

Serve should use signed cookie sessions rather than pushing access tokens into the frontend. Recommended approach:

- server-side session record stored in SQLite
- opaque session id in an `HttpOnly` cookie
- CSRF token for state-changing web requests
- session expiration and revocation support

Recommended DB location:

- `<workspace>/.serve/serve_state.db`

Suggested tables:

### `serve_users`

- `id`
- `provider`
- `subject`
- `login`
- `email`
- `display_name`
- `avatar_url`
- `created_at`
- `last_login_at`
- `disabled`

### `serve_roles`

- `id`
- `name`

### `serve_user_roles`

- `user_id`
- `role_id`
- `granted_by`
- `granted_at`

### `serve_sessions`

- `id`
- `user_id`
- `csrf_token`
- `expires_at`
- `created_at`
- `last_seen_at`
- `revoked_at`
- `ip_address`
- `user_agent`

## Authorization model

Permissions should be action-based and enforced server-side.

Suggested permission map:

| Action | Viewer | Operator | Editor | Admin |
|---|---:|---:|---:|---:|
| View dashboard / queue / results | yes | yes | yes | yes |
| View routes and config | yes | yes | yes | yes |
| Retry queue / clear queue | no | yes | yes | yes |
| Start / stop service | no | yes | yes | yes |
| Create route | no | no | yes | yes |
| Edit route | no | no | yes | yes |
| Delete / archive / promote route | no | no | yes | yes |
| Apply config | no | no | yes | yes |
| Manage users / roles | no | no | no | yes |
| Manage auth settings | no | no | no | yes |
| Manage email settings | no | no | no | yes |
| Start tunnel / public sharing | no | no | no | yes |

Implementation shape:

- add `require_permission("routes.write")` style dependencies in FastAPI
- keep role-to-permission mapping centralized
- return `403` with a human-readable message when blocked

## API changes

Add auth endpoints:

- `GET /api/auth/status`
- `POST /api/auth/login`
- `GET /api/auth/callback/{provider}`
- `POST /api/auth/logout`
- `GET /api/auth/me`

Add admin endpoints:

- `GET /api/admin/users`
- `POST /api/admin/users/{id}/roles`
- `DELETE /api/admin/users/{id}/roles/{role}`
- `GET /api/admin/auth/config`
- `PUT /api/admin/auth/config`

Add notification endpoints:

- `GET /api/admin/notifications/config`
- `PUT /api/admin/notifications/config`
- `POST /api/admin/notifications/test-email`

Protect existing route groups:

- `config.py`: read for `viewer+`, deploy/apply for `editor+`
- `serve_routes.py`: read for `viewer+`, write for `editor+`
- `service.py`: start/stop for `operator+`
- `queue.py`: retry/remove/clear for `operator+`
- `tunnel.py`: `admin+`
- `events.py` WebSocket: `viewer+`

## Frontend changes

### Login flow

- add a login screen or auth gate before the app shell
- show current user, role, and provider in the header
- expose logout
- wire CSRF token propagation into the shared API client

### Settings page

Add two new panels:

1. `Authentication`
2. `Email Notifications`

Authentication panel should allow an admin to:

- enable or disable auth
- select provider
- enter GitHub OAuth credentials
- configure allowed GitHub orgs/users
- view current role mappings

Email panel should allow an admin to:

- paste Resend API key
- set default sender name/address
- configure reply-to
- choose notification recipients
- send a test email

Sensitive values should never be fully echoed back after save.

### Permissions UX

- hide or disable buttons the current role cannot use
- still enforce everything on the backend
- show short permission errors, not generic failures

## Resend implementation

## Configuration

Store notification settings in a workspace-scoped file, for example:

`<workspace>/notifications.json`

Suggested shape:

```json
{
  "provider": "resend",
  "enabled": false,
  "resend": {
    "api_key": "",
    "from_email": "autoclean@examplelab.org",
    "from_name": "AutoClean",
    "reply_to": "scientist@examplelab.org"
  },
  "recipients": {
    "job_failed": ["pi@examplelab.org"],
    "route_disabled": ["admin@examplelab.org"],
    "service_stopped": ["ops@examplelab.org"],
    "daily_summary": []
  }
}
```

API keys should be stored carefully:

- acceptable short term: workspace config file with local file permissions and masked reads
- better medium term: OS keychain integration with only a key reference stored in workspace config

Notification delivery should also include cooldown and dedupe behavior to prevent alert storms.

## Notification service

Add a provider abstraction mirroring auth:

- `src/autoclean/api/notifications/base.py`
- `src/autoclean/api/notifications/resend.py`
- `src/autoclean/api/notifications/service.py`

Core interface:

```python
class NotificationProvider(Protocol):
    name: str
    async def send(self, message: NotificationMessage) -> NotificationResult: ...
```

Suggested initial triggers:

- service start
- service stop
- route validation failure
- repeated queue failure
- processing job failed
- daily summary

Email content should be concise and operational:

- workspace
- route id
- file path
- failure summary
- timestamp
- deep link back to Serve if available

Avoid attaching raw data or large reports in v1.

## Security notes

- block tunnel startup when auth is disabled
- use `HttpOnly`, `Secure`, and `SameSite=Lax` session cookies
- validate OAuth `state`
- rotate session ids on login
- rate-limit login and callback abuse points
- never send Resend API keys back to the client after save
- audit admin changes to auth, roles, routes, and notification settings
- authenticate WebSocket connections
- decide whether `/docs`, `/redoc`, and `/openapi.json` stay public

## Migration and rollout

## Phase 1

- add auth config model and provider abstraction
- implement GitHub OAuth
- add session persistence
- protect API endpoints with role checks
- create bootstrap admin flow

### Phase 1 Checklist

- [x] Add Serve-specific auth config loader/writer for `<workspace>/serve-auth.json`
- [x] Define normalized auth models: provider identity, user, session, role, permission
- [x] Create auth provider interface under `src/autoclean/api/auth/`
- [x] Implement GitHub OAuth provider
- [x] Add login, callback, logout, auth status, and `me` endpoints
- [x] Add SQLite-backed Serve auth/session store in `<workspace>/.serve/serve_state.db`
- [x] Keep Serve auth/session data separate from the existing `pipeline.db`
- [x] Add secure cookie handling
- [x] Add CSRF protection for state-changing requests
- [x] Add explicit bootstrap-admin assignment flow using allowlist or setup secret
- [x] Add role-to-permission map in one centralized module
- [x] Protect existing API endpoints with permission dependencies
- [x] Protect `/ws/events` with session validation and `viewer+` access
- [x] Decide whether `/docs`, `/redoc`, and `/openapi.json` are protected or disabled
- [x] Return clear `401` and `403` responses for auth and permission failures
- [x] Add backend tests for login flow, session validation, and permission checks

## Phase 2

- add admin user/role management UI
- add auth panel in Settings
- add disabled-auth warning banners
- block tunnel/public share without auth

### Phase 2 Checklist

- [x] Add frontend auth gate before the main Serve app shell
- [x] Show current user identity, provider, and role in the UI header
- [x] Add logout action in the web UI
- [x] Update `web/src/lib/api.ts` to send CSRF headers on write requests
- [x] Add `Authentication` section to the Settings page
- [x] Add admin UI to edit GitHub client id, client secret, allowed orgs, and allowed users
- [x] Add admin UI to enable auth, disable auth, and switch provider
- [x] Add persistent warning banner when auth is disabled
- [x] Add admin user list page or panel
- [x] Add role grant/revoke UI for `viewer`, `operator`, `editor`, `admin`
- [x] Hide or disable restricted actions for lower-permission users
- [x] Keep backend authorization enforcement unchanged even when UI hides buttons
- [x] Keep existing tunnel Basic Auth as an outer layer during migration
- [x] Block tunnel startup and public sharing when auth is disabled
- [x] Add frontend tests for auth-gated navigation and permissions-aware controls

## Phase 3

- add Resend provider
- add notifications settings UI
- add test-email action
- wire failure and service lifecycle events to notifications

### Phase 3 Checklist

- [x] Add notifications config loader/writer for `<workspace>/notifications.json`
- [x] Create notification provider interface under `src/autoclean/api/notifications/`
- [x] Implement Resend notification provider
- [x] Add backend validation for Resend API key and sender settings
- [x] Add admin notification config endpoints
- [x] Add test-email endpoint
- [x] Add `Email Notifications` panel in Settings
- [x] Add masked secret handling so saved API keys are not echoed back
- [x] Add recipient management for event categories
- [x] Add cooldown and dedupe rules for repeated alerts
- [x] Send notifications for service start and service stop
- [x] Send notifications for route validation failure
- [x] Send notifications for repeated queue failures
- [x] Send notifications for job failure events
- [x] Add backend tests for notification config and send flow
- [x] Add frontend tests for settings save and test-email flow

## Phase 4

- add second auth provider option such as generic OIDC
- add secret-storage hardening
- add notification digests and richer templates

### Phase 4 Checklist

- [x] Add a second provider implementation, preferably generic OIDC
- [x] Refactor provider selection so multiple configured providers are supported cleanly
- [x] Add provider-specific validation and health checks
- [x] Move secrets from plain workspace config toward OS keychain or equivalent secure storage
- [x] Store only secret references in workspace config where possible
- [x] Add audit logging for auth-config, user-role, and notification-config changes
- [x] Add daily digest or summary email job
- [x] Add richer email templates with deep links back into Serve
- [x] Evaluate per-route recipients or route-owner notifications if labs need it
- [x] Decide whether tunnel Basic Auth remains as a second outer layer or is retired
- [x] Add migration notes for existing Serve workspaces
- [x] Add end-to-end tests for multi-provider auth and notification workflows

## Open questions

- Should first-login admission be restricted to a GitHub org by default?
- Should local CLI commands bypass web auth when run from the same machine, or should they share the same auth config only conceptually?
- Should route ownership exist, or are lab-wide shared routes enough for now?
- Should role changes invalidate active sessions immediately?
- Should email recipient rules be global only, or also overridable per route?

## Final Decisions

- Tunnel Basic Auth stays as the outer layer for tunnel exposure. Serve auth remains the inner application auth layer. This preserves defense in depth and avoids weakening the public tunnel path.
- Multi-provider support is implemented and covered by end-to-end workflow tests using GitHub-style OAuth/OIDC login plus notification delivery flows.
- `/health` stays anonymous. It is used by local launchers, probes, and service supervision, and should remain stable even when Serve auth is enabled.
- Workspace setup remains available before login only for local first-run bootstrap flows. Remote clients cannot call workspace setup or recent-workspace discovery anonymously.
- Auth is enabled by default for workspace-backed Serve usage. Disabling auth is still an explicit operator choice for tightly controlled local environments.

## Recommended first implementation

If we want the fastest path that is still technically sound:

1. Build a Serve-specific auth subsystem, separate from the current compliance/Auth0 module.
2. Implement GitHub OAuth first.
3. Make auth enabled by default, but allow `auth.mode = disabled` for explicitly local installs.
4. Ship four roles only: `viewer`, `operator`, `editor`, `admin`.
5. Gate route creation/editing behind `editor`.
6. Gate tunnel sharing, auth settings, and email settings behind `admin`.
7. Add Resend as the first notification provider with test-email support and failure alerts.

That gives labs the safety they want without over-designing the first release.

## Implementation Order By Module

This section converts the plan into a practical engineering sequence. The order matters because auth, sessions, and permissions are cross-cutting concerns. Start with backend primitives, then wire enforcement, then update the frontend, then add notifications.

### Step 1. Add Serve state storage and config primitives

Files/modules:

- `src/autoclean/api/state.py`
- new `src/autoclean/api/auth/store.py`
- new `src/autoclean/api/auth/models.py`
- new `src/autoclean/api/auth/config.py`
- workspace files:
  - `<workspace>/serve-auth.json`
  - `<workspace>/.serve/serve_state.db`

Work:

- define where Serve auth/session state lives
- add config loader/writer for `serve-auth.json`
- create SQLite schema management for users, roles, sessions, and auth metadata
- keep this separate from `pipeline.db`

Why first:

- every later auth and permission feature depends on stable storage and models

### Step 2. Implement provider abstraction and GitHub OAuth

Files/modules:

- new `src/autoclean/api/auth/base.py`
- new `src/autoclean/api/auth/github.py`
- new `src/autoclean/api/auth/service.py`

Work:

- define provider interface
- implement GitHub login redirect, callback exchange, and normalized identity mapping
- enforce allowed GitHub org/user admission rules
- add bootstrap-admin resolution using allowlist or setup secret

Why second:

- this produces the authenticated identity object used by sessions and permissions

### Step 3. Add session, cookie, and CSRF infrastructure

Files/modules:

- new `src/autoclean/api/auth/session.py`
- `src/autoclean/api/server.py`
- possibly shared helpers in new `src/autoclean/api/auth/dependencies.py`

Work:

- create session records in SQLite
- issue `HttpOnly` session cookie
- issue CSRF token for write requests
- add auth/session middleware or FastAPI dependencies
- add logout and session revocation behavior
- decide behavior for `/docs`, `/redoc`, and `/openapi.json`

Why third:

- route protection should not land before sessions and CSRF are real

### Step 4. Add auth and admin endpoints

Files/modules:

- new `src/autoclean/api/routes/auth.py`
- new `src/autoclean/api/routes/admin_auth.py`
- new `src/autoclean/api/routes/admin_users.py`
- `src/autoclean/api/server.py`
- `src/autoclean/api/routes/__init__.py`

Work:

- add:
  - `GET /api/auth/status`
  - `POST /api/auth/login`
  - `GET /api/auth/callback/{provider}`
  - `POST /api/auth/logout`
  - `GET /api/auth/me`
- add admin endpoints for auth config and role assignment
- register the routers in `create_app()`

Why fourth:

- once auth primitives exist, the app needs stable API surfaces before page work starts

### Step 5. Add permission enforcement to existing backend routes

Files/modules:

- `src/autoclean/api/routes/config.py`
- `src/autoclean/api/routes/serve_routes.py`
- `src/autoclean/api/routes/service.py`
- `src/autoclean/api/routes/queue.py`
- `src/autoclean/api/routes/tunnel.py`
- `src/autoclean/api/routes/results.py`
- `src/autoclean/api/events.py`
- `src/autoclean/api/server.py`

Work:

- add `require_permission(...)` dependencies to each route group
- keep `viewer` access on read-only status/results paths
- gate route writes behind `editor`
- gate service and queue write actions behind `operator`
- gate tunnel and admin surfaces behind `admin`
- authenticate `/ws/events`
- preserve tunnel Basic Auth as the outer layer during migration

Why fifth:

- backend enforcement must be complete before the frontend starts hiding controls

### Step 6. Update the shared frontend API client

Files/modules:

- `web/src/lib/api.ts`

Work:

- add auth/session bootstrap helpers
- add CSRF header injection for `POST`, `PUT`, and `DELETE`
- normalize handling for `401` and `403`
- add types for current user, auth status, admin config, and notifications config

Why sixth:

- every page uses this module, so this is the frontend choke point

### Step 7. Add frontend auth gate and identity shell

Files/modules:

- likely `web/src/App.tsx`
- `web/src/components/TopBar.tsx`
- `web/src/components/Sidebar.tsx`
- new auth-specific page/component files under `web/src/`

Work:

- block the main app shell until auth state is known
- show login flow when auth is enabled and no session exists
- show current user, provider, and role in the top bar
- add logout
- show persistent auth-disabled warning banner when applicable

Why seventh:

- this establishes the user-facing session model before admin/settings pages are added

### Step 8. Add permissions-aware UI behavior

Files/modules:

- `web/src/pages/Routes.tsx`
- `web/src/pages/Settings.tsx`
- `web/src/pages/Service.tsx`
- `web/src/pages/Queue.tsx`
- `web/src/components/TopBar.tsx`

Work:

- hide or disable route-edit actions for non-editors
- hide or disable service controls for non-operators
- hide or disable tunnel and auth settings for non-admins
- keep error handling explicit when the backend returns `403`

Why eighth:

- the backend should already be authoritative, so this step becomes mostly UX cleanup

### Step 9. Add admin auth management UI

Files/modules:

- `web/src/pages/Settings.tsx`
- possibly new admin components under `web/src/components/`

Work:

- add `Authentication` settings section
- add GitHub OAuth config form
- add auth mode toggle
- add admin user list and role grant/revoke UI
- surface bootstrap-admin status clearly

Why ninth:

- by this point the auth APIs exist and the user session model is already live

### Step 10. Add notification provider abstraction and Resend backend

Files/modules:

- new `src/autoclean/api/notifications/base.py`
- new `src/autoclean/api/notifications/resend.py`
- new `src/autoclean/api/notifications/service.py`
- new `src/autoclean/api/routes/admin_notifications.py`

Work:

- add notification provider interface
- implement Resend sender
- add notification config persistence
- add test-email endpoint
- add cooldown and dedupe logic

Why tenth:

- email should land after auth/admin foundations, because only admins should configure it

### Step 11. Add notifications UI

Files/modules:

- `web/src/pages/Settings.tsx`
- possibly new form components under `web/src/components/`

Work:

- add `Email Notifications` section
- add sender and recipient config forms
- add masked API key handling
- add test-email flow

Why eleventh:

- this depends on the backend notification endpoints being complete

### Step 12. Wire event sources into notifications

Files/modules:

- `src/autoclean/api/routes/service.py`
- `src/autoclean/api/routes/config.py`
- `src/autoclean/api/routes/queue.py`
- `src/autoclean/api/events.py`
- notification service modules

Work:

- emit notification events on service start/stop
- emit notifications on route validation/apply failures where appropriate
- emit notifications on repeated queue or processing failures
- keep alerting deduped and rate-limited

Why twelfth:

- event wiring is safer once the provider, config, and throttling logic already exist

### Step 13. Test and harden

Files/modules:

- backend tests under `tests/`
- frontend tests under `web/src/**/*.test.tsx`

Recommended coverage:

- auth config parsing
- GitHub callback flow
- session creation, expiry, logout, and revocation
- CSRF rejection and success cases
- permission checks per route family
- WebSocket auth behavior
- tunnel behavior with auth enabled vs disabled
- Resend config validation
- notification cooldown/dedupe behavior
- permissions-aware frontend rendering

## File Ownership Suggestion

If multiple engineers work in parallel, keep write scopes separate:

- Engineer 1: backend auth core
  - `src/autoclean/api/auth/*`
  - `src/autoclean/api/state.py`
  - `src/autoclean/api/routes/auth.py`
  - `src/autoclean/api/routes/admin_auth.py`
  - `src/autoclean/api/routes/admin_users.py`
- Engineer 2: backend route enforcement and tunnel/WebSocket integration
  - `src/autoclean/api/server.py`
  - `src/autoclean/api/events.py`
  - existing route modules under `src/autoclean/api/routes/`
- Engineer 3: frontend auth shell and permissions UX
  - `web/src/lib/api.ts`
  - `web/src/components/*`
  - `web/src/pages/Settings.tsx`
  - `web/src/pages/Routes.tsx`
  - `web/src/pages/Service.tsx`
  - `web/src/pages/Queue.tsx`
- Engineer 4: notifications
  - `src/autoclean/api/notifications/*`
  - `src/autoclean/api/routes/admin_notifications.py`
  - notification UI in `web/src/pages/Settings.tsx`

## Recommended Milestones

### Milestone A

- Steps 1 through 5 complete
- backend auth and permission enforcement live
- frontend still rough is acceptable

#### Milestone A Phase Breakdown

##### Phase 1. Auth foundation and read-only enforcement

Goal:

- land the minimum backend foundation without taking on full admin UI or every mutation path at once

Include:

- Serve auth config models and loader/writer
- dedicated SQLite Serve auth/session store
- provider abstraction
- GitHub OAuth provider
- session cookie issuance and lookup
- CSRF support primitives
- auth endpoints:
  - `GET /api/auth/status`
  - `POST /api/auth/login`
  - `GET /api/auth/callback/{provider}`
  - `POST /api/auth/logout`
  - `GET /api/auth/me`
- `get_current_user()` and `require_permission()` dependencies
- protection for:
  - `/api/status`
  - `/health` if required by final decision
  - read-only route/config/results endpoints
  - `/ws/events`
- explicit bootstrap-admin mechanism
- backend tests for storage, session, OAuth callback, and read-only permission checks

Do not include yet:

- role-management endpoints
- auth-config admin endpoints
- mutation protection for every write path
- tunnel behavior changes beyond compatibility
- frontend app-shell work

Why this is the first phase:

- it proves the architecture
- it validates the auth/session model early
- it reduces the risk of designing the frontend around unstable backend assumptions

##### Phase 2. Write-path permission enforcement and admin endpoints

Goal:

- extend the backend from authenticated reads into controlled mutations

Include:

- admin auth config endpoints
- admin user/role endpoints
- enforcement on:
  - route create/update/delete/promote/archive
  - config deploy/apply
  - service start/stop
  - queue mutation actions
  - tunnel admin actions
- explicit `401` and `403` behavior across write paths
- backend tests for editor/operator/admin boundaries

Why separate from Phase 1:

- permission enforcement on mutations touches many route modules
- review is cleaner once auth/session primitives are already merged

##### Phase 3. Backend hardening and migration cleanup

Goal:

- finish the backend edges before frontend-heavy work starts

Include:

- final decision and implementation for `/docs`, `/redoc`, `/openapi.json`
- auth-disabled local-only constraints
- tunnel migration behavior validation
- session revocation edge cases
- role-change invalidation behavior if chosen
- any missing API error-shape cleanup
- additional backend tests for hardening scenarios

Why separate:

- these are important but should not block proving the core auth path
- they are easier to evaluate once Phase 1 and Phase 2 behavior is concrete

#### Milestone A Concrete Task List

##### A1. Create Serve auth storage primitives

Files:

- `src/autoclean/api/auth/models.py`
- `src/autoclean/api/auth/store.py`
- `src/autoclean/api/auth/config.py`

Tasks:

- [x] Define Pydantic models for auth config, user, role, session, and provider identity
- [x] Define the normalized permission ids used across the backend
- [x] Add loader/writer for `<workspace>/serve-auth.json`
- [x] Add bootstrap logic to create `<workspace>/.serve/`
- [x] Add SQLite schema creation for:
  - `serve_users`
  - `serve_roles`
  - `serve_user_roles`
  - `serve_sessions`
- [x] Seed default roles and role-permission mapping
- [x] Add helpers for session lookup, session create, session revoke, and role lookup
  - `serve_roles`
  - `serve_user_roles`
  - `serve_sessions`
- [x] Seed default roles and role-permission mapping
- [x] Add helpers for session lookup, session create, session revoke, and role lookup

##### A2. Implement provider abstraction and GitHub provider

Files:

- `src/autoclean/api/auth/base.py`
- `src/autoclean/api/auth/github.py`
- `src/autoclean/api/auth/service.py`

Tasks:

- [x] Define the provider interface and provider registry
- [x] Implement GitHub OAuth authorization URL generation
- [x] Implement callback code exchange
- [x] Normalize GitHub user payload into internal identity model
- [x] Enforce `allowed_orgs` and `allowed_users`
- [x] Implement bootstrap-admin assignment using explicit allowlist or setup secret
- [x] Add error mapping for rejected login, missing email, invalid callback, and provider misconfiguration

##### A3. Add auth/session dependencies into app wiring

Files:

- `src/autoclean/api/server.py`
- `src/autoclean/api/state.py`
- `src/autoclean/api/auth/session.py`
- `src/autoclean/api/auth/dependencies.py`

Tasks:

- [x] Extend app startup/helpers so auth config and Serve state DB are available for the active workspace
- [x] Add session cookie creation and parsing
- [x] Add CSRF token issuance and validation helpers
- [x] Add `get_current_user()` dependency
- [x] Add `require_permission()` dependency
- [x] Decide and implement protection behavior for `/docs`, `/redoc`, and `/openapi.json`
- [x] Keep tunnel Basic Auth middleware intact for now

##### A4. Add auth and admin backend endpoints

Files:

- `src/autoclean/api/routes/auth.py`
- `src/autoclean/api/routes/admin_auth.py`
- `src/autoclean/api/routes/admin_users.py`
- `src/autoclean/api/routes/__init__.py`
- `src/autoclean/api/server.py`

Tasks:

- [x] Add `GET /api/auth/status`
- [x] Add `POST /api/auth/login`
- [x] Add `GET /api/auth/callback/{provider}`
- [x] Add `POST /api/auth/logout`
- [x] Add `GET /api/auth/me`
- [x] Add `GET /api/admin/auth/config`
- [x] Add `PUT /api/admin/auth/config`
- [x] Add `GET /api/admin/users`
- [x] Add `POST /api/admin/users/{id}/roles`
- [x] Add `DELETE /api/admin/users/{id}/roles/{role}`
- [x] Register new routers in `create_app()`

##### A5. Enforce permissions on existing route families

Files:

- `src/autoclean/api/routes/config.py`
- `src/autoclean/api/routes/serve_routes.py`
- `src/autoclean/api/routes/service.py`
- `src/autoclean/api/routes/queue.py`
- `src/autoclean/api/routes/tunnel.py`
- `src/autoclean/api/routes/results.py`
- `src/autoclean/api/events.py`
- `src/autoclean/api/server.py`

Tasks:

- [x] Require `viewer` access for read-only config, status, results, and route reads
- [x] Require `editor` access for route writes and config apply/deploy actions
- [x] Require `operator` access for service control and queue mutation actions
- [x] Require `admin` access for tunnel and admin routes
- [x] Add authenticated session checks to `/ws/events`
- [x] Return explicit `401` for unauthenticated requests
- [x] Return explicit `403` for insufficient-role requests
- [x] Preserve local-only behavior when auth mode is disabled

##### A6. Add backend test coverage for Milestone A

Files:

- `tests/` additions in appropriate API/auth modules

Tasks:

- [x] Test auth config parsing and defaults
- [x] Test SQLite schema bootstrap and role seeding
- [x] Test GitHub provider validation and callback handling with mocked upstream responses
- [x] Test session creation, expiration, and revocation
- [x] Test CSRF enforcement for write endpoints
- [x] Test permission enforcement on:
  - route read vs write
  - service read vs start/stop
  - queue read vs mutation
  - tunnel access
- [x] Test WebSocket rejection without valid session
- [x] Test auth-disabled local mode behavior

##### A7. Milestone A exit criteria

- [x] A user can authenticate through GitHub when auth mode is enabled
- [x] A valid session cookie is required for protected HTTP endpoints
- [x] A valid session is required for `/ws/events`
- [x] Roles are enforced on route, config, queue, service, and tunnel actions
- [x] Admin can inspect and update auth config through backend endpoints
- [x] Existing tunnel Basic Auth still works as an outer layer
- [x] Backend test suite covers the new auth and permission paths

#### Recommended first Phase scope

If only one phase should start immediately, keep it to:

- `A1` storage/config primitives
- `A2` provider abstraction and GitHub provider
- the session/auth parts of `A3`
- auth endpoints from `A4`
- read-only enforcement from `A5`
- the corresponding subset of `A6`

That means the first phase should answer only these questions:

- can Serve authenticate a user safely
- can Serve issue and validate sessions correctly
- can Serve protect read surfaces and WebSocket connections
- can the backend establish a stable role/permission model

It should not yet try to finish the full admin surface or all mutation controls in one pass.

#### Phase 1 Exact Implementation Checklist

This is the recommended first backend build sequence. The goal is to make authentication real, keep scope tight, and avoid mixing in all mutation/admin work.

##### Step 1. Add auth models and config plumbing

Files to create:

- `src/autoclean/api/auth/models.py`
- `src/autoclean/api/auth/config.py`
- `src/autoclean/api/auth/__init__.py`

Files to update:

- `src/autoclean/api/state.py`

Checklist:

- [x] Create auth config models for:
  - auth mode
  - provider selection
  - session cookie settings
  - GitHub provider config
- [x] Create normalized runtime models for:
  - provider identity
  - auth user
  - auth session
  - role
  - permission
- [x] Add centralized permission ids for the first release
- [x] Add config loader/writer for `<workspace>/serve-auth.json`
- [x] Add helper to resolve `<workspace>/.serve/` paths
- [x] Extend `APIState` with accessors/helpers needed to locate Serve auth state safely

Step intent:

- no OAuth yet
- no DB writes yet
- just stable types and config plumbing

##### Step 2. Add SQLite-backed auth store

Files to create:

- `src/autoclean/api/auth/store.py`

Files to update:

- `src/autoclean/api/state.py`

Checklist:

- [x] Create store bootstrap for `<workspace>/.serve/serve_state.db`
- [x] Add schema creation for:
  - `serve_users`
  - `serve_roles`
  - `serve_user_roles`
  - `serve_sessions`
- [x] Seed default roles
- [x] Add centralized role-to-permission mapping
- [x] Add helpers for:
  - get/create user
  - assign bootstrap role
  - create session
  - get session by id
  - revoke session
  - list user permissions
- [x] Keep this store fully separate from `pipeline.db`

Step intent:

- persistence layer exists and is testable before any route wiring

##### Step 3. Add provider abstraction and GitHub OAuth service

Files to create:

- `src/autoclean/api/auth/base.py`
- `src/autoclean/api/auth/github.py`
- `src/autoclean/api/auth/service.py`

Checklist:

- [x] Define provider protocol and provider registry
- [x] Implement GitHub authorization URL generation
- [x] Implement callback code exchange
- [x] Fetch/normalize GitHub identity into internal model
- [x] Enforce allowed-user and allowed-org checks
- [x] Add bootstrap-admin resolution using allowlist or setup secret
- [x] Add clean failure paths for:
  - provider not configured
  - invalid callback state
  - rejected membership
  - token exchange failure

Step intent:

- make OAuth flow testable before cookies and route dependencies are added

##### Step 4. Add session and dependency layer

Files to create:

- `src/autoclean/api/auth/session.py`
- `src/autoclean/api/auth/dependencies.py`

Files to update:

- `src/autoclean/api/server.py`

Checklist:

- [x] Add secure session-id generation
- [x] Add session-cookie write/clear helpers
- [x] Add CSRF token generation helpers
- [x] Add `get_current_user()` dependency
- [x] Add `require_permission()` dependency
- [x] Add request-path logic for authenticated read access
- [x] Decide and implement Phase-1 behavior for:
  - `/docs`
  - `/redoc`
  - `/openapi.json`

Recorded decision:

- Leave `/docs`, `/redoc`, and `/openapi.json` outside Serve auth enforcement.
- Do not add permission checks to those endpoints.
- [x] Keep tunnel Basic Auth behavior unchanged

Step intent:

- route handlers should be able to ask for user/session/permission without owning auth logic

##### Step 5. Add auth routes and app registration

Files to create:

- `src/autoclean/api/routes/auth.py`

Files to update:

- `src/autoclean/api/routes/__init__.py`
- `src/autoclean/api/server.py`

Checklist:

- [x] Add `GET /api/auth/status`
- [x] Add `POST /api/auth/login`
- [x] Add `GET /api/auth/callback/{provider}`
- [x] Add `POST /api/auth/logout`
- [x] Add `GET /api/auth/me`
- [x] Register auth router in `create_app()`
- [x] Make sure auth endpoints work before the rest of the app is fully protected

Step intent:

- expose the minimal backend contract the later frontend work will depend on

##### Step 6. Protect read-only HTTP surfaces

Files to update:

- `src/autoclean/api/server.py`
- `src/autoclean/api/routes/config.py`
- `src/autoclean/api/routes/serve_routes.py`
- `src/autoclean/api/routes/results.py`
- any read-only status route code in `src/autoclean/api/server.py`

Checklist:

- [x] Require at least `viewer` for read-only config endpoints
- [x] Require at least `viewer` for route listing and route reads
- [x] Require at least `viewer` for results reads
- [x] Require at least `viewer` for dashboard/status reads
- [x] Decide whether `/health` stays anonymous or becomes gated
- [x] Return explicit `401` when auth is enabled and no valid session exists
- [x] Return explicit `403` when session exists but permissions are insufficient

Step intent:

- first phase should prove that auth is actually enforced somewhere meaningful without changing all write paths yet

##### Step 7. Protect WebSocket event stream

Files to update:

- `src/autoclean/api/events.py`

Checklist:

- [x] Validate session on `/ws/events` connect
- [x] Require at least `viewer`
- [x] Reject or close connection when session is missing, expired, or unauthorized
- [x] Preserve current ping/pong and broadcast behavior for valid clients

Step intent:

- close the most obvious auth gap in the current Serve runtime

##### Step 8. Add focused backend tests for Phase 1

Files to create/update:

- `tests/` files covering auth store, provider, auth routes, and protected reads

Suggested test files:

- `tests/unit/api/test_auth_config.py`
- `tests/unit/api/test_auth_store.py`
- `tests/unit/api/test_auth_routes.py`
- `tests/unit/api/test_auth_permissions.py`
- `tests/unit/api/test_events_auth.py`

Checklist:

- [x] Test config parsing and defaults
- [x] Test store bootstrap and schema creation
- [x] Test role seeding and permission lookup
- [x] Test GitHub callback flow with mocked HTTP responses
- [x] Test login success and failure paths
- [x] Test session cookie creation and logout
- [x] Test protected read endpoints with:
  - no session
  - valid viewer session
  - invalid/expired session
- [x] Test WebSocket rejection without valid session

Step intent:

- keep tests aligned to the exact scope of Phase 1 instead of writing speculative tests for future admin features

##### Phase 1 validation checklist

Historical note:

- Phase 1 originally excluded admin UI, broad write-path enforcement, and notifications to keep the first backend slice narrow.
- Later phases intentionally add those capabilities, so those original scope constraints no longer apply to the current codebase.
- [x] `pipeline.db` is untouched by Serve auth storage
- [x] Existing tunnel Basic Auth remains compatible
- [x] New auth code is isolated under `src/autoclean/api/auth/` where possible

##### Phase 1 done means

- [x] GitHub login works end-to-end in the backend
- [x] Serve can issue and revoke session cookies
- [x] Read-only backend surfaces can require authenticated `viewer` access
- [x] `/ws/events` is no longer anonymously accessible when auth is enabled
- [x] The backend contract is stable enough for the frontend auth shell in Milestone B

### Milestone B

- Steps 6 through 9 complete
- full auth-enabled Serve UI usable by labs

#### Milestone B Concrete Task List

##### B1. Upgrade the shared frontend API client for auth-aware requests

Files:

- `web/src/lib/api.ts`

Tasks:

- [x] Add auth-related response types:
  - auth status
  - current user
  - auth config
  - admin user list
- [x] Add request handling for session-aware fetches
- [x] Add CSRF header injection for `POST`, `PUT`, and `DELETE`
- [x] Normalize `401` handling so the UI can redirect to login cleanly
- [x] Normalize `403` handling so pages can show permission errors cleanly
- [x] Add client methods for auth and admin endpoints introduced in Milestone A

##### B2. Add auth gate and session bootstrap to the app shell

Files:

- `web/src/App.tsx`
- new auth helpers/components under `web/src/`

Tasks:

- [x] Load auth status before rendering the main app shell
- [x] Redirect or gate unauthenticated users when auth mode is enabled
- [x] Allow direct entry into the app when auth mode is disabled
- [x] Add loading state while auth status is being resolved
- [x] Add failure state for auth/session bootstrap errors

##### B3. Surface current identity and session controls in shared layout

Files:

- `web/src/components/TopBar.tsx`
- `web/src/components/Sidebar.tsx`
- new auth-related UI components if needed

Tasks:

- [x] Show current user login/display name
- [x] Show current provider
- [x] Show current effective role
- [x] Add logout action
- [x] Add persistent auth-disabled warning banner when applicable
- [x] Make permission-restricted share/tunnel affordances reflect the current role

##### B4. Add login and auth-management UI flows

Files:

- new auth-specific page/components under `web/src/`
- `web/src/pages/Settings.tsx`

Tasks:

- [x] Add login screen or login gate UI
- [x] Add GitHub sign-in action
- [x] Add clear messaging for denied org/user membership
- [x] Add clear messaging for expired or revoked sessions
- [x] Add `Authentication` settings panel for admins
- [x] Add auth mode toggle in the admin settings UI
- [x] Add GitHub OAuth config form:
  - client id
  - client secret
  - redirect URI display
  - allowed orgs
  - allowed users
- [x] Mask stored sensitive values after save

##### B5. Add admin user and role management UI

Files:

- `web/src/pages/Settings.tsx`
- optional new admin components under `web/src/components/`

Tasks:

- [x] Add admin-facing user list
- [x] Show each user’s current roles
- [x] Add role grant UI
- [x] Add role revoke UI
- [x] Show useful user fields such as login, provider, last login, and disabled state if available
- [x] Show clear success/error states for role changes

##### B6. Make role-aware controls explicit across the main pages

Files:

- `web/src/pages/Routes.tsx`
- `web/src/pages/Service.tsx`
- `web/src/pages/Queue.tsx`
- `web/src/pages/Settings.tsx`
- `web/src/components/TopBar.tsx`

Tasks:

- [x] Hide or disable route create/edit/delete/promote/archive actions for non-editors
- [x] Hide or disable config apply actions for non-editors
- [x] Hide or disable service start/stop actions for non-operators
- [x] Hide or disable queue retry/clear/remove actions for non-operators
- [x] Hide or disable tunnel controls for non-admins
- [x] Hide or disable auth settings and user-management controls for non-admins
- [x] Show short inline permission messages when an action is blocked

##### B7. Preserve current setup and workspace flows under auth

Files:

- pages/components that depend on:
  - `/health`
  - `/api/status`
  - `/api/setup/workspace`
  - `/api/workspaces/recent`

Tasks:

- [x] Decide whether workspace setup requires auth or is allowed pre-login during first-run bootstrap
- [x] Ensure the workspace picker still works in first-run scenarios
- [x] Ensure auth state refreshes correctly after workspace switch
- [x] Ensure role-aware UI updates correctly after workspace change

##### B8. Add frontend test coverage for Milestone B

Files:

- `web/src/**/*.test.tsx`

Tasks:

- [x] Test auth gate behavior for:
  - auth enabled + no session
  - auth enabled + valid session
  - auth disabled
- [x] Test login/logout UI behavior
- [x] Test admin-only settings visibility
- [x] Test editor/operator/viewer permission behavior on affected pages
- [x] Test `401` and `403` API error handling paths
- [x] Test auth-disabled warning banner behavior

##### B9. Milestone B exit criteria

- [x] A lab user can sign in through the Serve web UI when auth is enabled
- [x] The app shell correctly blocks unauthenticated access
- [x] The current user identity and role are visible in shared UI
- [x] Admins can manage auth config and user roles from the UI
- [x] Non-admins do not see or cannot use restricted settings and share controls
- [x] Editors can manage routes while viewers cannot
- [x] Operators can control service and queue actions while viewers cannot
- [x] Frontend tests cover auth gate and permissions-aware rendering

### Milestone C

- Steps 10 through 12 complete
- Resend notifications live with throttling

#### Milestone C Concrete Task List

##### C1. Add notification config and storage primitives

Files:

- `src/autoclean/api/notifications/service.py`
- new `src/autoclean/api/notifications/models.py`
- optionally new `src/autoclean/api/notifications/config.py`
- workspace file:
  - `<workspace>/notifications.json`

Tasks:

- [x] Define notification config models
- [x] Define notification message and delivery result models
- [x] Add loader/writer for `<workspace>/notifications.json`
- [x] Add validation for:
  - provider selection
  - sender email
  - reply-to
  - recipient lists
- [x] Add masking behavior for stored secret fields when returned to the UI

##### C2. Add provider abstraction and Resend implementation

Files:

- `src/autoclean/api/notifications/base.py`
- `src/autoclean/api/notifications/resend.py`
- `src/autoclean/api/notifications/service.py`

Tasks:

- [x] Define notification provider interface
- [x] Add provider registry
- [x] Implement Resend API client integration
- [x] Map notification message model to Resend payload shape
- [x] Handle delivery errors, provider misconfiguration, and network failures cleanly
- [x] Add sender-domain and API-key validation path where possible

##### C3. Add throttling, cooldown, and dedupe behavior

Files:

- `src/autoclean/api/notifications/service.py`
- optionally dedicated persistence helpers if needed

Tasks:

- [x] Define dedupe key strategy for repeated failures
- [x] Define cooldown windows by event type
- [x] Prevent repeated identical alerts from spamming recipients
- [x] Allow important state-change alerts through when they are meaningfully distinct
- [x] Add enough persistence to survive process restarts if cooldown behavior needs to persist

##### C4. Add backend admin notification endpoints

Files:

- `src/autoclean/api/routes/admin_notifications.py`
- `src/autoclean/api/routes/__init__.py`
- `src/autoclean/api/server.py`

Tasks:

- [x] Add `GET /api/admin/notifications/config`
- [x] Add `PUT /api/admin/notifications/config`
- [x] Add `POST /api/admin/notifications/test-email`
- [x] Require `admin` permission on all notification admin endpoints
- [x] Register new router in `create_app()`

##### C5. Wire notification triggers into Serve runtime paths

Files:

- `src/autoclean/api/routes/service.py`
- `src/autoclean/api/routes/config.py`
- `src/autoclean/api/routes/queue.py`
- `src/autoclean/api/events.py`
- any processing/job event integration points discovered during implementation

Tasks:

- [x] Send notification on service start if configured
- [x] Send notification on service stop if configured
- [x] Send notification on route validation/config apply failures where meaningful
- [x] Send notification on repeated queue failures
- [x] Send notification on job failure events
- [x] Ensure notification sending failures do not break the main Serve action path

##### C6. Add notification management UI

Files:

- `web/src/pages/Settings.tsx`
- optional new notification form components under `web/src/components/`
- `web/src/lib/api.ts`

Tasks:

- [x] Add notification config types and client methods to `web/src/lib/api.ts`
- [x] Add `Email Notifications` section to Settings
- [x] Add provider selection UI if future-proofing is desired now
- [x] Add Resend API key input with masked post-save behavior
- [x] Add sender name, sender email, and reply-to fields
- [x] Add recipient list management for supported event categories
- [x] Add test-email action and success/error feedback
- [x] Restrict all notification settings UI to admins

##### C7. Add backend and frontend tests for notifications

Files:

- backend tests under `tests/`
- frontend tests under `web/src/**/*.test.tsx`

Tasks:

- [x] Test notification config parsing and validation
- [x] Test Resend payload construction with mocked outbound calls
- [x] Test handling of provider errors and invalid credentials
- [x] Test cooldown and dedupe behavior
- [x] Test admin-only access to notification endpoints
- [x] Test Settings notification form save flow
- [x] Test masked secret behavior after save
- [x] Test test-email flow success and failure paths

##### C8. Milestone C exit criteria

- [x] Admin can configure Resend settings from the UI
- [x] Resend API key is never echoed back in plaintext after save
- [x] Test-email works through the Serve UI and backend
- [x] Service and failure notifications can be emitted without breaking core Serve operations
- [x] Cooldown/dedupe behavior prevents obvious alert spam
- [x] Notification settings and endpoints are restricted to admins
- [x] Tests cover provider behavior, config validation, and UI flows

### Milestone D

- Step 13 complete
- docs and migration notes complete

## Delivery Checklist

### Ready for implementation when

- [x] Phase 1 checklist is approved
- [x] Storage approach for sessions is chosen
- [x] GitHub OAuth app registration requirements are documented
- [x] Resend sender-domain expectations are documented

### Minimum shippable version

- [x] Phase 1 complete
- [x] Phase 2 complete
- [x] Auth enabled by default
- [x] Route edits protected by role checks
- [x] Tunnel blocked when auth is disabled

### Recommended first full release

- [x] Phase 1 complete
- [x] Phase 2 complete
- [x] Phase 3 complete
- [x] Admin can configure auth and Resend from the UI
- [x] Labs can assign undergrads limited roles
- [x] Failure emails can be sent without code changes
