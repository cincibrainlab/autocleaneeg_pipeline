"""GitHub OAuth provider for Serve."""

from __future__ import annotations

from urllib.parse import urlencode

import requests

from autoclean.api.auth.models import GitHubAuthConfig, ProviderIdentity


class GitHubAuthError(RuntimeError):
    """Raised when GitHub OAuth fails."""


class GitHubAuthProvider:
    """GitHub OAuth implementation."""

    name = "github"

    def __init__(self, config: GitHubAuthConfig):
        self.config = config

    def is_configured(self) -> bool:
        return bool(self.config.client_id.strip() and self.config.client_secret.strip())

    def build_login_url(self, *, state: str) -> str:
        if not self.is_configured():
            raise GitHubAuthError("GitHub auth is not configured")
        query = urlencode(
            {
                "client_id": self.config.client_id,
                "redirect_uri": self.config.redirect_uri,
                "scope": "read:user user:email read:org",
                "state": state,
            }
        )
        return f"https://github.com/login/oauth/authorize?{query}"

    def exchange_code(self, *, code: str) -> ProviderIdentity:
        if not self.is_configured():
            raise GitHubAuthError("GitHub auth is not configured")

        token_response = requests.post(
            "https://github.com/login/oauth/access_token",
            headers={"Accept": "application/json"},
            data={
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
                "code": code,
                "redirect_uri": self.config.redirect_uri,
            },
            timeout=10,
        )
        token_response.raise_for_status()
        token_payload = token_response.json()
        access_token = token_payload.get("access_token")
        if not access_token:
            raise GitHubAuthError(
                f"GitHub token exchange failed: {token_payload.get('error', 'missing access token')}"
            )

        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {access_token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        user_response = requests.get(
            "https://api.github.com/user",
            headers=headers,
            timeout=10,
        )
        user_response.raise_for_status()
        user_payload = user_response.json()

        emails_response = requests.get(
            "https://api.github.com/user/emails",
            headers=headers,
            timeout=10,
        )
        emails_response.raise_for_status()
        emails_payload = emails_response.json()
        primary_email = None
        if isinstance(emails_payload, list):
            for entry in emails_payload:
                if not isinstance(entry, dict):
                    continue
                if entry.get("primary") and entry.get("verified"):
                    primary_email = entry.get("email")
                    break
            if primary_email is None:
                for entry in emails_payload:
                    if isinstance(entry, dict) and entry.get("email"):
                        primary_email = entry.get("email")
                        break

        groups = self.load_orgs(access_token=access_token)

        return ProviderIdentity(
            provider=self.name,
            subject=str(user_payload["id"]),
            login=user_payload.get("login", ""),
            email=primary_email,
            display_name=user_payload.get("name"),
            avatar_url=user_payload.get("avatar_url"),
            groups=groups,
            raw_claims=user_payload,
        )

    def load_orgs(self, *, access_token: str) -> list[str]:
        """Future hook for org-based admission."""
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {access_token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        response = requests.get(
            "https://api.github.com/user/orgs",
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            return []
        return [
            entry.get("login", "")
            for entry in payload
            if isinstance(entry, dict) and entry.get("login")
        ]
