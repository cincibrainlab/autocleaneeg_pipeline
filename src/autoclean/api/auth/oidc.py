"""Generic OIDC provider for Serve."""

from __future__ import annotations

from urllib.parse import urlencode

import requests

from autoclean.api.auth.models import OIDCAuthConfig, ProviderIdentity


class OIDCAuthError(RuntimeError):
    """Raised when generic OIDC auth fails."""


class OIDCAuthProvider:
    """Generic OIDC implementation based on discovery."""

    name = "oidc"

    def __init__(self, config: OIDCAuthConfig):
        self.config = config

    def is_configured(self) -> bool:
        return bool(
            self.config.issuer_url.strip()
            and self.config.client_id.strip()
            and self.config.client_secret.strip()
        )

    def _well_known(self) -> dict[str, object]:
        if not self.is_configured():
            raise OIDCAuthError("OIDC auth is not configured")
        issuer = self.config.issuer_url.rstrip("/")
        response = requests.get(f"{issuer}/.well-known/openid-configuration", timeout=10)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise OIDCAuthError("Invalid OIDC discovery payload")
        return payload

    def build_login_url(self, *, state: str) -> str:
        metadata = self._well_known()
        authorization_endpoint = str(metadata.get("authorization_endpoint") or "")
        if not authorization_endpoint:
            raise OIDCAuthError("OIDC discovery is missing authorization_endpoint")
        query = urlencode(
            {
                "client_id": self.config.client_id,
                "redirect_uri": self.config.redirect_uri,
                "response_type": "code",
                "scope": " ".join(self.config.scopes),
                "state": state,
            }
        )
        return f"{authorization_endpoint}?{query}"

    def exchange_code(self, *, code: str) -> ProviderIdentity:
        metadata = self._well_known()
        token_endpoint = str(metadata.get("token_endpoint") or "")
        userinfo_endpoint = str(metadata.get("userinfo_endpoint") or "")
        if not token_endpoint or not userinfo_endpoint:
            raise OIDCAuthError("OIDC discovery is missing token or userinfo endpoint")
        token_response = requests.post(
            token_endpoint,
            data={
                "grant_type": "authorization_code",
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
            raise OIDCAuthError("OIDC token exchange failed: missing access token")
        user_response = requests.get(
            userinfo_endpoint,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        user_response.raise_for_status()
        claims = user_response.json()
        if not isinstance(claims, dict):
            raise OIDCAuthError("OIDC userinfo response was invalid")
        login = claims.get(self.config.username_claim) or claims.get("email") or claims.get("sub")
        if not login:
            raise OIDCAuthError("OIDC userinfo did not contain a usable login claim")
        groups_claim = claims.get(self.config.groups_claim, [])
        groups = groups_claim if isinstance(groups_claim, list) else []
        return ProviderIdentity(
            provider=self.name,
            subject=str(claims.get("sub") or login),
            login=str(login),
            email=claims.get("email") if isinstance(claims.get("email"), str) else None,
            display_name=claims.get("name") if isinstance(claims.get("name"), str) else None,
            avatar_url=claims.get("picture") if isinstance(claims.get("picture"), str) else None,
            groups=[str(group) for group in groups],
            raw_claims=claims,
        )
