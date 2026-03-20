"""Base protocol for Serve auth providers."""

from __future__ import annotations

from typing import Protocol

from autoclean.api.auth.models import ProviderIdentity


class AuthProvider(Protocol):
    """Protocol implemented by Serve auth providers."""

    name: str

    def build_login_url(self, *, state: str) -> str:
        """Return the provider login URL for a given state token."""

    def exchange_code(self, *, code: str) -> ProviderIdentity:
        """Exchange an authorization code for a normalized identity."""
