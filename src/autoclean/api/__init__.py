"""AutoClean API for serve workspace administration.

Provides a FastAPI-based REST API and RQ job queue for managing
the automation system remotely.
"""

from autoclean.api.server import create_app

__all__ = ["create_app"]
