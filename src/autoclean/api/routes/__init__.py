"""API route modules."""

from autoclean.api.routes import (
    admin_audit,
    admin_auth,
    admin_notifications,
    admin_users,
    auth,
    config,
    queue,
    serve_routes,
    service,
    worker,
)

__all__ = [
    "admin_audit",
    "admin_auth",
    "admin_notifications",
    "admin_users",
    "auth",
    "config",
    "queue",
    "serve_routes",
    "service",
    "worker",
]
