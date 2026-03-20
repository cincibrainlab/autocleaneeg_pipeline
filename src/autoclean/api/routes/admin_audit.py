"""Admin endpoints for Serve audit events."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from autoclean.api.audit import list_audit_events
from autoclean.api.auth.dependencies import require_permission
from autoclean.api.auth.models import Permission
from autoclean.api.state import api_state

router = APIRouter()


@router.get(
    "/audit",
    dependencies=[Depends(require_permission(Permission.AUTH_ADMIN))],
)
async def get_audit_log(limit: int = Query(default=100, ge=1, le=500)):
    """Return recent Serve admin audit events."""
    if api_state.workspace_dir is None:
        raise HTTPException(status_code=409, detail="Workspace not configured")
    return {"events": [event.model_dump(mode="json") for event in list_audit_events(api_state.get_auth_db_path(create_parent=True), limit=limit)]}
