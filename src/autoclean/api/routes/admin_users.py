"""Admin endpoints for Serve user and role management."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from autoclean.api.audit import log_audit_event
from autoclean.api.auth.dependencies import require_permission
from autoclean.api.auth.models import Permission, Role
from autoclean.api.auth.store import get_user_by_id, list_users, set_user_roles
from autoclean.api.state import api_state

router = APIRouter()


class RoleUpdateRequest(BaseModel):
    roles: list[Role] = Field(default_factory=list)


def _require_workspace_db():
    if api_state.workspace_dir is None:
        raise HTTPException(status_code=409, detail="Workspace not configured")
    return api_state.get_auth_db_path(create_parent=True)


def _admin_count(db_path) -> int:
    return sum(1 for user in list_users(db_path) if Role.ADMIN in user.roles)


def _ensure_admin_not_orphaned(db_path, *, user_id: str, next_roles: list[Role]) -> None:
    user = get_user_by_id(db_path, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")
    currently_admin = Role.ADMIN in user.roles
    remains_admin = Role.ADMIN in next_roles
    if currently_admin and not remains_admin and _admin_count(db_path) <= 1:
        raise HTTPException(status_code=400, detail="Cannot remove the last admin from this workspace")


@router.get(
    "/users",
    dependencies=[Depends(require_permission(Permission.USERS_ADMIN))],
)
async def get_users():
    """List Serve users with assigned roles."""
    return {
        "users": [
            user.model_dump(mode="json")
            for user in list_users(_require_workspace_db())
        ]
    }


@router.post(
    "/users/{user_id}/roles",
    dependencies=[Depends(require_permission(Permission.USERS_ADMIN))],
)
async def set_roles(user_id: str, body: RoleUpdateRequest, request: Request):
    """Replace all roles for a given user."""
    db_path = _require_workspace_db()
    _ensure_admin_not_orphaned(db_path, user_id=user_id, next_roles=body.roles)
    set_user_roles(
        db_path,
        user_id,
        body.roles,
        granted_by="admin",
        granted_at=datetime.now(timezone.utc).isoformat(),
    )
    admin_user = getattr(request.state, "auth_user", None)
    log_audit_event(
        db_path,
        action="set_roles",
        resource_type="user_role",
        resource_id=user_id,
        actor_user_id=getattr(admin_user, "id", None),
        actor_login=getattr(admin_user, "login", None),
        details_json=json.dumps({"roles": [role.value for role in body.roles]}),
        ip_address=request.client.host if request.client else None,
    )
    return {"success": True, "user_id": user_id, "roles": [role.value for role in body.roles]}


@router.delete(
    "/users/{user_id}/roles/{role}",
    dependencies=[Depends(require_permission(Permission.USERS_ADMIN))],
)
async def remove_role(user_id: str, role: Role, request: Request):
    """Remove a single role from a user."""
    db_path = _require_workspace_db()
    user = get_user_by_id(db_path, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")
    remaining = [existing_role for existing_role in user.roles if existing_role != role]
    _ensure_admin_not_orphaned(db_path, user_id=user_id, next_roles=remaining)
    set_user_roles(
        db_path,
        user_id,
        remaining,
        granted_by="admin",
        granted_at=datetime.now(timezone.utc).isoformat(),
    )
    admin_user = getattr(request.state, "auth_user", None)
    log_audit_event(
        db_path,
        action="remove_role",
        resource_type="user_role",
        resource_id=user_id,
        actor_user_id=getattr(admin_user, "id", None),
        actor_login=getattr(admin_user, "login", None),
        details_json=json.dumps({"removed_role": role.value, "remaining_roles": [r.value for r in remaining]}),
        ip_address=request.client.host if request.client else None,
    )
    return {"success": True, "user_id": user_id, "roles": [r.value for r in remaining]}
