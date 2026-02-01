"""Configuration API routes."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException

from autoclean.api.models import (
    ConfigResponse,
    DeployResponse,
    RouteInfo,
    ValidateResponse,
)
from autoclean.api.state import api_state

router = APIRouter()


def _load_config():
    """Load and parse the serve configuration."""
    from autoclean.utils.ingestion import (
        ServeConfigError,
        load_serve_config,
        parse_serve_config,
    )

    config_path = api_state.get_config_path(deployed=False)

    if not config_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Config file not found: {config_path}"
        )

    raw_config = load_serve_config(config_path)
    return raw_config, config_path


@router.get("", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """Get current configuration."""
    from autoclean.utils.ingestion import ServeConfigError, parse_serve_config

    raw_config, config_path = _load_config()

    errors = []
    warnings = []
    routes = []
    valid = False

    try:
        config, warnings = parse_serve_config(
            raw_config, api_state.workspace_dir, strict=False
        )
        valid = True

        for route in config.routes:
            routes.append(
                RouteInfo(
                    id=route.id,
                    enabled=route.enabled,
                    priority=route.priority,
                    taskfile=route.taskfile,
                    montage=route.montage,
                    version=route.version,
                    ingestion_folders=[str(f) for f in route.ingestion_folders],
                    file_globs=route.file_globs,
                    recursive=route.recursive,
                    sentinel_ext=route.sentinel_ext,
                )
            )

    except ServeConfigError as exc:
        errors = list(exc.errors)
        warnings = list(exc.warnings)
    except Exception as exc:
        errors = [str(exc)]

    return ConfigResponse(
        mode=api_state.mode,
        workspace_dir=str(api_state.workspace_dir),
        runtime_path=str(raw_config.get("runtime", "")),
        routes=routes,
        valid=valid,
        errors=errors,
        warnings=list(warnings),
    )


@router.get("/yaml")
async def get_config_yaml() -> dict[str, str]:
    """Get raw YAML configuration content."""
    config_path = api_state.get_config_path(deployed=False)

    if not config_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Config file not found: {config_path}"
        )

    content = config_path.read_text(encoding="utf-8")
    return {"content": content, "path": str(config_path)}


@router.post("/validate", response_model=ValidateResponse)
async def validate_config() -> ValidateResponse:
    """Validate the current configuration."""
    from autoclean.utils.ingestion import ServeConfigError, parse_serve_config

    raw_config, _ = _load_config()

    try:
        _, warnings = parse_serve_config(
            raw_config, api_state.workspace_dir, strict=True
        )
        return ValidateResponse(valid=True, errors=[], warnings=list(warnings))
    except ServeConfigError as exc:
        return ValidateResponse(
            valid=False, errors=list(exc.errors), warnings=list(exc.warnings)
        )
    except Exception as exc:
        return ValidateResponse(valid=False, errors=[str(exc)], warnings=[])


@router.post("/deploy", response_model=DeployResponse)
async def deploy_config() -> DeployResponse:
    """Deploy configuration from operator to deployed directory."""
    from autoclean.utils.ingestion import ServeConfigError, parse_serve_config

    source_path = api_state.get_config_path(deployed=False)
    target_path = api_state.get_config_path(deployed=True)

    if not source_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Source config not found: {source_path}"
        )

    # Validate before deploying
    raw_config, _ = _load_config()

    try:
        parse_serve_config(raw_config, api_state.workspace_dir, strict=True)
    except ServeConfigError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot deploy invalid config: {'; '.join(exc.errors)}",
        )

    # Ensure deploy directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy config
    try:
        shutil.copy2(source_path, target_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Deploy failed: {exc}")

    return DeployResponse(
        success=True,
        source=str(source_path),
        target=str(target_path),
        message=f"Configuration deployed to {target_path.name}",
    )


@router.get("/routes", response_model=list[RouteInfo])
async def get_routes() -> list[RouteInfo]:
    """Get list of configured routes."""
    from autoclean.utils.ingestion import parse_serve_config

    raw_config, _ = _load_config()

    try:
        config, _ = parse_serve_config(
            raw_config, api_state.workspace_dir, strict=False
        )

        routes = []
        for route in config.routes:
            routes.append(
                RouteInfo(
                    id=route.id,
                    enabled=route.enabled,
                    priority=route.priority,
                    taskfile=route.taskfile,
                    montage=route.montage,
                    version=route.version,
                    ingestion_folders=[str(f) for f in route.ingestion_folders],
                    file_globs=route.file_globs,
                    recursive=route.recursive,
                    sentinel_ext=route.sentinel_ext,
                )
            )

        return routes

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/routes/{route_id}", response_model=RouteInfo)
async def get_route(route_id: str) -> RouteInfo:
    """Get a specific route by ID."""
    routes = await get_routes()

    for route in routes:
        if route.id == route_id:
            return route

    raise HTTPException(status_code=404, detail=f"Route not found: {route_id}")
