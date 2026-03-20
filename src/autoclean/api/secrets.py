"""Workspace-local secret reference store for Serve."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

SECRET_REF_PREFIX = "secret://"
KEYCHAIN_REF_PREFIX = "keychain://"
_KEYCHAIN_ACCOUNT = "autoclean-serve"


def _secret_store_path(config_path: Path) -> Path:
    return config_path.parent / ".serve" / "secret_store.json"


def _secret_backend() -> str:
    forced = os.environ.get("AUTOCLEAN_SECRET_BACKEND", "auto").strip().lower()
    if forced in {"file", "keychain"}:
        return forced
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return "file"
    if platform.system() == "Darwin" and shutil.which("security"):
        return "keychain"
    return "file"


def _keychain_service(config_path: Path, key: str) -> str:
    workspace = str(config_path.parent.resolve())
    return f"autoclean-serve:{workspace}:{key}"


def _store_secret_in_keychain(config_path: Path, key: str, value: str) -> str:
    subprocess.run(
        [
            "security",
            "add-generic-password",
            "-a",
            _KEYCHAIN_ACCOUNT,
            "-s",
            _keychain_service(config_path, key),
            "-w",
            value,
            "-U",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return f"{KEYCHAIN_REF_PREFIX}{key}"


def _resolve_secret_from_keychain(config_path: Path, key: str) -> str:
    result = subprocess.run(
        [
            "security",
            "find-generic-password",
            "-a",
            _KEYCHAIN_ACCOUNT,
            "-s",
            _keychain_service(config_path, key),
            "-w",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def is_secret_ref(value: str | None) -> bool:
    return bool(
        value
        and (
            value.startswith(SECRET_REF_PREFIX)
            or value.startswith(KEYCHAIN_REF_PREFIX)
        )
    )


def store_secret(config_path: Path, key: str, value: str) -> str:
    """Persist a secret value and return its stable reference."""
    if _secret_backend() == "keychain":
        try:
            return _store_secret_in_keychain(config_path, key, value)
        except Exception:
            pass
    store_path = _secret_store_path(config_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = json.loads(store_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        payload = {}
    except json.JSONDecodeError:
        payload = {}
    payload[key] = value
    store_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        os.chmod(store_path, 0o600)
    except OSError:
        pass
    return f"{SECRET_REF_PREFIX}{key}"


def resolve_secret(config_path: Path, ref_or_value: str) -> str:
    """Resolve a secret reference to its stored value."""
    if not is_secret_ref(ref_or_value):
        return ref_or_value
    if ref_or_value.startswith(KEYCHAIN_REF_PREFIX):
        key = ref_or_value[len(KEYCHAIN_REF_PREFIX):]
        try:
            return _resolve_secret_from_keychain(config_path, key)
        except Exception:
            return ""
    key = ref_or_value[len(SECRET_REF_PREFIX):]
    store_path = _secret_store_path(config_path)
    try:
        payload = json.loads(store_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ""
    except json.JSONDecodeError:
        return ""
    value = payload.get(key, "")
    return value if isinstance(value, str) else ""
