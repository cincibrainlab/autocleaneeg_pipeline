"""Tests for task configuration validation."""

import pytest
from schema import SchemaError

from autoclean.task_config_schema import CONFIG_VERSION, validate_task_config


def test_validate_task_config_success():
    config = {
        "version": CONFIG_VERSION,
        "move_flagged_files": True,
        "resample_step": {"enabled": True, "value": 250},
        "filtering": {"enabled": False, "value": {}},
    }
    assert validate_task_config(config) == config


def test_validate_task_config_version_mismatch():
    config = {"version": "0.0", "resample_step": {"enabled": True, "value": 250}}
    with pytest.raises(ValueError):
        validate_task_config(config)


def test_validate_task_config_schema_error():
    config = {"version": CONFIG_VERSION, "resample_step": {"enabled": "yes", "value": 250}}
    with pytest.raises(SchemaError):
        validate_task_config(config)
