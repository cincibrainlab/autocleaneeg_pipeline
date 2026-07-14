from __future__ import annotations

import sys
import types

import pytest
from schema import SchemaError

from autoclean.configkit.schema import (
    SCHEMA_VERSION,
    _schema_error_path,
    format_task_config_error,
    validate_task_module_config,
)
from autoclean.core.task import Task


def _minimal_config() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "montage": {"enabled": True, "value": None},
        "resample_step": {"enabled": False, "value": None},
        "filtering": {
            "enabled": False,
            "value": {
                "l_freq": None,
                "h_freq": None,
                "notch_freqs": None,
                "notch_widths": None,
            },
        },
        "drop_outerlayer": {"enabled": False, "value": None},
        "eog_step": {"enabled": False, "value": None},
        "trim_step": {"enabled": False, "value": 0},
        "crop_step": {"enabled": False, "value": {"start": 0, "end": None}},
        "reference_step": {"enabled": False, "value": None},
        "ICA": {"enabled": False, "value": {"method": "fastica"}},
        "component_rejection": {
            "enabled": False,
            "method": "iclabel",
            "value": {
                "ic_flags_to_reject": [],
                "ic_rejection_threshold": 0,
            },
        },
        "epoch_settings": {
            "enabled": False,
            "value": {"tmin": None, "tmax": None},
            "event_id": None,
            "remove_baseline": {"enabled": False, "window": None},
            "threshold_rejection": {"enabled": False, "volt_threshold": {}},
        },
    }


def test_schema_accepts_prior_preprocessing_detection() -> None:
    config = _minimal_config()
    config["prior_preprocessing_detection"] = {
        "enabled": True,
        "strict": False,
    }

    validated = validate_task_module_config(config)

    assert validated["prior_preprocessing_detection"]["enabled"] is True


def _formatted_error(config: dict, *, debug: bool | None = None) -> str:
    with pytest.raises(Exception) as exc_info:
        validate_task_module_config(config)
    return format_task_config_error(
        exc_info.value,
        config,
        task_name="BadTask",
        task_file="bad_task.py",
        debug=debug,
    )


def test_format_task_config_error_shows_bad_montage_path_received_and_fix() -> None:
    config = _minimal_config()
    config["montage"] = {"enabled": True, "value": "not_a_montage"}

    message = _formatted_error(config)

    assert "Task config validation failed for BadTask" in message
    assert "Task file: bad_task.py" in message
    assert "Config path: config['montage']['value']" in message
    assert "Received: 'not_a_montage' (str)" in message
    assert "Expected: string (valid montage) | 'auto' | None" in message
    assert "Use a supported montage name" in message
    assert "Raw schema error" not in message


def test_format_task_config_error_prioritizes_enabled_suggestion_for_montage_enabled() -> (
    None
):
    config = _minimal_config()
    config["montage"]["enabled"] = "yes"

    message = _formatted_error(config)

    assert "Config path: config['montage']['enabled']" in message
    assert "Set `enabled` to true or false." in message
    assert "Use a supported montage name" not in message


def test_format_task_config_error_shows_nested_filter_value() -> None:
    config = _minimal_config()
    config["filtering"]["value"]["l_freq"] = "abc"

    message = _formatted_error(config)

    assert "Config path: config['filtering']['value']['l_freq']" in message
    assert "Received: 'abc' (str)" in message
    assert "Expected: number|null" in message
    assert "Use numeric filter frequencies" in message


def test_format_task_config_error_shows_missing_key() -> None:
    config = _minimal_config()
    del config["ICA"]

    message = _formatted_error(config)

    assert "Config path: config['ICA']" in message
    assert "Received: <missing>" in message
    assert "Add the missing required key" in message


def test_format_task_config_error_shows_wrong_key_without_dumping_config() -> None:
    config = _minimal_config()
    config["extra_key"] = 1

    message = _formatted_error(config)

    assert "Config path: config['extra_key']" in message
    assert "Received: 1 (int)" in message
    assert "Remove the extra key" in message
    assert "'schema_version':" not in message


def test_format_task_config_error_distinguishes_literal_missing_string() -> None:
    config = _minimal_config()
    config["extra_key"] = "<missing>"

    message = _formatted_error(config)

    assert "Config path: config['extra_key']" in message
    assert "Received: '<missing>' (str)" in message


def test_format_task_config_error_includes_raw_error_when_debug_enabled() -> None:
    config = _minimal_config()
    config["epoch_settings"]["value"]["tmin"] = "early"

    message = _formatted_error(config, debug=True)

    assert "Config path: config['epoch_settings']['value']['tmin']" in message
    assert "Raw schema error:" in message
    assert "'early' should be instance" in message


def test_format_task_config_error_includes_raw_error_when_env_debug_enabled(
    monkeypatch,
) -> None:
    config = _minimal_config()
    config["epoch_settings"]["value"]["tmin"] = "early"
    monkeypatch.setenv("AUTOCLEAN_CONFIG_DEBUG", "1")

    message = _formatted_error(config)

    assert "Raw schema error:" in message
    assert "'early' should be instance" in message


def test_format_task_config_error_handles_non_schema_exception() -> None:
    message = format_task_config_error(
        RuntimeError("plain failure"),
        {"schema_version": SCHEMA_VERSION},
        task_name="PlainTask",
    )

    assert "Task config validation failed for PlainTask" in message
    assert "Config path: config" in message
    assert "Received: {'schema_version':" in message
    assert "Raw schema error" not in message


def test_schema_error_path_parses_nested_autos() -> None:
    error = SchemaError(
        ["Key 'montage' error:", "Key 'enabled' error:"],
        None,
    )

    assert _schema_error_path(error) == ["montage", "enabled"]


def test_format_task_config_error_extracts_path_from_real_schema_error() -> None:
    config = _minimal_config()
    config["crop_step"]["value"]["start"] = "beginning"

    message = _formatted_error(config)

    assert "Config path: config['crop_step']['value']['start']" in message
    assert "Received: 'beginning' (str)" in message


def test_schema_error_path_extracts_path_from_real_schema_error() -> None:
    config = _minimal_config()
    config["crop_step"]["value"]["start"] = "beginning"

    with pytest.raises(SchemaError) as exc_info:
        validate_task_module_config(config)

    assert _schema_error_path(exc_info.value) == ["crop_step", "value", "start"]


def test_task_init_formats_legacy_iclabel_errors_against_migrated_config() -> None:
    module = types.ModuleType("autoclean_test_legacy_bad_task_module")
    module.__file__ = "legacy_bad_task_file.py"
    module.config = _minimal_config()
    del module.config["component_rejection"]
    module.config["ICLabel"] = {
        "enabled": True,
        "value": {"ic_flags_to_reject": "brain"},
    }
    sys.modules[module.__name__] = module
    try:

        class LegacyBadTask(Task):
            __module__ = module.__name__

            def run(self) -> None:
                return None

        setattr(module, "LegacyBadTask", LegacyBadTask)

        with pytest.raises(ValueError) as exc_info:
            LegacyBadTask({})

        message = str(exc_info.value)
        assert "Task config validation failed for LegacyBadTask" in message
        assert (
            "Config path: config['component_rejection']['value']['ic_flags_to_reject']"
            in message
        )
        assert "Received: 'brain' (str)" in message
        assert "Received: <missing>" not in message
    finally:
        sys.modules.pop(module.__name__, None)


def test_task_init_wraps_config_errors_with_actionable_message() -> None:
    module = types.ModuleType("autoclean_test_bad_task_module")
    module.__file__ = "bad_task_file.py"
    module.config = _minimal_config()
    module.config["montage"] = {"enabled": True, "value": "invalid"}
    sys.modules[module.__name__] = module
    try:

        class BadTask(Task):
            __module__ = module.__name__

            def run(self) -> None:
                return None

        setattr(module, "BadTask", BadTask)

        with pytest.raises(ValueError) as exc_info:
            BadTask({})

        message = str(exc_info.value)
        assert "Task config validation failed for BadTask" in message
        assert "Task file: bad_task_file.py" in message
        assert "Config path: config['montage']['value']" in message
        assert "Raw schema error" not in message
    finally:
        sys.modules.pop(module.__name__, None)
