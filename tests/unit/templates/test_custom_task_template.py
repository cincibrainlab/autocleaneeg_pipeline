from pathlib import Path

import pytest

from autoclean.configkit.schema import validate_task_module_config
from autoclean.templates.custom_task_template import config as custom_template_config
from autoclean.utils.template_renderer import (
    render_template,
    validate_python_identifier,
)


def test_custom_task_template_matches_python() -> None:
    templates_dir = (
        Path(__file__).resolve().parents[3] / "src" / "autoclean" / "templates"
    )
    jinja_template = templates_dir / "custom_task_template.jinja"
    canonical_python = templates_dir / "custom_task_template.py"

    rendered = render_template(jinja_template, {"class_name": "CustomTask"})

    assert canonical_python.read_text(encoding="utf-8") == rendered


def test_custom_task_template_config_matches_schema() -> None:
    validated = validate_task_module_config(custom_template_config)

    assert "input_path" not in validated
    assert validated["bad_channel_detection"]["value"]["preset"] == "auto"
    assert (
        validated["bad_channel_detection"]["value"]["cleaning_method"] == "interpolate"
    )
    assert "channel_count_bins" in validated["bad_channel_detection"]["value"]
    assert validated["schema_version"] == custom_template_config["schema_version"]


def test_render_template_missing_file(tmp_path: Path) -> None:
    missing_template = tmp_path / "does_not_exist.jinja"

    with pytest.raises(RuntimeError, match="Template not found"):
        render_template(missing_template, {})


def test_render_template_invalid_syntax(tmp_path: Path) -> None:
    bad_template = tmp_path / "bad_template.jinja"
    bad_template.write_text("{% for %}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Failed to load template"):
        render_template(bad_template, {})


def test_validate_python_identifier_rejects_invalid_names() -> None:
    with pytest.raises(ValueError, match="valid Python identifier"):
        validate_python_identifier("123Bad", label="Task class name")


def test_validate_python_identifier_accepts_valid_name() -> None:
    # Should not raise
    validate_python_identifier("CustomTask", label="Task class name")
