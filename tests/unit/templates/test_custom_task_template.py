from pathlib import Path

from autoclean.utils.template_renderer import render_template


def test_custom_task_template_matches_python() -> None:
    templates_dir = Path(__file__).resolve().parents[3] / "src" / "autoclean" / "templates"
    jinja_template = templates_dir / "custom_task_template.jinja"
    canonical_python = templates_dir / "custom_task_template.py"

    rendered = render_template(jinja_template, {"class_name": "CustomTask"})

    assert canonical_python.read_text(encoding="utf-8") == rendered
