"""Utilities for rendering package-managed templates safely."""

from __future__ import annotations

import keyword
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateError


@lru_cache(maxsize=None)
def _environment_for(directory: Path) -> Environment:
    """Return a memoized Jinja environment scoped to *directory*."""

    loader = FileSystemLoader(str(directory))
    return Environment(
        loader=loader,
        autoescape=False,
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )


def render_template(template_path: Path, context: Mapping[str, Any]) -> str:
    """Render *template_path* with *context* using Jinja2.

    Parameters
    ----------
    template_path:
        Absolute or relative path to the template file.
    context:
        Mapping of template variables.

    Returns
    -------
    str
        Rendered template content.

    Raises
    ------
    RuntimeError
        If the template cannot be located or rendered.
    """

    template_path = template_path.resolve()
    if not template_path.is_file():
        raise RuntimeError(f"Template not found: {template_path}")

    environment = _environment_for(template_path.parent)
    try:
        template = environment.get_template(template_path.name)
    except TemplateError as exc:  # pragma: no cover - exercised in rendering path
        raise RuntimeError(f"Failed to load template {template_path}: {exc}") from exc

    try:
        return template.render(**context)
    except TemplateError as exc:
        raise RuntimeError(f"Failed to render template {template_path}: {exc}") from exc


def validate_python_identifier(value: str, *, label: str = "value") -> None:
    """Ensure *value* is a valid, non-keyword Python identifier.

    Raises
    ------
    ValueError
        If the provided value is empty, not identifier-compatible, or a keyword.
    """

    if not value or not value.isidentifier() or keyword.iskeyword(value):
        raise ValueError(f"{label} must be a valid Python identifier and not a keyword.")
