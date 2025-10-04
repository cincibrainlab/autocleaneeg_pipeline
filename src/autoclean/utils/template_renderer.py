"""Utilities for rendering package-managed templates safely."""

from __future__ import annotations

import json
import keyword
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping

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


def render_reprocess_task_from_json(
    json_data: Dict[str, Any] | str | Path,
    class_name: str | None = None,
    output_path: Path | None = None,
    additional_context: Dict[str, Any] | None = None,
) -> str:
    """Render a reprocessing task file with manual override data from JSON.

    This function takes manual override data (bad channels and ICA components)
    from the review GUI and generates a complete task file that will reprocess
    the data from the beginning with those overrides applied.

    Parameters
    ----------
    json_data : dict, str, or Path
        Either a dictionary containing the override data, a JSON string, or a
        path to a JSON file. Expected structure:
        {
            "file_stem": "128_SteadyState_D3158",
            "fix_type": "both",
            "timestamp": "2025-10-03T09:01:56.152499",
            "modifications": {
                "bad_channels": {
                    "modified": ["E1", "E125", ...]
                },
                "rejected_ica": {
                    "modified": [0, 2, 6, ...]
                }
            }
        }
    class_name : str, optional
        Name for the generated task class. If None, uses "{file_stem}_Reprocess".
    output_path : Path, optional
        If provided, writes the rendered task to this file path.
    additional_context : dict, optional
        Additional template variables to override defaults (e.g., montage, filter settings).

    Returns
    -------
    str
        The rendered task file content.

    Raises
    ------
    RuntimeError
        If the template cannot be rendered or JSON data is invalid.
    ValueError
        If required fields are missing from the JSON data.

    Examples
    --------
    >>> json_file = Path("override_data.json")
    >>> task_code = render_reprocess_task_from_json(json_file)
    >>> # Or with direct dictionary
    >>> data = {"file_stem": "subject01", "modifications": {...}}
    >>> task_code = render_reprocess_task_from_json(data, output_path=Path("Reprocess.py"))
    """
    # Load JSON data if needed
    if isinstance(json_data, (str, Path)):
        json_path = Path(json_data)
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            # Try parsing as JSON string
            try:
                data = json.loads(str(json_data))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON data: {e}") from e
    else:
        data = json_data

    # Validate required fields
    if "file_stem" not in data:
        raise ValueError("JSON data must contain 'file_stem' field")
    if "modifications" not in data:
        raise ValueError("JSON data must contain 'modifications' field")

    # Extract override data
    modifications = data["modifications"]
    bad_channels = modifications.get("bad_channels", {}).get("modified", [])
    rejected_ica = modifications.get("rejected_ica", {}).get("modified", [])

    # Determine class name
    if class_name is None:
        base_name = data["file_stem"].replace("-", "_").replace(" ", "_")
        class_name = f"{base_name}_Reprocess"

    # Validate class name
    validate_python_identifier(class_name, label="class_name")

    # Build template context
    context = {
        "class_name": class_name,
        "original_file": data["file_stem"],
        "timestamp": data.get("timestamp", ""),
        "fix_type": data.get("fix_type", "both"),
        "bad_channels": bad_channels,
        "rejected_ica": rejected_ica,
    }

    # Merge additional context if provided
    if additional_context:
        context.update(additional_context)

    # Locate template
    template_name = "reprocess_with_overrides.jinja"
    # Assume template is in src/autoclean/templates/
    try:
        import autoclean
        package_dir = Path(autoclean.__file__).parent
        template_path = package_dir / "templates" / template_name
    except (ImportError, AttributeError):
        # Fallback: try relative to this module
        template_path = Path(__file__).parent.parent / "templates" / template_name

    if not template_path.exists():
        raise RuntimeError(f"Template not found: {template_path}")

    # Render template
    rendered = render_template(template_path, context)

    # Write to file if requested
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(rendered)

    return rendered
