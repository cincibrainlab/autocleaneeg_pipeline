"""Helpers for reading and updating task-file montage settings."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MontageBlock:
    """Located montage config block within a task file."""

    text: str
    start: int
    end: int


def locate_montage_block(text: str) -> MontageBlock | None:
    """Find the montage configuration block in Python task source."""

    match = re.search(r"[\"']montage[\"']\s*:\s*\{", text)
    if not match:
        return None

    brace_start = text.find("{", match.start())
    if brace_start == -1:
        return None

    depth = 0
    in_string: str | None = None
    escape = False

    for index in range(brace_start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == in_string:
                in_string = None
        else:
            if char in {'"', "'"}:
                in_string = char
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return MontageBlock(
                        text[match.start() : index + 1], match.start(), index + 1
                    )

    return None


def extract_montage_value(block_text: str) -> str | None:
    """Extract the montage value from a montage block."""

    value_match = re.search(
        r"([\"\']value[\"\']\s*:\s*)(?P<quote>[\"\'])(?P<val>.*?)(?P=quote)",
        block_text,
        re.DOTALL,
    )
    if value_match:
        return value_match.group("val")

    none_match = re.search(r"[\"\']value[\"\']\s*:\s*None", block_text)
    if none_match:
        return None

    return None


def replace_montage_value(block_text: str, new_value: str) -> str | None:
    """Return a montage block with only the value replaced."""

    string_match = re.search(
        r"([\"\']value[\"\']\s*)(:\s*)(?P<quote>[\"\'])(?P<val>.*?)(?P=quote)",
        block_text,
        re.DOTALL,
    )
    if string_match:
        prefix = block_text[: string_match.start()]
        suffix = block_text[string_match.end() :]
        quote = string_match.group("quote")
        replacement = (
            f"{string_match.group(1)}{string_match.group(2)}{quote}{new_value}{quote}"
        )
        return prefix + replacement + suffix

    none_match = re.search(r"([\"\']value[\"\']\s*:\s*)None", block_text)
    if none_match:
        prefix = block_text[: none_match.start()]
        suffix = block_text[none_match.end() :]
        replacement = f'{none_match.group(1)}"{new_value}"'
        return prefix + replacement + suffix

    return None


def read_task_montage(task_path: Path) -> str | None:
    """Read the configured montage value from a Python task file."""

    source = task_path.read_text(encoding="utf-8")
    block = locate_montage_block(source)
    if block is None:
        return None
    return extract_montage_value(block.text)


def update_task_montage_source(source: str, new_value: str) -> str:
    """Return task source with only config['montage']['value'] changed."""

    block = locate_montage_block(source)
    if block is None:
        raise ValueError("Task does not define an editable montage block")

    updated_block = replace_montage_value(block.text, new_value)
    if updated_block is None:
        raise ValueError("Task montage block does not define an editable value")

    return source[: block.start] + updated_block + source[block.end :]


def update_task_montage_file(task_path: Path, new_value: str) -> None:
    """Rewrite a task file, changing only its montage value."""

    source = task_path.read_text(encoding="utf-8")
    task_path.write_text(
        update_task_montage_source(source, new_value), encoding="utf-8"
    )


def replace_task_class_name(source: str, old_name: str, new_name: str) -> str:
    """Rename a task class declaration without touching unrelated identifiers."""

    pattern = re.compile(rf"(^\s*class\s+){re.escape(old_name)}(\s*\()", re.MULTILINE)
    updated, count = pattern.subn(rf"\1{new_name}\2", source, count=1)
    if count != 1:
        raise ValueError(f"Could not find class declaration for {old_name}")
    return updated
