"""Tests for reprocess override helpers."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from autoclean.utils.reprocess_overrides import (
    POST_EPOCH_ICA_FIX_TYPE,
    epoch_review_override_from_record,
    generate_reprocess_task_from_original,
)


def test_epoch_review_override_from_record_normalizes_values():
    result = epoch_review_override_from_record(
        {
            "bad_epochs_count": "2",
            "bad_epoch_indices": "1, 3",
            "bad_epoch_times": "0.500,1.500",
            "bad_epoch_events": "102,104",
        }
    )

    assert result == {
        "count": 2,
        "indices": [1, 3],
        "times": ["0.500", "1.500"],
        "events": ["102", "104"],
    }


@pytest.mark.parametrize(
    "epoch_method",
    [
        "create_regular_epochs",
        "create_eventid_epochs",
        "create_sl_epochs",
        "create_sl_randomized_epochs",
    ],
)
def test_generate_reprocess_task_injects_manual_epoch_drop(
    tmp_path: Path, epoch_method: str
):
    task_path = tmp_path / "ExampleTask.py"
    task_path.write_text(
        (
            "from autoclean.core.task import Task\n\n"
            "config = {}\n\n"
            "class ExampleTask(Task):\n"
            "    def run(self):\n"
            "        self.import_raw()\n"
            f"        self.{epoch_method}(export=True)\n"
            "        self.detect_outlier_epochs()\n"
        ),
        encoding="utf-8",
    )

    payload = {
        "file_stem": "subject01",
        "fix_type": "epoch",
        "timestamp": "2026-03-26T12:00:00",
        "modifications": {
            "epoch_review": {
                "count": 2,
                "indices": [1, 3],
                "times": ["0.500", "1.500"],
                "events": ["102", "104"],
            },
            "bad_channels": {
                "modified": [],
                "original": [],
                "added": [],
                "removed": [],
            },
            "rejected_ica": {
                "modified": [],
                "original": [],
                "added": [],
                "removed": [],
            },
        },
    }

    generated = generate_reprocess_task_from_original(
        task_path, payload, "ExampleTaskReprocess", "20260326_120000"
    )
    tree = ast.parse(generated)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ExampleTaskReprocess"
    )
    run_method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )

    run_calls = [
        stmt.value.func.attr
        for stmt in run_method.body
        if isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
    ]

    assert epoch_method in run_calls
    assert "drop_manual_bad_epochs" in run_calls
    assert run_calls.index("drop_manual_bad_epochs") > run_calls.index(epoch_method)
    assert "manual_bad_epoch_indices=[1, 3]" in generated


def test_generate_reprocess_task_replaces_existing_manual_bad_channels_keyword(
    tmp_path: Path,
):
    task_path = tmp_path / "ExampleTask.py"
    task_path.write_text(
        (
            "from autoclean.core.task import Task\n\n"
            "config = {}\n\n"
            "class ExampleTask(Task):\n"
            "    def run(self):\n"
            "        self.clean_bad_channels(manual_bad_channels=['OLD'])\n"
        ),
        encoding="utf-8",
    )

    payload = {
        "file_stem": "subject01",
        "fix_type": "channel",
        "timestamp": "2026-03-26T12:00:00",
        "modifications": {
            "epoch_review": {"count": 0, "indices": [], "times": [], "events": []},
            "bad_channels": {
                "modified": ["FP1", "FP2"],
                "original": [],
                "added": ["FP1", "FP2"],
                "removed": [],
            },
            "rejected_ica": {
                "modified": [],
                "original": [],
                "added": [],
                "removed": [],
            },
        },
    }

    generated = generate_reprocess_task_from_original(
        task_path, payload, "ExampleTaskReprocess", "20260326_120000"
    )

    assert generated.count("manual_bad_channels=") == 1
    assert "manual_bad_channels=['FP1', 'FP2']" in generated


def test_generate_reprocess_task_keeps_bad_channels_during_ica_reprocess(
    tmp_path: Path,
):
    task_path = tmp_path / "ExampleTask.py"
    task_path.write_text(
        (
            "from autoclean.core.task import Task\n\n"
            "config = {}\n\n"
            "class ExampleTask(Task):\n"
            "    def run(self):\n"
            "        self.clean_bad_channels()\n"
            "        self.classify_ica_components()\n"
        ),
        encoding="utf-8",
    )

    payload = {
        "file_stem": "subject01",
        "fix_type": "ica",
        "timestamp": "2026-03-26T12:00:00",
        "modifications": {
            "epoch_review": {"count": 0, "indices": [], "times": [], "events": []},
            "bad_channels": {
                "modified": ["FP1"],
                "original": ["FP1"],
                "added": [],
                "removed": [],
            },
            "rejected_ica": {
                "modified": [1],
                "original": [],
                "added": [1],
                "removed": [],
            },
        },
    }

    generated = generate_reprocess_task_from_original(
        task_path, payload, "ExampleTaskReprocess", "20260326_120000"
    )

    assert "manual_bad_channels=['FP1']" in generated
    assert "classify_ica_components(reject=False)" in generated
    assert "apply_ica_component_rejection(manual_rejected_components=[1])" in generated


def test_generate_reprocess_task_replaces_stale_manual_ica_rejection_calls(
    tmp_path: Path,
):
    task_path = tmp_path / "ExampleTask.py"
    task_path.write_text(
        (
            "from autoclean.core.task import Task\n\n"
            "config = {}\n\n"
            "class ExampleTask(Task):\n"
            "    def run(self):\n"
            "        self.classify_ica_components()\n"
            "        self.apply_ica_component_rejection(manual_rejected_components=[6, 9])\n"
            "        self.apply_ica_component_rejection(manual_rejected_components=[5, 10, 16])\n"
        ),
        encoding="utf-8",
    )

    payload = {
        "file_stem": "subject01",
        "fix_type": "ica",
        "timestamp": "2026-03-26T12:00:00",
        "modifications": {
            "epoch_review": {"count": 0, "indices": [], "times": [], "events": []},
            "bad_channels": {
                "modified": [],
                "original": [],
                "added": [],
                "removed": [],
            },
            "rejected_ica": {
                "modified": [3, 9, 13, 17],
                "original": [],
                "added": [3, 9, 13, 17],
                "removed": [],
            },
        },
    }

    generated = generate_reprocess_task_from_original(
        task_path, payload, "ExampleTaskReprocess", "20260326_120000"
    )

    assert generated.count("apply_ica_component_rejection(") == 1
    assert "manual_rejected_components=[3, 9, 13, 17]" in generated
    assert "manual_rejected_components=[6, 9]" not in generated
    assert "manual_rejected_components=[5, 10, 16]" not in generated


def test_generate_post_epoch_ica_task_fits_ica_after_manual_epoch_drop(
    tmp_path: Path,
):
    task_path = tmp_path / "ExampleTask.py"
    task_path.write_text(
        (
            "from autoclean.core.task import Task\n\n"
            "config = {}\n\n"
            "class ExampleTask(Task):\n"
            "    def run(self):\n"
            "        self.import_raw()\n"
            "        self.run_ica()\n"
            "        self.classify_ica_components(method='iclabel')\n"
            "        self.create_regular_epochs(export=True)\n"
            "        self.detect_outlier_epochs()\n"
        ),
        encoding="utf-8",
    )

    payload = {
        "file_stem": "subject01",
        "fix_type": POST_EPOCH_ICA_FIX_TYPE,
        "timestamp": "2026-03-26T12:00:00",
        "modifications": {
            "epoch_review": {
                "count": 2,
                "indices": [1, 3],
                "times": ["0.500", "1.500"],
                "events": ["102", "104"],
            },
            "bad_channels": {
                "modified": [],
                "original": [],
                "added": [],
                "removed": [],
            },
            "rejected_ica": {
                "modified": [],
                "original": [],
                "added": [],
                "removed": [],
            },
        },
    }

    generated = generate_reprocess_task_from_original(
        task_path, payload, "ExampleTaskReprocess", "20260326_120000"
    )
    tree = ast.parse(generated)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ExampleTaskReprocess"
    )
    run_method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    run_calls = [
        stmt.value.func.attr
        for stmt in run_method.body
        if isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
    ]

    assert run_calls.index("create_regular_epochs") < run_calls.index("run_ica")
    assert run_calls.index("drop_manual_bad_epochs") < run_calls.index("run_ica")
    assert run_calls.index("run_ica") < run_calls.index("classify_ica_components")
    assert (
        "run_ica(use_epochs=True, stage_name='post_epoch_rejection_ica_fit')"
        in generated
    )
    assert (
        "classify_ica_components(method='iclabel', reject=False, "
        "stage_name='post_epoch_rejection_ica_labeling')" in generated
    )
    assert "Post-epoch-rejection ICA: True" in generated


def test_generate_post_epoch_ica_task_adds_classifier_after_ica_when_missing(
    tmp_path: Path,
):
    task_path = tmp_path / "ExampleTask.py"
    task_path.write_text(
        (
            "from autoclean.core.task import Task\n\n"
            "config = {}\n\n"
            "class ExampleTask(Task):\n"
            "    def run(self):\n"
            "        self.import_raw()\n"
            "        self.run_ica()\n"
            "        self.create_regular_epochs(export=True)\n"
            "        self.generate_reports()\n"
        ),
        encoding="utf-8",
    )

    payload = {
        "file_stem": "subject01",
        "fix_type": POST_EPOCH_ICA_FIX_TYPE,
        "timestamp": "2026-03-26T12:00:00",
        "modifications": {
            "epoch_review": {"count": 1, "indices": [1], "times": [], "events": []},
            "bad_channels": {
                "modified": [],
                "original": [],
                "added": [],
                "removed": [],
            },
            "rejected_ica": {
                "modified": [],
                "original": [],
                "added": [],
                "removed": [],
            },
        },
    }

    generated = generate_reprocess_task_from_original(
        task_path, payload, "ExampleTaskReprocess", "20260326_120000"
    )
    tree = ast.parse(generated)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ExampleTaskReprocess"
    )
    run_method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    run_calls = [
        stmt.value.func.attr
        for stmt in run_method.body
        if isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
    ]

    assert run_calls.index("drop_manual_bad_epochs") < run_calls.index("run_ica")
    assert run_calls.index("run_ica") < run_calls.index("classify_ica_components")
    assert run_calls.index("classify_ica_components") < run_calls.index(
        "generate_reports"
    )
    assert (
        "classify_ica_components(method='iclabel', reject=False, "
        "stage_name='post_epoch_rejection_ica_labeling')" in generated
    )


def test_generate_post_epoch_ica_task_forces_iclabel_when_classifier_uses_config(
    tmp_path: Path,
):
    task_path = tmp_path / "ExampleTask.py"
    task_path.write_text(
        (
            "from autoclean.core.task import Task\n\n"
            "config = {}\n\n"
            "class ExampleTask(Task):\n"
            "    def run(self):\n"
            "        self.import_raw()\n"
            "        self.run_ica()\n"
            "        self.classify_ica_components()\n"
            "        self.create_regular_epochs(export=True)\n"
        ),
        encoding="utf-8",
    )

    payload = {
        "file_stem": "subject01",
        "fix_type": POST_EPOCH_ICA_FIX_TYPE,
        "timestamp": "2026-03-26T12:00:00",
        "modifications": {
            "epoch_review": {"count": 1, "indices": [1], "times": [], "events": []},
            "bad_channels": {
                "modified": [],
                "original": [],
                "added": [],
                "removed": [],
            },
            "rejected_ica": {
                "modified": [],
                "original": [],
                "added": [],
                "removed": [],
            },
        },
    }

    generated = generate_reprocess_task_from_original(
        task_path, payload, "ExampleTaskReprocess", "20260326_120000"
    )

    assert (
        "classify_ica_components(method='iclabel', reject=False, "
        "stage_name='post_epoch_rejection_ica_labeling')" in generated
    )


def test_generate_post_epoch_ica_task_replaces_positional_classifier_method(
    tmp_path: Path,
):
    task_path = tmp_path / "ExampleTask.py"
    task_path.write_text(
        (
            "from autoclean.core.task import Task\n\n"
            "config = {}\n\n"
            "class ExampleTask(Task):\n"
            "    def run(self):\n"
            "        self.import_raw()\n"
            "        self.run_ica()\n"
            "        self.classify_ica_components('icvision')\n"
            "        self.create_regular_epochs(export=True)\n"
        ),
        encoding="utf-8",
    )

    payload = {
        "file_stem": "subject01",
        "fix_type": POST_EPOCH_ICA_FIX_TYPE,
        "timestamp": "2026-03-26T12:00:00",
        "modifications": {
            "epoch_review": {"count": 1, "indices": [1], "times": [], "events": []},
            "bad_channels": {
                "modified": [],
                "original": [],
                "added": [],
                "removed": [],
            },
            "rejected_ica": {
                "modified": [],
                "original": [],
                "added": [],
                "removed": [],
            },
        },
    }

    generated = generate_reprocess_task_from_original(
        task_path, payload, "ExampleTaskReprocess", "20260326_120000"
    )

    assert (
        "classify_ica_components('iclabel', reject=False, "
        "stage_name='post_epoch_rejection_ica_labeling')" in generated
    )
    assert "method='iclabel'" not in generated
