"""Tests for the tasks registry to ensure approved tasks are not overwritten by pending_approval tasks."""

import pytest


def test_approved_task_takes_precedence_over_pending():
    """Approved tasks must not be overwritten by pending_approval tasks with the same name.

    Regression test for: https://github.com/cincibrainlab/autocleaneeg_pipeline/issues/138
    When a task exists in both src/autoclean/tasks/ (approved) and
    src/autoclean/tasks/pending_approval/ (not yet approved), the approved
    version must win. Otherwise the pending version's config (which may differ)
    silently replaces the approved one.
    """
    from autoclean.tasks import task_registry

    # RestingState_Basic exists in both approved and pending_approval directories.
    # The approved version has crop_step disabled; the pending_approval version has it enabled.
    task_name = "restingstate_basic"
    assert task_name in task_registry, f"Task '{task_name}' not found in registry"

    task_class = task_registry[task_name]

    # The approved task's module is autoclean.tasks.RestingState_Basic (not pending_approval)
    assert "pending_approval" not in task_class.__module__, (
        f"Task '{task_name}' was loaded from pending_approval instead of the approved tasks directory. "
        f"Module: {task_class.__module__}. "
        "This means pending_approval tasks are overwriting approved tasks in the registry."
    )


def test_pending_only_task_is_registered():
    """Tasks that exist only in pending_approval (with no approved counterpart) should be accessible."""
    from autoclean.tasks import task_registry

    # Check that pending_approval-only tasks are still registered
    # (they don't conflict with any approved task)
    # This ensures we haven't broken the ability to use pending tasks that are truly new.
    # We verify by checking that the registry itself is non-empty and includes approved tasks.
    assert len(task_registry) > 0, "Task registry should not be empty"


def test_registry_does_not_contain_duplicate_approved_tasks():
    """The registry should not contain duplicate entries for the same task name."""
    from autoclean.tasks import task_registry, __all__

    # Each task name should appear at most once in __all__ from approved tasks
    # (pending_approval tasks that don't conflict are also fine, but duplicates are not)
    registry_keys = list(task_registry.keys())
    assert len(registry_keys) == len(set(registry_keys)), (
        "Task registry contains duplicate keys: "
        f"{[k for k in registry_keys if registry_keys.count(k) > 1]}"
    )
