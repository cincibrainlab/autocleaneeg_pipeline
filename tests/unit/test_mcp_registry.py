from __future__ import annotations

from autoclean_mcp.registry import build_registry, get_registry_entry, registry_summary


def test_registry_covers_cli_inventory_and_has_summary_counts() -> None:
    entries = build_registry()
    summary = registry_summary()

    assert len(entries) == summary["count"]
    assert summary["count"] >= 100
    assert summary["mutating_count"] > 0
    assert summary["interactive_count"] > 0
    assert summary["long_running_count"] > 0


def test_registry_marks_known_command_behaviors() -> None:
    serve_up = get_registry_entry(["serve", "up"])
    route_delete = get_registry_entry(["serve", "route", "delete"])
    task_list = get_registry_entry(["task", "list"])
    mode_status = get_registry_entry(["serve", "mode", "status"])
    mode_test = get_registry_entry(["serve", "mode", "test"])
    share_status = get_registry_entry(["serve", "share", "status"])
    share_start = get_registry_entry(["serve", "share", "start"])
    queue_retry_failed = get_registry_entry(["serve", "queue", "retry-failed"])
    login = get_registry_entry(["login"])
    task_edit = get_registry_entry(["task", "edit"])
    workspace_cd = get_registry_entry(["workspace", "cd"])
    view = get_registry_entry(["view"])
    export_access_log = get_registry_entry(["export-access-log"])

    assert serve_up is not None
    assert serve_up.wrapper_kind == "managed_session"
    assert serve_up.execution_style == "long_running"

    assert route_delete is not None
    assert route_delete.mutating is True
    assert route_delete.destructive is True

    assert task_list is not None
    assert task_list.mutating is False
    assert task_list.family == "tasks"

    assert mode_status is not None
    assert mode_status.mutating is False

    assert mode_test is not None
    assert mode_test.mutating is True

    assert share_status is not None
    assert share_status.mutating is False

    assert share_start is not None
    assert share_start.mutating is True

    assert queue_retry_failed is not None
    assert queue_retry_failed.mutating is True

    assert login is not None
    assert login.mutating is True
    assert login.wrapper_kind == "compatibility_wrapper"

    assert task_edit is not None
    assert task_edit.mutating is True
    assert task_edit.wrapper_kind == "compatibility_wrapper"

    assert workspace_cd is not None
    assert workspace_cd.wrapper_kind == "compatibility_wrapper"
    assert workspace_cd.output_mode == "raw_compatible"

    assert view is not None
    assert view.wrapper_kind == "compatibility_wrapper"

    assert export_access_log is not None
    assert export_access_log.mutating is True
