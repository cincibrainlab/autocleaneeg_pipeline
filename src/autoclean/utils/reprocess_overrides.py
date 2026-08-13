"""Helpers for reprocess task generation with manual overrides."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Optional

EOG_CHANNEL_PATTERNS = {"EOG", "HEOG", "VEOG", "hEOG", "vEOG", "REOG", "LEOG"}
EPOCH_CREATION_METHODS = {
    "create_regular_epochs",
    "create_eventid_epochs",
    "create_sl_epochs",
    "create_sl_randomized_epochs",
}
POST_EPOCH_ICA_FIX_TYPE = "post_epoch_ica"


def epoch_review_override_from_record(
    record: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Normalize manual epoch-review data from an exclusion decision record."""
    record = record or {}
    return {
        "count": int(record.get("bad_epochs_count", 0) or 0),
        "indices": [
            int(v)
            for v in str(record.get("bad_epoch_indices", "")).split(",")
            if v.strip()
        ],
        "times": [
            v for v in str(record.get("bad_epoch_times", "")).split(",") if v.strip()
        ],
        "events": [
            v for v in str(record.get("bad_epoch_events", "")).split(",") if v.strip()
        ],
    }


def generate_reprocess_task_from_original(
    original_task_path: Path,
    payload: dict[str, Any],
    new_class_name: str,
    timestamp: str,
) -> str:
    """Generate a reprocess task by modifying the original task AST."""
    original_source = original_task_path.read_text(encoding="utf-8")
    tree = ast.parse(original_source)

    fix_type = payload.get("fix_type", "both")
    modifications = payload.get("modifications", {})
    bad_channels_raw = modifications.get("bad_channels", {}).get("modified", [])
    rejected_ica = modifications.get("rejected_ica", {}).get("modified", [])
    epoch_review = modifications.get("epoch_review", {})
    manual_bad_epoch_indices = [
        int(idx) for idx in epoch_review.get("indices", []) if str(idx).strip()
    ]
    manual_bad_epoch_times = [str(v) for v in epoch_review.get("times", []) if v]
    manual_bad_epoch_events = [str(v) for v in epoch_review.get("events", []) if v]
    file_stem = payload.get("file_stem", "unknown")
    dataset_name = f"{file_stem}_{timestamp}"

    bad_channels = [ch for ch in bad_channels_raw if ch not in EOG_CHANNEL_PATTERNS]

    class ConfigModifier(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign):  # type: ignore[override]
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "config"
                and isinstance(node.value, ast.Dict)
            ):
                node.value.keys.append(ast.Constant(value="dataset_name"))
                node.value.values.append(ast.Constant(value=dataset_name))
            return node

    class ClassRenamer(ast.NodeTransformer):
        def visit_ClassDef(self, node: ast.ClassDef):  # type: ignore[override]
            node.name = new_class_name

            doc_lines = [
                "Reprocessing task with manual overrides.",
                "",
                "This task reprocesses the original raw data from the beginning with:",
                f"- Manual bad channel list: {bad_channels}",
                f"- Manual ICA component rejection: {rejected_ica}",
            ]
            if manual_bad_epoch_indices:
                doc_lines.append(f"- Manual bad epochs: {manual_bad_epoch_indices}")
            new_docstring = ast.Expr(value=ast.Constant(value="\n".join(doc_lines)))

            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body[0] = new_docstring
            else:
                node.body.insert(0, new_docstring)

            return node

    class MethodModifier(ast.NodeTransformer):
        def __init__(self) -> None:
            self.in_run_method = False
            self.post_epoch_data_ready = False
            self.pending_post_epoch_ica_calls: list[ast.stmt] = []
            self.original_run_has_classification = False

        def visit_FunctionDef(self, node: ast.FunctionDef):  # type: ignore[override]
            if node.name != "run":
                return node

            self.in_run_method = True
            self.post_epoch_data_ready = False
            self.pending_post_epoch_ica_calls = []
            self.original_run_has_classification = any(
                self._is_self_call(stmt, "classify_ica_components")
                for stmt in ast.walk(node)
            )
            new_body: list[ast.stmt] = []
            for stmt in node.body:
                modified_stmt = self.visit(stmt)
                if self._is_manual_ica_rejection_call(modified_stmt):
                    continue
                if self._is_post_epoch_ica_call(modified_stmt):
                    call_stmt = self._post_epoch_ica_call(modified_stmt)
                    if self.post_epoch_data_ready:
                        self._append_post_epoch_ica_call(new_body, call_stmt)
                    else:
                        self.pending_post_epoch_ica_calls.append(call_stmt)
                    continue
                new_body.append(modified_stmt)

                if (
                    fix_type in ("ica", "both")
                    and isinstance(modified_stmt, ast.Expr)
                    and isinstance(modified_stmt.value, ast.Call)
                    and isinstance(modified_stmt.value.func, ast.Attribute)
                    and modified_stmt.value.func.attr == "classify_ica_components"
                ):
                    new_body.append(
                        ast.Expr(
                            value=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id="self", ctx=ast.Load()),
                                    attr="apply_ica_component_rejection",
                                    ctx=ast.Load(),
                                ),
                                args=[],
                                keywords=[
                                    ast.keyword(
                                        arg="manual_rejected_components",
                                        value=ast.List(
                                            elts=[
                                                ast.Constant(value=comp)
                                                for comp in rejected_ica
                                            ],
                                            ctx=ast.Load(),
                                        ),
                                    )
                                ],
                            )
                        )
                    )

                if (
                    manual_bad_epoch_indices
                    and isinstance(modified_stmt, ast.Expr)
                    and isinstance(modified_stmt.value, ast.Call)
                    and isinstance(modified_stmt.value.func, ast.Attribute)
                    and modified_stmt.value.func.attr in EPOCH_CREATION_METHODS
                ):
                    new_body.append(
                        ast.Expr(
                            value=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id="self", ctx=ast.Load()),
                                    attr="drop_manual_bad_epochs",
                                    ctx=ast.Load(),
                                ),
                                args=[],
                                keywords=[
                                    ast.keyword(
                                        arg="manual_bad_epoch_indices",
                                        value=ast.List(
                                            elts=[
                                                ast.Constant(value=idx)
                                                for idx in manual_bad_epoch_indices
                                            ],
                                            ctx=ast.Load(),
                                        ),
                                    ),
                                    ast.keyword(
                                        arg="manual_bad_epoch_times",
                                        value=ast.List(
                                            elts=[
                                                ast.Constant(value=value)
                                                for value in manual_bad_epoch_times
                                            ],
                                            ctx=ast.Load(),
                                        ),
                                    ),
                                    ast.keyword(
                                        arg="manual_bad_epoch_events",
                                        value=ast.List(
                                            elts=[
                                                ast.Constant(value=value)
                                                for value in manual_bad_epoch_events
                                            ],
                                            ctx=ast.Load(),
                                        ),
                                    ),
                                ],
                            )
                        )
                    )
                    if fix_type == POST_EPOCH_ICA_FIX_TYPE:
                        self.post_epoch_data_ready = True
                        self._append_pending_post_epoch_ica_calls(new_body)
                        self.pending_post_epoch_ica_calls = []

            if (
                fix_type == POST_EPOCH_ICA_FIX_TYPE
                and self.pending_post_epoch_ica_calls
            ):
                supported_methods = ", ".join(sorted(EPOCH_CREATION_METHODS))
                raise ValueError(
                    "Cannot generate post_epoch_ica reprocess task: no supported "
                    "epoch creation call was found before post-epoch ICA rewrite. "
                    f"Supported epoch creation methods: {supported_methods}."
                )

            node.body = new_body
            self.in_run_method = False
            return node

        def _is_self_call(self, node: ast.AST, name: str) -> bool:
            return (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == name
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
            )

        def _is_manual_ica_rejection_call(self, node: ast.AST) -> bool:
            if not (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "apply_ica_component_rejection"
            ):
                return False

            for keyword in node.value.keywords:
                if keyword.arg == "manual_rejected_components":
                    return True

            return False

        def _is_post_epoch_ica_call(self, node: ast.AST) -> bool:
            if fix_type != POST_EPOCH_ICA_FIX_TYPE:
                return False
            return (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr in {"run_ica", "classify_ica_components"}
            )

        def _replace_keyword(self, node: ast.Call, name: str, value: ast.expr) -> None:
            for keyword in node.keywords:
                if keyword.arg == name:
                    keyword.value = value
                    return
            node.keywords.append(ast.keyword(arg=name, value=value))

        def _replace_arg_or_keyword(
            self, node: ast.Call, name: str, arg_index: int, value: ast.expr
        ) -> None:
            if len(node.args) > arg_index:
                node.args[arg_index] = value
                return
            self._replace_keyword(node, name, value)

        def _post_epoch_ica_call(self, node: ast.stmt) -> ast.stmt:
            if not (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
            ):
                return node

            call = node.value
            if call.func.attr == "run_ica":
                self._replace_keyword(call, "use_epochs", ast.Constant(value=True))
                self._replace_keyword(
                    call,
                    "stage_name",
                    ast.Constant(value="post_epoch_rejection_ica_fit"),
                )
            elif call.func.attr == "classify_ica_components":
                self._replace_arg_or_keyword(
                    call, "method", 0, ast.Constant(value="iclabel")
                )
                self._replace_keyword(call, "reject", ast.Constant(value=False))
                self._replace_keyword(
                    call,
                    "stage_name",
                    ast.Constant(value="post_epoch_rejection_ica_labeling"),
                )
            return node

        def _synthetic_post_epoch_classification_call(self) -> ast.stmt:
            return ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()),
                        attr="classify_ica_components",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[
                        ast.keyword(arg="method", value=ast.Constant(value="iclabel")),
                        ast.keyword(arg="reject", value=ast.Constant(value=False)),
                        ast.keyword(
                            arg="stage_name",
                            value=ast.Constant(
                                value="post_epoch_rejection_ica_labeling"
                            ),
                        ),
                    ],
                )
            )

        def _append_post_epoch_ica_call(
            self, body: list[ast.stmt], call_stmt: ast.stmt
        ) -> None:
            body.append(call_stmt)
            if (
                not self.original_run_has_classification
                and isinstance(call_stmt, ast.Expr)
                and isinstance(call_stmt.value, ast.Call)
                and isinstance(call_stmt.value.func, ast.Attribute)
                and call_stmt.value.func.attr == "run_ica"
            ):
                body.append(self._synthetic_post_epoch_classification_call())

        def _append_pending_post_epoch_ica_calls(self, body: list[ast.stmt]) -> None:
            for call_stmt in self.pending_post_epoch_ica_calls:
                self._append_post_epoch_ica_call(body, call_stmt)

        def visit_Call(self, node: ast.Call):  # type: ignore[override]
            if not self.in_run_method:
                return node

            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "clean_bad_channels"
                and bad_channels
            ):
                found_manual_bad_channels = False
                for kw in node.keywords:
                    if kw.arg == "manual_bad_channels":
                        kw.value = ast.List(
                            elts=[ast.Constant(value=ch) for ch in bad_channels],
                            ctx=ast.Load(),
                        )
                        found_manual_bad_channels = True
                        break
                if not found_manual_bad_channels:
                    node.keywords.append(
                        ast.keyword(
                            arg="manual_bad_channels",
                            value=ast.List(
                                elts=[ast.Constant(value=ch) for ch in bad_channels],
                                ctx=ast.Load(),
                            ),
                        )
                    )

            if (
                fix_type in ("ica", "both")
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "classify_ica_components"
            ):
                found_reject = False
                for kw in node.keywords:
                    if kw.arg == "reject":
                        kw.value = ast.Constant(value=False)
                        found_reject = True
                        break
                if not found_reject:
                    node.keywords.append(
                        ast.keyword(arg="reject", value=ast.Constant(value=False))
                    )

            return node

    tree = ConfigModifier().visit(tree)
    tree = ClassRenamer().visit(tree)
    tree = MethodModifier().visit(tree)
    ast.fix_missing_locations(tree)
    modified_code = ast.unparse(tree)

    eog_note = ""
    if len(bad_channels) != len(bad_channels_raw):
        filtered_out = [ch for ch in bad_channels_raw if ch not in bad_channels]
        eog_note = (
            f"\n# Note: EOG channels {filtered_out} excluded"
            " (dropped earlier in pipeline)"
        )

    header = f"""# =============================================================================
#  REPROCESSING TASK WITH MANUAL OVERRIDES
# =============================================================================
# This task was automatically generated to reprocess EEG data with manual
# overrides from the review GUI.
#
# Generated: {payload.get("timestamp", "")}
# Original file: {file_stem}
# Fix type: {fix_type}
#
# Manual Overrides:
# - Bad channels: {len(bad_channels)} channels{eog_note}
# - ICA components: {len(rejected_ica)} components
# - Manual bad epochs: {len(manual_bad_epoch_indices)} epochs
# - Post-epoch-rejection ICA: {fix_type == POST_EPOCH_ICA_FIX_TYPE}
# =============================================================================

"""

    return header + modified_code + "\n"
