"""AutoClean EEG inclusion/exclusion review helper.

This tool extends the classic :mod:`autoclean.tools.autoclean_review` window
so reviewers can keep the familiar full-screen MNE browser while tracking
Pass/Fail/Review decisions and notes for every exported ``.set`` file.  The
script keeps everything self-contained to make distribution easy for labs that
copy the helper into bespoke environments.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, Optional


def check_gui_dependencies() -> None:
    """Fail fast if the optional GUI stack is missing."""

    missing: list[str] = []
    try:  # pragma: no cover - import guard only
        import PyQt5  # noqa: F401
    except ImportError:  # pragma: no cover - runtime dependency guard
        missing.append("PyQt5")

    if missing:
        print("Error: Missing required GUI dependencies for the exclusion tool.")
        print("Install the extras bundle first:")
        print("    pip install autocleaneeg-pipeline[gui]")
        print(f"Missing packages: {', '.join(missing)}")
        sys.exit(1)


check_gui_dependencies()


from PyQt5.QtCore import Qt, QTimer  # noqa: E402
from PyQt5.QtGui import QColor, QKeySequence  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QShortcut,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from autoclean.tools import autoclean_review  # noqa: E402
from autoclean.utils.user_config import user_config  # noqa: E402


STATUS_DEFINITIONS: dict[str, dict[str, str]] = {
    "UNSET": {"label": "Not Started", "color": "#bdc3c7", "shortcut": ""},
    "PASS": {"label": "Pass", "color": "#2ecc71", "shortcut": "P"},
    "FAIL": {"label": "Fail", "color": "#e74c3c", "shortcut": "F"},
    "REVIEW": {
        "label": "Needs Review",
        "color": "#f1c40f",
        "shortcut": "R",
    },
}

STATUS_ORDER: tuple[str, ...] = ("PASS", "FAIL", "REVIEW", "UNSET")


def _human_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _open_path(path: Path) -> None:
    """Open *path* using the default OS handler."""

    if sys.platform.startswith("darwin"):
        subprocess.run(["open", str(path)], check=False)
    elif os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


class ExclusionFileSelector(autoclean_review.FileSelector):
    """Subclass of the classic review widget with exclusion helpers."""

    def __init__(
        self,
        exports_dir: Optional[Path] = None,
        task_root: Optional[Path] = None,
    ) -> None:
        self.task_root = Path(task_root).resolve() if task_root else None
        self.exports_dir = Path(exports_dir).resolve() if exports_dir else None

        self.decisions_path: Optional[Path] = None
        self.decisions_csv_path: Optional[Path] = None
        self.decisions: Dict[str, dict[str, str]] = {}
        self.row_lookup: dict[str, QTreeWidgetItem] = {}
        self.all_keys: set[str] = set()
        self.current_key: Optional[str] = None
        self.current_display_name: Optional[str] = None

        self.status_label: Optional[QLabel] = None
        self.current_file_label: Optional[QLabel] = None
        self.save_state_label: Optional[QLabel] = None
        self.summary_table: Optional[QTableWidget] = None
        self.notes_edit: Optional[QTextEdit] = None
        self.related_list: Optional[QListWidget] = None
        self.detail_panel: Optional[QWidget] = None
        self.save_timer: Optional[QTimer] = None

        self._updating_notes = False

        super().__init__(
            str(self.exports_dir) if self.exports_dir is not None else None
        )

        # Base ``__init__`` calls ``loadFiles`` once; run our extensions after.
        self._extend_ui()
        self._configure_directory(self.current_dir)
        self._load_decisions()
        self.loadFiles()  # Refresh now that status metadata exists
        self.updateStatusBar()
        self._update_decision_controls(None)

    # ------------------------------------------------------------------
    # UI bootstrapping helpers
    # ------------------------------------------------------------------
    def _extend_ui(self) -> None:
        """Inject decision widgets while keeping the base layout intact."""

        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.setInterval(400)
        self.save_timer.timeout.connect(self._commit_decisions)

        decision_group = QGroupBox("Decision")
        decision_layout = QVBoxLayout()

        self.current_file_label = QLabel("No file selected")
        self.current_file_label.setWordWrap(True)
        decision_layout.addWidget(self.current_file_label)

        self.status_label = QLabel("Status: Not Started")
        self.status_label.setStyleSheet("font-weight: bold; color: #2c3e50")
        decision_layout.addWidget(self.status_label)

        button_row = QHBoxLayout()
        self._shortcuts: dict[str, QShortcut] = {}
        for status in ("PASS", "FAIL", "REVIEW"):
            meta = STATUS_DEFINITIONS[status]
            btn = QPushButton(f"{meta['label']} ({meta['shortcut']})")
            btn.clicked.connect(partial(self._set_status, status))
            button_row.addWidget(btn)
            shortcut = QShortcut(QKeySequence(meta["shortcut"]), self)
            shortcut.activated.connect(partial(self._set_status, status))
            self._shortcuts[status] = shortcut

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(partial(self._set_status, "UNSET"))
        button_row.addWidget(clear_btn)
        decision_layout.addLayout(button_row)

        self.save_state_label = QLabel("")
        self.save_state_label.setStyleSheet("color: #7f8c8d; font-style: italic")
        decision_layout.addWidget(self.save_state_label)

        decision_group.setLayout(decision_layout)

        # Insert decision controls right above the close/exit buttons
        insert_index = self.left_layout.indexOf(self.close_plot_btn)
        if insert_index < 0:
            insert_index = self.left_layout.count() - 1
        self.left_layout.insertWidget(insert_index, decision_group)

        summary_group = QGroupBox("Summary")
        summary_layout = QVBoxLayout()
        self.summary_table = QTableWidget(len(STATUS_ORDER), 2)
        self.summary_table.setHorizontalHeaderLabels(["Status", "Count"])
        self.summary_table.verticalHeader().setVisible(False)
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        self.summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.summary_table.setSelectionMode(QTableWidget.NoSelection)
        for row, status in enumerate(STATUS_ORDER):
            meta = STATUS_DEFINITIONS[status]
            status_item = QTableWidgetItem(meta["label"])
            status_item.setFlags(Qt.ItemIsEnabled)
            status_item.setForeground(QColor(meta["color"]))
            self.summary_table.setItem(row, 0, status_item)
            count_item = QTableWidgetItem("0")
            count_item.setTextAlignment(Qt.AlignCenter)
            count_item.setFlags(Qt.ItemIsEnabled)
            self.summary_table.setItem(row, 1, count_item)
        summary_layout.addWidget(self.summary_table)
        summary_group.setLayout(summary_layout)

        exit_index = self.left_layout.indexOf(self.exit_btn)
        self.left_layout.insertWidget(exit_index, summary_group)

        # Detail panel (notes + related exports)
        self.detail_panel = QWidget()
        detail_layout = QVBoxLayout()

        notes_group = QGroupBox("Reviewer Notes")
        notes_layout = QVBoxLayout()
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText(
            "Summarize observations, reasons for exclusion, or follow-up items."
        )
        self.notes_edit.textChanged.connect(self._handle_notes_changed)
        notes_layout.addWidget(self.notes_edit)
        notes_group.setLayout(notes_layout)
        detail_layout.addWidget(notes_group)

        related_group = QGroupBox("Related Exports & Reports")
        related_layout = QVBoxLayout()
        self.related_list = QListWidget()
        self.related_list.itemActivated.connect(self._open_related_item)
        related_layout.addWidget(self.related_list)
        related_group.setLayout(related_layout)
        detail_layout.addWidget(related_group)
        detail_layout.addStretch(1)

        self.detail_panel.setLayout(detail_layout)
        self.detail_panel.hide()
        self.right_layout.addWidget(self.detail_panel)

    # ------------------------------------------------------------------
    # Directory + persistence helpers
    # ------------------------------------------------------------------
    def _configure_directory(self, directory: Optional[str]) -> None:
        if directory:
            root = Path(directory).resolve()
        elif self.exports_dir:
            root = self.exports_dir
        else:
            root = Path.cwd()
        self.exports_dir = root
        self.current_dir = str(root)
        self.decisions_path = root / "autoclean_exclusion_decisions.json"
        self.decisions_csv_path = root / "autoclean_exclusion_decisions.csv"

    def _load_decisions(self) -> None:
        self.decisions = {}
        if self.decisions_path and self.decisions_path.exists():
            try:
                data = json.loads(self.decisions_path.read_text())
                if isinstance(data, dict):
                    self.decisions = data
            except Exception as exc:  # pragma: no cover - defensive
                print(f"Warning: could not load decisions file: {exc}")
        self._update_summary()

    def _schedule_save(self) -> None:
        if self.save_state_label is not None:
            self.save_state_label.setText("Saving...")
        if self.save_timer is not None:
            self.save_timer.start()

    def _commit_decisions(self) -> None:
        if not self.decisions_path:
            return

        self.decisions_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.decisions, indent=2, sort_keys=True)
        self.decisions_path.write_text(payload)

        if self.decisions_csv_path:
            rows = []
            for key, record in sorted(self.decisions.items()):
                rows.append(
                    {
                        "entry": key,
                        "status": record.get("status", "UNSET"),
                        "notes": record.get("notes", ""),
                        "relative_path": record.get("relative_path", ""),
                        "last_updated": record.get("last_updated", ""),
                    }
                )
            with self.decisions_csv_path.open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(
                    fp, fieldnames=["entry", "status", "notes", "relative_path", "last_updated"]
                )
                writer.writeheader()
                writer.writerows(rows)

        if self.save_state_label is not None:
            self.save_state_label.setText(f"Saved {_human_timestamp()}")

    # ------------------------------------------------------------------
    # Tree + selection logic
    # ------------------------------------------------------------------
    def loadFiles(self) -> None:  # noqa: N802 - inherited public API
        self.file_tree.clear()
        self.row_lookup.clear()
        self.all_keys.clear()
        if self.current_dir:
            root_path = Path(self.current_dir)
            root_item = QTreeWidgetItem(self.file_tree, [root_path.name])
            root_item.setData(0, Qt.UserRole, str(root_path))
            self._populate_tree(root_item, root_path)
            root_item.setExpanded(True)
            if root_item.childCount() > 0:
                root_item.child(0).setExpanded(True)
        self._update_summary()

    def _populate_tree(
        self,
        parent_item: QTreeWidgetItem,
        directory: Path,
    ) -> None:
        entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        for entry in entries:
            if entry.is_dir():
                folder = QTreeWidgetItem(parent_item, [entry.name])
                folder.setIcon(0, self.style().standardIcon(self.style().SP_DirIcon))
                folder.setData(0, Qt.UserRole, str(entry))
                self._populate_tree(folder, entry)
            elif entry.suffix.lower() == ".set":
                item = QTreeWidgetItem(parent_item, [entry.name])
                item.setIcon(0, self.style().standardIcon(self.style().SP_FileIcon))
                item.setData(0, Qt.UserRole, str(entry))
                key = self._record_key(entry)
                item.setData(0, Qt.UserRole + 1, key)
                base_label = entry.name
                if entry.name in self.modified_files:
                    base_label = f"{base_label} *"
                item.setData(0, Qt.UserRole + 2, base_label)
                self.row_lookup[key] = item
                self.all_keys.add(key)
                status = self.decisions.get(key, {}).get("status", "UNSET")
                self._apply_status_to_item(item, status)

    def selectDirectory(self) -> None:  # noqa: N802 - inherited public API
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Directory", self.current_dir or str(Path.cwd())
        )
        if dir_path:
            self._configure_directory(dir_path)
            self._load_decisions()
            self.current_key = None
            self.current_display_name = None
            if self.related_list is not None:
                self.related_list.clear()
            self._update_decision_controls(None)
            super().updateStatusBar()
            self.loadFiles()

    def onFileSelect(self, item):  # noqa: N802 - inherited public API
        file_path_str = item.data(0, Qt.UserRole)
        if not file_path_str:
            self.plot_btn.setEnabled(False)
            self.view_record_btn.setEnabled(False)
            self.current_key = None
            self.current_display_name = None
            self._update_decision_controls(None)
            return

        file_path = Path(file_path_str)
        if file_path.suffix.lower() != ".set":
            self.plot_btn.setEnabled(False)
            self.view_record_btn.setEnabled(False)
            self.current_key = None
            self.current_display_name = None
            self._update_decision_controls(None)
            return

        self.selected_item = item
        self.selected_file = file_path.name
        self.selected_file_path = str(file_path)
        self.plot_btn.setEnabled(True)
        self.current_display_name = self._relative_path(file_path)
        try:
            self.current_run_id = self.getRunId(self.selected_file_path)
            self.current_run_record = autoclean_review.get_run_record(self.current_run_id)
            self.view_record_btn.setEnabled(True)
        except Exception:
            self.view_record_btn.setEnabled(False)
            self.current_run_record = None

        self.current_key = self._record_key(file_path)
        record = self.decisions.get(self.current_key)
        self._update_decision_controls(record)
        self._refresh_related_list(file_path)
        if self.detail_panel is not None:
            self.detail_panel.show()

    def plotFile(self) -> None:  # noqa: N802 - inherited public API
        super().plotFile()
        if self.detail_panel is not None and self.detail_panel.isHidden():
            self.detail_panel.show()
        if self.detail_panel is not None:
            # Ensure the detail panel sits underneath the plot widget
            self.right_layout.removeWidget(self.detail_panel)
            self.right_layout.addWidget(self.detail_panel)

    def closePlot(self) -> None:  # noqa: N802 - inherited public API
        super().closePlot()

    # ------------------------------------------------------------------
    # Decision management
    # ------------------------------------------------------------------
    def _set_status(self, status: str) -> None:
        if not self.current_key or not hasattr(self, "selected_file_path"):
            return

        record = self.decisions.setdefault(
            self.current_key,
            {
                "status": "UNSET",
                "notes": "",
                "relative_path": self._relative_path(Path(self.selected_file_path)),
                "last_updated": "",
            },
        )
        record["status"] = status
        record["last_updated"] = _human_timestamp()
        if self.notes_edit is not None:
            record["notes"] = self.notes_edit.toPlainText().strip()

        item = self.row_lookup.get(self.current_key)
        if item is not None:
            self._apply_status_to_item(item, status)

        self._update_decision_controls(record)
        self._update_summary()
        self._schedule_save()

    def _handle_notes_changed(self) -> None:
        if self._updating_notes or not self.current_key or self.notes_edit is None:
            return
        record = self.decisions.setdefault(
            self.current_key,
            {
                "status": "UNSET",
                "notes": "",
                "relative_path": self._relative_path(Path(self.selected_file_path)),
                "last_updated": "",
            },
        )
        record["notes"] = self.notes_edit.toPlainText().strip()
        record["last_updated"] = _human_timestamp()
        self._schedule_save()

    def _update_decision_controls(self, record: Optional[dict[str, str]]) -> None:
        status = record.get("status") if record else "UNSET"
        meta = STATUS_DEFINITIONS.get(status or "UNSET", STATUS_DEFINITIONS["UNSET"])
        if self.status_label is not None:
            self.status_label.setText(f"Status: {meta['label']}")
            self.status_label.setStyleSheet(
                f"font-weight: bold; color: {meta['color'] if status != 'UNSET' else '#2c3e50'}"
            )
        if self.current_file_label is not None:
            if self.current_display_name:
                self.current_file_label.setText(self.current_display_name)
            else:
                self.current_file_label.setText("No file selected")
        if self.notes_edit is not None:
            self._updating_notes = True
            self.notes_edit.setPlainText(record.get("notes", "") if record else "")
            self._updating_notes = False

    def _apply_status_to_item(self, item, status: str) -> None:
        base_label = item.data(0, Qt.UserRole + 2) or item.text(0)
        meta = STATUS_DEFINITIONS.get(status, STATUS_DEFINITIONS["UNSET"])
        display = base_label
        if status and status != "UNSET":
            display = f"{base_label} [{meta['label']}]"
        item.setText(0, display)
        if status and status != "UNSET":
            color = QColor(meta["color"])
            color.setAlpha(60)
            item.setBackground(0, color)
        else:
            item.setBackground(0, QColor())

    def _update_summary(self) -> None:
        counts = Counter({key: 0 for key in STATUS_DEFINITIONS})
        for key in self.all_keys:
            status = self.decisions.get(key, {}).get("status", "UNSET")
            counts[status] += 1
        if self.summary_table is not None:
            for row, status in enumerate(STATUS_ORDER):
                item = self.summary_table.item(row, 1)
                if item is not None:
                    item.setText(str(counts[status]))

    # ------------------------------------------------------------------
    # Related files + helpers
    # ------------------------------------------------------------------
    def _refresh_related_list(self, file_path: Path) -> None:
        if self.related_list is None:
            return
        self.related_list.clear()
        for related in self._gather_related_files(file_path):
            item = QListWidgetItem(related.name)
            item.setToolTip(str(related))
            item.setData(Qt.UserRole, str(related))
            self.related_list.addItem(item)

    def _open_related_item(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.UserRole)
        if data:
            _open_path(Path(str(data)))

    def _gather_related_files(self, file_path: Path) -> Iterable[Path]:
        base_stem = file_path.stem
        results: list[Path] = []

        for sibling in sorted(file_path.parent.iterdir()):
            if sibling == file_path:
                continue
            if sibling.name.startswith(base_stem):
                results.append(sibling)

        if self.task_root and self.task_root.exists():
            reports_root = self.task_root / "reports"
            if reports_root.exists():
                for report in sorted(reports_root.rglob("*")):
                    if report.is_file() and base_stem in report.stem:
                        if report not in results and report != file_path:
                            results.append(report)

        return results

    def _record_key(self, file_path: Path) -> str:
        root = self.exports_dir or Path(self.current_dir or "")
        try:
            relative = file_path.resolve().relative_to(root.resolve())
        except Exception:
            relative = file_path.name
            return str(Path(relative).with_suffix(""))
        return str(Path(relative).with_suffix(""))

    def _relative_path(self, file_path: Path) -> str:
        root = self.exports_dir or Path(self.current_dir or "")
        try:
            return str(file_path.resolve().relative_to(root.resolve()))
        except Exception:
            return str(file_path.name)


def determine_paths(args: argparse.Namespace) -> tuple[Path, Optional[Path]]:
    """Infer exports/task directories from CLI arguments."""

    if args.exports:
        exports_dir = Path(args.exports).expanduser().resolve()
        task_root: Optional[Path] = (
            Path(args.task_root).expanduser().resolve()
            if args.task_root
            else exports_dir.parent
        )
        return exports_dir, task_root

    if args.path:
        candidate = Path(args.path).expanduser().resolve()
        exports_candidate = candidate / "exports"
        if exports_candidate.exists():
            task_root = candidate
            return exports_candidate, task_root
        return candidate, candidate.parent if candidate.parent.exists() else None

    # Default: prefer most recent task exports in workspace, then ./exports, then current directory
    workspace_output = user_config.get_default_output_dir()
    if workspace_output.exists():
        # Find the most recent task directory in workspace output
        task_dirs = [d for d in workspace_output.iterdir() if d.is_dir()]
        if task_dirs:
            # Sort by modification time, most recent first
            task_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
            most_recent_task = task_dirs[0]
            task_exports = most_recent_task / "exports"
            if task_exports.exists():
                return task_exports, most_recent_task

    cwd = Path.cwd()
    default_exports = cwd / "exports"
    if default_exports.exists():
        return default_exports, cwd
    return cwd, cwd


def run_autoclean_exclusion_tool(
    exports_dir: Optional[Path] = None,
    task_root: Optional[Path] = None,
) -> None:
    """Launch the Qt inclusion/exclusion helper."""

    app = QApplication(sys.argv)
    window = ExclusionFileSelector(exports_dir=exports_dir, task_root=task_root)
    window.setWindowTitle("Autoclean - Inclusion/Exclusion Review")
    window.showMaximized()
    if not app.styleSheet():
        app.setStyleSheet("")
    sys.exit(app.exec_())


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the Autoclean inclusion/exclusion review interface with the "
            "full timeseries browser from autoclean_review."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        help=(
            "Optional path to a run folder or exports directory. Defaults to the "
            "current working directory or ./exports if present."
        ),
    )
    parser.add_argument(
        "--exports",
        help="Directly specify the exports directory to inspect.",
    )
    parser.add_argument(
        "--task-root",
        help="Optional task/run root directory so related reports can be listed.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    exports_dir, task_root = determine_paths(args)
    if not exports_dir.exists():
        raise SystemExit(f"Exports directory not found: {exports_dir}")
    run_autoclean_exclusion_tool(exports_dir, task_root)


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    main()
