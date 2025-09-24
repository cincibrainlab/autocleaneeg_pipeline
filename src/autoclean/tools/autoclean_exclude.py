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


from PyQt5.QtCore import Qt, QTimer, QSize  # noqa: E402
from PyQt5.QtGui import QColor, QKeySequence  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QFrame,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QStackedLayout,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QStyle,
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
        self._status_buttons: dict[str, QPushButton] = {}
        self._clear_button: Optional[QPushButton] = None
        self._decision_stack: Optional[QStackedLayout] = None

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

        # Refresh the directory controls with a compact toolbar treatment
        self._modify_top_buttons()

        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.setInterval(400)
        self.save_timer.timeout.connect(self._commit_decisions)

        decision_card = QFrame()
        decision_card.setObjectName("decisionCard")
        decision_layout = QVBoxLayout()
        decision_layout.setContentsMargins(18, 18, 18, 18)
        decision_layout.setSpacing(14)
        decision_card.setLayout(decision_layout)

        header_row = QHBoxLayout()
        header_row.setSpacing(10)

        header_label = QLabel("Review Decision")
        header_label.setObjectName("decisionHeader")
        header_row.addWidget(header_label)
        header_row.addStretch(1)

        self.status_label = QLabel("Not Started")
        self.status_label.setObjectName("decisionStatusChip")
        self.status_label.setAlignment(Qt.AlignCenter)
        header_row.addWidget(self.status_label)
        decision_layout.addLayout(header_row)

        self.current_file_label = QLabel("No file selected")
        self.current_file_label.setObjectName("decisionFileLabel")
        self.current_file_label.setWordWrap(True)
        decision_layout.addWidget(self.current_file_label)

        button_container = QFrame()
        button_container.setObjectName("decisionButtonRow")
        button_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(10)
        button_container.setLayout(button_row)

        self._shortcuts: dict[str, QShortcut] = {}
        self._status_buttons.clear()
        for status in ("PASS", "FAIL", "REVIEW"):
            meta = STATUS_DEFINITIONS[status]
            btn = QPushButton(meta["label"])
            btn.setCursor(Qt.PointingHandCursor)
            btn.setCheckable(True)
            btn.setMinimumHeight(34)
            btn.setToolTip(
                f"Mark the selection as {meta['label']}. Shortcut: {meta['shortcut']}"
            )
            btn.clicked.connect(partial(self._set_status, status))
            button_row.addWidget(btn)
            self._status_buttons[status] = btn

            shortcut = QShortcut(QKeySequence(meta["shortcut"]), self)
            shortcut.activated.connect(partial(self._set_status, status))
            self._shortcuts[status] = shortcut

        button_row.addStretch(1)

        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("decisionClearButton")
        clear_btn.setCursor(Qt.PointingHandCursor)
        clear_btn.setMinimumHeight(34)
        clear_btn.setToolTip("Reset decision to Not Started. Shortcut: C")
        clear_btn.clicked.connect(partial(self._set_status, "UNSET"))
        self._clear_button = clear_btn
        button_row.addWidget(clear_btn)

        clear_shortcut = QShortcut(QKeySequence("C"), self)
        clear_shortcut.activated.connect(partial(self._set_status, "UNSET"))
        self._shortcuts["CLEAR"] = clear_shortcut

        shortcut_hint = QLabel("Shortcuts: P Pass • F Fail • R Needs Review • C Clear")
        shortcut_hint.setObjectName("decisionShortcutHint")
        shortcut_hint.setWordWrap(True)

        self.save_state_label = QLabel("Select a file to assign a decision.")
        self.save_state_label.setObjectName("decisionSaveLabel")

        actions_widget = QWidget()
        actions_layout = QVBoxLayout()
        actions_layout.setContentsMargins(0, 4, 0, 0)
        actions_layout.setSpacing(12)
        actions_widget.setLayout(actions_layout)
        actions_layout.addWidget(button_container)
        actions_layout.addWidget(shortcut_hint)
        actions_layout.addWidget(self.save_state_label)

        empty_widget = QFrame()
        empty_widget.setObjectName("decisionEmptyState")
        empty_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        empty_layout = QVBoxLayout()
        empty_layout.setContentsMargins(0, 8, 0, 8)
        empty_layout.setSpacing(6)
        empty_widget.setLayout(empty_layout)
        empty_layout.addStretch(1)
        empty_title = QLabel("Waiting for a selection")
        empty_title.setObjectName("decisionEmptyTitle")
        empty_title.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(empty_title)
        empty_body = QLabel(
            "Pick an export on the left to assign Pass, Fail, or Needs Review decisions."
        )
        empty_body.setObjectName("decisionEmptyBody")
        empty_body.setAlignment(Qt.AlignCenter)
        empty_body.setWordWrap(True)
        empty_layout.addWidget(empty_body)
        empty_layout.addStretch(1)

        self._decision_stack = QStackedLayout()
        self._decision_stack.setContentsMargins(0, 0, 0, 0)
        self._decision_stack.setSpacing(0)
        self._decision_stack.addWidget(empty_widget)
        self._decision_stack.addWidget(actions_widget)
        self._decision_stack.setCurrentIndex(0)
        decision_layout.addLayout(self._decision_stack)

        decision_card.setStyleSheet(
            """
            #decisionCard {
                background-color: #ffffff;
                border: 1px solid #d9e2ec;
                border-radius: 12px;
            }
            #decisionHeader {
                font-size: 13px;
                text-transform: uppercase;
                letter-spacing: 0.8px;
                color: #51606f;
                font-weight: 700;
            }
            #decisionFileLabel {
                color: #1f2d3d;
                font-weight: 600;
            }
            #decisionStatusChip {
                padding: 4px 12px;
                border-radius: 12px;
                font-weight: 600;
                background-color: #edf2f7;
                color: #5b6c7c;
            }
            #decisionButtonRow QPushButton {
                background-color: #f6f9ff;
                border: 1px solid #d4e2ff;
                border-radius: 8px;
                padding: 6px 18px;
                font-weight: 600;
                color: #1f2d3d;
            }
            #decisionButtonRow QPushButton:hover {
                border-color: #3a7bd5;
                color: #1a4fa3;
            }
            #decisionButtonRow QPushButton:checked {
                background-color: #1a4fa3;
                border-color: #1a4fa3;
                color: #ffffff;
            }
            #decisionButtonRow QPushButton:disabled {
                background-color: #f0f4f8;
                color: #9aa5b1;
                border-color: #dfe4ea;
            }
            #decisionClearButton {
                background-color: #ffffff;
                border: 1px solid #d9e2ec;
                color: #5b6c7c;
            }
            #decisionClearButton:hover {
                border-color: #94a3b8;
                color: #2c3e50;
            }
            #decisionShortcutHint {
                color: #64748b;
                font-size: 12px;
            }
            #decisionSaveLabel {
                color: #64748b;
                font-style: italic;
            }
            #decisionEmptyState {
                background-color: #f8fafc;
                border: 1px dashed #d0d7e2;
                border-radius: 10px;
            }
            #decisionEmptyTitle {
                font-size: 13px;
                font-weight: 700;
                color: #52606d;
            }
            #decisionEmptyBody {
                color: #7b8794;
                font-size: 12px;
            }
            """
        )

        # Insert decision controls right above the close/exit buttons
        insert_index = self.left_layout.indexOf(self.close_plot_btn)
        if insert_index < 0:
            insert_index = self.left_layout.count() - 1
        self.left_layout.insertWidget(insert_index, decision_card)

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

    def _modify_top_buttons(self) -> None:
        """Replace the default directory buttons with a polished toolbar."""

        if getattr(self, "_directory_toolbar_initialized", False):
            return

        toolbar = QFrame()
        toolbar.setObjectName("directoryToolbar")

        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(12, 10, 12, 10)
        toolbar_layout.setSpacing(10)
        toolbar_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        header = QLabel("Workspace")
        header.setObjectName("directoryToolbarLabel")
        header.setAlignment(Qt.AlignVCenter)
        header.setStyleSheet(
            "font-size: 11px; letter-spacing: 0.6px; text-transform: uppercase; "
            "color: #5b6c7c; font-weight: 600;"
        )
        toolbar_layout.addWidget(header)

        button_specs = (
            (
                self.select_dir_btn,
                "Choose Folder…",
                self.style().standardIcon(QStyle.SP_DialogOpenButton),
                "Browse for a directory containing exported .set files.",
            ),
            (
                self.open_folder_btn,
                "Open Folder",
                self.style().standardIcon(QStyle.SP_DirIcon),
                "Reveal the current review folder in your file browser.",
            ),
            (
                self.refresh_btn,
                "Refresh List",
                self.style().standardIcon(QStyle.SP_BrowserReload),
                "Reload the file tree to pick up new or modified exports.",
            ),
        )

        for button, label, icon, tooltip in button_specs:
            index = self.left_layout.indexOf(button)
            if index >= 0:
                self.left_layout.removeWidget(button)
            button.setText(label)
            if not icon.isNull():
                button.setIcon(icon)
                button.setIconSize(QSize(18, 18))
            button.setToolTip(tooltip)
            button.setCursor(Qt.PointingHandCursor)
            button.setMinimumHeight(34)
            button.setMaximumWidth(180)
            button.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
            toolbar_layout.addWidget(button)

        toolbar_layout.addStretch(1)

        toolbar.setLayout(toolbar_layout)
        toolbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        toolbar.setStyleSheet(
            """
            #directoryToolbar {
                background-color: #f7f9fc;
                border: 1px solid #d9e2ec;
                border-radius: 8px;
            }
            #directoryToolbar QPushButton {
                background-color: #ffffff;
                border: 1px solid #d9e2ec;
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: 600;
                color: #1f2d3d;
            }
            #directoryToolbar QPushButton:hover {
                border-color: #3a7bd5;
                color: #1a4fa3;
            }
            #directoryToolbar QPushButton:pressed {
                background-color: #ecf2fb;
            }
            #directoryToolbar QPushButton:disabled {
                background-color: #f1f3f6;
                color: #9aa5b1;
                border-color: #dfe4ea;
            }
            """
        )

        self.left_layout.insertWidget(0, toolbar)
        self.directory_toolbar = toolbar
        self._directory_toolbar_initialized = True

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
        has_selection = self.current_key is not None

        if self._decision_stack is not None:
            self._decision_stack.setCurrentIndex(1 if has_selection else 0)

        if self.status_label is not None:
            display_label = meta["label"] if status and status != "UNSET" else "Not Started"
            if status and status != "UNSET":
                chip_color = QColor(meta["color"])
                chip_bg = QColor(chip_color)
                chip_bg.setAlpha(48)
                chip_bg_value = chip_bg.name(QColor.HexArgb)
            else:
                chip_color = QColor("#5b6c7c")
                chip_bg = QColor("#edf2f7")
                chip_bg_value = chip_bg.name()

            self.status_label.setText(display_label)
            self.status_label.setStyleSheet(
                "padding: 4px 12px; border-radius: 12px; font-weight: 600; "
                f"background-color: {chip_bg_value}; color: {chip_color.name()};"
            )

        for key, button in self._status_buttons.items():
            button.blockSignals(True)
            button.setEnabled(has_selection)
            button.setChecked(has_selection and status == key)
            button.blockSignals(False)

        if self._clear_button is not None:
            self._clear_button.setEnabled(has_selection)
        if self.current_file_label is not None:
            if self.current_display_name:
                self.current_file_label.setText(self.current_display_name)
            else:
                self.current_file_label.setText("No file selected")
        if self.notes_edit is not None:
            self._updating_notes = True
            self.notes_edit.setPlainText(record.get("notes", "") if record else "")
            self._updating_notes = False

        if self.save_state_label is not None:
            if has_selection:
                if self.save_state_label.text() in {"Select a file to assign a decision.", ""}:
                    self.save_state_label.setText("Changes auto-save after a short pause.")
            else:
                self.save_state_label.setText("Select a file to assign a decision.")

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
