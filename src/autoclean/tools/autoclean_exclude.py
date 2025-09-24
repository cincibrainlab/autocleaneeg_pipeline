"""AutoClean EEG inclusion/exclusion review tool.

This module provides a lightweight Qt application that mirrors the spirit of
the legacy :mod:`autoclean.tools.autoclean_review` helper but focuses on
quickly classifying exported EEG deliverables as *pass*, *fail*, or *needs
review*.  The tool keeps everything in a single script so researchers can copy
it into bespoke environments without chasing additional modules.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def check_gui_dependencies() -> None:
    """Ensure the optional GUI stack is available before importing Qt."""

    missing: list[str] = []
    try:  # pragma: no cover - import side effect only
        import PyQt5  # noqa: F401
    except ImportError:  # pragma: no cover - runtime dependency guard
        missing.append("PyQt5")

    if missing:
        print("Error: Missing required GUI dependencies for the exclusion tool.")
        print("Install the extras bundle first:")
        print("    pip install autocleaneeg-pipeline[gui]")
        print(f"Missing packages: {', '.join(missing)}")
        sys.exit(1)


# Guard Qt imports behind the dependency check so running this module from a
# minimal environment fails fast with a readable error message.
check_gui_dependencies()

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtGui import QKeySequence

from autoclean.utils.user_config import UserConfigManager


PRIMARY_EXTENSIONS: set[str] = {
    ".set",
    ".fif",
    ".edf",
    ".bdf",
    ".vhdr",
    ".eeg",
    ".xdf",
    ".gdf",
    ".mat",
    ".parquet",
    ".csv",
    ".tsv",
}

STATUS_STYLES: Dict[str, Dict[str, str]] = {
    "UNSET": {"label": "Not Started", "color": "#bdc3c7", "shortcut": ""},
    "PASS": {"label": "Pass", "color": "#2ecc71", "shortcut": "P"},
    "FAIL": {"label": "Fail", "color": "#e74c3c", "shortcut": "F"},
    "REVIEW": {
        "label": "Needs Review",
        "color": "#f1c40f",
        "shortcut": "R",
    },
}


@dataclass
class ExportRecord:
    """Container describing a single exported deliverable."""

    base_name: str
    primary_files: list[Path] = field(default_factory=list)
    additional_files: list[Path] = field(default_factory=list)
    reports: list[Path] = field(default_factory=list)
    status: str = "UNSET"
    notes: str = ""
    last_updated: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "base_name": self.base_name,
            "status": self.status,
            "notes": self.notes,
            "last_updated": self.last_updated,
            "primary_files": [str(p) for p in self.primary_files],
            "additional_files": [str(p) for p in self.additional_files],
            "reports": [str(p) for p in self.reports],
        }


def _open_in_file_browser(path: Path) -> None:
    """Open *path* in the platform's default file browser."""

    if sys.platform.startswith("darwin"):
        subprocess.run(["open", str(path)], check=False)
    elif os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


class InclusionExclusionWindow(QMainWindow):
    """Qt window that manages the inclusion/exclusion review workflow."""

    def __init__(
        self,
        exports_dir: Optional[Path] = None,
        task_root: Optional[Path] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self.exports_dir: Optional[Path] = exports_dir
        self.task_root: Optional[Path] = task_root
        self.reports_dir: Optional[Path] = None
        self.records: list[ExportRecord] = []
        self.row_lookup: Dict[str, int] = {}
        self.unsaved_changes = False
        self.status_counts: Counter[str] = Counter()

        self.decisions_path: Optional[Path] = None
        self.decisions_csv_path: Optional[Path] = None

        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self.save_decisions)

        self._updating_notes = False
        self.current_record: Optional[ExportRecord] = None

        self.setWindowTitle("AutoClean EEG – Inclusion/Exclusion Review")
        self.setMinimumSize(1200, 720)

        self._build_ui()

        if self.exports_dir:
            self._configure_directory(self.exports_dir)

    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        central_layout = QVBoxLayout()
        central_layout.setContentsMargins(12, 12, 12, 12)
        central.setLayout(central_layout)

        # Directory controls -------------------------------------------------
        controls_row = QHBoxLayout()
        controls_row.setSpacing(8)

        self.directory_label = QLabel("No exports directory selected")
        self.directory_label.setStyleSheet(
            "font-weight:600; color:#2c3e50;" "padding:4px 8px;"
        )
        controls_row.addWidget(self.directory_label, 1)

        self.select_dir_btn = QPushButton("Choose Directory…")
        self.select_dir_btn.clicked.connect(self.select_directory)
        controls_row.addWidget(self.select_dir_btn, 0)

        self.open_dir_btn = QPushButton("Open Folder")
        self.open_dir_btn.clicked.connect(self.open_exports_folder)
        self.open_dir_btn.setEnabled(False)
        controls_row.addWidget(self.open_dir_btn, 0)

        central_layout.addLayout(controls_row)

        # Progress summary ---------------------------------------------------
        self.summary_box = QGroupBox("Progress Overview")
        summary_layout = QHBoxLayout()
        summary_layout.setContentsMargins(12, 8, 12, 8)
        self.summary_labels: Dict[str, QLabel] = {}
        for status_key in ("PASS", "FAIL", "REVIEW", "UNSET"):
            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumWidth(140)
            label.setStyleSheet("font-size:14px; font-weight:600;")
            self.summary_labels[status_key] = label
            summary_layout.addWidget(label)
        summary_layout.addStretch(1)
        self.summary_box.setLayout(summary_layout)
        central_layout.addWidget(self.summary_box)

        # Search + table -----------------------------------------------------
        splitter = QSplitter(Qt.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_panel.setLayout(left_layout)

        filter_row = QHBoxLayout()
        filter_label = QLabel("Filter exports:")
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search by basename or status…")
        self.search_box.textChanged.connect(self.apply_filter)
        filter_row.addWidget(filter_label)
        filter_row.addWidget(self.search_box)
        left_layout.addLayout(filter_row)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["#", "Recording", "Primary Files", "Status"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.itemSelectionChanged.connect(self.display_selected_record)
        self.table.setSortingEnabled(False)
        self.table.setStyleSheet(
            "QTableWidget::item { padding:6px; }"
            "QHeaderView::section { background-color:#ecf0f1; padding:6px; }"
        )
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        left_layout.addWidget(self.table)

        navigation_row = QHBoxLayout()
        self.previous_btn = QPushButton("◀ Previous")
        self.previous_btn.clicked.connect(self.goto_previous)
        self.previous_btn.setEnabled(False)
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self.goto_next)
        self.next_btn.setEnabled(False)
        navigation_row.addWidget(self.previous_btn)
        navigation_row.addWidget(self.next_btn)
        left_layout.addLayout(navigation_row)

        splitter.addWidget(left_panel)

        # Detail pane --------------------------------------------------------
        detail_panel = QWidget()
        detail_layout = QVBoxLayout()
        detail_panel.setLayout(detail_layout)

        self.record_title = QLabel("Select a recording to begin")
        self.record_title.setStyleSheet("font-size:20px; font-weight:600; color:#2c3e50;")
        detail_layout.addWidget(self.record_title)

        status_group = QGroupBox("Set classification")
        status_layout = QHBoxLayout()
        status_group.setLayout(status_layout)

        self.status_buttons: Dict[str, QPushButton] = {}
        for status_key in ("PASS", "FAIL", "REVIEW"):
            btn = QPushButton(
                f"{STATUS_STYLES[status_key]['label']} ({STATUS_STYLES[status_key]['shortcut']})"
            )
            btn.setEnabled(False)
            btn.clicked.connect(lambda _checked=False, s=status_key: self.mark_status(s))
            btn.setStyleSheet(
                f"background-color:{STATUS_STYLES[status_key]['color']};"
                "color:#1b1b1b; font-weight:600; padding:10px 14px; border-radius:6px;"
            )
            self.status_buttons[status_key] = btn
            status_layout.addWidget(btn)
        status_layout.addStretch(1)
        detail_layout.addWidget(status_group)

        notes_group = QGroupBox("Reviewer notes")
        notes_layout = QVBoxLayout()
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText("Add optional context or reminders…")
        self.notes_edit.textChanged.connect(self._handle_notes_changed)
        self.notes_edit.setEnabled(False)
        notes_layout.addWidget(self.notes_edit)
        notes_group.setLayout(notes_layout)
        detail_layout.addWidget(notes_group, 2)

        files_group = QGroupBox("Available files")
        files_layout = QHBoxLayout()

        exports_column = QVBoxLayout()
        exports_label = QLabel("Exports directory")
        exports_label.setStyleSheet("font-weight:600;")
        self.exports_list = QListWidget()
        self.exports_list.itemDoubleClicked.connect(self._open_selected_item)
        self.exports_list.setContextMenuPolicy(Qt.CustomContextMenu)
        exports_column.addWidget(exports_label)
        exports_column.addWidget(self.exports_list)

        reports_column = QVBoxLayout()
        reports_label = QLabel("Reports & QA artifacts")
        reports_label.setStyleSheet("font-weight:600;")
        self.reports_list = QListWidget()
        self.reports_list.itemDoubleClicked.connect(self._open_selected_item)
        reports_column.addWidget(reports_label)
        reports_column.addWidget(self.reports_list)

        files_layout.addLayout(exports_column)
        files_layout.addLayout(reports_column)
        files_group.setLayout(files_layout)
        detail_layout.addWidget(files_group, 3)

        button_row = QHBoxLayout()
        self.open_reports_btn = QPushButton("Open reports folder")
        self.open_reports_btn.clicked.connect(self.open_reports_folder)
        self.open_reports_btn.setEnabled(False)
        self.save_btn = QPushButton("Save summary")
        self.save_btn.clicked.connect(self.save_decisions)
        self.save_btn.setEnabled(False)
        button_row.addWidget(self.open_reports_btn)
        button_row.addStretch(1)
        button_row.addWidget(self.save_btn)
        detail_layout.addLayout(button_row)

        instructions = QLabel(
            "<b>Keyboard shortcuts</b>: "
            "<code>P</code> = Pass, <code>F</code> = Fail, <code>R</code> = Review, "
            "<code>Ctrl+S</code> = Save, <code>Alt+Right</code>/<code>Alt+Left</code> = Next/Previous"
        )
        instructions.setStyleSheet("color:#555; padding-top:8px;")
        detail_layout.addWidget(instructions)

        splitter.addWidget(detail_panel)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        central_layout.addWidget(splitter, 1)

        # Status bar ---------------------------------------------------------
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.setCentralWidget(central)

        # Global shortcuts ---------------------------------------------------
        QShortcut(QKeySequence("P"), self, activated=lambda: self.mark_status("PASS"))
        QShortcut(QKeySequence("F"), self, activated=lambda: self.mark_status("FAIL"))
        QShortcut(QKeySequence("R"), self, activated=lambda: self.mark_status("REVIEW"))
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save_decisions)
        QShortcut(QKeySequence("Alt+Right"), self, activated=self.goto_next)
        QShortcut(QKeySequence("Alt+Left"), self, activated=self.goto_previous)

    # ------------------------------------------------------------------
    # Directory handling
    # ------------------------------------------------------------------
    def _configure_directory(self, exports_dir: Path) -> None:
        exports_dir = exports_dir.resolve()
        self.exports_dir = exports_dir
        self.decisions_path = exports_dir / "autocleaneeg_exclusion_decisions.json"
        self.decisions_csv_path = exports_dir / "autocleaneeg_exclusion_summary.csv"

        if exports_dir.name == "exports":
            self.task_root = exports_dir.parent
        elif self.task_root is None:
            self.task_root = exports_dir.parent if exports_dir.parent != exports_dir else None

        potential_reports = None
        if self.task_root and (self.task_root / "reports").exists():
            potential_reports = self.task_root / "reports"
        elif exports_dir.parent != exports_dir and (exports_dir.parent / "reports").exists():
            potential_reports = exports_dir.parent / "reports"
        self.reports_dir = potential_reports

        self.directory_label.setText(str(exports_dir))
        self.open_dir_btn.setEnabled(True)
        self.open_reports_btn.setEnabled(self.reports_dir is not None)
        self.save_btn.setEnabled(True)

        self._load_records()
        self._load_saved_decisions()
        self._refresh_table()
        self.status_bar.showMessage(
            f"Loaded {len(self.records)} recordings from {exports_dir}", 5000
        )

    def select_directory(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Select exports or task directory")
        if not chosen:
            return

        selected = Path(chosen)
        if (selected / "exports").exists():
            exports_dir = selected / "exports"
            self.task_root = selected
        else:
            exports_dir = selected
            if exports_dir.name == "exports":
                self.task_root = exports_dir.parent
        self._configure_directory(exports_dir)

    def open_exports_folder(self) -> None:
        if self.exports_dir:
            _open_in_file_browser(self.exports_dir)

    def open_reports_folder(self) -> None:
        if self.reports_dir:
            _open_in_file_browser(self.reports_dir)

    # ------------------------------------------------------------------
    # Data loading and persistence
    # ------------------------------------------------------------------
    def _load_records(self) -> None:
        assert self.exports_dir is not None
        records: Dict[str, ExportRecord] = {}

        for file_path in sorted(self.exports_dir.glob("**/*")):
            if not file_path.is_file():
                continue
            base_name = file_path.stem
            record = records.setdefault(base_name, ExportRecord(base_name))
            if file_path.suffix.lower() in PRIMARY_EXTENSIONS or not record.primary_files:
                record.primary_files.append(file_path)
            else:
                record.additional_files.append(file_path)

        if self.reports_dir and self.reports_dir.exists():
            report_files = [f for f in self.reports_dir.rglob("*") if f.is_file()]
            for report_file in report_files:
                for record in records.values():
                    if record.base_name in report_file.stem:
                        record.reports.append(report_file)

        self.records = sorted(records.values(), key=lambda r: r.base_name.lower())
        self.row_lookup = {record.base_name: idx for idx, record in enumerate(self.records)}

    def _load_saved_decisions(self) -> None:
        if not self.decisions_path or not self.decisions_path.exists():
            self.unsaved_changes = False
            self.update_summary()
            return

        try:
            with self.decisions_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            QMessageBox.warning(
                self,
                "Decisions file",
                "Existing decisions file is corrupt – starting fresh.",
            )
            self.unsaved_changes = False
            self.update_summary()
            return

        stored_records = {rec["base_name"]: rec for rec in payload.get("records", [])}
        for record in self.records:
            saved = stored_records.get(record.base_name)
            if not saved:
                continue
            record.status = saved.get("status", "UNSET")
            record.notes = saved.get("notes", "")
            record.last_updated = saved.get("last_updated")

        self.unsaved_changes = False
        self.update_summary()

    def save_decisions(self) -> None:
        if not self.decisions_path:
            return
        if not self.exports_dir:
            return

        payload = {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
            "exports_directory": str(self.exports_dir),
            "task_root": str(self.task_root) if self.task_root else None,
            "records": [record.to_dict() for record in self.records],
            "summary": dict(self.status_counts),
        }

        self.decisions_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if self.decisions_csv_path:
            with self.decisions_csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["base_name", "status", "notes", "primary_files", "additional_files", "reports"])
                for record in self.records:
                    writer.writerow(
                        [
                            record.base_name,
                            record.status,
                            record.notes,
                            " | ".join(p.name for p in record.primary_files),
                            " | ".join(p.name for p in record.additional_files),
                            " | ".join(r.name for r in record.reports),
                        ]
                    )

        self.unsaved_changes = False
        self.status_bar.showMessage("Decisions saved", 3000)

    # ------------------------------------------------------------------
    # Table + detail interactions
    # ------------------------------------------------------------------
    def _refresh_table(self) -> None:
        self.table.setRowCount(len(self.records))
        for row, record in enumerate(self.records):
            row_item = QTableWidgetItem(str(row + 1))
            row_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(row, 0, row_item)

            name_item = QTableWidgetItem(record.base_name)
            name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(row, 1, name_item)

            primary_text = "\n".join(p.name for p in record.primary_files)
            primary_item = QTableWidgetItem(primary_text)
            primary_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(row, 2, primary_item)

            status_item = QTableWidgetItem(STATUS_STYLES[record.status]["label"])
            status_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(row, 3, status_item)
            self._apply_status_format(row, record.status)

        if self.records:
            self.table.selectRow(0)
            self.previous_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
        else:
            self.record_title.setText("No exports discovered in the selected folder")
            self.previous_btn.setEnabled(False)
            self.next_btn.setEnabled(False)

        self.update_summary()

    def _apply_status_format(self, row: int, status: str) -> None:
        color = QColor(STATUS_STYLES[status]["color"])
        muted = QColor(color)
        muted.setAlpha(60)
        for col in range(self.table.columnCount()):
            item = self.table.item(row, col)
            if not item:
                continue
            if col == 3:
                item.setBackground(color.lighter(140))
                item.setForeground(QColor("black"))
            else:
                item.setBackground(muted)

    def display_selected_record(self) -> None:
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            self.current_record = None
            return
        row = selected_rows[0].row()
        record = self.records[row]
        self.current_record = record

        self.record_title.setText(record.base_name)
        for btn in self.status_buttons.values():
            btn.setEnabled(True)
        self.notes_edit.setEnabled(True)

        self._updating_notes = True
        self.notes_edit.setPlainText(record.notes)
        self._updating_notes = False

        self.exports_list.clear()
        for file_path in record.primary_files + record.additional_files:
            item = QListWidgetItem(file_path.name)
            item.setData(Qt.UserRole, file_path)
            self.exports_list.addItem(item)

        self.reports_list.clear()
        for file_path in record.reports:
            item = QListWidgetItem(file_path.name)
            item.setData(Qt.UserRole, file_path)
            self.reports_list.addItem(item)

    def _open_selected_item(self, item: QListWidgetItem) -> None:
        path: Path = item.data(Qt.UserRole)
        if path:
            _open_in_file_browser(path)

    def mark_status(self, status: str) -> None:
        if not self.current_record:
            return
        if status not in STATUS_STYLES:
            return

        if self.current_record.status == status:
            return

        self.current_record.status = status
        self.current_record.last_updated = datetime.utcnow().isoformat(timespec="seconds")
        row = self.row_lookup.get(self.current_record.base_name)
        if row is not None:
            status_item = self.table.item(row, 3)
            if status_item:
                status_item.setText(STATUS_STYLES[status]["label"])
            self._apply_status_format(row, status)

        self.unsaved_changes = True
        self.update_summary()
        self.save_timer.start(1200)

    def _handle_notes_changed(self) -> None:
        if self._updating_notes:
            return
        if not self.current_record:
            return
        self.current_record.notes = self.notes_edit.toPlainText()
        self.unsaved_changes = True
        self.save_timer.start(1500)

    def apply_filter(self, text: str) -> None:
        text_lower = text.lower().strip()
        for row, record in enumerate(self.records):
            should_hide = False
            if text_lower:
                matches = (
                    text_lower in record.base_name.lower()
                    or text_lower in STATUS_STYLES[record.status]["label"].lower()
                )
                should_hide = not matches
            self.table.setRowHidden(row, should_hide)

    def goto_next(self) -> None:
        if not self.records:
            return
        current_row = self.table.currentRow()
        next_row = 0 if current_row == len(self.records) - 1 else current_row + 1
        self.table.selectRow(next_row)

    def goto_previous(self) -> None:
        if not self.records:
            return
        current_row = self.table.currentRow()
        previous_row = len(self.records) - 1 if current_row <= 0 else current_row - 1
        self.table.selectRow(previous_row)

    def update_summary(self) -> None:
        self.status_counts = Counter(record.status for record in self.records)
        total = len(self.records) or 1
        for status_key, label in self.summary_labels.items():
            count = self.status_counts.get(status_key, 0)
            percent = int((count / total) * 100)
            label.setText(f"{STATUS_STYLES[status_key]['label']}\n<b>{count}</b> ({percent}%)")
            label.setStyleSheet(
                f"font-size:14px; font-weight:600; color:{STATUS_STYLES[status_key]['color']};"
            )

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.unsaved_changes:
            self.save_decisions()
        super().closeEvent(event)


# ----------------------------------------------------------------------
# CLI helpers
# ----------------------------------------------------------------------


def discover_default_exports_dir(user_supplied: Optional[str] = None) -> tuple[Optional[Path], Optional[Path]]:
    """Best-effort discovery of the exports directory for the active task."""

    manager = UserConfigManager()
    exports_dir: Optional[Path] = None
    task_root: Optional[Path] = None

    if user_supplied:
        supplied_path = Path(user_supplied).expanduser().resolve()
        if supplied_path.is_dir():
            if supplied_path.name == "exports":
                exports_dir = supplied_path
                task_root = supplied_path.parent
            elif (supplied_path / "exports").is_dir():
                task_root = supplied_path
                exports_dir = supplied_path / "exports"
            else:
                exports_dir = supplied_path
                if (exports_dir.parent / "reports").is_dir():
                    task_root = exports_dir.parent

    if exports_dir and exports_dir.exists():
        return exports_dir, task_root

    default_output = manager.get_default_output_dir()
    active_task = manager.get_active_task()

    candidate_dirs: List[Path] = []
    if active_task:
        candidate_dirs.append(default_output / active_task)

    if default_output.exists():
        subdirs = [p for p in default_output.iterdir() if p.is_dir()]
        subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        candidate_dirs.extend(subdirs)

    for candidate in candidate_dirs:
        if not candidate.exists():
            continue
        if (candidate / "exports").is_dir():
            return candidate / "exports", candidate

    if default_output.exists():
        return default_output, default_output

    return None, None


def run_autoclean_exclusion_tool(exports_dir: Optional[Path] = None, task_root: Optional[Path] = None) -> None:
    """Entry point that creates the Qt application."""

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationDisplayName("AutoClean EEG Exclusion Tool")
    window = InclusionExclusionWindow(exports_dir=exports_dir, task_root=task_root)
    window.show()
    app.exec_()


def main(argv: Optional[Iterable[str]] = None) -> None:
    """Console entry point used by ``autocleaneeg-exclude``."""

    parser = argparse.ArgumentParser(
        description=(
            "Launch the AutoClean EEG inclusion/exclusion helper. "
            "Point to a task root or exports directory to preload recordings."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Optional task root or exports directory to review.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    exports_dir, task_root = discover_default_exports_dir(args.path)

    if exports_dir is None:
        print("Could not locate an exports directory. Launching without preload…")

    run_autoclean_exclusion_tool(exports_dir, task_root)


if __name__ == "__main__":  # pragma: no cover - manual launch helper
    main()

