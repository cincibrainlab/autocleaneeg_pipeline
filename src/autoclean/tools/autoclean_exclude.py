import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


def check_gui_dependencies() -> None:
    """Ensure optional GUI dependencies are available."""

    missing = []
    for package, import_name in [("PyQt5", "PyQt5"), ("pymupdf", "fitz")]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package)

    if missing:
        print("Error: Missing required GUI dependencies.")
        print("Install the GUI extras with: pip install autocleaneeg-pipeline[gui]")
        print(f"\nMissing packages: {', '.join(sorted(set(missing)))}")
        sys.exit(1)


check_gui_dependencies()

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QShortcut,
    QSplitter,
    QStatusBar,
    QStyle,
    QVBoxLayout,
    QWidget,
)

import fitz

try:  # pragma: no cover - convenience for interactive sessions
    from PyQt5.QtCore import pyqtRemoveInputHook
except ImportError:  # pragma: no cover - available on most platforms
    pyqtRemoveInputHook = None

if pyqtRemoveInputHook is not None:  # pragma: no cover - best effort only
    try:
        pyqtRemoveInputHook()
    except Exception:
        pass


EXPORT_EXTENSIONS = {
    ".set",
    ".fif",
    ".edf",
    ".bdf",
    ".vhdr",
    ".eeg",
    ".mat",
    ".tsv",
    ".csv",
}

ATTACHMENT_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".json",
    ".html",
    ".htm",
    ".txt",
    ".xlsx",
}

SKIP_DIRECTORIES = {
    ".git",
    "__pycache__",
    "env",
    "venv",
    ".venv",
    "node_modules",
    "build",
    "dist",
    "site-packages",
}

STATUS_ORDER = ["PASS", "FAIL", "REVIEW"]
STATUS_STYLES: Dict[str, Tuple[str, str]] = {
    "PASS": ("#2ecc71", "#ffffff"),
    "FAIL": ("#e74c3c", "#ffffff"),
    "REVIEW": ("#f1c40f", "#2c3e50"),
}
STATUS_LABELS = {
    "PASS": "Pass",
    "FAIL": "Fail",
    "REVIEW": "Review",
}


@dataclass
class ExportRecord:
    basename: str
    export_path: Path
    attachments: List[Path] = field(default_factory=list)
    status: str = "REVIEW"
    notes: str = ""


class AutocleanExclusionTool(QWidget):
    """Qt helper that tallies pass/fail/review decisions for exported files."""

    def __init__(self, root_path: Union[str, Path]):
        super().__init__()
        self.root_path = Path(root_path).expanduser().resolve()
        self.task_dir, self.exports_dir = self._discover_directories(self.root_path)
        self.status_file = self.root_path / "autoclean_exclusion_decisions.csv"

        self.records: Dict[str, ExportRecord] = {}
        self.record_order: List[str] = []
        self.attachment_index: Dict[str, List[Path]] = {}
        self.current_record_key: Optional[str] = None

        self.note_timer = QTimer(self)
        self.note_timer.setSingleShot(True)
        self.note_timer.timeout.connect(self._commit_note_from_timer)
        self._note_updates_blocked = False

        self._build_ui()
        self._connect_shortcuts()
        self.refresh_records(initial=True)

    # ------------------------------------------------------------------
    # UI setup helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.setWindowTitle("Autoclean Exclusion Review")
        self.setMinimumSize(1100, 700)
        self.setWindowState(Qt.WindowMaximized)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(12, 12, 12, 12)

        # Top controls ----------------------------------------------------
        top_bar = QHBoxLayout()
        self.select_dir_btn = QPushButton("Select Directory…")
        self.select_dir_btn.clicked.connect(self.select_directory)
        top_bar.addWidget(self.select_dir_btn)

        self.open_task_btn = QPushButton("Open Task Folder")
        self.open_task_btn.clicked.connect(lambda: self._open_path(self.task_dir))
        top_bar.addWidget(self.open_task_btn)

        self.open_exports_btn = QPushButton("Open Exports Folder")
        self.open_exports_btn.clicked.connect(lambda: self._open_path(self.exports_dir))
        top_bar.addWidget(self.open_exports_btn)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_records)
        self.refresh_btn.setShortcut("F5")
        self.refresh_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        top_bar.addWidget(self.refresh_btn)

        top_bar.addStretch(1)

        self.root_label = QLabel(str(self.root_path))
        self.root_label.setWordWrap(True)
        self.root_label.setStyleSheet("font-weight: bold;")
        top_bar.addWidget(self.root_label)

        main_layout.addLayout(top_bar)

        # Summary ---------------------------------------------------------
        summary_layout = QHBoxLayout()
        self.summary_label = QLabel()
        self.summary_label.setFont(QFont("Helvetica", 11, QFont.Bold))
        summary_layout.addWidget(self.summary_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(260)
        self.progress_bar.setFormat("0/0 decided")
        summary_layout.addWidget(self.progress_bar)
        summary_layout.addStretch(1)

        self.position_label = QLabel("No selection")
        summary_layout.addWidget(self.position_label)

        main_layout.addLayout(summary_layout)

        # Splitter --------------------------------------------------------
        self.splitter = QSplitter(Qt.Horizontal)

        # Left pane - export list
        left_container = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)

        list_label = QLabel("Exports to review")
        list_label.setFont(QFont("Helvetica", 12, QFont.Bold))
        left_layout.addWidget(list_label)

        self.record_list = QListWidget()
        self.record_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.record_list.setAlternatingRowColors(True)
        self.record_list.setUniformItemSizes(True)
        self.record_list.currentItemChanged.connect(self._on_record_changed)
        self.record_list.itemDoubleClicked.connect(self.open_selected_export)
        left_layout.addWidget(self.record_list, 1)

        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Previous (Shift+Tab)")
        self.prev_btn.clicked.connect(self.go_previous)
        nav_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next (Tab) ▶")
        self.next_btn.clicked.connect(self.go_next)
        nav_layout.addWidget(self.next_btn)

        left_layout.addLayout(nav_layout)
        left_container.setLayout(left_layout)
        self.splitter.addWidget(left_container)

        # Middle pane - decision + notes
        middle_container = QWidget()
        middle_layout = QVBoxLayout()
        middle_layout.setContentsMargins(6, 0, 6, 0)

        decision_box = QGroupBox("Review decision")
        decision_layout = QVBoxLayout()

        button_row = QHBoxLayout()
        self.pass_btn = QPushButton("Mark PASS (P)")
        self.pass_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
        self.pass_btn.clicked.connect(lambda: self.set_status("PASS"))
        button_row.addWidget(self.pass_btn)

        self.fail_btn = QPushButton("Mark FAIL (F)")
        self.fail_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogCancelButton))
        self.fail_btn.clicked.connect(lambda: self.set_status("FAIL"))
        button_row.addWidget(self.fail_btn)

        self.review_btn = QPushButton("Mark REVIEW (R)")
        self.review_btn.setIcon(
            self.style().standardIcon(QStyle.SP_MessageBoxInformation)
        )
        self.review_btn.clicked.connect(lambda: self.set_status("REVIEW"))
        button_row.addWidget(self.review_btn)

        decision_layout.addLayout(button_row)

        self.open_export_btn = QPushButton("Open export file (Enter)")
        self.open_export_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.open_export_btn.clicked.connect(self.open_selected_export)
        decision_layout.addWidget(self.open_export_btn)

        shortcuts_label = QLabel(
            """
            <ul>
                <li><b>P</b> → Pass, <b>F</b> → Fail, <b>R</b> → Review</li>
                <li><b>Tab</b>/<b>Shift+Tab</b> → Navigate between exports</li>
                <li><b>F5</b> → Rescan directories</li>
                <li>Double-click or press <b>Enter</b> to open the export file</li>
            </ul>
            """
        )
        shortcuts_label.setWordWrap(True)
        decision_layout.addWidget(shortcuts_label)

        decision_box.setLayout(decision_layout)
        middle_layout.addWidget(decision_box)

        notes_box = QGroupBox("Reviewer notes")
        notes_layout = QVBoxLayout()
        self.note_edit = QPlainTextEdit()
        self.note_edit.setPlaceholderText("Optional notes for the selected export…")
        self.note_edit.textChanged.connect(self._on_note_changed)
        notes_layout.addWidget(self.note_edit)
        notes_box.setLayout(notes_layout)
        middle_layout.addWidget(notes_box, 1)

        middle_container.setLayout(middle_layout)
        self.splitter.addWidget(middle_container)

        # Right pane - attachments & preview
        right_container = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)

        attachments_box = QGroupBox("Related reports & artifacts")
        attachments_layout = QVBoxLayout()
        self.attachments_list = QListWidget()
        self.attachments_list.itemDoubleClicked.connect(self.open_attachment)
        self.attachments_list.currentItemChanged.connect(self._on_attachment_selected)
        attachments_layout.addWidget(self.attachments_list)
        attachments_box.setLayout(attachments_layout)
        right_layout.addWidget(attachments_box, 1)

        preview_box = QGroupBox("Preview")
        preview_layout = QVBoxLayout()
        self.preview_area = QScrollArea()
        self.preview_area.setWidgetResizable(True)
        self._show_preview_message("Select an attachment to preview PDFs or images.")
        preview_layout.addWidget(self.preview_area)
        preview_box.setLayout(preview_layout)
        right_layout.addWidget(preview_box, 2)

        right_container.setLayout(right_layout)
        self.splitter.addWidget(right_container)

        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setStretchFactor(2, 3)

        main_layout.addWidget(self.splitter)

        self.status_bar = QStatusBar()
        main_layout.addWidget(self.status_bar)

        self.setLayout(main_layout)
        self.update_status_bar()

    def _connect_shortcuts(self) -> None:
        QShortcut(QKeySequence("P"), self, activated=lambda: self.set_status("PASS"))
        QShortcut(QKeySequence("F"), self, activated=lambda: self.set_status("FAIL"))
        QShortcut(QKeySequence("R"), self, activated=lambda: self.set_status("REVIEW"))
        QShortcut(QKeySequence(Qt.Key_Tab), self, activated=self.go_next)
        QShortcut(QKeySequence(Qt.Key_Backtab), self, activated=self.go_previous)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save_statuses)

    # ------------------------------------------------------------------
    # Directory discovery & scanning
    # ------------------------------------------------------------------
    @staticmethod
    def _discover_directories(root_path: Path) -> Tuple[Path, Path]:
        """Best-effort discovery of task and exports directories."""

        root_path = root_path.expanduser().resolve()

        if root_path.name.lower() == "exports":
            return root_path.parent, root_path

        direct_exports = root_path / "exports"
        if direct_exports.is_dir():
            return root_path, direct_exports

        for child in root_path.iterdir():
            if not child.is_dir():
                continue
            candidate = child / "exports"
            if candidate.is_dir():
                return child, candidate

        for current_root, dirs, _ in os.walk(root_path):
            rel_depth = len(Path(current_root).relative_to(root_path).parts)
            if rel_depth > 3:
                dirs[:] = []
                continue
            if "exports" in dirs:
                exports_dir = Path(current_root) / "exports"
                return Path(current_root), exports_dir

        return root_path, root_path

    def refresh_records(self, initial: bool = False) -> None:
        if not initial:
            self.status_bar.showMessage("Refreshing file list…", 2000)

        self.records.clear()
        self.record_order.clear()
        self.attachment_index = self._build_attachment_index(self.root_path)

        if self.exports_dir.is_dir():
            for export_path in sorted(self.exports_dir.rglob("*")):
                if export_path.is_dir():
                    continue
                if export_path.suffix.lower() not in EXPORT_EXTENSIONS:
                    continue
                key = str(export_path.relative_to(self.root_path))
                basename = export_path.stem
                attachments = self._resolve_attachments(basename, export_path)
                self.records[key] = ExportRecord(
                    basename=basename,
                    export_path=export_path,
                    attachments=attachments,
                )
                self.record_order.append(key)

        self._load_statuses()
        self._populate_record_list()
        self.update_summary()
        self.update_status_bar()

        if not self.records:
            self.status_bar.showMessage(
                "No exports were found – choose a directory with processed files.",
                5000,
            )

    def _build_attachment_index(self, root: Path) -> Dict[str, List[Path]]:
        index: Dict[str, List[Path]] = defaultdict(list)
        if not root.exists():
            return index

        for current_root, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRECTORIES]
            for filename in files:
                path = Path(current_root) / filename
                suffix = path.suffix.lower()
                if suffix not in EXPORT_EXTENSIONS and suffix not in ATTACHMENT_EXTENSIONS:
                    continue
                index[path.stem].append(path)
        return index

    def _resolve_attachments(self, basename: str, export_path: Path) -> List[Path]:
        candidates = self.attachment_index.get(basename, [])
        cleaned: List[Path] = []
        seen = set()
        for candidate in sorted(
            candidates,
            key=lambda p: (p.parent != export_path.parent, str(p)),
        ):
            if candidate == export_path:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            cleaned.append(candidate)
        return cleaned

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load_statuses(self) -> None:
        if not self.status_file.exists():
            return
        try:
            with self.status_file.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    key = row.get("export_path")
                    if not key or key not in self.records:
                        continue
                    record = self.records[key]
                    status = row.get("status", "REVIEW").upper()
                    if status in STATUS_ORDER:
                        record.status = status
                    record.notes = row.get("notes", "")
        except Exception as exc:  # pragma: no cover - defensive IO
            self.status_bar.showMessage(f"Failed to read decisions file: {exc}", 8000)

    def save_statuses(self, quiet: bool = False) -> None:
        if not self.records:
            return
        try:
            with self.status_file.open("w", encoding="utf-8", newline="") as handle:
                fieldnames = ["basename", "status", "export_path", "notes"]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for key in self.record_order:
                    record = self.records[key]
                    writer.writerow(
                        {
                            "basename": record.basename,
                            "status": record.status,
                            "export_path": key,
                            "notes": record.notes,
                        }
                    )
            if not quiet:
                self.status_bar.showMessage(
                    f"Decisions saved to {self.status_file.name}",
                    2500,
                )
        except Exception as exc:  # pragma: no cover - defensive IO
            QMessageBox.warning(self, "Save failed", str(exc))

    # ------------------------------------------------------------------
    # Record list interactions
    # ------------------------------------------------------------------
    def _populate_record_list(self) -> None:
        self.record_list.blockSignals(True)
        self.record_list.clear()
        for index, key in enumerate(self.record_order, start=1):
            record = self.records[key]
            item = QListWidgetItem(self._format_record_label(index, record))
            item.setData(Qt.UserRole, key)
            item.setToolTip(str(record.export_path))
            self._apply_status_style(item, record.status)
            self.record_list.addItem(item)
        self.record_list.blockSignals(False)

        if self.record_order:
            self.record_list.setCurrentRow(0)
            self.record_list.setFocus()
        else:
            self._on_record_changed(None, None)

    def _format_record_label(self, position: int, record: ExportRecord) -> str:
        return f"{position:02d}. [{STATUS_LABELS[record.status]}] {record.basename}"

    def _apply_status_style(self, item: QListWidgetItem, status: str) -> None:
        background, foreground = STATUS_STYLES.get(status, ("#bdc3c7", "#2c3e50"))
        item.setBackground(QColor(background))
        item.setForeground(QColor(foreground))
        font = item.font()
        font.setBold(status != "REVIEW")
        item.setFont(font)

    def _on_record_changed(
        self, current: Optional[QListWidgetItem], previous: Optional[QListWidgetItem]
    ) -> None:
        if previous is not None:
            prev_key = previous.data(Qt.UserRole)
            if prev_key:
                self._commit_note_immediately(prev_key)

        if current is None:
            self.current_record_key = None
            self.note_edit.setPlainText("")
            self.note_edit.setEnabled(False)
            self.pass_btn.setEnabled(False)
            self.fail_btn.setEnabled(False)
            self.review_btn.setEnabled(False)
            self.open_export_btn.setEnabled(False)
            self.attachments_list.clear()
            self.attachments_list.setEnabled(False)
            self._show_preview_message("Select an export to start reviewing.")
            self.position_label.setText("No selection")
            return

        key = current.data(Qt.UserRole)
        self.current_record_key = key
        record = self.records.get(key)
        if not record:
            return

        index = self.record_order.index(key) + 1
        total = len(self.record_order)
        self.position_label.setText(f"Viewing {index}/{total}")

        self._note_updates_blocked = True
        self.note_edit.setPlainText(record.notes)
        self._note_updates_blocked = False
        self.note_edit.setEnabled(True)

        self.pass_btn.setEnabled(True)
        self.fail_btn.setEnabled(True)
        self.review_btn.setEnabled(True)
        self.open_export_btn.setEnabled(record.export_path.exists())

        self.attachments_list.blockSignals(True)
        self.attachments_list.clear()
        if record.attachments:
            for attachment in record.attachments:
                item = QListWidgetItem(attachment.name)
                item.setData(Qt.UserRole, attachment)
                item.setToolTip(str(attachment))
                self.attachments_list.addItem(item)
            self.attachments_list.setEnabled(True)
            self.attachments_list.blockSignals(False)
            self.attachments_list.setCurrentRow(0)
        else:
            placeholder = QListWidgetItem("No related reports found.")
            placeholder.setFlags(Qt.NoItemFlags)
            self.attachments_list.addItem(placeholder)
            self.attachments_list.setEnabled(False)
            self.attachments_list.blockSignals(False)
            self._show_preview_message(
                "No related reports were located for this export."
            )
            return

        self.attachments_list.blockSignals(False)
        first_attachment = self.attachments_list.currentItem()
        if first_attachment is not None:
            self._on_attachment_selected(first_attachment, None)

    def _commit_note_immediately(self, key: str) -> None:
        if self._note_updates_blocked:
            return
        record = self.records.get(key)
        if not record:
            return
        record.notes = self.note_edit.toPlainText()
        self.save_statuses(quiet=True)

    def _on_note_changed(self) -> None:
        if self._note_updates_blocked or self.current_record_key is None:
            return
        self.note_timer.start(600)

    def _commit_note_from_timer(self) -> None:
        if self.current_record_key is None:
            return
        record = self.records.get(self.current_record_key)
        if not record:
            return
        record.notes = self.note_edit.toPlainText()
        self.save_statuses(quiet=True)

    # ------------------------------------------------------------------
    # Attachments & preview
    # ------------------------------------------------------------------
    def _show_preview_message(self, message: str) -> None:
        label = QLabel(message)
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        self.preview_area.takeWidget()
        self.preview_area.setWidget(label)

    def _on_attachment_selected(
        self, current: Optional[QListWidgetItem], _: Optional[QListWidgetItem]
    ) -> None:
        if current is None:
            self._show_preview_message(
                "Select an attachment to preview PDFs or images."
            )
            return
        path = current.data(Qt.UserRole)
        if not isinstance(path, Path):
            self._show_preview_message(
                "Preview not available. Double-click to open externally."
            )
            return
        self._render_attachment_preview(path)

    def _render_attachment_preview(self, path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".svg"}:
            pixmap = QPixmap(str(path))
            if pixmap.isNull():
                self._show_preview_message(f"Unable to display {path.name}.")
                return
            if pixmap.width() > 1280:
                pixmap = pixmap.scaledToWidth(1280, Qt.SmoothTransformation)
            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            label.setPixmap(pixmap)
            self.preview_area.takeWidget()
            self.preview_area.setWidget(label)
            return

        if suffix == ".pdf":
            try:
                with fitz.open(str(path)) as doc:
                    if doc.page_count == 0:
                        self._show_preview_message(f"{path.name} is empty.")
                        return
                    page = doc.load_page(0)
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            except Exception as exc:  # pragma: no cover - preview best-effort
                self._show_preview_message(f"Unable to preview PDF: {exc}")
                return
            image = QImage(
                pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGBA8888
            )
            pixmap = QPixmap.fromImage(image)
            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            label.setPixmap(pixmap)
            self.preview_area.takeWidget()
            self.preview_area.setWidget(label)
            return

        self._show_preview_message(
            "Preview not available. Double-click to open in the default application."
        )

    def open_attachment(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        if isinstance(path, Path):
            self._open_path(path)

    # ------------------------------------------------------------------
    # Decision helpers & navigation
    # ------------------------------------------------------------------
    def set_status(self, status: str) -> None:
        if self.current_record_key is None:
            return
        record = self.records.get(self.current_record_key)
        if not record or record.status == status:
            return
        record.status = status
        current_row = self.record_list.currentRow()
        current_item = self.record_list.currentItem()
        if current_item is not None:
            current_item.setText(
                self._format_record_label(current_row + 1, record)
            )
            self._apply_status_style(current_item, status)
        self.update_summary()
        self.save_statuses()

    def update_summary(self) -> None:
        total = len(self.record_order)
        counts = {status: 0 for status in STATUS_ORDER}
        for record in self.records.values():
            counts[record.status] += 1
        decided = counts["PASS"] + counts["FAIL"]
        self.summary_label.setText(
            " | ".join(
                [
                    f"Total: {total}",
                    f"Pass: {counts['PASS']}",
                    f"Fail: {counts['FAIL']}",
                    f"Review: {counts['REVIEW']}",
                ]
            )
        )
        if total:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(decided)
            self.progress_bar.setFormat(f"{decided}/{total} decided")
        else:
            self.progress_bar.setMaximum(1)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("No exports found")

    def update_status_bar(self) -> None:
        task_text = str(self.task_dir) if self.task_dir.exists() else "<missing>"
        export_text = (
            str(self.exports_dir) if self.exports_dir.exists() else "<missing>"
        )
        message = (
            f"Root: {self.root_path} | Task: {task_text} | Exports: {export_text}"
        )
        self.status_bar.showMessage(message)
        self.open_task_btn.setEnabled(self.task_dir.exists())
        self.open_exports_btn.setEnabled(self.exports_dir.exists())

    def go_next(self) -> None:
        if not self.record_order:
            return
        row = self.record_list.currentRow()
        if row < self.record_list.count() - 1:
            self.record_list.setCurrentRow(row + 1)

    def go_previous(self) -> None:
        if not self.record_order:
            return
        row = self.record_list.currentRow()
        if row > 0:
            self.record_list.setCurrentRow(row - 1)

    def open_selected_export(self, item: Optional[QListWidgetItem] = None) -> None:
        if item is None:
            item = self.record_list.currentItem()
        if item is None:
            return
        key = item.data(Qt.UserRole)
        record = self.records.get(key)
        if not record:
            return
        self._open_path(record.export_path)

    # ------------------------------------------------------------------
    # Misc utilities
    # ------------------------------------------------------------------
    def select_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Autoclean directory",
            str(self.root_path),
        )
        if not directory:
            return
        self.root_path = Path(directory).expanduser().resolve()
        self.task_dir, self.exports_dir = self._discover_directories(self.root_path)
        self.status_file = self.root_path / "autoclean_exclusion_decisions.csv"
        self.root_label.setText(str(self.root_path))
        self.refresh_records()

    def _open_path(self, path: Union[str, Path]) -> None:
        if not path:
            return
        candidate = Path(path)
        if not candidate.exists():
            QMessageBox.warning(self, "Path not found", f"{candidate} does not exist.")
            return
        if sys.platform.startswith("darwin"):
            subprocess.run(["open", str(candidate)], check=False)
        elif os.name == "nt":  # pragma: no cover - Windows specific
            os.startfile(str(candidate))  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(candidate)], check=False)


def run_autoclean_exclude(autoclean_dir: Optional[Union[str, Path]] = None):
    """Launch the exclusion helper. Returns the widget when embedding."""

    target_dir = Path(autoclean_dir or Path.cwd()).expanduser().resolve()
    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication(sys.argv)
        owns_app = True
    app.setStyleSheet("")

    window = AutocleanExclusionTool(target_dir)
    window.show()

    if owns_app:
        return app.exec_()
    return window


def run_autoclean_exclude_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Review exports and mark pass/fail/review decisions.",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=None,
        help="Autoclean output directory to load (default: current directory)",
    )
    args = parser.parse_args()

    result = run_autoclean_exclude(args.directory)
    if isinstance(result, int):
        sys.exit(result)


if __name__ == "__main__":
    run_autoclean_exclude_cli()
