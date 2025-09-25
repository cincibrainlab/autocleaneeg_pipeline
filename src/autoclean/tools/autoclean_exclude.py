"""AutoClean EEG inclusion/exclusion review helper.

This tool extends the classic :mod:`autoclean.tools.autoclean_review` window
so reviewers can keep the familiar full-screen MNE browser while tracking
Pass/Fail/Review decisions and notes for every exported ``.set`` file.  The
script keeps everything self-contained to make distribution easy for labs that
copy the helper into bespoke environments.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import subprocess
import sys
from collections import Counter, OrderedDict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, Optional, List, Tuple


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
from PyQt5.QtGui import QColor, QKeySequence, QPalette  # noqa: E402
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
    QTabWidget,
    QTextEdit,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QStyle,
)

try:  # noqa: E402
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover - matplotlib optional for metrics
    FigureCanvas = None  # type: ignore
    Figure = None  # type: ignore

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


def _coerce_list(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass
    if ";" in text:
        return [part.strip() for part in text.split(";") if part.strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def _safe_int(value: Optional[str]) -> int:
    try:
        if value is None:
            return 0
        text = str(value).strip()
        if not text:
            return 0
        return int(float(text))
    except (ValueError, TypeError):
        return 0


def _longest_common_prefix_length(a: str, b: str) -> int:
    length = 0
    for char_a, char_b in zip(a, b):
        if char_a != char_b:
            break
        length += 1
    return length


class ProcessingMetricsWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        self.setLayout(layout)

        self.message_label = QLabel("")
        self.message_label.setObjectName("decisionMetricsMessage")
        self.message_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.message_label)

        if Figure is None or FigureCanvas is None:
            self.figure = None
            self.canvas = None
            self.message_label.setText("Matplotlib is required to display processing metrics.")
            self.metrics = []
            return

        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setObjectName("decisionMetricsCanvas")
        layout.addWidget(self.canvas)

        self.metrics: List[Tuple[str, Dict[str, int]]] = []
        self._render_no_data("No processing metrics available.")

    def show_message(self, message: str) -> None:
        self._render_no_data(message)

    def _render_no_data(self, message: str) -> None:
        if self.figure is not None and self.canvas is not None:
            self.figure.clear()
            self.canvas.draw()
            self.canvas.hide()
        self.message_label.setText(message)
        self.message_label.show()

    def update_metrics(self, metrics: List[Tuple[str, Dict[str, int]]]) -> None:
        if self.figure is None or self.canvas is None:
            return

        if not metrics:
            self._render_no_data("No processing metrics available.")
            return

        metrics = metrics[:4]

        has_data = any(counter for _, counter in metrics if sum(counter.values()) > 0)
        if not has_data:
            self._render_no_data("No processing metrics available.")
            return

        self.message_label.hide()
        self.canvas.show()
        self.figure.clear()
        axes = self.figure.subplots(2, 2).flatten()
        metrics = metrics + [("", {})] * (len(axes) - len(metrics))

        palette = [
            "#264653",
            "#2a9d8f",
            "#e9c46a",
            "#f4a261",
            "#e76f51",
            "#8ab9ff",
            "#b8c1ec",
        ]

        for ax, (title, counter) in zip(axes, metrics):
            total = sum(counter.values())
            if total == 0:
                ax.axis("off")
                if title:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10)
                    ax.set_title(title, fontsize=11, fontweight="bold")
                continue

            labels = list(counter.keys())
            sizes = list(counter.values())
            colors = palette[: len(labels)]
            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels,
                autopct="%1.0f%%",
                startangle=90,
                colors=colors,
                textprops={"fontsize": 9},
            )
            for text in autotexts:
                text.set_fontsize(9)
            ax.axis("equal")
            if title:
                ax.set_title(title, fontsize=11, fontweight="bold")

        self.canvas.draw()



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
        self.summary_chip_labels: Dict[str, QLabel] = {}
        self.notes_edit: Optional[QTextEdit] = None
        self.related_list: Optional[QListWidget] = None
        self.detail_panel: Optional[QFrame] = None
        self.save_timer: Optional[QTimer] = None
        self._status_buttons: dict[str, QPushButton] = {}
        self._clear_button: Optional[QPushButton] = None
        self._decision_stack: Optional[QStackedLayout] = None
        self._current_plot_path: Optional[str] = None
        self._workspace_path_label: Optional[QLabel] = None
        self.metrics_widget: Optional[ProcessingMetricsWidget] = None

        self._updating_notes = False
        self._suppress_selection_autoload = False
        self._plot_in_progress = False
        self._pending_plot_refresh = False
        self._pending_selection_item: Optional[QTreeWidgetItem] = None
        self._selection_timer: Optional[QTimer] = None

        super().__init__(
            str(self.exports_dir) if self.exports_dir is not None else None
        )

        # Base ``__init__`` calls ``loadFiles`` once; run our extensions after.
        self._extend_ui()
        # Ensure file tree uses a consistent light theme even when expanded
        self._apply_file_tree_theme()
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

        if hasattr(self, "plot_btn"):
            try:
                self.left_layout.removeWidget(self.plot_btn)
            except Exception:
                pass
            self.plot_btn.deleteLater()
            self.plot_btn = None

        for attr in ("view_record_btn", "close_plot_btn", "exit_btn"):
            btn = getattr(self, attr, None)
            if btn is None:
                continue
            try:
                btn.hide()
                parent = btn.parentWidget()
                if parent is not None and parent.layout() is not None:
                    parent.layout().removeWidget(btn)
                btn.setParent(self)
            except Exception:
                pass

        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.setInterval(400)
        self.save_timer.timeout.connect(self._commit_decisions)

        self._selection_timer = QTimer(self)
        self._selection_timer.setSingleShot(True)
        self._selection_timer.setInterval(180)
        self._selection_timer.timeout.connect(self._process_pending_selection)

        if hasattr(self, "file_tree") and self.file_tree is not None:
            try:
                self.file_tree.itemSelectionChanged.connect(self._handle_tree_selection_changed)
            except Exception:
                pass

        decision_card = QFrame()
        decision_card.setObjectName("decisionCard")
        decision_layout = QVBoxLayout()
        decision_layout.setContentsMargins(12, 12, 12, 12)
        decision_layout.setSpacing(8)
        decision_card.setLayout(decision_layout)

        header_row = QHBoxLayout()
        header_row.setSpacing(6)

        header_label = QLabel("Review Decision")
        header_label.setObjectName("decisionHeader")
        header_row.addWidget(header_label)

        shortcut_hint = QLabel("P Pass • F Fail • R Review • C Clear")
        shortcut_hint.setObjectName("decisionShortcutHint")
        shortcut_hint.setAlignment(Qt.AlignCenter)
        header_row.addWidget(shortcut_hint, 0, Qt.AlignRight)
        header_row.addSpacing(4)

        self.status_label = QLabel("Not Started")
        self.status_label.setObjectName("decisionStatusChip")
        self.status_label.setAlignment(Qt.AlignCenter)
        header_row.addWidget(self.status_label)
        decision_layout.addLayout(header_row)

        self.current_file_label = QLabel("No file selected")
        self.current_file_label.setObjectName("decisionFileLabel")
        self.current_file_label.setWordWrap(True)
        decision_layout.addWidget(self.current_file_label)

        button_panel = QWidget()
        button_panel.setObjectName("decisionButtonPanel")
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(6)
        button_panel.setLayout(button_layout)

        self._shortcuts: dict[str, QShortcut] = {}
        self._status_buttons.clear()
        for status in ("PASS", "FAIL", "REVIEW"):
            meta = STATUS_DEFINITIONS[status]
            btn = QPushButton(meta["label"])
            btn.setCursor(Qt.PointingHandCursor)
            btn.setCheckable(True)
            btn.setMinimumHeight(28)
            btn.setMinimumWidth(0)
            btn.setToolTip(
                f"Mark the selection as {meta['label']}. Shortcut: {meta['shortcut']}"
            )
            btn.clicked.connect(partial(self._set_status, status))
            button_layout.addWidget(btn)
            self._status_buttons[status] = btn

            shortcut = QShortcut(QKeySequence(meta["shortcut"]), self)
            shortcut.activated.connect(partial(self._set_status, status))
            self._shortcuts[status] = shortcut

        button_layout.addSpacing(2)

        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("decisionClearButton")
        clear_btn.setCursor(Qt.PointingHandCursor)
        clear_btn.setMinimumHeight(28)
        clear_btn.setFixedWidth(64)
        clear_btn.setToolTip("Reset decision to Not Started. Shortcut: C")
        clear_btn.clicked.connect(partial(self._set_status, "UNSET"))
        self._clear_button = clear_btn
        button_layout.addWidget(clear_btn)

        clear_shortcut = QShortcut(QKeySequence("C"), self)
        clear_shortcut.activated.connect(partial(self._set_status, "UNSET"))
        self._shortcuts["CLEAR"] = clear_shortcut

        self.save_state_label = QLabel("Select a file to assign a decision.")
        self.save_state_label.setObjectName("decisionSaveLabel")
        self.save_state_label.setAlignment(Qt.AlignLeft)

        actions_widget = QWidget()
        actions_layout = QVBoxLayout()
        actions_layout.setContentsMargins(0, 2, 0, 0)
        actions_layout.setSpacing(6)
        actions_widget.setLayout(actions_layout)
        actions_layout.addWidget(button_panel)

        summary_panel = QWidget()
        summary_panel.setObjectName("decisionSummaryPanel")
        summary_layout = QVBoxLayout()
        summary_layout.setContentsMargins(0, 4, 0, 0)
        summary_layout.setSpacing(2)
        summary_panel.setLayout(summary_layout)

        self.summary_chip_labels = {}
        for status in STATUS_ORDER:
            meta = STATUS_DEFINITIONS[status]
            color_hex = STATUS_DEFINITIONS[status]["color"]

            row = QFrame()
            row.setObjectName("decisionSummaryRow")
            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(6, 4, 6, 4)
            row_layout.setSpacing(8)
            row.setLayout(row_layout)

            indicator = QFrame()
            indicator.setFixedWidth(3)
            indicator.setObjectName("decisionSummaryBar")
            indicator.setStyleSheet(
                f"background-color: {color_hex}; border-radius: 2px;"
            )
            row_layout.addWidget(indicator)

            name_label = QLabel(meta["label"])
            name_label.setObjectName("decisionSummaryName")
            row_layout.addWidget(name_label)

            row_layout.addStretch(1)

            count_label = QLabel("0")
            count_label.setObjectName("decisionSummaryCount")
            count_label.setStyleSheet(f"color: {color_hex};")
            row_layout.addWidget(count_label, 0, Qt.AlignRight)

            summary_layout.addWidget(row)
            self.summary_chip_labels[status] = count_label

        actions_layout.addWidget(summary_panel)
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
                border: 1px solid #d3deea;
                border-radius: 10px;
            }
            #decisionHeader {
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.7px;
                color: #425466;
                font-weight: 700;
            }
            #decisionShortcutHint {
                color: #7a8ca3;
                font-size: 11px;
                letter-spacing: 0.3px;
            }
            #decisionFileLabel {
                color: #1d2939;
                font-weight: 500;
                font-size: 12px;
            }
            #decisionStatusChip {
                padding: 2px 10px;
                border-radius: 12px;
                font-weight: 600;
                background-color: #ecf2ff;
                color: #1a4fa3;
            }
            #decisionButtonPanel QPushButton {
                background-color: #eef3ff;
                border: 1px solid #c9d6ff;
                border-radius: 6px;
                padding: 4px 12px;
                font-weight: 600;
                color: #1f2d3d;
            }
            #decisionButtonPanel QPushButton:hover {
                border-color: #93aaff;
                color: #1a4fa3;
            }
            #decisionButtonPanel QPushButton:checked {
                background-color: #1a4fa3;
                border-color: #1a4fa3;
                color: #ffffff;
            }
            #decisionButtonPanel QPushButton:disabled {
                background-color: #f1f4fb;
                color: #98a4b5;
                border-color: #d7deeb;
            }
            #decisionClearButton {
                background-color: #ffffff;
                border: 1px solid #d9e2ec;
                color: #5b6c7c;
            }
            #decisionClearButton:hover {
                border-color: #a5b4c7;
                color: #2f3d4f;
            }
            #decisionSaveLabel {
                color: #738194;
                font-style: italic;
                font-size: 11px;
            }
            #decisionSummaryPanel,
            #decisionSummaryPanelLarge {
                background-color: transparent;
            }
            #decisionSummaryRow {
                background-color: #f7f9ff;
                border: 1px solid #e1e8f5;
                border-radius: 6px;
            }
            #decisionSummaryRowLarge {
                background-color: #ffffff;
                border: 1px solid #dbe3f3;
                border-radius: 8px;
            }
            #decisionSummaryName,
            #decisionSummaryNameLarge {
                color: #4b5567;
                font-size: 12px;
                font-weight: 600;
            }
            #decisionSummaryCount {
                font-size: 12px;
                font-weight: 700;
            }
            #decisionSummaryCountLarge {
                font-size: 16px;
                font-weight: 700;
            }
            #decisionInfoPanel {
                background-color: transparent;
                border: none;
            }
            #decisionInfoPanel QLabel {
                color: #1f2933;
                font-size: 12px;
            }
            #decisionInfoPanel QTextEdit {
                background-color: #ffffff;
                color: #1f2933;
                border: 1px solid #d7e0ed;
                border-radius: 6px;
            }
            #decisionInfoPanel QListWidget {
                background-color: #ffffff;
                color: #1f2933;
                border: 1px solid #d7e0ed;
                border-radius: 6px;
            }
            #decisionSummaryGroup {
                background-color: #f8fafc;
                border: 1px solid #dde4ef;
                border-radius: 6px;
                padding: 4px 6px;
            }
            #decisionSummaryGroup:title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 6px;
                font-size: 11px;
                font-weight: 700;
                color: #51606f;
                text-transform: uppercase;
                letter-spacing: 0.4px;
            }
            #decisionNotesGroup {
                background-color: #ffffff;
                border: 1px solid #dde4ef;
                border-radius: 6px;
                padding: 4px 6px;
            }
            #decisionNotesGroup:title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 6px;
                font-size: 11px;
                font-weight: 700;
                color: #51606f;
                text-transform: uppercase;
                letter-spacing: 0.4px;
            }
            #decisionRelatedGroup {
                background-color: #ffffff;
                border: 1px solid #dde4ef;
                border-radius: 6px;
                padding: 4px 6px;
            }
            #decisionRelatedGroup:title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 6px;
                font-size: 11px;
                font-weight: 700;
                color: #51606f;
                text-transform: uppercase;
                letter-spacing: 0.4px;
            }
            QTabWidget#decisionInfoTabs QWidget {
                background-color: #ffffff;
                color: #1f2933;
            }
            QTabWidget#decisionInfoTabs::pane {
                border: 1px solid #dde4ef;
                background-color: #ffffff;
                margin-top: -4px;
            }
            QTabWidget#decisionInfoTabs QTabBar::tab {
                background-color: transparent;
                border: 1px solid transparent;
                border-bottom: none;
                padding: 4px 10px;
                margin-right: 6px;
                color: #5b6c7c;
                font-size: 11px;
                font-weight: 600;
            }
            QTabWidget#decisionInfoTabs QTabBar::tab:selected {
                border-color: #a5b9d5;
                background-color: #eef3ff;
                color: #1f2d3d;
                border-radius: 6px 6px 0 0;
            }
            QTabWidget#decisionInfoTabs QTabBar::tab:hover {
                color: #1a56db;
            }
            #decisionEmptyState {
                background-color: #f8fafc;
                border: 1px dashed #d0d7e2;
                border-radius: 6px;
                padding: 10px;
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

        # Place decision controls directly below the file actions button row
        anchor = getattr(self, "file_tree", None)
        insert_index = self.left_layout.indexOf(anchor) if anchor is not None else -1
        if insert_index == -1:
            insert_index = self.left_layout.count()
        else:
            insert_index += 1
        self.left_layout.insertWidget(insert_index, decision_card)

        # Detail panel (notes + related exports)
        self.detail_panel = QFrame()
        self.detail_panel.setObjectName("decisionInfoPanel")
        self.detail_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        detail_layout = QVBoxLayout()
        detail_layout.setContentsMargins(0, 8, 0, 0)
        detail_layout.setSpacing(10)

        notes_group = QGroupBox("Notes")
        notes_group.setObjectName("decisionNotesGroup")
        notes_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        notes_layout = QVBoxLayout()
        notes_layout.setContentsMargins(4, 6, 4, 6)
        notes_layout.setSpacing(4)
        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText(
            "Summarize observations, reasons for exclusion, or follow-up items."
        )
        self.notes_edit.setMinimumHeight(110)
        self.notes_edit.textChanged.connect(self._handle_notes_changed)
        self.notes_edit.setStyleSheet("font-size: 12px;")
        notes_layout.addWidget(self.notes_edit)
        notes_group.setLayout(notes_layout)

        related_group = QGroupBox("Related")
        related_group.setObjectName("decisionRelatedGroup")
        related_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        related_group.setMinimumHeight(120)
        related_layout = QVBoxLayout()
        related_layout.setContentsMargins(4, 6, 4, 6)
        related_layout.setSpacing(4)
        self.related_list = QListWidget()
        self.related_list.itemActivated.connect(self._open_related_item)
        self.related_list.setMinimumHeight(110)
        self.related_list.setStyleSheet("font-size: 12px;")
        related_layout.addWidget(self.related_list)
        related_group.setLayout(related_layout)

        detail_tabs = QTabWidget()
        detail_tabs.setObjectName("decisionInfoTabs")
        detail_tabs.setTabPosition(QTabWidget.North)
        detail_tabs.setElideMode(Qt.ElideRight)
        detail_tabs.setDocumentMode(True)
        detail_tabs.setFocusPolicy(Qt.NoFocus)
        detail_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.metrics_widget = ProcessingMetricsWidget()
        detail_tabs.addTab(self.metrics_widget, "Processing Metrics")
        detail_tabs.addTab(notes_group, "Notes")
        detail_tabs.addTab(related_group, "Related")

        detail_layout.addWidget(detail_tabs)
        detail_layout.addStretch(1)

        self._apply_light_palette(self.detail_panel)
        self._apply_light_palette(detail_tabs)

        self.detail_panel.setLayout(detail_layout)
        self.detail_panel.hide()

        insert_index = self.left_layout.indexOf(decision_card)
        if insert_index == -1:
            insert_index = self.left_layout.count()
        else:
            insert_index += 1
        self.left_layout.insertWidget(insert_index, self.detail_panel)

    def _modify_top_buttons(self) -> None:
        """Replace the default directory buttons with a polished toolbar."""

        if getattr(self, "_directory_toolbar_initialized", False):
            return

        original_action_bar = None
        if hasattr(self, "close_plot_btn"):
            original_action_bar = self.close_plot_btn.parentWidget()

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

        toolbar_layout.addStretch(1)

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

        header_container = QFrame()
        header_container.setObjectName("directoryHeaderContainer")
        header_layout = QVBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        header_container.setLayout(header_layout)

        header_layout.addWidget(toolbar)
        self.directory_toolbar = toolbar

        self.left_layout.insertWidget(0, header_container)
        self.directory_toolbar_container = header_container

        # Add folder path label on its own row under the toolbar rows
        path_label = QLabel()
        path_label.setObjectName("directoryPathLabel")
        path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        path_label.setWordWrap(True)
        self.left_layout.insertWidget(1, path_label)
        self._workspace_path_label = path_label
        self._refresh_workspace_path_label()

        if hasattr(self, "file_tree") and self.file_tree is not None:
            try:
                self.file_tree.setRootIsDecorated(False)
                self.file_tree.setIndentation(10)
                self.file_tree.setAlternatingRowColors(True)
            except Exception:
                pass

        if original_action_bar is not None:
            self.left_layout.removeWidget(original_action_bar)
            original_action_bar.deleteLater()

        self._directory_toolbar_initialized = True

    def _apply_light_palette(self, widget: Optional[QWidget]) -> None:
        if widget is None:
            return

        try:
            palette = widget.palette()
            palette.setColor(QPalette.Window, QColor("#ffffff"))
            palette.setColor(QPalette.Base, QColor("#ffffff"))
            palette.setColor(QPalette.AlternateBase, QColor("#f6f9ff"))
            palette.setColor(QPalette.Text, QColor("#1f2933"))
            palette.setColor(QPalette.WindowText, QColor("#1f2933"))
            palette.setColor(QPalette.Button, QColor("#ffffff"))
            palette.setColor(QPalette.ButtonText, QColor("#1f2933"))
            widget.setPalette(palette)
            widget.setAutoFillBackground(True)
            for child in widget.findChildren(QWidget):
                try:
                    child.setPalette(palette)
                except Exception:
                    continue
        except Exception:
            pass

    def _apply_file_tree_theme(self) -> None:
        """Force the navigation file tree to a light background.

        Some platforms/styles render the QTreeWidget viewport with a dark
        background when branches expand. Apply explicit palette + stylesheet
        to keep it consistently light with dark text.
        """
        if not hasattr(self, "file_tree") or self.file_tree is None:
            return

        # Palette for viewport/base colors
        try:
            pal = self.file_tree.palette()
            pal.setColor(QPalette.Base, QColor("#ffffff"))
            pal.setColor(QPalette.AlternateBase, QColor("#f6f9ff"))
            pal.setColor(QPalette.Text, QColor("#1f2933"))
            pal.setColor(QPalette.WindowText, QColor("#1f2933"))
            self.file_tree.setPalette(pal)
            self.file_tree.viewport().setAutoFillBackground(True)
        except Exception:
            pass

        # Targeted stylesheet to cover viewport and branches only
        self.file_tree.setStyleSheet(
            """
            QTreeWidget, QTreeView {
                background-color: #ffffff;
                color: #1f2933;
                alternate-background-color: #f6f9ff;
            }
            QTreeWidget::viewport, QTreeView::viewport {
                background-color: #ffffff;
            }
            QTreeWidget::item, QTreeView::item {
                /* no background here so per-item brushes (status shading) show */
                color: #1f2933;
            }
            QTreeWidget::item:selected, QTreeView::item:selected {
                background-color: #e3ecff;
                color: #0b3d91;
            }
            QTreeWidget::branch, QTreeView::branch {
                background: #ffffff;
            }
            """
        )

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
        # Update the workspace path label whenever directory changes
        self._refresh_workspace_path_label()

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
        self._update_processing_metrics_panel()

    def _read_processing_log_file(self, log_path: Path) -> List[Dict[str, str]]:
        try:
            with log_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                return [dict(row) for row in reader]
        except Exception:
            return []

    def _limit_segments(self, counter: Counter, limit: int = 6) -> OrderedDict[str, int]:
        if not counter:
            return OrderedDict()
        if len(counter) <= limit:
            return OrderedDict(counter.most_common())
        most_common = counter.most_common(limit - 1)
        used = {key for key, _ in most_common}
        other_total = sum(value for key, value in counter.items() if key not in used)
        ordered = OrderedDict((key, value) for key, value in most_common)
        ordered["Other"] = other_total
        return ordered

    def _counter_from_columns(
        self,
        rows: List[Dict[str, str]],
        columns: Iterable[str],
        limit: int = 6,
    ) -> OrderedDict[str, int]:
        for column in columns:
            values = [str(row.get(column, "")).strip() for row in rows if row.get(column)]
            values = [val for val in values if val]
            if values:
                normalized = [value or "Unspecified" for value in values]
                counter = Counter(normalized)
                counter = Counter({k: v for k, v in counter.items() if v > 0})
                if counter:
                    return self._limit_segments(counter, limit)
        return OrderedDict()

    def _counter_from_list_columns(
        self,
        rows: List[Dict[str, str]],
        columns: Iterable[str],
        limit: int = 6,
    ) -> OrderedDict[str, int]:
        aggregated: Counter = Counter()
        for column in columns:
            column_counter = Counter()
            for row in rows:
                items = _coerce_list(row.get(column))
                for item in items:
                    column_counter[item or "Unspecified"] += 1
            if column_counter:
                aggregated = column_counter
                break
        aggregated = Counter({k: v for k, v in aggregated.items() if v > 0})
        return self._limit_segments(aggregated, limit)

    def _channel_retention_metrics(self, row: Dict[str, str]) -> OrderedDict[str, int]:
        orig = _safe_int(row.get("net_nbchan_orig"))
        post = _safe_int(row.get("net_nbchan_post"))
        bad_list = len(_coerce_list(row.get("proc_badchans")))
        if orig <= 0 and post <= 0 and bad_list == 0:
            return OrderedDict()
        if orig <= 0:
            orig = post + bad_list
        removed = max(orig - post, bad_list, 0)
        retained = max(orig - removed, 0)
        counter = OrderedDict()
        if retained > 0:
            counter["Retained"] = retained
        if removed > 0:
            counter["Removed"] = removed
        return counter

    def _epoch_retention_metrics(self, row: Dict[str, str]) -> OrderedDict[str, int]:
        total = _safe_int(row.get("epoch_trials"))
        bad = _safe_int(row.get("epoch_badtrials"))
        if total <= 0 and bad <= 0:
            return OrderedDict()
        bad = min(bad, total) if total > 0 else bad
        kept = max(total - bad, 0)
        if total <= 0:
            total = kept + bad
        counter = OrderedDict()
        if kept > 0:
            counter["Kept"] = kept
        if bad > 0:
            counter["Rejected"] = bad
        return counter

    def _ica_component_metrics(self, row: Dict[str, str]) -> OrderedDict[str, int]:
        total = _safe_int(row.get("proc_nComps"))
        removed = len(_coerce_list(row.get("proc_removeComps")))
        if total <= 0 and removed <= 0:
            return OrderedDict()
        removed = min(removed, total) if total > 0 else removed
        retained = max(total - removed, 0)
        counter = OrderedDict()
        if retained > 0:
            counter["Retained"] = retained
        if removed > 0:
            counter["Removed"] = removed
        return counter

    def _build_processing_metrics(
        self, rows: List[Dict[str, str]]
    ) -> List[Tuple[str, Dict[str, int]]]:
        if not rows:
            return []

        metrics: List[Tuple[str, Dict[str, int]]] = []
        metrics.append(
            (
                "Step Outcomes",
                self._counter_from_columns(rows, ("proc_state", "status", "outcome", "result"), limit=6),
            )
        )
        metrics.append(
            (
                "Channels",
                self._channel_retention_metrics(rows[-1]),
            )
        )
        metrics.append(
            (
                "Epochs",
                self._epoch_retention_metrics(rows[-1]),
            )
        )
        metrics.append(
            (
                "ICA Components",
                self._ica_component_metrics(rows[-1]),
            )
        )
        return metrics

    def _update_processing_metrics_panel(self) -> None:
        if self.metrics_widget is not None:
            self.metrics_widget.show_message("Select a file to view processing metrics")

    def _find_processing_log_for_file(self, file_path: Path) -> Optional[Path]:
        parent = file_path.parent
        candidates = list(parent.glob("*_processing_log.csv"))
        if not candidates:
            return None

        stem = file_path.stem
        variants = {stem}
        suffixes = ["_comp_epo", "_comp", "_epo", "_postedit", "_preproc", "_raw"]
        for suffix in suffixes:
            if stem.endswith(suffix):
                variants.add(stem[: -len(suffix)])

        parts = stem.split("_")
        if len(parts) >= 3:
            variants.add("_".join(parts[:3]))
        if len(parts) >= 2:
            variants.add("_".join(parts[:2]))

        best_score = -1
        best_path: Optional[Path] = None
        for log_path in candidates:
            log_stem = log_path.stem
            if "_processing_log" in log_stem:
                log_prefix = log_stem.rsplit("_processing_log", 1)[0]
            else:
                log_prefix = log_stem

            score = max(_longest_common_prefix_length(log_prefix, variant) for variant in variants)
            if score > best_score:
                best_score = score
                best_path = log_path

        if best_score <= 0:
            return None
        return best_path

    def _update_processing_metrics_for_file(self, file_path: Path) -> None:
        if self.metrics_widget is None:
            return

        log_path = self._find_processing_log_for_file(file_path)
        if log_path is None or not log_path.exists():
            self.metrics_widget.show_message(
                "Processing log not available for this file"
            )
            return

        rows = self._read_processing_log_file(log_path)
        if not rows:
            self.metrics_widget.show_message(
                "Processing log not available for this file"
            )
            return

        metrics = self._build_processing_metrics(rows)
        if not any(sum(counter.values()) for _, counter in metrics):
            self.metrics_widget.show_message(
                "Processing log does not include recognized metrics"
            )
            return
        self.metrics_widget.update_metrics(metrics)

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

    def _handle_tree_selection_changed(self) -> None:
        if getattr(self, "_suppress_selection_autoload", False):
            return
        if not hasattr(self, "file_tree") or self.file_tree is None:
            return
        items = self.file_tree.selectedItems()
        if not items:
            return
        item = items[0]
        self._pending_selection_item = item
        if self._selection_timer is not None:
            self._selection_timer.stop()
            self._selection_timer.start()

    def _process_pending_selection(self) -> None:
        if self._pending_selection_item is None:
            return
        item = self._pending_selection_item
        self._pending_selection_item = None

        if not hasattr(self, "file_tree") or self.file_tree is None:
            return
        if item not in self.file_tree.selectedItems():
            return

        try:
            self.onFileSelect(item)
        except Exception:
            return

        self._auto_plot_current()

    def _auto_plot_current(self) -> None:
        if not getattr(self, "selected_file_path", None):
            return
        if self._plot_in_progress:
            self._pending_plot_refresh = True
            return
        self._pending_plot_refresh = False
        self._render_plot(reason="auto")

    # ------------------------------------------------------------------
    # Tree + selection logic
    # ------------------------------------------------------------------
    def loadFiles(self) -> None:  # noqa: N802 - inherited public API
        self._suppress_selection_autoload = True
        first_item: Optional[QTreeWidgetItem] = None
        try:
            if self.file_tree is not None:
                self.file_tree.clear()
            self.row_lookup.clear()
            self.all_keys.clear()

            current_dir = self.current_dir
            if not current_dir:
                self._update_summary()
                if self.instruction_widget is not None:
                    self.instruction_widget.show()
                if self.status_bar is not None:
                    self.status_bar.showMessage("No folder selected")
                if self.metrics_widget is not None:
                    self.metrics_widget.show_message("Select a folder with processing logs")
                return

            root_path = Path(current_dir)
            if not root_path.exists():
                self._update_summary()
                if self.instruction_widget is not None:
                    self.instruction_widget.show()
                if self.status_bar is not None:
                    self.status_bar.showMessage("Folder not found")
                if self.metrics_widget is not None:
                    self.metrics_widget.show_message("Processing folder not found")
                return
            file_icon = self.style().standardIcon(self.style().SP_FileIcon)

            def _sort_key(path: Path) -> tuple[str, str]:
                relative = path.relative_to(root_path)
                folder = "/".join(relative.parts[:-1])
                return (folder.lower(), relative.name.lower())

            set_files = sorted(root_path.rglob("*.set"), key=_sort_key)

            for file_path in set_files:
                relative_path = file_path.relative_to(root_path)
                display_name = relative_path.as_posix()
                base_label = display_name
                if file_path.name in self.modified_files:
                    base_label = f"{base_label} *"

                item = QTreeWidgetItem([base_label])
                if not file_icon.isNull():
                    item.setIcon(0, file_icon)
                item.setData(0, Qt.UserRole, str(file_path))
                key = self._record_key(file_path)
                item.setData(0, Qt.UserRole + 1, key)
                item.setData(0, Qt.UserRole + 2, base_label)
                if self.file_tree is not None:
                    self.file_tree.addTopLevelItem(item)
                self.row_lookup[key] = item
                self.all_keys.add(key)
                status = self.decisions.get(key, {}).get("status", "UNSET")
                self._apply_status_to_item(item, status)
                if first_item is None:
                    first_item = item

            self._update_summary()
            self._update_processing_metrics_panel()

            if not set_files and self.instruction_widget is not None:
                self.instruction_widget.show()
                if self.status_bar is not None:
                    self.status_bar.showMessage("No .set files found in the selected folder")
                if self.metrics_widget is not None:
                    self.metrics_widget.show_message(
                        "Processing log not available for this folder"
                    )
        finally:
            self._suppress_selection_autoload = False

        if first_item is not None:
            if self.instruction_widget is not None:
                self.instruction_widget.hide()

            def _select_initial() -> None:
                if not hasattr(self, "file_tree") or self.file_tree is None:
                    return
                self.file_tree.setCurrentItem(first_item)

            QTimer.singleShot(0, _select_initial)
        else:
            if self.status_bar is not None:
                self.status_bar.showMessage("Select a folder with .set files")

    def selectDirectory(self) -> None:  # noqa: N802 - inherited public API
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Directory", self.current_dir or str(Path.cwd())
        )
        if dir_path:
            self.closePlot()
            self._configure_directory(dir_path)
            self._load_decisions()
            self.current_key = None
            self.current_display_name = None
            if self.related_list is not None:
                self.related_list.clear()
            self._update_decision_controls(None)
            super().updateStatusBar()
            self.loadFiles()
            self._refresh_workspace_path_label()

    def _refresh_workspace_path_label(self) -> None:
        """Update the workspace path label with the current directory."""
        label = self._workspace_path_label
        if label is None:
            return
        text = self.current_dir or ""
        label.setToolTip(text)
        label.setText(text)

    def onFileSelect(self, item):  # noqa: N802 - inherited public API
        file_path_str = item.data(0, Qt.UserRole)
        if not file_path_str:
            if self.view_record_btn is not None:
                self.view_record_btn.setEnabled(False)
            self.current_key = None
            self.current_display_name = None
            self._update_decision_controls(None)
            if self.detail_panel is not None:
                self.detail_panel.hide()
            self._update_processing_metrics_panel()
            return

        file_path = Path(file_path_str)
        if file_path.suffix.lower() != ".set":
            if self.view_record_btn is not None:
                self.view_record_btn.setEnabled(False)
            self.current_key = None
            self.current_display_name = None
            self._update_decision_controls(None)
            if self.detail_panel is not None:
                self.detail_panel.hide()
            self._update_processing_metrics_panel()
            return

        previous_plot = getattr(self, "_plotted_file_path", None)
        if self.plot_widget is not None and (
            previous_plot is None or Path(previous_plot) != file_path
        ):
            self.closePlot(reason="selection_changed")

        self.selected_item = item
        self.selected_file = file_path.name
        self.selected_file_path = str(file_path)
        self.current_display_name = self._relative_path(file_path)
        try:
            self.current_run_id = self.getRunId(self.selected_file_path)
            self.current_run_record = autoclean_review.get_run_record(self.current_run_id)
            if self.view_record_btn is not None:
                self.view_record_btn.setEnabled(True)
        except Exception:
            if self.view_record_btn is not None:
                self.view_record_btn.setEnabled(False)
            self.current_run_record = None

        self.current_key = self._record_key(file_path)
        record = self.decisions.get(self.current_key)
        self._update_decision_controls(record)
        self._refresh_related_list(file_path)
        if self.detail_panel is not None:
            self.detail_panel.show()

        self._update_processing_metrics_for_file(file_path)

        if self.status_bar is not None and self.current_display_name:
            self.status_bar.showMessage(f"Queued · {self.current_display_name}")

    def _render_plot(self, *, reason: str) -> None:
        current_path = getattr(self, "selected_file_path", None)
        if not current_path:
            return

        if self._plot_in_progress:
            self._pending_plot_refresh = True
            return

        if (
            reason != "manual"
            and self.plot_widget is not None
            and self._current_plot_path == current_path
        ):
            if self.status_bar is not None and self.current_display_name:
                self.status_bar.showMessage(f"Ready · {self.current_display_name}")
            return

        self._plot_in_progress = True
        try:
            reload_reason = "reload" if reason == "manual" else "selection_changed"
            if self.plot_widget is not None:
                self.closePlot(reason=reload_reason)

            if self.status_bar is not None and self.current_display_name:
                verb = "Reloading" if reason == "manual" else "Loading"
                self.status_bar.showMessage(f"{verb} {self.current_display_name}…")

            if self.instruction_widget is not None:
                self.instruction_widget.show()

            super().plotFile()

            if self.detail_panel is not None and self.detail_panel.isHidden():
                self.detail_panel.show()

            self._current_plot_path = getattr(self, "selected_file_path", None)

            if self.status_bar is not None and self.current_display_name:
                self.status_bar.showMessage(f"Ready · {self.current_display_name}")
        finally:
            self._plot_in_progress = False
            if self._pending_plot_refresh:
                self._pending_plot_refresh = False
                self._auto_plot_current()

    def plotFile(self) -> None:  # noqa: N802 - inherited public API
        self._render_plot(reason="manual")

    def closePlot(self, reason: str = "close_button") -> None:  # noqa: N802
        if self.plot_widget is None:
            return

        try:
            self._auto_save_pending_epochs(reason=reason)
        except TypeError:
            self._auto_save_pending_epochs()

        try:
            if hasattr(self, "right_layout") and self.right_layout is not None:
                self.right_layout.removeWidget(self.plot_widget)
        except Exception:
            pass

        self.plot_widget.close()
        self.plot_widget.deleteLater()
        self.plot_widget = None
        if self.close_plot_btn is not None:
            self.close_plot_btn.setEnabled(False)
        self._plotted_file_path = None
        self._plot_is_raw = False
        self._current_plot_path = None
        self._pending_plot_refresh = False

        suppress_placeholder = reason == "selection_changed"
        if not suppress_placeholder and self.instruction_widget is not None:
            self.instruction_widget.show()
        if suppress_placeholder and self.instruction_widget is not None:
            self.instruction_widget.hide()

        if self.status_bar is not None and not suppress_placeholder:
            self.status_bar.showMessage("Select a file to review")

        QApplication.processEvents()

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
            # Explicit white so unmarked rows are light, not dark/transparent.
            item.setBackground(0, QColor("#ffffff"))

    def _update_summary(self) -> None:
        counts = Counter({key: 0 for key in STATUS_DEFINITIONS})
        for key in self.all_keys:
            status = self.decisions.get(key, {}).get("status", "UNSET")
            counts[status] += 1
        for status, label in self.summary_chip_labels.items():
            if label is not None:
                label.setText(str(counts.get(status, 0)))

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
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    pal = app.palette()
    pal.setColor(QPalette.Window, QColor("#f7f9fc"))
    pal.setColor(QPalette.Base, QColor("#ffffff"))
    pal.setColor(QPalette.AlternateBase, QColor("#f1f5ff"))
    pal.setColor(QPalette.Text, QColor("#1f2933"))
    pal.setColor(QPalette.WindowText, QColor("#1f2933"))
    pal.setColor(QPalette.Button, QColor("#ffffff"))
    pal.setColor(QPalette.ButtonText, QColor("#1f2933"))
    pal.setColor(QPalette.Highlight, QColor("#cde3ff"))
    pal.setColor(QPalette.HighlightedText, QColor("#0b3d91"))
    app.setPalette(pal)
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
