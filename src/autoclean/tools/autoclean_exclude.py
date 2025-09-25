"""AutoClean EEG inclusion/exclusion review helper.

The exclusion assistant ships as a fully standalone PyQt application so
reviewers can triage exported ``.set`` files while preserving the familiar
full-screen MNE browser.  The window collects Pass/Fail/Review decisions,
notes, and context without relying on the legacy ``autoclean_review`` module.
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

import pandas as pd
import yaml


def check_gui_dependencies() -> None:
    """Fail fast if the optional GUI stack is missing."""

    missing: list[str] = []
    try:  # pragma: no cover - import guard only
        import PyQt6  # noqa: F401
    except ImportError:  # pragma: no cover - runtime dependency guard
        missing.append("PyQt6")

    try:  # pragma: no cover - import guard only
        from PyQt6 import QtPdf  # noqa: F401
    except ImportError:  # pragma: no cover - runtime dependency guard
        missing.append("QtPdf")

    if missing:
        print("Error: Missing required GUI dependencies for the exclusion tool.")
        print("Install the extras bundle first:")
        print("    pip install autocleaneeg-pipeline[gui]")
        print(f"Missing packages: {', '.join(missing)}")
        sys.exit(1)


check_gui_dependencies()


from qtpy.QtCore import (  # noqa: E402
    QAbstractItemModel,
    QModelIndex,
    QObject,
    QEvent,
    QPointF,
    QSize,
    Qt,
    QTimer,
)
from qtpy.QtGui import QColor, QKeySequence, QPalette, QPixmap  # noqa: E402
from qtpy.QtWidgets import (  # noqa: E402
    QApplication,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSplitter,
    QSizePolicy,
    QStackedLayout,
    QStatusBar,
    QStyle,
    QTabWidget,
    QTextEdit,
    QTreeView,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from PyQt6.QtCore import pyqtRemoveInputHook  # noqa: E402
from PyQt6.QtPdf import QPdfDocument  # noqa: E402
from PyQt6.QtPdfWidgets import QPdfView  # noqa: E402

from autoclean.io.export import save_epochs_to_set  # noqa: E402
from autoclean.utils.database import get_run_record  # noqa: E402
from autoclean.utils.logging import message  # noqa: E402
from autoclean.utils.path_resolution import resolve_moved_path  # noqa: E402
from autoclean.utils.user_config import user_config  # noqa: E402

import mne  # noqa: E402
import scipy.io as sio  # noqa: E402


pyqtRemoveInputHook()

os.environ["MNE_BROWSER_THEME"] = "light"
mne.viz.set_browser_backend("qt")


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


log_debug = partial(message, "debug")
log_info = partial(message, "info")
log_warning = partial(message, "warning")


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


def _safe_float(value: Optional[str]) -> float:
    try:
        if value is None:
            return 0.0
        text = str(value).strip()
        if not text:
            return 0.0
        return float(text)
    except (ValueError, TypeError):
        return 0.0


def _longest_common_prefix_length(a: str, b: str) -> int:
    length = 0
    for char_a, char_b in zip(a, b):
        if char_a != char_b:
            break
        length += 1
    return length


def _normalized_prefix_score(a: str, b: str) -> int:
    def normalize(s: str) -> str:
        return "".join(ch for ch in s if ch.isalnum()).lower()

    return _longest_common_prefix_length(normalize(a), normalize(b))

def _enum_name(value: object) -> str:
    """Return the Enum name if available, otherwise string form."""
    if value is None:
        return "None"
    name = getattr(value, 'name', None)
    if name:
        return str(name)
    return str(value)


# --- Configuration and Asset Resolution ---

def _load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Configuration file not found at {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"Warning: Error parsing configuration file: {e}")
        return {}


def strip_suffixes(stem: str, asset_type: str = None, config: dict = None) -> str:
    """Strip suffixes from filename stem based on configuration.
    
    Args:
        stem: The filename stem to process
        asset_type: Optional asset type for asset-specific suffixes
        config: Configuration dictionary (loaded if not provided)
    
    Returns:
        Stem with suffixes stripped
    """
    if config is None:
        config = _load_config()
    
    if not config:
        return stem
    
    # Get global suffixes
    all_suffixes = list(config.get("suffixes", {}).get("global", []))
    
    # Add asset-specific suffixes if available
    if asset_type and asset_type in config.get("suffixes", {}):
        all_suffixes.extend(config["suffixes"][asset_type])
    
    # Sort by length (longest first) to handle overlapping suffixes correctly
    all_suffixes = sorted(all_suffixes, key=len, reverse=True)
    
    # Strip suffixes
    for suffix in all_suffixes:
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    
    return stem


def resolve_asset(file_path: Path, asset_type: str, log_df: pd.DataFrame = None, config: dict = None) -> Optional[Path]:
    """Resolve asset path using configuration-based approach.
    
    Args:
        file_path: The source file path
        asset_type: Type of asset to resolve (processing_log, psd_overview, run_report, ica_report)
        log_df: DataFrame containing preprocessing log (optional for backward compatibility)
        config: Configuration dictionary (loaded if not provided)
    
    Returns:
        Resolved asset path or None if not found
    """
    if config is None:
        config = _load_config()
    
    if not config:
        return None
    
    # Strip suffixes to get normalized stem
    stem = strip_suffixes(file_path.stem, asset_type, config)
    
    # Get configuration for this asset type
    postfixes = config.get("postfixes", {})
    directories = config.get("directories", {})
    logfile_config = config.get("logfile", {})
    
    if asset_type not in postfixes or asset_type not in directories:
        return None
    
    postfix = postfixes[asset_type]
    subdir = directories[asset_type]
    
    # Determine base directory
    if subdir == ".":
        base_dir = file_path.parent
    else:
        base_dir = file_path.parent / subdir
    
    # Construct the asset path
    asset_path = base_dir / f"{stem}{postfix}"
    
    # If log DataFrame is provided, verify the stem exists in the log
    if log_df is not None and not log_df.empty:
        key_column = logfile_config.get("key_column", "subj_basename")
        if key_column in log_df.columns:
            if stem not in log_df[key_column].values:
                return None
    
    return asset_path


def _load_preprocessing_log(task_root: Optional[Path] = None, exports_dir: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Load preprocessing log DataFrame.
    
    Args:
        task_root: Task root directory
        exports_dir: Exports directory
    
    Returns:
        DataFrame with preprocessing log or None if not found
    """
    config = _load_config()
    if not config:
        return None
    
    logfile_name = config.get("logfile", {}).get("name", "preprocessing_log.csv")
    
    # Try to find the log file in common locations
    search_paths = []
    
    if task_root:
        search_paths.append(task_root / logfile_name)
        search_paths.append(task_root / "logs" / logfile_name)
    
    if exports_dir:
        search_paths.append(exports_dir / logfile_name)
        search_paths.append(exports_dir.parent / logfile_name)
        search_paths.append(exports_dir.parent / "logs" / logfile_name)
    
    for log_path in search_paths:
        if log_path.exists():
            try:
                return pd.read_csv(log_path)
            except Exception as e:
                print(f"Warning: Could not load preprocessing log from {log_path}: {e}")
                continue
    
    return None


class PdfPreviewWidget(QWidget):
    """Lightweight PDF viewer that embeds Qt's native renderer."""

    def __init__(self, placeholder: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._placeholder = placeholder
        self._document = QPdfDocument(self)
        self._view = QPdfView(self)
        self._view.setDocument(self._document)
        self._view.setZoomMode(QPdfView.ZoomMode.FitToWidth)
        self._view.setPageMode(QPdfView.PageMode.SinglePage)

        self._message = QLabel(placeholder)
        self._message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._message.setWordWrap(True)

        self._status_label = QLabel("No document loaded")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setWordWrap(True)
        self._status_label.setObjectName("pdfStatusLabel")
        self._status_label.hide()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._message)
        layout.addWidget(self._view, 1)
        layout.addWidget(self._status_label)
        self.setLayout(layout)

        self._current_path: Optional[Path] = None
        self._total_pages = 0
        self._current_page = 0

        self._navigator = self._view.pageNavigator()
        if self._navigator is not None:
            self._navigator.currentPageChanged.connect(self._on_page_changed)

        self._next_page_shortcut = QShortcut(
            QKeySequence(QKeySequence.StandardKey.MoveToNextPage), self
        )
        self._next_page_shortcut.activated.connect(lambda: self._step_page(+1))

        self._prev_page_shortcut = QShortcut(
            QKeySequence(QKeySequence.StandardKey.MoveToPreviousPage), self
        )
        self._prev_page_shortcut.activated.connect(lambda: self._step_page(-1))

        self._view.hide()
        self.clear(suppress_log=True)

    def clear(self, suppress_log: bool = False) -> None:
        self._document.close()
        self._current_path = None
        self._total_pages = 0
        self._current_page = 0
        self._status_label.hide()
        if not suppress_log:
            log_debug(
                f"[{_human_timestamp()}] PDF preview cleared; placeholder restored."
            )
        self.show_message(self._placeholder)

    def show_message(self, message: str) -> None:
        self._message.setText(message)
        self._message.show()
        self._view.hide()
        self._status_label.hide()

    def load(self, path: Path) -> None:
        log_debug(f"[{_human_timestamp()}] Attempting to load PDF preview: {path}")
        status = self._document.load(str(path))
        status_name = _enum_name(status)

        try:
            doc_status = self._document.status()
        except Exception as exc:
            doc_status = None
            doc_status_name = f"status_call_failed={exc}"
        else:
            doc_status_name = _enum_name(doc_status)

        error_value: Optional[object] = None
        error_name = 'unsupported'
        if hasattr(self._document, 'error'):
            try:
                error_value = self._document.error()
                error_name = _enum_name(error_value)
            except Exception as exc:
                error_name = f"error_call_failed={exc}"

        ready = doc_status_name == 'Ready'
        error_ok = error_name in {'None', 'None_', 'NoError', 'NoError_', 'unsupported'}

        if not ready or not error_ok:
            diagnostics: list[str] = []
            try:
                exists = path.exists()
            except Exception as exc:
                diagnostics.append(f"exists_check_failed={exc}")
                exists = False

            if exists:
                try:
                    stat = path.stat()
                except OSError as exc:
                    diagnostics.append(f"stat_error={exc}")
                else:
                    diagnostics.append(f"size={stat.st_size} bytes")
                    diagnostics.append(
                        f"mtime={datetime.fromtimestamp(stat.st_mtime).isoformat(timespec='seconds')}"
                    )
            else:
                diagnostics.append('file_missing')

            if hasattr(self._document, 'errorString'):
                try:
                    error_string = self._document.errorString()
                except Exception as exc:
                    diagnostics.append(f"error_string_failed={exc}")
                else:
                    if error_string:
                        diagnostics.append(f"error_string={error_string!r}")

            diag_text = ", ".join(diagnostics) if diagnostics else 'no file diagnostics'
            log_warning(
                f"[{_human_timestamp()}] PDF load failed for {path}; "
                f"requested_status={status_name}, document_status={doc_status_name}, "
                f"error={error_name}, {diag_text}."
            )
            self.clear(suppress_log=True)
            self.show_message('Failed to load preview')
            return

        try:
            page_count = self._document.pageCount()
        except Exception as exc:
            page_count = f"page_count_failed={exc}"
        log_info(
            f"[{_human_timestamp()}] PDF load succeeded: {path} "
            f"(requested_status={status_name}, document_status={doc_status_name}, "
            f"error={error_name}, pages={page_count})."
        )

        if isinstance(page_count, int) and page_count > 0:
            self._total_pages = page_count
        else:
            self._total_pages = 0
        self._current_page = 0

        if self._navigator is not None:
            self._navigator.jump(0, QPointF(0, 0))

        if self._total_pages > 1:
            self._view.setPageMode(QPdfView.PageMode.MultiPage)
        else:
            self._view.setPageMode(QPdfView.PageMode.SinglePage)

        navigator = self._view.pageNavigator()
        if navigator is not None:
            navigator.jump(0, QPointF(0, 0))

        self._current_path = path
        self._message.hide()
        self._view.show()
        self._status_label.show()
        self._update_status_label()


    def _update_status_label(self) -> None:
        if self._total_pages <= 0:
            self._status_label.setText("No document loaded")
            return
        page_text = f"Page {self._current_page + 1} / {self._total_pages}"
        try:
            zoom = self._view.zoomFactor()
        except Exception:
            zoom = 1.0
        zoom_pct = int(round(zoom * 100))
        mode = self._view.pageMode()
        mode_name = getattr(mode, 'name', str(mode))
        self._status_label.setText(
            f"{page_text} · Zoom {zoom_pct}% · Mode {mode_name}"
        )

    def _step_page(self, delta: int) -> None:
        if self._navigator is None or self._total_pages <= 0:
            return
        try:
            current = self._navigator.currentPage()
        except Exception:
            current = self._current_page
        target = max(0, min(self._total_pages - 1, current + delta))
        if target == current:
            return
        self._navigator.jump(target, QPointF(0, 0))
        self._current_page = target
        self._update_status_label()

    def _on_page_changed(self, page: int) -> None:
        if self._total_pages <= 0:
            return
        self._current_page = max(0, min(self._total_pages - 1, page))
        self._update_status_label()


class ReviewBase(QWidget):
    """Minimal review surface providing file tree + plotting helpers."""

    def __init__(self, autoclean_dir: Optional[str]) -> None:
        super().__init__()
        self.current_dir = autoclean_dir
        self.modified_files: set[str] = set()
        self.current_run_id: Optional[str] = None
        self.current_run_record: Optional[dict] = None
        self.current_run_record_window: Optional[QWidget] = None
        self.plot_widget: Optional[QWidget] = None
        self.current_epochs: Optional[mne.BaseEpochs] = None
        self.current_raw: Optional[mne.io.BaseRaw] = None
        self._plotted_file_path: Optional[str] = None
        self._plot_is_raw = False
        self._auto_saving_epochs = False
        
        # New configuration-based asset resolution
        self.preprocessing_log_df: Optional[pd.DataFrame] = None
        self.config: dict = _load_config()

        self.selected_item: Optional[QTreeWidgetItem] = None
        self.selected_file: Optional[str] = None
        self.selected_file_path: Optional[str] = None

        self.left_layout: Optional[QVBoxLayout] = None
        self.file_tree: Optional[QTreeWidget] = None
        self.status_bar: Optional[QStatusBar] = None
        self.instruction_widget: Optional[QWidget] = None
        self.right_container = QWidget()
        self.right_layout = QVBoxLayout()
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(0)
        self.right_container.setLayout(self.right_layout)

        self.select_dir_btn: Optional[QPushButton] = None
        self.open_folder_btn: Optional[QPushButton] = None
        self.refresh_btn: Optional[QPushButton] = None
        self.plot_btn: Optional[QPushButton] = None
        self.close_plot_btn: Optional[QPushButton] = None
        self.view_record_btn: Optional[QPushButton] = None
        self.exit_btn: Optional[QPushButton] = None

        self._apply_global_theme()
        self._init_ui()

        if self.current_dir:
            self._load_preprocessing_log()
            self.loadFiles()
            self.updateStatusBar()

    # ------------------------------------------------------------------
    # UI bootstrapping
    # ------------------------------------------------------------------
    def _init_ui(self) -> None:
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(8)

        content_splitter = QSplitter(Qt.Orientation.Horizontal)

        navigation = QWidget()
        navigation.setObjectName("navigationPanel")
        self.left_layout = QVBoxLayout()
        self.left_layout.setContentsMargins(12, 12, 12, 12)
        self.left_layout.setSpacing(10)
        navigation.setLayout(self.left_layout)

        self.select_dir_btn = QPushButton("Select Directory")
        self.select_dir_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.select_dir_btn.clicked.connect(self.selectDirectory)
        self.left_layout.addWidget(self.select_dir_btn)

        self.open_folder_btn = QPushButton("Open Folder")
        self.open_folder_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_folder_btn.clicked.connect(self._open_current_directory)
        self.left_layout.addWidget(self.open_folder_btn)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_btn.clicked.connect(self.refreshFileTree)
        self.left_layout.addWidget(self.refresh_btn)

        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderHidden(True)
        self.file_tree.itemClicked.connect(self.onFileSelect)
        self.file_tree.setObjectName("fileTree")
        self.left_layout.addWidget(self.file_tree, 1)

        action_bar = QHBoxLayout()
        action_bar.setContentsMargins(0, 0, 0, 0)
        action_bar.setSpacing(6)

        self.plot_btn = QPushButton("Plot Selected")
        self.plot_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.plot_btn.clicked.connect(self.plotFile)
        self.plot_btn.setEnabled(False)
        action_bar.addWidget(self.plot_btn)

        self.close_plot_btn = QPushButton("Close Plot")
        self.close_plot_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_plot_btn.clicked.connect(self.closePlot)
        self.close_plot_btn.setEnabled(False)
        action_bar.addWidget(self.close_plot_btn)

        self.view_record_btn = QPushButton("View Run Record")
        self.view_record_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.view_record_btn.clicked.connect(self.viewRunRecord)
        self.view_record_btn.setEnabled(False)
        action_bar.addWidget(self.view_record_btn)

        self.exit_btn = QPushButton("Exit")
        self.exit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.exit_btn.clicked.connect(self.close)
        action_bar.addWidget(self.exit_btn)

        action_container = QWidget()
        action_container.setObjectName("actionBar")
        action_container.setLayout(action_bar)
        self.left_layout.addWidget(action_container)

        navigation.setMinimumWidth(320)
        content_splitter.addWidget(navigation)
        content_splitter.addWidget(self.right_container)
        content_splitter.setStretchFactor(1, 1)

        self.instruction_widget = QLabel("Select a file to review")
        self.instruction_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.instruction_widget.setWordWrap(True)
        self.instruction_widget.setObjectName("instructionPanel")
        self.right_layout.addWidget(self.instruction_widget, 1)

        root_layout.addWidget(content_splitter, 1)

        self.status_bar = QStatusBar()
        root_layout.addWidget(self.status_bar)

        self.setLayout(root_layout)
        self.setWindowTitle("AutoClean Exclusion Review")
        self.resize(1280, 720)

    def _apply_global_theme(self) -> None:
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#f5f7fb"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#eef2f8"))
        palette.setColor(QPalette.ColorRole.Button, QColor("#ffffff"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#1f2933"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#1f2933"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#1f2933"))
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#cce0ff"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#0b3d91"))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

    # ------------------------------------------------------------------
    # File tree + status helpers
    # ------------------------------------------------------------------
    def loadFiles(self) -> None:  # noqa: N802 - public API compatibility
        if self.file_tree is None:
            return
        self.file_tree.clear()
        if not self.current_dir:
            return
        root_path = Path(self.current_dir)
        if not root_path.exists():
            return
        for file_path in sorted(root_path.rglob("*.set")):
            item = QTreeWidgetItem([file_path.name])
            item.setData(0, Qt.ItemDataRole.UserRole, str(file_path))
            self.file_tree.addTopLevelItem(item)

    def updateStatusBar(self) -> None:  # noqa: N802 - public API compatibility
        if self.status_bar is None:
            return
        if self.current_dir:
            self.status_bar.showMessage(f"Workspace: {self.current_dir}")
        else:
            self.status_bar.showMessage("Select a folder with .set files")

    def selectDirectory(self) -> None:  # noqa: N802 - public API compatibility
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Directory", self.current_dir or str(Path.cwd())
        )
        if dir_path:
            self.closePlot()
            self.current_dir = dir_path
            self._load_preprocessing_log()
            self.loadFiles()
            self.updateStatusBar()

    def _open_current_directory(self) -> None:
        if not self.current_dir:
            return
        _open_path(Path(self.current_dir))

    def refreshFileTree(self) -> None:
        if self.current_dir:
            self.loadFiles()
            self.updateStatusBar()

    def _load_preprocessing_log(self) -> None:
        """Load preprocessing log DataFrame for asset resolution."""
        if not self.current_dir:
            return
        
        # Try to determine task_root and exports_dir from current_dir
        current_path = Path(self.current_dir)
        
        # Look for exports directory
        exports_dir = None
        if current_path.name == "exports":
            exports_dir = current_path
        elif (current_path / "exports").exists():
            exports_dir = current_path / "exports"
        
        # Look for task root (parent of exports or current directory)
        task_root = None
        if exports_dir:
            task_root = exports_dir.parent
        else:
            task_root = current_path
        
        self.preprocessing_log_df = _load_preprocessing_log(task_root, exports_dir)
        
        if self.preprocessing_log_df is not None:
            print(f"Loaded preprocessing log with {len(self.preprocessing_log_df)} entries")
        else:
            print("No preprocessing log found - using fallback asset resolution")

    # ------------------------------------------------------------------
    # Selection + plotting hooks
    # ------------------------------------------------------------------
    def getRunId(self, file_path: str) -> str:  # noqa: N802 - public API
        eeg = sio.loadmat(file_path)
        return str(eeg["etc"]["run_id"][0][0][0])

    def onFileSelect(self, item: QTreeWidgetItem) -> None:  # noqa: N802
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        self.selected_item = item
        self.selected_file = item.text(0)
        self.selected_file_path = str(data)
        if self.plot_btn is not None:
            self.plot_btn.setEnabled(True)
        try:
            self.current_run_id = self.getRunId(self.selected_file_path)
            self.current_run_record = get_run_record(self.current_run_id)
            if self.view_record_btn is not None:
                self.view_record_btn.setEnabled(True)
        except Exception:
            if self.view_record_btn is not None:
                self.view_record_btn.setEnabled(False)

    def viewRunRecord(self) -> None:  # noqa: N802 - legacy public API
        if not self.current_run_record:
            QMessageBox.information(self, "Run Record", "No record available.")
            return

        window = QWidget()
        window.setWindowTitle("Run Record")
        window.resize(1000, 800)

        layout = QVBoxLayout()
        splitter = QSplitter(Qt.Orientation.Horizontal)

        scroll_tree = QScrollArea()
        tree_view = QTreeView()
        model = _JsonTreeModel(self.current_run_record)
        tree_view.setModel(model)
        tree_view.expandAll()
        scroll_tree.setWidget(tree_view)
        scroll_tree.setWidgetResizable(True)
        splitter.addWidget(scroll_tree)

        placeholder = QLabel("Artifacts preview is not available in this build.")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        splitter.addWidget(placeholder)

        layout.addWidget(splitter)
        window.setLayout(layout)
        window.show()
        self.current_run_record_window = window

    def plotFile(self) -> None:  # noqa: N802 - legacy public API
        if not self.selected_file_path:
            return

        file_path = Path(self.selected_file_path)
        if not file_path.exists():
            QMessageBox.warning(self, "Missing File", f"{file_path} not found")
            return

        try:
            resolved_path = resolve_moved_path(file_path)
        except Exception:
            resolved_path = file_path

        self.instruction_widget.hide() if self.instruction_widget else None

        try:
            epochs = mne.io.read_epochs_eeglab(resolved_path)
            self.current_epochs = epochs
            self.current_raw = None
            is_raw = False
        except Exception:
            raw = mne.io.read_raw_eeglab(resolved_path, preload=True)
            self.current_raw = raw
            self.current_epochs = None
            is_raw = True

        if self.plot_widget is not None:
            self.right_layout.removeWidget(self.plot_widget)
            self.plot_widget.close()

        if not is_raw and self.current_epochs is not None:
            self.plot_widget = self.current_epochs.plot(
                n_epochs=10,
                show=False,
                block=False,
                picks="all",
                events=True,
                show_scalebars=True,
                scalings={"eeg": 25e-6},
                n_channels=self.current_epochs.info["nchan"],
            )
        elif self.current_raw is not None:
            self.plot_widget = self.current_raw.plot(
                show=False,
                block=True,
                show_scalebars=True,
                scalings={"eeg": 25e-6},
                n_channels=self.current_raw.info["nchan"],
                show_options=True,
            )
        else:
            return

        self._plot_is_raw = is_raw
        self._plotted_file_path = str(resolved_path)
        self._enforce_light_browser_theme(self.plot_widget)
        self.right_layout.addWidget(self.plot_widget)
        self.plot_widget.show()
        if self.close_plot_btn is not None:
            self.close_plot_btn.setEnabled(True)

    def closePlot(self) -> None:  # noqa: N802 - legacy public API
        if self.plot_widget is None:
            return
        self._auto_save_pending_epochs()
        self.right_layout.removeWidget(self.plot_widget)
        self.plot_widget.close()
        self.plot_widget.deleteLater()
        self.plot_widget = None
        if self.close_plot_btn is not None:
            self.close_plot_btn.setEnabled(False)
        self._plotted_file_path = None
        self._plot_is_raw = False
        if self.instruction_widget is not None:
            self.instruction_widget.show()

    def _auto_save_pending_epochs(self, reason: str = "") -> None:
        if self._auto_saving_epochs:
            return
        if self._plot_is_raw or self._plotted_file_path is None:
            return
        if self.current_epochs is None or self.current_run_id is None:
            return
        if not hasattr(self.plot_widget, "mne") or self.plot_widget.mne is None:
            return
        if self.current_dir is None:
            return

        try:
            bad_epochs = sorted(self.plot_widget.mne.bad_epochs)
        except AttributeError:
            bad_epochs = []

        if not bad_epochs:
            return

        self._auto_saving_epochs = True
        try:
            run_record = get_run_record(self.current_run_id)
            original_stage_dir = Path(
                run_record["metadata"]["step_prepare_directories"]["stage"]
            )
            try:
                task_name = original_stage_dir.parent.name
                container_stage_dir = Path(self.current_dir) / task_name / "stage"
            except Exception:
                container_stage_dir = Path(self.current_dir) / "stage"

            autoclean_dict = {
                "run_id": self.current_run_id,
                "stage_files": run_record["metadata"]["entrypoint"]["stage_files"],
                "stage_dir": container_stage_dir,
                "unprocessed_file": run_record["unprocessed_file"],
            }

            message(
                "info",
                f"Auto-saving {len(bad_epochs)} marked epochs for {Path(self._plotted_file_path).name}",
            )

            self.current_epochs.drop(bad_epochs)
            container_stage_dir.mkdir(parents=True, exist_ok=True)
            save_epochs_to_set(self.current_epochs, autoclean_dict, stage="post_edit")

            if self.status_bar is not None:
                self.status_bar.showMessage(
                    f"Auto-saved exclusions for {Path(self._plotted_file_path).name}",
                    5000,
                )
        finally:
            self._auto_saving_epochs = False

    def _enforce_light_browser_theme(self, widget: Optional[QWidget]) -> None:
        if widget is None:
            return
        try:
            if hasattr(widget, "set_theme"):
                widget.set_theme("light")
                return
        except Exception:
            pass
        try:
            if hasattr(widget, "set_dark_mode"):
                widget.set_dark_mode(False)
                return
        except Exception:
            pass
        try:
            palette = widget.palette()
            palette.setColor(QPalette.ColorRole.Window, QColor("#ffffff"))
            palette.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
            palette.setColor(QPalette.ColorRole.Text, QColor("#1f2933"))
            palette.setColor(QPalette.ColorRole.WindowText, QColor("#1f2933"))
            widget.setPalette(palette)
            widget.setStyleSheet("background-color: #ffffff; color: #1f2933;")
            for child in widget.findChildren(QWidget):
                child.setPalette(palette)
        except Exception:
            pass


class _JsonTreeModel(QAbstractItemModel):
    """Simplified tree model for displaying nested JSON-like data."""

    class TreeItem:
        def __init__(self, key, value, children=None):
            self.key = key
            self.value = value
            self.children = children or []
            self.parent = None
            for child in self.children:
                child.parent = self

    def __init__(self, data):
        super().__init__()
        self._root = self.TreeItem("root", "")
        self._root.children = self._process_data(data)
        for child in self._root.children:
            child.parent = self._root

    def _process_data(self, data):
        items = []
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    item = self.TreeItem(str(key), "")
                    item.children = self._process_data(value)
                    for child in item.children:
                        child.parent = item
                else:
                    item = self.TreeItem(str(key), str(value))
                items.append(item)
        elif isinstance(data, list):
            for i, value in enumerate(data):
                if isinstance(value, (dict, list)):
                    item = self.TreeItem(str(i), "")
                    item.children = self._process_data(value)
                    for child in item.children:
                        child.parent = item
                else:
                    item = self.TreeItem(str(i), str(value))
                items.append(item)
        return items

    def index(self, row, column, parent=QModelIndex()):
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        parent_item = parent.internalPointer() if parent.isValid() else self._root
        if row < len(parent_item.children):
            return self.createIndex(row, column, parent_item.children[row])
        return QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()
        child_item = index.internalPointer()
        parent_item = child_item.parent
        if parent_item is self._root or parent_item is None:
            return QModelIndex()
        row = (
            parent_item.parent.children.index(parent_item) if parent_item.parent else 0
        )
        return self.createIndex(row, 0, parent_item)

    def rowCount(self, parent=QModelIndex()):
        if parent.column() > 0:
            return 0
        parent_item = parent.internalPointer() if parent.isValid() else self._root
        return len(parent_item.children)

    def columnCount(self, parent=QModelIndex()):
        return 2

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        item = index.internalPointer()
        return item.key if index.column() == 0 else item.value

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return ["Key", "Value"][section]
        return None

class ProcessingMetricsWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.setLayout(layout)

        self.message_label = QLabel("")
        self.message_label.setObjectName("decisionMetricsMessage")
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.message_label)

        self.rows_container = QVBoxLayout()
        self.rows_container.setContentsMargins(0, 0, 0, 0)
        self.rows_container.setSpacing(0)
        layout.addLayout(self.rows_container)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMaximumHeight(140)

        self._render_no_data("No processing metrics available.")

    def show_message(self, message: str) -> None:
        self._render_no_data(message)

    def _render_no_data(self, message: str) -> None:
        self._clear_rows()
        self.message_label.setText(message)
        self.message_label.show()

    def _clear_rows(self) -> None:
        while self.rows_container.count():
            item = self.rows_container.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def update_metrics(self, metrics: List[Tuple[str, str, str]]) -> None:
        self._clear_rows()

        if not metrics:
            self._render_no_data("No processing metrics available.")
            return

        self.message_label.hide()

        for label_text, value_text, color in metrics:
            row = QFrame()
            row.setObjectName("decisionMetricsRow")
            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(6, 2, 6, 2)
            row_layout.setSpacing(6)
            row.setLayout(row_layout)

            indicator = QFrame()
            indicator.setFixedWidth(4)
            indicator.setObjectName("decisionMetricsBar")
            indicator.setStyleSheet(f"background-color: {color}; border-radius: 2px;")
            row_layout.addWidget(indicator)

            name_label = QLabel(label_text)
            name_label.setObjectName("decisionMetricsName")
            row_layout.addWidget(name_label)
            row_layout.addStretch(1)

            value_label = QLabel(value_text)
            value_label.setObjectName("decisionMetricsValue")
            value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row_layout.addWidget(value_label)

            self.rows_container.addWidget(row)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.rows_container.addWidget(spacer)



def _open_path(path: Path) -> None:
    """Open *path* using the default OS handler."""

    if sys.platform.startswith("darwin"):
        subprocess.run(["open", str(path)], check=False)
    elif os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


class ExclusionFileSelector(ReviewBase):
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
        self.plot_tabs: Optional[QTabWidget] = None
        self.plot_tab_layout: Optional[QVBoxLayout] = None
        self.psd_message_label: Optional[QLabel] = None
        self.psd_image_label: Optional[QLabel] = None
        self.psd_scroll: Optional[QScrollArea] = None
        self.psd_original_pixmap: Optional[QPixmap] = None
        self.run_report_preview: Optional[PdfPreviewWidget] = None
        self.ica_preview: Optional[PdfPreviewWidget] = None
        self.time_series_tab_index: Optional[int] = None
        self.psd_tab_index: Optional[int] = None
        self.run_report_tab_index: Optional[int] = None
        self.ica_tab_index: Optional[int] = None

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

        # Configure plot + PSD tabs on the right side
        self.plot_tabs = QTabWidget()
        self.plot_tabs.setObjectName("decisionPlotTabs")

        plot_tab_container = QWidget()
        plot_tab_layout = QVBoxLayout()
        plot_tab_layout.setContentsMargins(0, 0, 0, 0)
        plot_tab_layout.setSpacing(0)
        plot_tab_container.setLayout(plot_tab_layout)
        self.plot_tab_layout = plot_tab_layout

        existing_right_layout = getattr(self, "right_layout", None)
        if existing_right_layout is not None:
            while existing_right_layout.count():
                item = existing_right_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    plot_tab_layout.addWidget(widget)

        self.time_series_tab_index = self.plot_tabs.addTab(plot_tab_container, "Time Series")
        self.right_layout = plot_tab_layout

        psd_tab_container = QWidget()
        psd_layout = QVBoxLayout()
        psd_layout.setContentsMargins(12, 12, 12, 12)
        psd_layout.setSpacing(8)
        psd_tab_container.setLayout(psd_layout)

        self.psd_message_label = QLabel("Select a file to view PSD overview")
        self.psd_message_label.setObjectName("psdOverviewMessage")
        self.psd_message_label.setAlignment(Qt.AlignCenter)
        psd_layout.addWidget(self.psd_message_label)

        self.psd_scroll = QScrollArea()
        self.psd_scroll.setObjectName("psdOverviewScroll")
        self.psd_scroll.setWidgetResizable(True)
        self.psd_image_label = QLabel()
        self.psd_image_label.setObjectName("psdOverviewImage")
        self.psd_image_label.setAlignment(Qt.AlignCenter)
        self.psd_scroll.setWidget(self.psd_image_label)
        self.psd_scroll.hide()
        psd_layout.addWidget(self.psd_scroll, 1)

        if self.psd_scroll is not None:
            self.psd_scroll.viewport().installEventFilter(self)
        self.psd_tab_index = self.plot_tabs.addTab(psd_tab_container, "PSD Overview")

        run_tab_container = QWidget()
        run_layout = QVBoxLayout()
        run_layout.setContentsMargins(12, 12, 12, 12)
        run_layout.setSpacing(8)
        run_tab_container.setLayout(run_layout)

        self.run_report_preview = PdfPreviewWidget(
            "Select a file to view run report"
        )
        self.run_report_preview.setObjectName("runReportPreview")
        run_layout.addWidget(self.run_report_preview, 1)
        self.run_report_tab_index = self.plot_tabs.addTab(run_tab_container, "Run Report")

        ica_tab_container = QWidget()
        ica_layout = QVBoxLayout()
        ica_layout.setContentsMargins(12, 12, 12, 12)
        ica_layout.setSpacing(8)
        ica_tab_container.setLayout(ica_layout)

        self.ica_preview = PdfPreviewWidget("Select a file to view ICA overview")
        self.ica_preview.setObjectName("icaOverviewPreview")
        ica_layout.addWidget(self.ica_preview, 1)
        self.ica_tab_index = self.plot_tabs.addTab(ica_tab_container, "ICA Components")

        container_layout = self.right_container.layout()
        if container_layout is None:
            container_layout = QVBoxLayout()
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(0)
            self.right_container.setLayout(container_layout)
        container_layout.addWidget(self.plot_tabs)
        self.plot_tabs.currentChanged.connect(self._handle_plot_tab_changed)

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
            #decisionMetricsMessage {
                color: #6b778c;
                font-style: italic;
                font-size: 12px;
            }
            #decisionMetricsRow {
                background-color: #f7f9ff;
                border: 1px solid #e0e7f5;
                border-radius: 6px;
            }
            #decisionMetricsBar {
                background-color: #98a4b5;
            }
            #decisionMetricsName {
                color: #445066;
                font-size: 12px;
                font-weight: 600;
            }
            #decisionMetricsValue {
                color: #2f3b52;
                font-size: 12px;
                font-weight: 600;
            }
            QTabWidget#decisionPlotTabs::pane {
                border: none;
                background-color: #ffffff;
            }
            QTabWidget#decisionPlotTabs QTabBar::tab {
                background-color: transparent;
                border: 1px solid transparent;
                padding: 6px 12px;
                margin-right: 4px;
                color: #5b6c7c;
                font-size: 11px;
                font-weight: 600;
            }
            QTabWidget#decisionPlotTabs QTabBar::tab:selected {
                background-color: #eef3ff;
                border: 1px solid #a5b9d5;
                border-radius: 6px 6px 0 0;
                color: #1f2d3d;
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
        self.detail_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.detail_panel.setMaximumHeight(180)
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

    def _limit_segments(self, counter: Dict[str, int], limit: int = 6) -> OrderedDict[str, int]:
        if not counter:
            return OrderedDict()

        if isinstance(counter, Counter):
            items = counter.most_common()
        else:
            items = list(counter.items())
            items.sort(key=lambda item: item[1], reverse=True)

        if len(items) <= limit:
            return OrderedDict(items)

        kept = items[: limit - 1]
        other_total = sum(value for _, value in items[limit - 1 :])
        ordered = OrderedDict(kept)
        ordered["Other"] = other_total
        return ordered

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
        counter["Retained"] = max(retained, 0)
        counter["Removed"] = max(removed, 0)
        return self._limit_segments(counter, 2)

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
        counter["Kept"] = max(kept, 0)
        counter["Rejected"] = max(bad, 0)
        return self._limit_segments(counter, 2)

    def _ica_component_metrics(self, row: Dict[str, str]) -> OrderedDict[str, int]:
        total = _safe_int(row.get("proc_nComps"))
        removed = len(_coerce_list(row.get("proc_removeComps")))
        if total <= 0 and removed <= 0:
            return OrderedDict()
        removed = min(removed, total) if total > 0 else removed
        retained = max(total - removed, 0)
        counter = OrderedDict()
        counter["Retained"] = max(retained, 0)
        counter["Removed"] = max(removed, 0)
        return self._limit_segments(counter, 2)

    def _build_processing_metrics(
        self, rows: List[Dict[str, str]]
    ) -> List[Tuple[str, str, str]]:
        if not rows:
            return []

        latest = rows[-1]
        metrics: List[Tuple[str, str, str]] = []

        raw_duration = _safe_float(latest.get("proc_xmax_raw"))
        post_duration = _safe_float(latest.get("proc_xmax_post"))
        if raw_duration > 0 or post_duration > 0:
            if raw_duration <= 0:
                raw_duration = post_duration
            retained_pct = (post_duration / raw_duration * 100.0) if raw_duration > 0 else 0.0
            metrics.append(
                (
                    "Data Retained",
                    f"{post_duration:.1f}s of {raw_duration:.1f}s ({retained_pct:.1f}%)",
                    "#2ecc71",
                )
            )

        channel_counts = self._channel_retention_metrics(latest)
        if channel_counts:
            retained = channel_counts.get("Retained", 0)
            removed = channel_counts.get("Removed", 0)
            total = retained + removed
            if total > 0:
                metrics.append(
                    (
                        "Channels Retained",
                        f"{retained} / {total}",
                        "#3498db",
                    )
                )

        epoch_counts = self._epoch_retention_metrics(latest)
        if epoch_counts:
            kept = epoch_counts.get("Kept", 0)
            rejected = epoch_counts.get("Rejected", 0)
            total = kept + rejected
            if total > 0:
                kept_pct = kept / total * 100.0
                metrics.append(
                    (
                        "Epochs Kept",
                        f"{kept} / {total} ({kept_pct:.1f}%)",
                        "#2a9d8f",
                    )
                )

        ica_counts = self._ica_component_metrics(latest)
        if ica_counts:
            retained = ica_counts.get("Retained", 0)
            removed = ica_counts.get("Removed", 0)
            total = retained + removed
            if total > 0:
                metrics.append(
                    (
                        "ICA Components Retained",
                        f"{retained} / {total}",
                        "#6c5ce7",
                    )
                )

        return [metric for metric in metrics if metric[1]]

    def _update_processing_metrics_panel(self) -> None:
        if self.metrics_widget is not None:
            self.metrics_widget.show_message("Select a file to view processing metrics")
        self._update_psd_preview_for_file(None)
        self._update_run_report_preview_for_file(None)
        self._update_ica_preview_for_file(None)
        self._update_run_report_preview_for_file(None)

    def _find_processing_log_for_file(self, file_path: Path) -> Optional[Path]:
        """Find processing log for a file using configuration-based resolution."""
        try:
            asset_path = resolve_asset(file_path, "processing_log", self.preprocessing_log_df, self.config)
            if asset_path and asset_path.exists():
                return asset_path
        except Exception as e:
            print(f"Warning: Error resolving processing log for {file_path}: {e}")
        
        # Fallback to legacy method if configuration-based approach fails
        return self._find_processing_log_for_file_legacy(file_path)
    
    def _find_processing_log_for_file_legacy(self, file_path: Path) -> Optional[Path]:
        """Legacy processing log finding method (kept as fallback)."""
        parent = file_path.parent
        candidates = list(parent.glob("*_processing_log.csv"))
        if not candidates:
            return None

        stem = file_path.stem
        variants = {stem}
        suffixes = ["_comp_epo", "_comp", "_epo", "_postedit", "_preproc", "_raw", "_clean"]
        for suffix in suffixes:
            if stem.endswith(suffix):
                variants.add(stem[: -len(suffix)])

        parts = stem.split("_")
        if len(parts) >= 3:
            variants.add("_".join(parts[:3]))
        if len(parts) >= 2:
            variants.add("_".join(parts[:2]))
        if parts:
            variants.add(parts[0])
            variants.add(f"sub-{parts[0]}")
            variants.add(f"sub_{parts[0]}")

        best_score = -1
        best_path: Optional[Path] = None
        for log_path in candidates:
            log_stem = log_path.stem
            if "_processing_log" in log_stem:
                log_prefix = log_stem.rsplit("_processing_log", 1)[0]
            else:
                log_prefix = log_stem

            score = max(_normalized_prefix_score(log_prefix, variant) for variant in variants)
            if score > best_score:
                best_score = score
                best_path = log_path

        if best_score <= 0:
            return None
        return best_path

    def _psd_reports_dir(self) -> Optional[Path]:
        if self.task_root and (self.task_root / "reports" / "psd_topo").exists():
            return self.task_root / "reports" / "psd_topo"
        if self.exports_dir:
            candidate = self.exports_dir.parent / "reports" / "psd_topo"
            if candidate.exists():
                return candidate
        return None

    def _run_reports_dir(self) -> Optional[Path]:
        if self.task_root and (self.task_root / "reports" / "run_reports").exists():
            return self.task_root / "reports" / "run_reports"
        if self.exports_dir:
            candidate = self.exports_dir.parent / "reports" / "run_reports"
            if candidate.exists():
                return candidate
        return None

    def _ica_reports_dir(self) -> Optional[Path]:
        if self.task_root and (self.task_root / "reports" / "ica_components").exists():
            return self.task_root / "reports" / "ica_components"
        if self.exports_dir:
            candidate = self.exports_dir.parent / "reports" / "ica_components"
            if candidate.exists():
                return candidate
        return None

    def _find_psd_overview_for_file(self, file_path: Path) -> Optional[Path]:
        """Find PSD overview for a file using configuration-based resolution."""
        try:
            asset_path = resolve_asset(file_path, "psd_overview", self.preprocessing_log_df, self.config)
            if asset_path and asset_path.exists():
                return asset_path
        except Exception as e:
            print(f"Warning: Error resolving PSD overview for {file_path}: {e}")
        
        # Fallback to legacy method if configuration-based approach fails
        return self._find_psd_overview_for_file_legacy(file_path)
    
    def _find_psd_overview_for_file_legacy(self, file_path: Path) -> Optional[Path]:
        """Legacy PSD overview finding method (kept as fallback)."""
        psd_dir = self._psd_reports_dir()
        if psd_dir is None:
            return None

        candidates = list(psd_dir.glob("*_psd_topo_figure.png"))
        if not candidates:
            return None

        stem = file_path.stem
        variants = {stem}
        suffixes = ["_comp_epo", "_comp", "_epo", "_postedit", "_preproc", "_raw", "_clean"]
        for suffix in suffixes:
            if stem.endswith(suffix):
                variants.add(stem[: -len(suffix)])

        parts = stem.split("_")
        for length in range(len(parts), 1, -1):
            variants.add("_".join(parts[:length]))
        if parts:
            variants.add(parts[0])
            variants.add(f"sub-{parts[0]}")
            variants.add(f"sub_{parts[0]}")

        best_score = -1
        best_path: Optional[Path] = None
        for candidate in candidates:
            c_stem = candidate.stem
            if "_psd_topo" in c_stem:
                c_stem = c_stem.split("_psd_topo", 1)[0]
            score = max(_normalized_prefix_score(c_stem, variant) for variant in variants)
            if score > best_score:
                best_score = score
                best_path = candidate

        if best_score <= 0:
            return None
        return best_path

    def _find_run_report_for_file(self, file_path: Path) -> Optional[Path]:
        """Find run report for a file using configuration-based resolution."""
        try:
            asset_path = resolve_asset(file_path, "run_report", self.preprocessing_log_df, self.config)
            if asset_path and asset_path.exists():
                return asset_path
        except Exception as e:
            print(f"Warning: Error resolving run report for {file_path}: {e}")
        
        # Fallback to legacy method if configuration-based approach fails
        return self._find_run_report_for_file_legacy(file_path)
    
    def _find_run_report_for_file_legacy(self, file_path: Path) -> Optional[Path]:
        """Legacy run report finding method (kept as fallback)."""
        reports_dir = self._run_reports_dir()
        if reports_dir is None:
            log_warning(
                f"[{_human_timestamp()}] Run report directory missing for {file_path}."
            )
            return None

        log_debug(
            f"[{_human_timestamp()}] Searching run reports in {reports_dir}."
        )
        candidates = list(reports_dir.glob("*_autoclean_report.pdf"))
        if not candidates:
            candidates = list(reports_dir.glob("*.pdf"))
        if not candidates:
            log_info(
                f"[{_human_timestamp()}] No run report PDFs found in {reports_dir}."
            )
            return None

        log_debug(
            f"[{_human_timestamp()}] Evaluating {len(candidates)} run report candidates for {file_path}."
        )
        stem = file_path.stem
        variants = {stem}
        suffixes = ["_comp_epo", "_comp", "_epo", "_postedit", "_preproc", "_raw", "_clean"]
        for suffix in suffixes:
            if stem.endswith(suffix):
                variants.add(stem[: -len(suffix)])

        parts = stem.split("_")
        for length in range(len(parts), 1, -1):
            variants.add("_".join(parts[:length]))
        if parts:
            variants.add(parts[0])
            variants.add(f"sub-{parts[0]}")
            variants.add(f"sub_{parts[0]}")

        best_score = -1
        best_path: Optional[Path] = None
        for candidate in candidates:
            c_stem = candidate.stem
            for needle in ("_autoclean_report", "_report"):
                if needle in c_stem:
                    c_stem = c_stem.split(needle, 1)[0]
                    break
            score = max(_normalized_prefix_score(c_stem, variant) for variant in variants)
            if score > best_score:
                best_score = score
                best_path = candidate

        if best_score <= 0:
            log_info(
                f"[{_human_timestamp()}] No suitable run report match for {file_path}."
            )
            return None
        log_debug(
            f"[{_human_timestamp()}] Best run report match for {file_path}: {best_path} (score={best_score})."
        )
        return best_path

    def _find_ica_overview_for_file(self, file_path: Path) -> Optional[Path]:
        """Find ICA overview for a file using configuration-based resolution."""
        try:
            asset_path = resolve_asset(file_path, "ica_report", self.preprocessing_log_df, self.config)
            if asset_path and asset_path.exists():
                return asset_path
        except Exception as e:
            print(f"Warning: Error resolving ICA overview for {file_path}: {e}")
        
        # Fallback to legacy method if configuration-based approach fails
        return self._find_ica_overview_for_file_legacy(file_path)
    
    def _find_ica_overview_for_file_legacy(self, file_path: Path) -> Optional[Path]:
        """Legacy ICA overview finding method (kept as fallback)."""
        ica_dir = self._ica_reports_dir()
        if ica_dir is None:
            log_warning(
                f"[{_human_timestamp()}] ICA report directory missing for {file_path}."
            )
            return None

        log_debug(
            f"[{_human_timestamp()}] Searching ICA reports in {ica_dir}."
        )
        candidates = list(ica_dir.glob("*.pdf"))
        if not candidates:
            log_info(f"[{_human_timestamp()}] No ICA PDFs found in {ica_dir}.")
            return None

        log_debug(
            f"[{_human_timestamp()}] Evaluating {len(candidates)} ICA report candidates for {file_path}."
        )
        stem = file_path.stem
        variants = {stem}
        suffixes = ["_comp_epo", "_comp", "_epo", "_postedit", "_preproc", "_raw", "_clean"]
        for suffix in suffixes:
            if stem.endswith(suffix):
                variants.add(stem[: -len(suffix)])

        parts = stem.split("_")
        for length in range(len(parts), 1, -1):
            variants.add("_".join(parts[:length]))
        if parts:
            variants.add(parts[0])
            variants.add(f"sub-{parts[0]}")
            variants.add(f"sub_{parts[0]}")

        best_score = -1
        best_path: Optional[Path] = None
        for candidate in candidates:
            c_stem = candidate.stem
            for needle in ("_ica_components", "_components", "_report"):
                if needle in c_stem:
                    c_stem = c_stem.split(needle, 1)[0]
                    break
            score = max(_normalized_prefix_score(c_stem, variant) for variant in variants)
            if score > best_score:
                best_score = score
                best_path = candidate

        if best_score <= 0:
            log_info(
                f"[{_human_timestamp()}] No suitable ICA report match for {file_path}."
            )
            return None
        log_debug(
            f"[{_human_timestamp()}] Best ICA report match for {file_path}: {best_path} (score={best_score})."
        )
        return best_path

    def _update_psd_preview_for_file(self, file_path: Optional[Path]) -> None:
        if self.psd_message_label is None or self.psd_image_label is None or self.psd_scroll is None:
            return

        if file_path is None:
            self.psd_original_pixmap = None
            self.psd_scroll.hide()
            self.psd_message_label.setText("Select a file to view PSD overview")
            self.psd_message_label.show()
            return

        psd_path = self._find_psd_overview_for_file(file_path)
        if psd_path is None:
            self.psd_original_pixmap = None
            self.psd_scroll.hide()
            self.psd_message_label.setText("PSD overview not available for this file")
            self.psd_message_label.show()
            return

        pixmap = QPixmap(str(psd_path))
        if pixmap.isNull():
            self.psd_original_pixmap = None
            self.psd_scroll.hide()
            self.psd_message_label.setText("PSD overview failed to load")
            self.psd_message_label.show()
            return

        self.psd_original_pixmap = pixmap
        self._set_psd_pixmap()
        self.psd_scroll.show()
        self.psd_message_label.hide()

    def _set_psd_pixmap(self, pixmap: Optional[QPixmap] = None) -> None:
        if self.psd_image_label is None or self.psd_scroll is None:
            return
        if pixmap is not None:
            self.psd_original_pixmap = pixmap
        source = self.psd_original_pixmap
        if source is None:
            return
        viewport = self.psd_scroll.viewport()
        if viewport.width() > 0:
            scaled = source.scaled(
                viewport.width(),
                viewport.width(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        else:
            scaled = source
        self.psd_image_label.setPixmap(scaled)

    def _update_run_report_preview_for_file(self, file_path: Optional[Path]) -> None:
        if self.run_report_preview is None:
            return

        if file_path is None:
            self.run_report_preview.clear()
            self.run_report_preview.show_message("Select a file to view run report")
            return

        pdf_path = self._find_run_report_for_file(file_path)
        if pdf_path is None or not pdf_path.exists():
            log_info(
                f"[{_human_timestamp()}] Run report not available for {file_path}; found path={pdf_path}."
            )
            self.run_report_preview.clear()
            self.run_report_preview.show_message("Run report not available for this file")
            return

        try:
            log_debug(
                f"[{_human_timestamp()}] Loading run report preview from {pdf_path}."
            )
            self.run_report_preview.load(pdf_path)
        except Exception as exc:
            log_warning(
                f"[{_human_timestamp()}] Exception while loading run report {pdf_path}: {exc}"
            )
            self.run_report_preview.clear()
            self.run_report_preview.show_message("Failed to load run report preview")

    def _update_ica_preview_for_file(self, file_path: Optional[Path]) -> None:
        if self.ica_preview is None:
            return

        if file_path is None:
            self.ica_preview.clear()
            self.ica_preview.show_message("Select a file to view ICA overview")
            return

        ica_path = self._find_ica_overview_for_file(file_path)
        if ica_path is None or not ica_path.exists():
            log_info(
                f"[{_human_timestamp()}] ICA overview not available for {file_path}; found path={ica_path}."
            )
            self.ica_preview.clear()
            self.ica_preview.show_message("ICA overview not available for this file")
            return

        try:
            log_debug(
                f"[{_human_timestamp()}] Loading ICA overview preview from {ica_path}."
            )
            self.ica_preview.load(ica_path)
        except Exception as exc:
            log_warning(
                f"[{_human_timestamp()}] Exception while loading ICA overview {ica_path}: {exc}"
            )
            self.ica_preview.clear()
            self.ica_preview.show_message("Failed to load ICA overview")


    def _handle_plot_tab_changed(self, index: int) -> None:
        if self.plot_tabs is None:
            return
        if index == self.psd_tab_index:
            self._set_psd_pixmap()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Resize:
            if self.psd_scroll is not None and obj is self.psd_scroll.viewport():
                self._set_psd_pixmap()
        return super().eventFilter(obj, event)

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
        if not metrics:
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
        first_key: Optional[str] = None
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
                    first_key = key

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
                target = first_item
                if first_key is not None:
                    target = self.row_lookup.get(first_key, target)
                if target is not None:
                    self.file_tree.setCurrentItem(target)

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
            self.current_run_record = get_run_record(self.current_run_id)
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

        self._update_psd_preview_for_file(file_path)
        self._update_run_report_preview_for_file(file_path)
        self._update_ica_preview_for_file(file_path)
        self._update_processing_metrics_for_file(file_path)

        if self.status_bar is not None and self.current_display_name:
            self.status_bar.showMessage(f"Queued · {self.current_display_name}")

        self._auto_plot_current()

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
    window.setWindowTitle("AutocleanEEG - Inclusion/Exclusion Review")
    window.showMaximized()
    if not app.styleSheet():
        app.setStyleSheet("")
    sys.exit(app.exec())


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the AutocleanEEG inclusion/exclusion review interface with the "
            "embedded MNE time-series browser."
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
