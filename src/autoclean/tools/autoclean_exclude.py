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
import hashlib
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

    try:  # pragma: no cover - import guard only
        import PyQt6  # noqa: F401
        from PyQt6 import QtPdf  # noqa: F401
    except ImportError as e:  # pragma: no cover - runtime dependency guard
        print("Error: Missing required GUI dependencies for the exclusion tool.")
        print("Reinstall the package to get all dependencies:")
        print("    uv tool install --force autocleaneeg-pipeline")
        print("Or with pip:")
        print("    pip install --force-reinstall autocleaneeg-pipeline")
        print(f"Import error: {e}")
        sys.exit(1)


check_gui_dependencies()


from qtpy.QtCore import (  # noqa: E402
    QAbstractItemModel,
    QModelIndex,
    QObject,
    QEvent,
    QPointF,
    QProcess,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from qtpy.QtGui import QColor, QKeySequence, QPalette, QPixmap  # noqa: E402
from qtpy.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSpinBox,
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


def _enum_name(value: object) -> str:
    """Return the Enum name if available, otherwise string form."""
    if value is None:
        return "None"
    name = getattr(value, 'name', None)
    if name:
        return str(name)
    return str(value)


def _group_channel_removals(channel_removals: List[Dict]) -> Dict[str, List[str]]:
    """Group channel removals by reason code.

    Parameters
    ----------
    channel_removals : list of dict
        List of channel removal entries from metadata["channel_removals"]
        Each entry has: {"channel": str, "reason": str, "source_step": str, "timestamp": str}

    Returns
    -------
    dict
        Dictionary mapping reason codes to lists of channel names
        Example: {"EOG_DROPPED": ["E1", "E8"], "UNCORRELATED": ["E34"]}
    """
    grouped = {}
    for removal in channel_removals:
        reason = removal.get("reason", "UNKNOWN")
        channel = removal.get("channel", "")
        if channel:
            if reason not in grouped:
                grouped[reason] = []
            grouped[reason].append(channel)
    return grouped


def _get_removal_reason_display(reason_code: str) -> Tuple[str, str]:
    """Get human-readable label and color for a removal reason code.

    Parameters
    ----------
    reason_code : str
        Reason code from channel_removals (e.g., "EOG_DROPPED", "UNCORRELATED")

    Returns
    -------
    tuple of (label, color)
        label: Human-readable string (e.g., "EOG")
        color: Hex color code for display (e.g., "#9b59b6")
    """
    reason_map = {
        "EOG_DROPPED": ("EOG", "#9b59b6"),          # Purple
        "OUTER_LAYER": ("Outer Layer", "#3498db"),  # Blue
        "UNCORRELATED": ("Uncorrelated", "#e67e22"), # Orange
        "DEVIATION": ("Deviation", "#e74c3c"),       # Red
        "RANSAC": ("RANSAC", "#d35400"),            # Dark Orange
        "BRIDGED": ("Bridged", "#c0392b"),          # Dark Red
        "RANK": ("Rank", "#8e44ad"),                # Purple
        "MANUAL_EXCLUDE": ("Manual", "#7f8c8d"),    # Gray
        "TEMPLATE_EXCLUDE": ("Template", "#95a5a6"), # Light Gray
        "NOISY": ("Noisy", "#e74c3c"),              # Red
    }
    return reason_map.get(reason_code, (reason_code, "#95a5a6"))


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


def _parse_task_file_config(task_file_path: Path) -> Optional[dict]:
    """Parse a task file and extract its config dictionary.

    Parameters
    ----------
    task_file_path : Path
        Path to the task file to parse

    Returns
    -------
    dict or None
        The config dictionary from the task file, or None if parsing fails
    """
    try:
        with open(task_file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        # Parse the file as AST
        tree = ast.parse(file_content)

        # Look for 'config = {...}' assignment
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'config':
                        # Found the config assignment - evaluate it safely
                        return ast.literal_eval(node.value)

        print(f"Warning: No 'config' dictionary found in {task_file_path}")
        return None

    except Exception as e:
        print(f"Warning: Failed to parse task file {task_file_path}: {e}")
        return None


def _generate_reprocess_task_from_original(
    original_task_path: Path,
    payload: dict,
    new_class_name: str,
    timestamp: str
) -> str:
    """Generate reprocess task by modifying the original task file's AST.

    Parameters
    ----------
    original_task_path : Path
        Path to the original task file
    payload : dict
        Manual fix payload with modifications and metadata
    new_class_name : str
        Name for the reprocess task class
    timestamp : str
        Timestamp string to use for dataset_name

    Returns
    -------
    str
        Complete reprocess task file content
    """
    # Read original task file
    with open(original_task_path, 'r', encoding='utf-8') as f:
        original_source = f.read()

    # Parse AST
    tree = ast.parse(original_source)

    # Extract override data from payload
    fix_type = payload.get('fix_type', 'both')
    bad_channels_raw = payload['modifications']['bad_channels']['modified']
    rejected_ica = payload['modifications']['rejected_ica']['modified']
    file_stem = payload.get('file_stem', 'unknown')

    # Filter out EOG channels that will be dropped before clean_bad_channels runs
    # Common EOG channel names that are typically dropped early in pipeline
    eog_channel_patterns = ['EOG', 'HEOG', 'VEOG', 'hEOG', 'vEOG', 'REOG', 'LEOG']
    bad_channels = [ch for ch in bad_channels_raw if ch not in eog_channel_patterns]

    if len(bad_channels) != len(bad_channels_raw):
        filtered_out = [ch for ch in bad_channels_raw if ch not in bad_channels]
        print(f"[AST DEBUG] Filtered out EOG channels from bad channel list: {filtered_out}")
        print(f"[AST DEBUG] These channels are dropped earlier in the pipeline and cannot be marked as bad")

    print(f"[AST DEBUG] fix_type from payload: '{fix_type}'")
    print(f"[AST DEBUG] bad_channels (after EOG filter): {bad_channels}")
    print(f"[AST DEBUG] rejected_ica: {rejected_ica}")

    # Use provided timestamp to create dataset_name
    dataset_name = f"{file_stem}_{timestamp}"
    print(f"[AST DEBUG] Setting dataset_name in config: {dataset_name}")

    # Add dataset_name to config dictionary
    class ConfigModifier(ast.NodeTransformer):
        def visit_Assign(self, node):
            # Look for 'config = {...}' assignment
            if (len(node.targets) == 1 and
                isinstance(node.targets[0], ast.Name) and
                node.targets[0].id == 'config' and
                isinstance(node.value, ast.Dict)):
                # Add dataset_name to the config dict
                node.value.keys.append(ast.Constant(value='dataset_name'))
                node.value.values.append(ast.Constant(value=dataset_name))
                print(f"[AST DEBUG] Added dataset_name to config dict")
            return node

    tree = ConfigModifier().visit(tree)

    # Find and rename class, update docstring
    class ClassRenamer(ast.NodeTransformer):
        def visit_ClassDef(self, node):
            # Rename class
            node.name = new_class_name

            # Update docstring with override info
            if len(bad_channels) != len(bad_channels_raw):
                docstring = f'''
    Reprocessing task with manual bad channel and ICA component overrides.

    This task reprocesses the original raw data from the beginning with:
    - Manual bad channel list: {bad_channels} (EOG channels excluded, dropped earlier in pipeline)
    - Manual ICA component rejection: {rejected_ica}
    '''
            else:
                docstring = f'''
    Reprocessing task with manual bad channel and ICA component overrides.

    This task reprocesses the original raw data from the beginning with:
    - Manual bad channel list: {bad_channels}
    - Manual ICA component rejection: {rejected_ica}
    '''
            node.body[0] = ast.Expr(value=ast.Constant(value=docstring.strip()))

            return node

    tree = ClassRenamer().visit(tree)

    # Modify run() method based on fix_type
    class MethodModifier(ast.NodeTransformer):
        def __init__(self):
            self.in_run_method = False
            self.ica_classify_modified = False

        def visit_FunctionDef(self, node):
            if node.name == 'run':
                print(f"[AST DEBUG] Entering run() method, setting in_run_method=True")
                self.in_run_method = True

                # Manually visit each statement and build new body with modifications
                new_body = []
                for i, stmt in enumerate(node.body):
                    print(f"[AST DEBUG] Visiting statement {i}: {type(stmt).__name__}")
                    # Visit the statement to apply modifications from visit_Call
                    modified_stmt = self.visit(stmt)
                    new_body.append(modified_stmt)

                    # If ICA fix and this is the classify_ica_components call, insert apply after it
                    if (fix_type in ('ica', 'both') and
                        isinstance(modified_stmt, ast.Expr) and
                        isinstance(modified_stmt.value, ast.Call) and
                        isinstance(modified_stmt.value.func, ast.Attribute) and
                        modified_stmt.value.func.attr == 'classify_ica_components'):
                        # Insert apply_ica_component_rejection call after this
                        print(f"[AST DEBUG] Inserting apply_ica_component_rejection after classify_ica_components")
                        apply_call = ast.Expr(
                            value=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id='self', ctx=ast.Load()),
                                    attr='apply_ica_component_rejection',
                                    ctx=ast.Load()
                                ),
                                args=[],
                                keywords=[
                                    ast.keyword(
                                        arg='manual_rejected_components',
                                        value=ast.List(
                                            elts=[ast.Constant(value=comp) for comp in rejected_ica],
                                            ctx=ast.Load()
                                        )
                                    )
                                ]
                            )
                        )
                        new_body.append(apply_call)

                node.body = new_body
                self.in_run_method = False
            return node

        def visit_Call(self, node):
            func_name = node.func.attr if isinstance(node.func, ast.Attribute) else '?'
            print(f"[AST DEBUG] visit_Call: func={func_name}, in_run_method={self.in_run_method}")

            if not self.in_run_method:
                print(f"[AST DEBUG] Skipping {func_name} (not in run method)")
                return node

            # Modify clean_bad_channels() for channel fixes
            # Support both 'channel' and 'channels' variants
            if (fix_type in ('channel', 'channels', 'both') and
                isinstance(node.func, ast.Attribute) and
                node.func.attr == 'clean_bad_channels'):
                # Only add parameter if there are non-EOG bad channels
                if bad_channels:
                    print(f"[AST DEBUG] Adding manual_bad_channels parameter: {bad_channels}")
                    # Add manual_bad_channels parameter
                    node.keywords.append(
                        ast.keyword(
                            arg='manual_bad_channels',
                            value=ast.List(
                                elts=[ast.Constant(value=ch) for ch in bad_channels],
                                ctx=ast.Load()
                            )
                        )
                    )
                else:
                    print(f"[AST DEBUG] Skipping manual_bad_channels parameter (empty after filtering EOG channels)")

            # Modify classify_ica_components() for ICA fixes
            if (fix_type in ('ica', 'both') and
                isinstance(node.func, ast.Attribute) and
                node.func.attr == 'classify_ica_components'):
                print(f"[AST DEBUG] Modifying classify_ica_components to reject=False")
                self.ica_classify_modified = True
                # Set reject=False
                found_reject = False
                for kw in node.keywords:
                    if kw.arg == 'reject':
                        kw.value = ast.Constant(value=False)
                        found_reject = True
                        break
                if not found_reject:
                    node.keywords.append(
                        ast.keyword(arg='reject', value=ast.Constant(value=False))
                    )

            return node

    tree = MethodModifier().visit(tree)

    # Fix missing locations in AST
    ast.fix_missing_locations(tree)

    # Unparse back to Python code
    modified_code = ast.unparse(tree)

    # Create header comment
    timestamp = payload.get('timestamp', '')
    file_stem = payload.get('file_stem', 'unknown')

    # Build header with EOG filter note if applicable
    eog_note = ""
    if len(bad_channels) != len(bad_channels_raw):
        filtered_out = [ch for ch in bad_channels_raw if ch not in bad_channels]
        eog_note = f"\n# Note: EOG channels {filtered_out} excluded (dropped earlier in pipeline)"

    header = f'''# =============================================================================
#  REPROCESSING TASK WITH MANUAL OVERRIDES
# =============================================================================
# This task was automatically generated to reprocess EEG data with manual
# bad channel and ICA component overrides from the review GUI.
#
# Generated: {timestamp}
# Original file: {file_stem}
# Fix type: {fix_type}
#
# Manual Overrides:
# - Bad channels: {len(bad_channels)} channels{eog_note}
# - ICA components: {len(rejected_ica)} components
# =============================================================================

'''

    return header + modified_code


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

        # Create tabs for file list and reprocess
        left_tabs = QTabWidget()
        left_tabs.setObjectName("leftPanelTabs")

        # File List Tab
        file_list_container = QWidget()
        file_list_layout = QVBoxLayout()
        file_list_layout.setContentsMargins(0, 0, 0, 0)
        file_list_layout.setSpacing(0)
        file_list_container.setLayout(file_list_layout)

        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderHidden(True)
        self.file_tree.itemClicked.connect(self.onFileSelect)
        self.file_tree.setObjectName("fileTree")
        file_list_layout.addWidget(self.file_tree, 1)

        left_tabs.addTab(file_list_container, "File List")

        # Reprocess Tab
        self.reprocess_widget = ReprocessWidget()
        left_tabs.addTab(self.reprocess_widget, "Reprocess")

        self._apply_light_palette(left_tabs)
        self.left_layout.addWidget(left_tabs, 1)

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
            # Restore previously marked bad epochs before plotting
            self._restore_bad_epochs_to_plot()

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

            # Sync browser's bad_epochs list from drop_log and update visuals
            self._sync_browser_bad_epochs()

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

    def _restore_bad_epochs_to_plot(self) -> None:
        """Restore previously marked bad epochs to the current epochs before plotting."""
        # This method will be overridden in ExclusionFileSelector to restore bad epochs
        pass

    def _sync_browser_bad_epochs(self) -> None:
        """Sync browser's bad_epochs list from drop_log and update visuals."""
        if not self.plot_widget or not hasattr(self.plot_widget, 'mne'):
            return

        # Extract bad epoch numbers from drop_log
        bad_epoch_nums = []
        if hasattr(self.current_epochs, 'drop_log') and hasattr(self.current_epochs, 'selection'):
            for idx, log in enumerate(self.current_epochs.drop_log):
                # Check if this epoch is marked as USER-rejected
                if log and any(isinstance(entry, str) and entry.upper() == 'USER' for entry in log):
                    # Get the actual epoch number from selection
                    if idx < len(self.current_epochs.selection):
                        bad_epoch_nums.append(self.current_epochs.selection[idx])

        # Update browser's bad_epochs list
        self.plot_widget.mne.bad_epochs = sorted(bad_epoch_nums)
        print(f"[EPOCH DEBUG] Synced {len(bad_epoch_nums)} bad epochs to browser: {bad_epoch_nums}")

        # Trigger visual updates
        if hasattr(self.plot_widget.mne, 'overview_bar'):
            self.plot_widget.mne.overview_bar.update_bad_epochs()
        if hasattr(self.plot_widget, 'update_bad_epoch_highlights'):
            self.plot_widget.update_bad_epoch_highlights()

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

        # Add styled button toolbar
        button_toolbar = QWidget()
        button_toolbar.setObjectName("qaButtonToolbar")
        button_toolbar_layout = QHBoxLayout()
        button_toolbar_layout.setContentsMargins(8, 8, 8, 8)
        button_toolbar_layout.setSpacing(8)

        self.export_to_qa_btn = QPushButton("Export to QA")
        self.export_to_qa_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_to_qa_btn.setMinimumHeight(34)
        self.export_to_qa_btn.setMaximumWidth(180)
        self.export_to_qa_btn.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.export_to_qa_btn.setToolTip("Export cleaned .set files (bad epochs removed) to qa/ folder with unified preprocessing log")
        button_toolbar_layout.addWidget(self.export_to_qa_btn)

        self.open_qa_folder_btn = QPushButton("Open QA Folder")
        self.open_qa_folder_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_qa_folder_btn.setMinimumHeight(34)
        self.open_qa_folder_btn.setMaximumWidth(180)
        self.open_qa_folder_btn.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.open_qa_folder_btn.setToolTip("Open the QA folder in your file browser")
        button_toolbar_layout.addWidget(self.open_qa_folder_btn)

        button_toolbar_layout.addStretch(1)

        button_toolbar.setLayout(button_toolbar_layout)
        button_toolbar.setStyleSheet(
            """
            #qaButtonToolbar {
                background-color: #f7f9fc;
                border: 1px solid #d9e2ec;
                border-radius: 8px;
            }
            #qaButtonToolbar QPushButton {
                background-color: #ffffff;
                border: 1px solid #d9e2ec;
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: 600;
                color: #1f2d3d;
            }
            #qaButtonToolbar QPushButton:hover {
                border-color: #3a7bd5;
                color: #1a4fa3;
            }
            #qaButtonToolbar QPushButton:pressed {
                background-color: #ecf2fb;
            }
            #qaButtonToolbar QPushButton:disabled {
                background-color: #f1f3f6;
                color: #9aa5b1;
                border-color: #dfe4ea;
            }
            """
        )
        layout.addWidget(button_toolbar)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMaximumHeight(200)

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


class JsonMetadataViewer(QWidget):
    """Interactive JSON metadata viewer with tree display and search."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        self.setLayout(layout)

        # Toolbar with controls
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(8)
        toolbar.setLayout(toolbar_layout)

        # Search bar
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search keys...")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.textChanged.connect(self._on_search_changed)
        toolbar_layout.addWidget(self.search_input, 1)

        # Expand/Collapse buttons
        self.expand_btn = QPushButton("Expand All")
        self.expand_btn.clicked.connect(self._expand_all)
        self.expand_btn.setMaximumWidth(100)
        toolbar_layout.addWidget(self.expand_btn)

        self.collapse_btn = QPushButton("Collapse All")
        self.collapse_btn.clicked.connect(self._collapse_all)
        self.collapse_btn.setMaximumWidth(100)
        toolbar_layout.addWidget(self.collapse_btn)

        layout.addWidget(toolbar)

        # Tree view for JSON
        self.tree_view = QTreeView()
        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setHeaderHidden(False)
        self.tree_view.setSortingEnabled(False)
        self.tree_view.setStyleSheet("""
            QTreeView {
                background-color: #ffffff;
                border: 1px solid #d9e2ec;
                border-radius: 6px;
                font-size: 12px;
            }
            QTreeView::item {
                padding: 4px;
            }
            QTreeView::item:selected {
                background-color: #e3f2fd;
                color: #1565c0;
            }
            QTreeView::branch:has-children:closed {
                image: url(none);
            }
            QTreeView::branch:has-children:open {
                image: url(none);
            }
        """)
        layout.addWidget(self.tree_view)

        # Message label for empty state
        self.message_label = QLabel("Select a file to view JSON metadata")
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setStyleSheet("color: #95a5a6; font-size: 14px;")
        layout.addWidget(self.message_label)

        self.tree_view.hide()
        self._current_data = None

    def load_json(self, json_path: Optional[Path]) -> None:
        """Load and display JSON from file."""
        if not json_path or not json_path.exists():
            self.message_label.setText("JSON metadata not found")
            self.tree_view.hide()
            self.message_label.show()
            self._current_data = None
            return

        try:
            data = json.loads(json_path.read_text())
            self._current_data = data
            model = _JsonTreeModel(data)
            self.tree_view.setModel(model)
            self.tree_view.expandToDepth(1)  # Expand first level by default
            self.tree_view.resizeColumnToContents(0)
            self.tree_view.show()
            self.message_label.hide()
        except Exception as e:
            self.message_label.setText(f"Error loading JSON: {e}")
            self.tree_view.hide()
            self.message_label.show()
            self._current_data = None

    def _expand_all(self) -> None:
        """Expand all tree nodes."""
        if self.tree_view.model():
            self.tree_view.expandAll()

    def _collapse_all(self) -> None:
        """Collapse all tree nodes."""
        if self.tree_view.model():
            self.tree_view.collapseAll()

    def _on_search_changed(self, text: str) -> None:
        """Filter tree view based on search text."""
        # TODO: Implement search/filter functionality
        pass


class ReprocessWidget(QWidget):
    """Widget for editing bad channels and rejected ICA components for reprocessing."""

    # Signal emitted when reprocess values change
    values_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)
        self.setLayout(layout)

        # Store current state
        self.valid_channels: list[str] = []
        self.max_components: int = 0
        self.original_bad_channels: list[str] = []
        self.original_rejected_ica: list[int] = []
        self._suppress_change_signal: bool = False
        self._modification_mode: Optional[str] = None  # 'channels', 'components', or None

        # Create horizontal layout for side-by-side groups
        groups_layout = QHBoxLayout()
        groups_layout.setSpacing(12)

        # Bad Channels Section
        self.channels_group = QGroupBox("Bad Channels")
        channels_layout = QVBoxLayout()
        channels_layout.setSpacing(6)

        self.channels_list = QListWidget()
        self.channels_list.setStyleSheet("font-size: 12px;")
        channels_layout.addWidget(self.channels_list, 1)

        channels_controls = QHBoxLayout()
        self.channel_combo = QComboBox()
        self.channel_combo.setEditable(True)
        self.channel_combo.setPlaceholderText("Select channel...")
        channels_controls.addWidget(self.channel_combo, 1)

        self.add_channel_btn = QPushButton("Add")
        self.add_channel_btn.clicked.connect(self._add_channel)
        self.add_channel_btn.setMaximumWidth(60)
        channels_controls.addWidget(self.add_channel_btn)

        self.remove_channel_btn = QPushButton("Remove")
        self.remove_channel_btn.clicked.connect(self._remove_channel)
        self.remove_channel_btn.setMaximumWidth(80)
        channels_controls.addWidget(self.remove_channel_btn)

        channels_layout.addLayout(channels_controls)
        self.channels_group.setLayout(channels_layout)
        groups_layout.addWidget(self.channels_group, 1)

        # Rejected ICA Components Section
        self.ica_group = QGroupBox("Rejected ICA Components")
        ica_layout = QVBoxLayout()
        ica_layout.setSpacing(6)

        self.ica_list = QListWidget()
        self.ica_list.setStyleSheet("font-size: 12px;")
        ica_layout.addWidget(self.ica_list, 1)

        ica_controls = QHBoxLayout()
        self.ica_spinbox = QSpinBox()
        self.ica_spinbox.setMinimum(0)
        self.ica_spinbox.setMaximum(0)
        self.ica_spinbox.setPrefix("Component ")
        ica_controls.addWidget(self.ica_spinbox, 1)

        self.add_ica_btn = QPushButton("Add")
        self.add_ica_btn.clicked.connect(self._add_ica_component)
        self.add_ica_btn.setMaximumWidth(60)
        ica_controls.addWidget(self.add_ica_btn)

        self.remove_ica_btn = QPushButton("Remove")
        self.remove_ica_btn.clicked.connect(self._remove_ica_component)
        self.remove_ica_btn.setMaximumWidth(80)
        ica_controls.addWidget(self.remove_ica_btn)

        ica_layout.addLayout(ica_controls)
        self.ica_group.setLayout(ica_layout)
        groups_layout.addWidget(self.ica_group, 1)

        # Add side-by-side groups to main layout
        layout.addLayout(groups_layout, 1)

        # Create overlay labels for mutual exclusivity feedback
        # Channels overlay (shown when ICA is being modified)
        self.channels_overlay = QLabel(self.channels_group)
        self.channels_overlay.setAlignment(Qt.AlignCenter)
        self.channels_overlay.setWordWrap(True)
        self.channels_overlay.setText(
            "⚠️ Channel Selection Disabled\n\n"
            "ICA component changes are active.\n"
            "Channel modifications are unavailable\n"
            "until you reset."
        )
        self.channels_overlay.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 0.92);
                border: 2px solid #e67e22;
                border-radius: 6px;
                padding: 20px;
                font-size: 13px;
                font-weight: 600;
                color: #d35400;
            }
        """)
        self.channels_overlay.hide()

        # ICA overlay (shown when channels are being modified)
        self.ica_overlay = QLabel(self.ica_group)
        self.ica_overlay.setAlignment(Qt.AlignCenter)
        self.ica_overlay.setWordWrap(True)
        self.ica_overlay.setText(
            "⚠️ ICA Components Disabled\n\n"
            "Channel modifications require new\n"
            "ICA decomposition. ICA component\n"
            "selection is unavailable until you reset."
        )
        self.ica_overlay.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 0.92);
                border: 2px solid #e67e22;
                border-radius: 6px;
                padding: 20px;
                font-size: 13px;
                font-weight: 600;
                color: #d35400;
            }
        """)
        self.ica_overlay.hide()

        # Button bar for reset and reprocess
        button_bar = QHBoxLayout()
        button_bar.setSpacing(8)

        # Reset button
        reset_btn = QPushButton("Reset to Original")
        reset_btn.clicked.connect(self._reset_to_original)
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f4f8;
                border: 1px solid #cbd5e0;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #e2e8f0;
                border-color: #a0aec0;
            }
        """)
        button_bar.addWidget(reset_btn)

        # Reprocess button
        self.reprocess_btn = QPushButton("Reprocess with Overrides")
        self.reprocess_btn.clicked.connect(self._handle_reprocess_clicked)
        self.reprocess_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: 1px solid #2980b9;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #2980b9;
                border-color: #21618c;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                border-color: #95a5a6;
                color: #7f8c8d;
            }
        """)
        self.reprocess_btn.setEnabled(False)  # Disabled until changes are made
        button_bar.addWidget(self.reprocess_btn)

        layout.addLayout(button_bar)

        # Changes summary widget
        self.changes_summary_widget = QWidget()
        self.changes_summary_layout = QVBoxLayout()
        self.changes_summary_layout.setContentsMargins(12, 12, 12, 12)
        self.changes_summary_layout.setSpacing(8)
        self.changes_summary_widget.setLayout(self.changes_summary_layout)

        # Summary title
        self.summary_title = QLabel("Changes Summary")
        self.summary_title.setAlignment(Qt.AlignCenter)
        self.summary_title.setStyleSheet("""
            font-size: 14px;
            font-weight: 600;
            color: #2c3e50;
            padding-bottom: 6px;
            border-bottom: 2px solid #ecf0f1;
        """)
        self.changes_summary_layout.addWidget(self.summary_title)

        # Channels changes section
        self.channels_changes_widget = QWidget()
        self.channels_changes_layout = QVBoxLayout()
        self.channels_changes_layout.setContentsMargins(0, 4, 0, 4)
        self.channels_changes_layout.setSpacing(4)
        self.channels_changes_widget.setLayout(self.channels_changes_layout)

        self.channels_summary_label = QLabel()
        self.channels_summary_label.setStyleSheet("font-size: 12px; font-weight: 600; color: #34495e;")
        self.channels_changes_layout.addWidget(self.channels_summary_label)

        self.channels_chips_widget = QWidget()
        self.channels_chips_layout = QHBoxLayout()
        self.channels_chips_layout.setContentsMargins(0, 0, 0, 0)
        self.channels_chips_layout.setSpacing(4)
        self.channels_chips_layout.setAlignment(Qt.AlignLeft)
        self.channels_chips_widget.setLayout(self.channels_chips_layout)
        self.channels_changes_layout.addWidget(self.channels_chips_widget)

        self.changes_summary_layout.addWidget(self.channels_changes_widget)

        # ICA changes section
        self.ica_changes_widget = QWidget()
        self.ica_changes_layout = QVBoxLayout()
        self.ica_changes_layout.setContentsMargins(0, 4, 0, 4)
        self.ica_changes_layout.setSpacing(4)
        self.ica_changes_widget.setLayout(self.ica_changes_layout)

        self.ica_summary_label = QLabel()
        self.ica_summary_label.setStyleSheet("font-size: 12px; font-weight: 600; color: #34495e;")
        self.ica_changes_layout.addWidget(self.ica_summary_label)

        self.ica_chips_widget = QWidget()
        self.ica_chips_layout = QHBoxLayout()
        self.ica_chips_layout.setContentsMargins(0, 0, 0, 0)
        self.ica_chips_layout.setSpacing(4)
        self.ica_chips_layout.setAlignment(Qt.AlignLeft)
        self.ica_chips_widget.setLayout(self.ica_chips_layout)
        self.ica_changes_layout.addWidget(self.ica_chips_widget)

        self.changes_summary_layout.addWidget(self.ica_changes_widget)

        # Message label for empty state
        self.message_label = QLabel("Select a file to edit reprocessing parameters")
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setStyleSheet("color: #95a5a6; font-size: 13px; padding: 20px;")

        # Stack to switch between summary and message
        self.bottom_stack = QStackedLayout()
        self.bottom_stack.addWidget(self.message_label)
        self.bottom_stack.addWidget(self.changes_summary_widget)
        self.bottom_stack.setCurrentWidget(self.message_label)
        layout.addLayout(self.bottom_stack)

    def resizeEvent(self, event) -> None:
        """Handle resize events to position overlays correctly."""
        super().resizeEvent(event)
        # Position overlays to cover their respective group boxes
        if hasattr(self, 'channels_overlay'):
            self.channels_overlay.setGeometry(self.channels_group.rect())
        if hasattr(self, 'ica_overlay'):
            self.ica_overlay.setGeometry(self.ica_group.rect())

    def _update_section_states(self) -> None:
        """Update enabled/disabled state of sections based on modification mode."""
        if self._modification_mode == 'channels':
            # Channels being modified - disable ICA section
            self.ica_spinbox.setEnabled(False)
            self.add_ica_btn.setEnabled(False)
            self.remove_ica_btn.setEnabled(False)
            self.ica_list.setEnabled(False)
            self.ica_overlay.show()
            self.ica_overlay.raise_()

            # Enable channels section
            self.channel_combo.setEnabled(True)
            self.add_channel_btn.setEnabled(True)
            self.remove_channel_btn.setEnabled(True)
            self.channels_list.setEnabled(True)
            self.channels_overlay.hide()

        elif self._modification_mode == 'components':
            # Components being modified - disable channels section
            self.channel_combo.setEnabled(False)
            self.add_channel_btn.setEnabled(False)
            self.remove_channel_btn.setEnabled(False)
            self.channels_list.setEnabled(False)
            self.channels_overlay.show()
            self.channels_overlay.raise_()

            # Enable ICA section
            self.ica_spinbox.setEnabled(True)
            self.add_ica_btn.setEnabled(True)
            self.remove_ica_btn.setEnabled(True)
            self.ica_list.setEnabled(True)
            self.ica_overlay.hide()

        else:
            # No modifications - enable both sections
            self.channel_combo.setEnabled(True)
            self.add_channel_btn.setEnabled(True)
            self.remove_channel_btn.setEnabled(True)
            self.channels_list.setEnabled(True)
            self.channels_overlay.hide()

            self.ica_spinbox.setEnabled(True)
            self.add_ica_btn.setEnabled(True)
            self.remove_ica_btn.setEnabled(True)
            self.ica_list.setEnabled(True)
            self.ica_overlay.hide()

    def load_from_metadata(self, metadata: dict) -> None:
        """Load bad channels and ICA components from metadata."""
        # Suppress change signals during loading
        self._suppress_change_signal = True

        # Clear current state
        self.channels_list.clear()
        self.ica_list.clear()
        self.channel_combo.clear()

        # Extract data from metadata
        bad_channels = metadata.get("bad_channels", [])
        rejected_ica = metadata.get("rejected_ica", [])
        valid_channels = metadata.get("valid_channels", [])
        max_components = metadata.get("max_components", 0)

        # Store original values and validation data
        self.original_bad_channels = bad_channels.copy()
        self.original_rejected_ica = rejected_ica.copy()
        self.valid_channels = valid_channels
        self.max_components = max_components

        # Populate channel combo
        self.channel_combo.addItems(valid_channels)

        # Set ICA spinbox range
        if max_components > 0:
            self.ica_spinbox.setMaximum(max_components - 1)

        # Populate lists
        for channel in bad_channels:
            self.channels_list.addItem(channel)

        for component in rejected_ica:
            self.ica_list.addItem(f"Component {component}")

        # Clear modification mode and re-enable both sections when loading new file
        self._modification_mode = None
        self._update_section_states()

        # Re-enable change signals
        self._suppress_change_signal = False

        # Update changes summary display
        self._update_changes_summary()

    def _add_channel(self) -> None:
        """Add selected channel to bad channels list."""
        channel = self.channel_combo.currentText().strip().upper()
        if not channel:
            return

        # Validate channel
        if channel not in self.valid_channels:
            QMessageBox.warning(
                self,
                "Invalid Channel",
                f"'{channel}' is not a valid channel.\nValid channels: {', '.join(self.valid_channels[:10])}..."
            )
            return

        # Check if already in list
        items = [self.channels_list.item(i).text() for i in range(self.channels_list.count())]
        if channel in items:
            return

        self.channels_list.addItem(channel)
        self.channel_combo.setCurrentIndex(-1)

        # Set modification mode to channels and update UI states
        if self._modification_mode is None:
            self._modification_mode = 'channels'
            self._update_section_states()

        self._emit_change_signal()

    def _remove_channel(self) -> None:
        """Remove selected channel from list."""
        current_item = self.channels_list.currentItem()
        if current_item:
            self.channels_list.takeItem(self.channels_list.row(current_item))

            # Set modification mode to channels and update UI states
            if self._modification_mode is None:
                self._modification_mode = 'channels'
                self._update_section_states()

            self._emit_change_signal()

    def _add_ica_component(self) -> None:
        """Add ICA component to rejected list."""
        component = self.ica_spinbox.value()
        component_text = f"Component {component}"

        # Check if already in list
        items = [self.ica_list.item(i).text() for i in range(self.ica_list.count())]
        if component_text in items:
            return

        self.ica_list.addItem(component_text)

        # Set modification mode to components and update UI states
        if self._modification_mode is None:
            self._modification_mode = 'components'
            self._update_section_states()

        self._emit_change_signal()

    def _remove_ica_component(self) -> None:
        """Remove selected ICA component from list."""
        current_item = self.ica_list.currentItem()
        if current_item:
            self.ica_list.takeItem(self.ica_list.row(current_item))

            # Set modification mode to components and update UI states
            if self._modification_mode is None:
                self._modification_mode = 'components'
                self._update_section_states()

            self._emit_change_signal()

    def _reset_to_original(self) -> None:
        """Reset to original values from metadata."""
        self.channels_list.clear()
        self.ica_list.clear()

        for channel in self.original_bad_channels:
            self.channels_list.addItem(channel)

        for component in self.original_rejected_ica:
            self.ica_list.addItem(f"Component {component}")

        # Clear modification mode and re-enable both sections
        self._modification_mode = None
        self._update_section_states()

        self._emit_change_signal()

    def get_current_values(self) -> dict:
        """Get current bad channels and rejected ICA components."""
        bad_channels = [
            self.channels_list.item(i).text()
            for i in range(self.channels_list.count())
        ]

        rejected_ica = [
            int(self.ica_list.item(i).text().replace("Component ", ""))
            for i in range(self.ica_list.count())
        ]

        return {
            "bad_channels": bad_channels,
            "rejected_ica": rejected_ica
        }

    def has_changes(self) -> bool:
        """Check if current values differ from original values."""
        current = self.get_current_values()
        return (
            set(current["bad_channels"]) != set(self.original_bad_channels)
            or set(current["rejected_ica"]) != set(self.original_rejected_ica)
        )

    def get_changes_diff(self) -> dict:
        """Get a diff of changes from original values.

        Returns
        -------
        dict
            Dictionary containing:
            - bad_channels: {original, modified, added, removed}
            - rejected_ica: {original, modified, added, removed}
            - has_channel_changes: bool
            - has_ica_changes: bool
        """
        current = self.get_current_values()
        current_channels = set(current["bad_channels"])
        original_channels = set(self.original_bad_channels)
        current_ica = set(current["rejected_ica"])
        original_ica = set(self.original_rejected_ica)

        return {
            "bad_channels": {
                "original": sorted(self.original_bad_channels),
                "modified": sorted(current["bad_channels"]),
                "added": sorted(current_channels - original_channels),
                "removed": sorted(original_channels - current_channels),
            },
            "rejected_ica": {
                "original": sorted(self.original_rejected_ica),
                "modified": sorted(current["rejected_ica"]),
                "added": sorted(current_ica - original_ica),
                "removed": sorted(original_ica - current_ica),
            },
            "has_channel_changes": current_channels != original_channels,
            "has_ica_changes": current_ica != original_ica,
        }

    def _update_changes_summary(self) -> None:
        """Update the changes summary widget with current additions/deletions."""
        if not self.valid_channels:
            # No file loaded - show message
            self.bottom_stack.setCurrentWidget(self.message_label)
            self.reprocess_btn.setEnabled(False)
            return

        diff = self.get_changes_diff()
        has_any_changes = diff["has_channel_changes"] or diff["has_ica_changes"]

        if not has_any_changes:
            # No changes - show message
            self.bottom_stack.setCurrentWidget(self.message_label)
            self.message_label.setText("No modifications yet")
            self.message_label.setStyleSheet("color: #95a5a6; font-size: 13px; padding: 20px;")
            self.reprocess_btn.setEnabled(False)
            return

        # Show changes summary and enable reprocess button
        self.bottom_stack.setCurrentWidget(self.changes_summary_widget)
        self.reprocess_btn.setEnabled(True)

        # Update channels section
        ch_added = diff["bad_channels"]["added"]
        ch_removed = diff["bad_channels"]["removed"]

        if ch_added or ch_removed:
            self.channels_summary_label.setText(
                f"Channels:  +{len(ch_added)}  -{len(ch_removed)}"
            )
            self.channels_changes_widget.show()

            # Clear existing chips
            while self.channels_chips_layout.count():
                child = self.channels_chips_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

            # Add chips for added channels
            for ch in ch_added[:10]:  # Limit to 10 to avoid overflow
                chip = QLabel(f"+ {ch}")
                chip.setStyleSheet("""
                    background-color: #d4edda;
                    color: #155724;
                    border: 1px solid #c3e6cb;
                    border-radius: 4px;
                    padding: 3px 8px;
                    font-size: 11px;
                    font-weight: 600;
                """)
                self.channels_chips_layout.addWidget(chip)

            # Add chips for removed channels
            for ch in ch_removed[:10]:  # Limit to 10 to avoid overflow
                chip = QLabel(f"− {ch}")
                chip.setStyleSheet("""
                    background-color: #f8d7da;
                    color: #721c24;
                    border: 1px solid #f5c6cb;
                    border-radius: 4px;
                    padding: 3px 8px;
                    font-size: 11px;
                    font-weight: 600;
                """)
                self.channels_chips_layout.addWidget(chip)

            # Add ellipsis if there are more items
            total_shown = min(len(ch_added), 10) + min(len(ch_removed), 10)
            total_items = len(ch_added) + len(ch_removed)
            if total_items > total_shown:
                more_label = QLabel(f"... +{total_items - total_shown} more")
                more_label.setStyleSheet("color: #7f8c8d; font-size: 11px; padding: 3px 8px;")
                self.channels_chips_layout.addWidget(more_label)

        else:
            self.channels_changes_widget.hide()

        # Update ICA section
        ica_added = diff["rejected_ica"]["added"]
        ica_removed = diff["rejected_ica"]["removed"]

        if ica_added or ica_removed:
            self.ica_summary_label.setText(
                f"ICA Components:  +{len(ica_added)}  -{len(ica_removed)}"
            )
            self.ica_changes_widget.show()

            # Clear existing chips
            while self.ica_chips_layout.count():
                child = self.ica_chips_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

            # Add chips for added components
            for ic in ica_added[:10]:  # Limit to 10 to avoid overflow
                chip = QLabel(f"+ IC{ic}")
                chip.setStyleSheet("""
                    background-color: #d4edda;
                    color: #155724;
                    border: 1px solid #c3e6cb;
                    border-radius: 4px;
                    padding: 3px 8px;
                    font-size: 11px;
                    font-weight: 600;
                """)
                self.ica_chips_layout.addWidget(chip)

            # Add chips for removed components
            for ic in ica_removed[:10]:  # Limit to 10 to avoid overflow
                chip = QLabel(f"− IC{ic}")
                chip.setStyleSheet("""
                    background-color: #f8d7da;
                    color: #721c24;
                    border: 1px solid #f5c6cb;
                    border-radius: 4px;
                    padding: 3px 8px;
                    font-size: 11px;
                    font-weight: 600;
                """)
                self.ica_chips_layout.addWidget(chip)

            # Add ellipsis if there are more items
            total_shown = min(len(ica_added), 10) + min(len(ica_removed), 10)
            total_items = len(ica_added) + len(ica_removed)
            if total_items > total_shown:
                more_label = QLabel(f"... +{total_items - total_shown} more")
                more_label.setStyleSheet("color: #7f8c8d; font-size: 11px; padding: 3px 8px;")
                self.ica_chips_layout.addWidget(more_label)

        else:
            self.ica_changes_widget.hide()

    def _emit_change_signal(self) -> None:
        """Emit values_changed signal if not suppressed."""
        if not self._suppress_change_signal:
            self.values_changed.emit()
            self._update_changes_summary()

    def _handle_reprocess_clicked(self) -> None:
        """Handle reprocess button click - trigger reprocessing with current overrides."""
        # Get parent ExclusionFileSelector instance
        parent = self.parent()
        while parent and not isinstance(parent, ExclusionFileSelector):
            parent = parent.parent()

        if not parent:
            QMessageBox.warning(
                self,
                "Error",
                "Could not access parent window for reprocessing."
            )
            return

        # Trigger reprocessing through parent
        parent._trigger_reprocess_with_overrides()


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
        self.reprocess_widget: Optional[ReprocessWidget] = None
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
        self.json_metadata_viewer: Optional[JsonMetadataViewer] = None
        self.time_series_tab_index: Optional[int] = None
        self.psd_tab_index: Optional[int] = None
        self.run_report_tab_index: Optional[int] = None
        self.ica_tab_index: Optional[int] = None
        self.json_tab_index: Optional[int] = None

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

        # JSON Metadata tab
        self.json_metadata_viewer = JsonMetadataViewer()
        self.json_tab_index = self.plot_tabs.addTab(self.json_metadata_viewer, "Metadata")

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

        # No more timers - we'll use event-driven approach

        if hasattr(self, "file_tree") and self.file_tree is not None:
            try:
                self.file_tree.itemSelectionChanged.connect(self._handle_tree_selection_changed)
            except Exception:
                pass

        # Connect reprocess widget signal
        if hasattr(self, "reprocess_widget") and self.reprocess_widget is not None:
            try:
                self.reprocess_widget.values_changed.connect(self._handle_reprocess_changed)
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

        shortcut_hint = QLabel("P Pass • F Fail • R Review • C Clear • ↑↓ Navigate")
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

            # Create global shortcuts that work regardless of focus
            shortcut = QShortcut(QKeySequence(meta["shortcut"]), self)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
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

        # Create global shortcut for clear that works regardless of focus
        clear_shortcut = QShortcut(QKeySequence("C"), self)
        clear_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        clear_shortcut.activated.connect(partial(self._set_status, "UNSET"))
        self._shortcuts["CLEAR"] = clear_shortcut

        # Add up/down arrow navigation shortcuts
        up_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.MoveToPreviousLine), self)
        up_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        up_shortcut.activated.connect(self._navigate_up)
        self._shortcuts["UP"] = up_shortcut

        down_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.MoveToNextLine), self)
        down_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        down_shortcut.activated.connect(self._navigate_down)
        self._shortcuts["DOWN"] = down_shortcut

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
            QTabWidget#leftPanelTabs::pane {
                border: 1px solid #dde4ef;
                background-color: #ffffff;
                border-radius: 4px;
            }
            QTabWidget#leftPanelTabs QTabBar::tab {
                background-color: transparent;
                border: 1px solid transparent;
                padding: 6px 12px;
                margin-right: 4px;
                color: #5b6c7c;
                font-size: 11px;
                font-weight: 600;
            }
            QTabWidget#leftPanelTabs QTabBar::tab:selected {
                background-color: #eef3ff;
                border: 1px solid #a5b9d5;
                border-radius: 6px 6px 0 0;
                border-bottom: none;
                color: #1f2d3d;
            }
            QTabWidget#leftPanelTabs QTabBar::tab:hover {
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

        # Connect and style Export to QA button
        self.metrics_widget.export_to_qa_btn.clicked.connect(self._batch_export_to_qa)
        export_icon = self.style().standardIcon(QStyle.SP_DialogSaveButton)
        if not export_icon.isNull():
            self.metrics_widget.export_to_qa_btn.setIcon(export_icon)
            self.metrics_widget.export_to_qa_btn.setIconSize(QSize(18, 18))

        # Connect and style Open QA Folder button
        self.metrics_widget.open_qa_folder_btn.clicked.connect(self._open_qa_folder)
        folder_icon = self.style().standardIcon(QStyle.SP_DirIcon)
        if not folder_icon.isNull():
            self.metrics_widget.open_qa_folder_btn.setIcon(folder_icon)
            self.metrics_widget.open_qa_folder_btn.setIconSize(QSize(18, 18))

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

        # Update task_root to point to parent directory (task output folder)
        # exports_dir is typically: /path/to/output/TaskName/exports
        # task_root should be: /path/to/output/TaskName
        self.task_root = root.parent

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
        self._update_json_metadata_for_file(None)
        self._update_reprocess_for_file(None)
        self._update_run_report_preview_for_file(None)

    def _find_processing_log_for_file(self, file_path: Path) -> Optional[Path]:
        """Find processing log for a file using configuration-based resolution."""
        try:
            asset_path = resolve_asset(file_path, "processing_log", self.preprocessing_log_df, self.config)
            if asset_path and asset_path.exists():
                return asset_path
        except Exception as e:
            print(f"Warning: Error resolving processing log for {file_path}: {e}")
        
        return None

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
        """Find PSD overview for a file using task root."""
        if not self.task_root:
            return None

        stem = strip_suffixes(file_path.stem, config=self.config)
        psd_path = self.task_root / "reports" / "psd_topo" / f"{stem}_psd_topo_figure.png"

        if psd_path.exists():
            return psd_path
        return None

    def _find_run_report_for_file(self, file_path: Path) -> Optional[Path]:
        """Find run report for a file using task root."""
        if not self.task_root:
            return None

        stem = strip_suffixes(file_path.stem, config=self.config)
        report_path = self.task_root / "reports" / "run_reports" / f"{stem}_autoclean_report.pdf"

        if report_path.exists():
            return report_path
        return None

    def _find_ica_overview_for_file(self, file_path: Path) -> Optional[Path]:
        """Find ICA overview for a file using task root."""
        if not self.task_root:
            return None

        stem = strip_suffixes(file_path.stem, config=self.config)
        ica_path = self.task_root / "reports" / "ica_components" / f"{stem}_ica_components_all.pdf"

        if ica_path.exists():
            return ica_path
        return None

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

    def _update_json_metadata_for_file(self, file_path: Optional[Path]) -> None:
        """Update JSON metadata viewer with file's metadata."""
        if self.json_metadata_viewer is None:
            return

        if file_path is None:
            self.json_metadata_viewer.load_json(None)
            return

        # Get the normalized stem and construct JSON path
        stem = strip_suffixes(file_path.stem, config=self.config)
        if self.task_root:
            json_path = self.task_root / "reports" / "run_reports" / f"{stem}_autoclean_metadata.json"
            self.json_metadata_viewer.load_json(json_path)
        else:
            self.json_metadata_viewer.load_json(None)

    def _update_reprocess_for_file(self, file_path: Optional[Path]) -> None:
        """Update reprocess widget with file's metadata."""
        if self.reprocess_widget is None:
            return

        if file_path is None:
            return

        # Get the normalized stem and construct JSON path
        stem = strip_suffixes(file_path.stem, config=self.config)
        if not self.task_root:
            return

        json_path = self.task_root / "reports" / "run_reports" / f"{stem}_autoclean_metadata.json"
        if not json_path.exists():
            return

        try:
            # Load JSON and extract reprocessing parameters
            data = json.loads(json_path.read_text())
            metadata_section = data.get("metadata", {})

            # Extract unified channel removals (preferred)
            channel_removals = metadata_section.get("channel_removals", [])
            if channel_removals:
                # Build bad_channels list from unified removals
                bad_channels = [r["channel"] for r in channel_removals]
            else:
                # Fallback to legacy bad channels extraction
                bad_channels = metadata_section.get("step_clean_bad_channels", {}).get("bads", [])

            # Extract rejected ICA components
            ica_rejection = metadata_section.get("step_apply_ica_component_rejection", {})
            rejected_ica = ica_rejection.get("ica", {}).get("final_excluded_indices", [])

            # Extract original channel names from import (BEFORE any removals)
            # This allows users to select ANY original channel to mark as bad
            import_details = data.get("import_details", {})
            valid_channels = import_details.get("original_channel_names", [])

            # Fallback to legacy methods if original_channel_names not available
            if not valid_channels:
                # Try metadata.import_eeg.originalChannelNames
                valid_channels = metadata_section.get("import_eeg", {}).get("originalChannelNames", [])

            if not valid_channels:
                # Final fallback: use scalp_channels_used (old behavior)
                gfp_section = metadata_section.get("step_gfp_clean_epochs", {})
                valid_channels = gfp_section.get("scalp_channels_used", [])

            # Extract total ICA components
            ica_section = metadata_section.get("step_run_ica", {})
            ica_components_str = ica_section.get("ica", {}).get("ica_components", "0")
            max_components = int(ica_components_str) if ica_components_str else 0

            # Load data into widget
            self.reprocess_widget.load_from_metadata({
                "bad_channels": bad_channels,
                "rejected_ica": rejected_ica,
                "valid_channels": valid_channels,
                "max_components": max_components,
                "channel_removals": channel_removals  # Pass unified metadata
            })

        except Exception as e:
            print(f"Warning: Could not load reprocess data from {json_path}: {e}")

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
            print(f"[EPOCH DEBUG] No decisions path available for saving")
            return

        print(f"[EPOCH DEBUG] Committing decisions to: {self.decisions_path}")
        print(f"[EPOCH DEBUG] Total decisions to save: {len(self.decisions)}")
        
        # Log epoch information for each decision
        for key, record in self.decisions.items():
            if record.get("epochs_reviewed", False):
                print(f"[EPOCH DEBUG] Decision {key}: {record.get('bad_epochs_count', 0)} bad epochs, {record.get('total_epochs', 0)} total epochs")

        self.decisions_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.decisions, indent=2, sort_keys=True)
        self.decisions_path.write_text(payload)

        if self.decisions_csv_path:
            print(f"[EPOCH DEBUG] Writing CSV to: {self.decisions_csv_path}")
            rows = []
            for key, record in sorted(self.decisions.items()):
                # Skip invalid records that are strings instead of dicts
                if not isinstance(record, dict):
                    print(f"[WARNING] Skipping invalid record for {key}: {type(record)}")
                    continue

                row_data = {
                    "entry": key,
                    "status": record.get("status", "UNSET"),
                    "notes": record.get("notes", ""),
                    "relative_path": record.get("relative_path", ""),
                    "last_updated": record.get("last_updated", ""),
                    "epochs_reviewed": record.get("epochs_reviewed", False),
                    "bad_epochs_count": record.get("bad_epochs_count", 0),
                    "bad_epoch_indices": record.get("bad_epoch_indices", ""),
                    "bad_epoch_times": record.get("bad_epoch_times", ""),
                    "bad_epoch_events": record.get("bad_epoch_events", ""),
                    "total_epochs": record.get("total_epochs", 0),
                    "epoch_rejection_rate": record.get("epoch_rejection_rate", 0.0),
                    "qa_export_hash": record.get("qa_export_hash", ""),
                    "qa_export_timestamp": record.get("qa_export_timestamp", ""),
                    "qa_export_path": record.get("qa_export_path", ""),
                    "reprocess_modified": record.get("reprocess_modified", False),
                    "reprocess_fix_type": record.get("reprocess_fix_type", ""),
                    "reprocess_timestamp": record.get("reprocess_timestamp", ""),
                }
                rows.append(row_data)

                # Log epoch data for CSV rows
                if row_data["epochs_reviewed"]:
                    print(f"[EPOCH DEBUG] CSV row {key}: {row_data['bad_epochs_count']} bad epochs, indices={row_data['bad_epoch_indices']}")

            with self.decisions_csv_path.open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(
                    fp, fieldnames=[
                        "entry", "status", "notes", "relative_path", "last_updated",
                        "epochs_reviewed", "bad_epochs_count", "bad_epoch_indices",
                        "bad_epoch_times", "bad_epoch_events", "total_epochs", "epoch_rejection_rate",
                        "qa_export_hash", "qa_export_timestamp", "qa_export_path",
                        "reprocess_modified", "reprocess_fix_type", "reprocess_timestamp"
                    ]
                )
                writer.writeheader()
                writer.writerows(rows)
            print(f"[EPOCH DEBUG] CSV written successfully with {len(rows)} rows")

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
            # Capture bad epochs from current plot before closing it
            if hasattr(self, '_previous_key') and self._previous_key:
                print(f"[EPOCH DEBUG] Capturing epochs from current plot before closing: {self._previous_key}")
                self._capture_bad_epochs_for_current_file()
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
        
        # Note: Epochs are captured before closing the plot above
        
        record = self.decisions.get(self.current_key)
        self._update_decision_controls(record)
        self._refresh_related_list(file_path)
        if self.detail_panel is not None:
            self.detail_panel.show()

        # Update visual indicators for the current file
        self._apply_status_to_item(item, record.get("status", "UNSET") if record else "UNSET")

        self._update_psd_preview_for_file(file_path)
        self._update_run_report_preview_for_file(file_path)
        self._update_ica_preview_for_file(file_path)
        self._update_processing_metrics_for_file(file_path)
        self._update_json_metadata_for_file(file_path)
        self._update_reprocess_for_file(file_path)

        if self.status_bar is not None and self.current_display_name:
            self.status_bar.showMessage(f"Queued · {self.current_display_name}")

        # Store current key for next file switch
        self._previous_key = self.current_key
        
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

            # Call the base class plotFile but ensure our restoration method is used
            self._call_base_plotFile_with_restoration()

            if self.detail_panel is not None and self.detail_panel.isHidden():
                self.detail_panel.show()

            self._current_plot_path = getattr(self, "selected_file_path", None)

            if self.status_bar is not None and self.current_display_name:
                self.status_bar.showMessage(f"Ready · {self.current_display_name}")
            
            # Set up event-driven epoch capture
            self._setup_epoch_event_handlers()
                
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

        # Clean up event handlers
        self._cleanup_epoch_event_handlers()
        
        # Capture bad epochs before closing plot
        if self.current_key:
            print(f"[EPOCH DEBUG] Closing plot - capturing epochs for current file: {self.current_key}")
            self._capture_bad_epochs_for_current_file()
        else:
            print(f"[EPOCH DEBUG] Closing plot - no current_key to capture epochs")

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
                "epochs_reviewed": False,
                "bad_epochs_count": 0,
                "bad_epoch_indices": "",
                "bad_epoch_times": "",
                "bad_epoch_events": "",
                "total_epochs": 0,
                "epoch_rejection_rate": 0.0,
                "qa_export_hash": "",
                "qa_export_timestamp": "",
                "qa_export_path": "",
            },
        )
        record["status"] = status
        record["last_updated"] = _human_timestamp()
        if self.notes_edit is not None:
            record["notes"] = self.notes_edit.toPlainText().strip()

        # Capture current bad epoch state when status is set
        print(f"[EPOCH DEBUG] Setting status '{status}' for file: {self.current_key}")
        self._capture_bad_epochs_for_current_file()

        item = self.row_lookup.get(self.current_key)
        if item is not None:
            self._apply_status_to_item(item, status)

        self._update_decision_controls(record)
        self._update_summary()
        self._schedule_save()

    def _capture_bad_epochs_for_current_file(self) -> None:
        """Capture bad epoch information for the current file independently of decision status."""
        if not self.current_key:
            print(f"[EPOCH DEBUG] No current_key available for epoch capture")
            return
        
        print(f"[EPOCH DEBUG] Capturing epochs for current file: {self.current_key}")
        
        # Initialize record if it doesn't exist
        record = self.decisions.setdefault(
            self.current_key,
            {
                "status": "UNSET",
                "notes": "",
                "relative_path": self._relative_path(Path(self.selected_file_path)) if hasattr(self, "selected_file_path") else "",
                "last_updated": "",
                "epochs_reviewed": False,
                "bad_epochs_count": 0,
                "bad_epoch_indices": "",
                "bad_epoch_times": "",
                "bad_epoch_events": "",
                "total_epochs": 0,
                "epoch_rejection_rate": 0.0,
                "qa_export_hash": "",
                "qa_export_timestamp": "",
                "qa_export_path": "",
            },
        )

        # Check if we have epochs available
        bad_epochs = []
        total_epochs = 0
        epoch_times = []
        epoch_events = []

        print(f"[EPOCH DEBUG] Current epochs exists: {self.current_epochs is not None}")

        if self.current_epochs is not None:
            try:
                # Read from browser's live bad_epochs list if plot is open
                has_plot = self.plot_widget is not None
                has_mne = hasattr(self.plot_widget, 'mne') if has_plot else False
                has_bad_epochs = hasattr(self.plot_widget.mne, 'bad_epochs') if has_mne else False

                print(f"[EPOCH DEBUG] Plot widget check: has_plot={has_plot}, has_mne={has_mne}, has_bad_epochs={has_bad_epochs}")

                if has_plot and has_mne and has_bad_epochs:
                    bad_epochs = list(self.plot_widget.mne.bad_epochs)
                    print(f"[EPOCH DEBUG] Found {len(bad_epochs)} bad epochs from browser: {bad_epochs}")
                else:
                    # Fall back to drop_log if no plot widget
                    bad_epochs = self._extract_user_bad_epoch_indices(self.current_epochs)
                    print(f"[EPOCH DEBUG] Found {len(bad_epochs)} bad epochs from drop_log: {bad_epochs}")

                total_epochs = len(self.current_epochs)
                print(f"[EPOCH DEBUG] Total epochs: {total_epochs}")

                # Extract timing and event information for bad epochs
                if bad_epochs and hasattr(self.current_epochs, 'events'):
                    print(f"[EPOCH DEBUG] Extracting timing and event info for {len(bad_epochs)} bad epochs")
                    for idx in bad_epochs:
                        if idx < len(self.current_epochs.events):
                            # Convert sample to time
                            time_sec = self.current_epochs.events[idx, 0] / self.current_epochs.info['sfreq']
                            epoch_times.append(f"{time_sec:.3f}")
                            epoch_events.append(str(self.current_epochs.events[idx, 2]))
                            print(f"[EPOCH DEBUG] Bad epoch {idx}: time={time_sec:.3f}s, event={self.current_epochs.events[idx, 2]}")

                record["epochs_reviewed"] = True
                print(f"[EPOCH DEBUG] Marked epochs as reviewed for {self.current_key}")

            except (AttributeError, IndexError) as e:
                print(f"[EPOCH DEBUG] Error extracting epoch information from drop_log: {e}")
                print(f"[EPOCH DEBUG] Exception type: {type(e).__name__}")
                bad_epochs = []
        else:
            print(f"[EPOCH DEBUG] No epochs available for capture")
        
        # Update record with epoch information
        record["bad_epochs_count"] = len(bad_epochs)
        record["bad_epoch_indices"] = ",".join(map(str, bad_epochs)) if bad_epochs else ""
        record["bad_epoch_times"] = ",".join(epoch_times) if epoch_times else ""
        record["bad_epoch_events"] = ",".join(epoch_events) if epoch_events else ""
        record["total_epochs"] = total_epochs
        record["epoch_rejection_rate"] = (len(bad_epochs) / total_epochs * 100.0) if total_epochs > 0 else 0.0
        
        print(f"[EPOCH DEBUG] Updated record for {self.current_key}:")
        print(f"[EPOCH DEBUG]   - Bad epochs count: {record['bad_epochs_count']}")
        print(f"[EPOCH DEBUG]   - Bad epoch indices: {record['bad_epoch_indices']}")
        print(f"[EPOCH DEBUG]   - Bad epoch times: {record['bad_epoch_times']}")
        print(f"[EPOCH DEBUG]   - Bad epoch events: {record['bad_epoch_events']}")
        print(f"[EPOCH DEBUG]   - Total epochs: {record['total_epochs']}")
        print(f"[EPOCH DEBUG]   - Rejection rate: {record['epoch_rejection_rate']:.1f}%")
        
        # Update visual indicator if we have bad epochs
        item = self.row_lookup.get(self.current_key)
        if item is not None:
            print(f"[EPOCH DEBUG] Updating visual indicator for item: {item.text(0)}")
            self._apply_status_to_item(item, record.get("status", "UNSET"))
        else:
            print(f"[EPOCH DEBUG] No item found in row_lookup for key: {self.current_key}")
        
        # Schedule save to persist the epoch information
        print(f"[EPOCH DEBUG] Scheduling save for epoch data")
        self._schedule_save()

    def _capture_bad_epochs_for_key(self, key: str) -> None:
        """Capture bad epoch information for a specific file key."""
        if not key:
            print(f"[EPOCH DEBUG] No key provided for epoch capture")
            return
        
        print(f"[EPOCH DEBUG] Capturing epochs for key: {key}")
        
        # Initialize record if it doesn't exist
        record = self.decisions.setdefault(
            key,
            {
                "status": "UNSET",
                "notes": "",
                "relative_path": "",
                "last_updated": "",
                "epochs_reviewed": False,
                "bad_epochs_count": 0,
                "bad_epoch_indices": "",
                "bad_epoch_times": "",
                "bad_epoch_events": "",
                "total_epochs": 0,
                "epoch_rejection_rate": 0.0,
                "qa_export_hash": "",
                "qa_export_timestamp": "",
                "qa_export_path": "",
            },
        )

        # Check if we have epochs available
        bad_epochs = []
        total_epochs = 0
        epoch_times = []
        epoch_events = []

        print(f"[EPOCH DEBUG] Current epochs exists: {self.current_epochs is not None}")

        if self.current_epochs is not None:
            try:
                bad_epochs = self._extract_user_bad_epoch_indices(self.current_epochs)
                total_epochs = len(self.current_epochs)

                print(f"[EPOCH DEBUG] Found {len(bad_epochs)} bad epochs for key {key} from drop_log: {bad_epochs}")
                print(f"[EPOCH DEBUG] Total epochs: {total_epochs}")

                # Extract timing and event information for bad epochs
                if bad_epochs and hasattr(self.current_epochs, 'events'):
                    print(f"[EPOCH DEBUG] Extracting timing and event info for {len(bad_epochs)} bad epochs")
                    for idx in bad_epochs:
                        if idx < len(self.current_epochs.events):
                            # Convert sample to time
                            time_sec = self.current_epochs.events[idx, 0] / self.current_epochs.info['sfreq']
                            epoch_times.append(f"{time_sec:.3f}")
                            epoch_events.append(str(self.current_epochs.events[idx, 2]))
                            print(f"[EPOCH DEBUG] Bad epoch {idx}: time={time_sec:.3f}s, event={self.current_epochs.events[idx, 2]}")

                record["epochs_reviewed"] = True
                print(f"[EPOCH DEBUG] Marked epochs as reviewed for key {key}")

            except (AttributeError, IndexError) as e:
                print(f"[EPOCH DEBUG] Error extracting epoch information for {key}: {e}")
                print(f"[EPOCH DEBUG] Exception type: {type(e).__name__}")
                bad_epochs = []
        else:
            print(f"[EPOCH DEBUG] No epochs available for capture for key {key}")
        
        # Update record with epoch information
        record["bad_epochs_count"] = len(bad_epochs)
        record["bad_epoch_indices"] = ",".join(map(str, bad_epochs)) if bad_epochs else ""
        record["bad_epoch_times"] = ",".join(epoch_times) if epoch_times else ""
        record["bad_epoch_events"] = ",".join(epoch_events) if epoch_events else ""
        record["total_epochs"] = total_epochs
        record["epoch_rejection_rate"] = (len(bad_epochs) / total_epochs * 100.0) if total_epochs > 0 else 0.0
        
        print(f"[EPOCH DEBUG] Updated record for key {key}:")
        print(f"[EPOCH DEBUG]   - Bad epochs count: {record['bad_epochs_count']}")
        print(f"[EPOCH DEBUG]   - Bad epoch indices: {record['bad_epoch_indices']}")
        print(f"[EPOCH DEBUG]   - Bad epoch times: {record['bad_epoch_times']}")
        print(f"[EPOCH DEBUG]   - Bad epoch events: {record['bad_epoch_events']}")
        print(f"[EPOCH DEBUG]   - Total epochs: {record['total_epochs']}")
        print(f"[EPOCH DEBUG]   - Rejection rate: {record['epoch_rejection_rate']:.1f}%")
        
        # Update visual indicator if we have bad epochs
        item = self.row_lookup.get(key)
        if item is not None:
            print(f"[EPOCH DEBUG] Updating visual indicator for item: {item.text(0)}")
            self._apply_status_to_item(item, record.get("status", "UNSET"))
        else:
            print(f"[EPOCH DEBUG] No item found in row_lookup for key: {key}")
        
        # Schedule save to persist the epoch information
        print(f"[EPOCH DEBUG] Scheduling save for epoch data for key {key}")
        self._schedule_save()

    def _call_base_plotFile_with_restoration(self) -> None:
        """Call the base class plotFile method but ensure our restoration is used."""
        # Temporarily replace the base class restoration method with ours
        original_method = ReviewBase._restore_bad_epochs_to_plot
        ReviewBase._restore_bad_epochs_to_plot = self._restore_bad_epochs_to_plot
        
        try:
            # Call the base class plotFile
            super().plotFile()
        finally:
            # Restore the original method
            ReviewBase._restore_bad_epochs_to_plot = original_method

    def _restore_bad_epochs_to_plot(self) -> None:
        """Restore previously marked bad epochs to the current epochs before plotting."""
        if not self.current_key or not self.current_epochs:
            print(f"[EPOCH DEBUG] No current key or epochs to restore bad epochs")
            return
        
        # Get the saved bad epoch information for this file
        record = self.decisions.get(self.current_key, {})
        bad_epoch_indices_str = record.get("bad_epoch_indices", "")
        
        if not bad_epoch_indices_str:
            print(f"[EPOCH DEBUG] No bad epochs to restore for {self.current_key}")
            return
        
        try:
            # Parse the bad epoch indices
            bad_epoch_indices = [int(idx.strip()) for idx in bad_epoch_indices_str.split(",") if idx.strip()]
            print(f"[EPOCH DEBUG] Restoring {len(bad_epoch_indices)} bad epochs for {self.current_key}: {bad_epoch_indices}")

            # Update drop_log to match the desired state
            drop_log_list = []
            bad_epoch_set = set(bad_epoch_indices)
            for idx in range(len(self.current_epochs)):
                drop_log_list.append(('USER',) if idx in bad_epoch_set else tuple())
            self.current_epochs.drop_log = tuple(drop_log_list)

            print(f"[EPOCH DEBUG] Successfully restored {len(bad_epoch_indices)} bad epochs via drop_log")

        except (ValueError, IndexError) as e:
            print(f"[EPOCH DEBUG] Error restoring bad epochs: {e}")
            print(f"[EPOCH DEBUG] Bad epoch indices string: {bad_epoch_indices_str}")

    def _setup_epoch_event_handlers(self) -> None:
        """Set up event handlers for epoch marking/unmarking."""
        if not self.plot_widget:
            print(f"[EPOCH DEBUG] Cannot set up epoch event handlers - no plot widget")
            return

        print(f"[EPOCH DEBUG] Setting up epoch event handlers")

        # Set up a timer to check for epoch changes (polling drop_log)
        self._epoch_check_timer = QTimer(self)
        self._epoch_check_timer.setInterval(800)  # Check roughly once per second
        self._epoch_check_timer.timeout.connect(self._check_epoch_changes)
        self._epoch_check_timer.start()

        # Store the last known drop_log snapshot for this snapshot
        self._last_drop_log_snapshot = self._snapshot_drop_log()

    def _cleanup_epoch_event_handlers(self) -> None:
        """Clean up event handlers."""
        if hasattr(self, '_epoch_check_timer'):
            self._epoch_check_timer.stop()
            self._epoch_check_timer.deleteLater()
            delattr(self, '_epoch_check_timer')

        if hasattr(self, '_last_drop_log_snapshot'):
            delattr(self, '_last_drop_log_snapshot')

    def _check_epoch_changes(self) -> None:
        """Check if epochs have been marked/unmarked and save immediately."""
        current_snapshot = self._snapshot_drop_log()

        if current_snapshot != getattr(self, '_last_drop_log_snapshot', None):
            print(f"[EPOCH DEBUG] Drop log changed; saving epoch updates")
            self._last_drop_log_snapshot = current_snapshot

            current_bad_epochs = self._get_user_marked_bad_epochs()

            # Save immediately
            self._save_epoch_changes_immediately(current_bad_epochs)

    def _extract_user_bad_epoch_indices(self, epochs) -> list[int]:
        """Extract indices marked as bad by the user from the epochs drop_log."""
        if epochs is None or not hasattr(epochs, 'drop_log'):
            return []

        marked_indices: list[int] = []
        for idx, log in enumerate(epochs.drop_log):
            if not log:
                continue

            # Normalise log entries to an iterable collection
            if isinstance(log, (tuple, list)):
                entries = log
            else:
                entries = [log]

            if any(isinstance(entry, str) and entry.upper() == 'USER' for entry in entries):
                marked_indices.append(idx)

        return marked_indices

    def _snapshot_drop_log(self) -> tuple:
        """Return a hashable snapshot of the current drop_log state."""
        epochs = getattr(self, 'current_epochs', None)
        if epochs is None or not hasattr(epochs, 'drop_log'):
            return tuple()

        snapshot = []
        for log in epochs.drop_log:
            if isinstance(log, (tuple, list)):
                snapshot.append(tuple(log))
            elif log is None:
                snapshot.append(tuple())
            else:
                snapshot.append((log,))
        return tuple(snapshot)

    def _get_user_marked_bad_epochs(self) -> set[int]:
        """Return a set of epoch indices currently marked as bad by the user."""
        epochs = getattr(self, 'current_epochs', None)
        return set(self._extract_user_bad_epoch_indices(epochs))

    def _save_epoch_changes_immediately(self, bad_epochs: set) -> None:
        """Save epoch changes immediately when they occur."""
        if not self.current_key:
            print(f"[EPOCH DEBUG] No current key for immediate save")
            return
        
        print(f"[EPOCH DEBUG] Saving epoch changes immediately for {self.current_key}: {sorted(bad_epochs)}")
        
        # Initialize record if it doesn't exist
        record = self.decisions.setdefault(
            self.current_key,
            {
                "status": "UNSET",
                "notes": "",
                "relative_path": self._relative_path(Path(self.selected_file_path)) if hasattr(self, "selected_file_path") else "",
                "last_updated": "",
                "epochs_reviewed": False,
                "bad_epochs_count": 0,
                "bad_epoch_indices": "",
                "bad_epoch_times": "",
                "bad_epoch_events": "",
                "total_epochs": 0,
                "epoch_rejection_rate": 0.0,
                "qa_export_hash": "",
                "qa_export_timestamp": "",
                "qa_export_path": "",
            },
        )

        # Update record with current epoch information
        bad_epochs_list = sorted(list(bad_epochs))
        total_epochs = len(self.current_epochs) if self.current_epochs else 0
        
        # Extract timing and event information for bad epochs
        epoch_times = []
        epoch_events = []
        if bad_epochs_list and self.current_epochs and hasattr(self.current_epochs, 'events'):
            for idx in bad_epochs_list:
                if idx < len(self.current_epochs.events):
                    # Convert sample to time
                    time_sec = self.current_epochs.events[idx, 0] / self.current_epochs.info['sfreq']
                    epoch_times.append(f"{time_sec:.3f}")
                    epoch_events.append(str(self.current_epochs.events[idx, 2]))
        
        # Update record
        record["epochs_reviewed"] = True
        record["bad_epochs_count"] = len(bad_epochs_list)
        record["bad_epoch_indices"] = ",".join(map(str, bad_epochs_list))
        record["bad_epoch_times"] = ",".join(epoch_times)
        record["bad_epoch_events"] = ",".join(epoch_events)
        record["total_epochs"] = total_epochs
        record["epoch_rejection_rate"] = (len(bad_epochs_list) / total_epochs * 100.0) if total_epochs > 0 else 0.0
        
        print(f"[EPOCH DEBUG] Updated record for {self.current_key}:")
        print(f"[EPOCH DEBUG]   - Bad epochs count: {record['bad_epochs_count']}")
        print(f"[EPOCH DEBUG]   - Bad epoch indices: {record['bad_epoch_indices']}")
        
        # Update visual indicator
        item = self.row_lookup.get(self.current_key)
        if item is not None:
            self._apply_status_to_item(item, record.get("status", "UNSET"))
        
        # Save immediately
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

    def _handle_reprocess_changed(self) -> None:
        """Handle changes to reprocess widget values."""
        if not self.current_key or self.reprocess_widget is None:
            return

        if not hasattr(self, "selected_file_path"):
            return

        # Check if there are actual changes
        if not self.reprocess_widget.has_changes():
            # No changes - clear reprocess fields if they exist
            record = self.decisions.get(self.current_key)
            if record and record.get("reprocess_modified"):
                record["reprocess_modified"] = False
                record["reprocess_fix_type"] = ""
                record["reprocess_timestamp"] = ""
                self._schedule_save()
            return

        # Get changes diff
        diff = self.reprocess_widget.get_changes_diff()
        has_channel_changes = diff["has_channel_changes"]
        has_ica_changes = diff["has_ica_changes"]

        # Determine fix type
        if has_channel_changes and has_ica_changes:
            fix_type = "both"
        elif has_channel_changes:
            fix_type = "channel"
        elif has_ica_changes:
            fix_type = "ica"
        else:
            fix_type = ""

        # Update decisions record
        record = self.decisions.setdefault(
            self.current_key,
            {
                "status": "UNSET",
                "notes": "",
                "relative_path": self._relative_path(Path(self.selected_file_path)),
                "last_updated": "",
            },
        )

        record["reprocess_modified"] = True
        record["reprocess_fix_type"] = fix_type
        record["reprocess_timestamp"] = _human_timestamp()
        record["last_updated"] = _human_timestamp()

        # Save the detailed payload to QA directory
        self._save_reprocess_payload(diff)

        # Schedule save of decisions JSON
        self._schedule_save()

    def _save_reprocess_payload(self, diff: dict) -> None:
        """Save reprocess payload JSON to QA directory.

        Parameters
        ----------
        diff : dict
            Changes diff from reprocess widget
        """
        if not self.task_root or not hasattr(self, "selected_file_path"):
            return

        from datetime import datetime

        # Get normalized file stem
        file_path = Path(self.selected_file_path)
        stem = strip_suffixes(file_path.stem, config=self.config)

        # Create qa/manual_fixes directory
        qa_fixes_dir = self.task_root / "qa" / "manual_fixes"
        qa_fixes_dir.mkdir(parents=True, exist_ok=True)

        # Determine fix type
        has_channel_changes = diff["has_channel_changes"]
        has_ica_changes = diff["has_ica_changes"]

        if has_channel_changes and has_ica_changes:
            fix_type = "both"
        elif has_channel_changes:
            fix_type = "channel"
        elif has_ica_changes:
            fix_type = "ica"
        else:
            return  # No changes, shouldn't happen

        # Check if ICA file exists
        ica_file_path = self.task_root / "ica" / f"{stem}-ica.fif"
        ica_file_exists = ica_file_path.exists()
        ica_file_relative = f"ica/{stem}-ica.fif" if ica_file_exists else None

        # Validation: can we apply ICA-only fix?
        # Only if no channel changes and ICA file exists
        can_apply_ica_fix = (not has_channel_changes) and ica_file_exists

        # Requires full reprocess if channels changed
        requires_full_reprocess = has_channel_changes

        # Find task file in status directory
        task_file_path = None
        task_file_relative = None
        task_file_hash = None
        status_dir = self.task_root / "status"
        if status_dir.exists():
            # Find .py files in status directory
            py_files = list(status_dir.glob("*.py"))
            if py_files:
                # Use the first .py file found (typically only one task file)
                task_file_path = py_files[0]
                task_file_relative = f"status/{task_file_path.name}"
                # Calculate SHA256 hash of task file for integrity
                if task_file_path.exists():
                    task_file_hash = hashlib.sha256(task_file_path.read_bytes()).hexdigest()

        # Build payload
        payload = {
            "file_key": self.current_key,
            "file_stem": stem,
            "fix_type": fix_type,
            "timestamp": datetime.now().isoformat(),
            "modifications": {
                "bad_channels": diff["bad_channels"],
                "rejected_ica": diff["rejected_ica"],
            },
            "ica_file_path": ica_file_relative,
            "metadata_source": f"reports/run_reports/{stem}_autoclean_metadata.json",
            "task_file_path": task_file_relative,
            "task_file_hash": task_file_hash,
            "validation": {
                "can_apply_ica_fix": can_apply_ica_fix,
                "requires_full_reprocess": requires_full_reprocess,
                "ica_file_exists": ica_file_exists,
                "task_file_exists": task_file_path.exists() if task_file_path else False,
            },
        }

        # Save payload to JSON
        payload_path = qa_fixes_dir / f"{stem}_manual_fix.json"
        payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

        print(f"[REPROCESS] Saved manual fix payload: {payload_path}")
        print(f"[REPROCESS]   Fix type: {fix_type}")
        print(f"[REPROCESS]   Can apply ICA-only: {can_apply_ica_fix}")
        print(f"[REPROCESS]   Requires full reprocess: {requires_full_reprocess}")
        if task_file_relative:
            print(f"[REPROCESS]   Task file: {task_file_relative}")
            print(f"[REPROCESS]   Task file hash: {task_file_hash[:16]}...")
        else:
            print(f"[REPROCESS]   WARNING: Task file not found in status/ directory")

    def _trigger_reprocess_with_overrides(self) -> None:
        """Trigger reprocessing with manual overrides from the current file's payload."""
        if not self.task_root or not hasattr(self, "selected_file_path"):
            QMessageBox.warning(
                self,
                "Reprocess Error",
                "No file selected or task root not configured."
            )
            return

        # Check if we're in a reprocess subfolder (not the original task folder)
        if "reprocess" in self.task_root.parts:
            QMessageBox.warning(
                self,
                "Reprocess Error",
                f"Cannot reprocess from a reprocess subfolder.\n\n"
                f"Current folder: {self.task_root}\n\n"
                f"Please open the exclude GUI on the original task folder "
                f"(e.g., 'BiotrialResting1020'), not the reprocess temp subfolder.\n\n"
                f"Reprocess temp folders are automatically created under reprocess/ "
                f"and can be deleted after results are copied."
            )
            return

        # Get file stem
        file_path = Path(self.selected_file_path)
        stem = strip_suffixes(file_path.stem, config=self.config)

        # Check if manual fix payload exists
        payload_path = self.task_root / "qa" / "manual_fixes" / f"{stem}_manual_fix.json"
        if not payload_path.exists():
            QMessageBox.warning(
                self,
                "Reprocess Error",
                f"Manual fix payload not found:\n{payload_path}\n\n"
                "Please make changes in the Reprocess tab first."
            )
            return

        try:
            # Load manual fix payload
            payload = json.loads(payload_path.read_text())

            # Load metadata to get original raw file path
            metadata_path = self.task_root / "reports" / "run_reports" / f"{stem}_autoclean_metadata.json"
            if not metadata_path.exists():
                QMessageBox.warning(
                    self,
                    "Reprocess Error",
                    f"Metadata file not found:\n{metadata_path}"
                )
                return

            metadata = json.loads(metadata_path.read_text())
            original_raw_file = metadata.get("unprocessed_file")
            if not original_raw_file:
                QMessageBox.warning(
                    self,
                    "Reprocess Error",
                    "Could not find original raw file path in metadata."
                )
                return

            # Verify raw file exists
            raw_file_path = Path(original_raw_file)
            if not raw_file_path.exists():
                QMessageBox.warning(
                    self,
                    "Reprocess Error",
                    f"Original raw file not found:\n{raw_file_path}"
                )
                return

            # Show confirmation dialog
            fix_type = payload.get("fix_type", "unknown")
            bad_ch_count = len(payload["modifications"]["bad_channels"]["modified"])
            ica_count = len(payload["modifications"]["rejected_ica"]["modified"])

            confirm_msg = (
                f"<b>Reprocess: {stem}</b><br><br>"
                f"<b>Fix Type:</b> {fix_type}<br>"
                f"<b>Bad Channels:</b> {bad_ch_count}<br>"
                f"<b>ICA Components:</b> {ica_count}<br><br>"
                f"<b>Raw File:</b><br>{raw_file_path}<br><br>"
                "This will generate a new reprocessing task and run the pipeline.<br>"
                "Do you want to continue?"
            )

            reply = QMessageBox.question(
                self,
                "Confirm Reprocessing",
                confirm_msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply != QMessageBox.StandardButton.Yes:
                return

            # Generate reprocessing task file with sanitized class name
            # This creates a temporary task folder that we'll copy from later
            sanitized = stem.replace('-', '_').replace(' ', '_')
            sanitized = ''.join(c if c.isalnum() or c == '_' else '_' for c in sanitized)
            if sanitized and sanitized[0].isdigit():
                class_name = f"Task_{sanitized}_Reprocess"
            else:
                class_name = f"{sanitized}_Reprocess"

            task_output_path = self.task_root / "status" / f"{stem}_Reprocess.py"

            # Get original task file path
            task_file_path = payload.get("task_file_path")
            if not task_file_path:
                QMessageBox.warning(
                    self,
                    "Reprocess Error",
                    "No original task file reference in payload. Cannot generate reprocess task."
                )
                return

            # Resolve path relative to task_root
            original_task_path = self.task_root / task_file_path
            if not original_task_path.exists():
                QMessageBox.warning(
                    self,
                    "Reprocess Error",
                    f"Original task file not found:\n{original_task_path}"
                )
                return

            # Generate timestamp for consistent folder naming
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Generate reprocess task by modifying original task's AST
            print(f"[REPROCESS] Generating task from original: {task_file_path}")
            rendered_task = _generate_reprocess_task_from_original(
                original_task_path=original_task_path,
                payload=payload,
                new_class_name=class_name,
                timestamp=timestamp
            )

            # Write generated task to file
            with open(task_output_path, 'w', encoding='utf-8') as f:
                f.write(rendered_task)

            print(f"[REPROCESS] Generated task file: {task_output_path}")

            # Start non-blocking reprocess
            self._start_reprocess(stem, task_output_path, raw_file_path, payload, timestamp)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Reprocessing Error",
                f"An error occurred during reprocessing:\n\n{str(e)}"
            )
            import traceback
            traceback.print_exc()

    def _merge_reprocess_database(
        self,
        original_db_path: Path,
        reprocess_db_path: Path,
        stem: str,
        manifest_path: Optional[Path] = None
    ) -> tuple[Optional[str], Optional[str]]:
        """Merge reprocess database into original task database.

        Parameters
        ----------
        original_db_path : Path
            Path to the original task's run_database.db
        reprocess_db_path : Path
            Path to the reprocess run's run_database.db
        stem : str
            File stem (e.g., '101001_C1D1BL_EO')
        manifest_path : Path, optional
            Path to backup manifest JSON to update with run IDs

        Returns
        -------
        tuple[str | None, str | None]
            (original_run_id, reprocess_run_id) if successful, (None, None) otherwise
        """
        import sqlite3
        import json
        from autoclean.utils.audit import calculate_access_log_hash, get_user_context

        try:
            print(f"[REPROCESS] Merging databases...")

            # Connect to both databases
            original_conn = sqlite3.connect(str(original_db_path))
            original_conn.row_factory = sqlite3.Row
            reprocess_conn = sqlite3.connect(str(reprocess_db_path))
            reprocess_conn.row_factory = sqlite3.Row

            original_cursor = original_conn.cursor()
            reprocess_cursor = reprocess_conn.cursor()

            # 1. Add supersession columns if they don't exist
            try:
                original_cursor.execute("ALTER TABLE pipeline_runs ADD COLUMN superseded_by TEXT")
                print("[REPROCESS] Added superseded_by column")
            except sqlite3.OperationalError:
                pass  # Column already exists

            try:
                original_cursor.execute("ALTER TABLE pipeline_runs ADD COLUMN supersedes_run_id TEXT")
                print("[REPROCESS] Added supersedes_run_id column")
            except sqlite3.OperationalError:
                pass  # Column already exists

            # 2. Find the original run for this file in original database
            original_cursor.execute(
                "SELECT run_id FROM pipeline_runs WHERE unprocessed_file LIKE ? ORDER BY created_at DESC LIMIT 1",
                (f"%{stem}%",)
            )
            original_run = original_cursor.fetchone()
            original_run_id = original_run['run_id'] if original_run else None

            if not original_run_id:
                print(f"[REPROCESS] Warning: No original run found for {stem}")

            # 3. Get the reprocess run from reprocess database
            reprocess_cursor.execute("SELECT * FROM pipeline_runs LIMIT 1")
            reprocess_run = reprocess_cursor.fetchone()

            if not reprocess_run:
                print("[REPROCESS] Error: No run found in reprocess database")
                return None, None

            reprocess_run_id = reprocess_run['run_id']

            # 4. Insert reprocess run into original database
            columns = [desc[0] for desc in reprocess_cursor.description]
            placeholders = ", ".join(["?" for _ in columns])

            # Add supersedes_run_id to the insert
            if 'supersedes_run_id' in columns:
                values = list(reprocess_run)
                supersedes_idx = columns.index('supersedes_run_id')
                values[supersedes_idx] = original_run_id
            else:
                columns.append('supersedes_run_id')
                values = list(reprocess_run) + [original_run_id]
                placeholders += ", ?"

            original_cursor.execute(
                f"INSERT INTO pipeline_runs ({', '.join(columns)}) VALUES ({placeholders})",
                values
            )
            print(f"[REPROCESS] Inserted reprocess run: {reprocess_run_id}")

            # 5. Mark original run as superseded (if found)
            if original_run_id:
                original_cursor.execute(
                    "UPDATE pipeline_runs SET superseded_by = ? WHERE run_id = ?",
                    (reprocess_run_id, original_run_id)
                )
                print(f"[REPROCESS] Marked original run as superseded: {original_run_id}")

            # 6. Copy update_audit_log entries
            reprocess_cursor.execute("SELECT * FROM update_audit_log")
            audit_logs = reprocess_cursor.fetchall()

            if audit_logs:
                columns = [desc[0] for desc in reprocess_cursor.description]
                placeholders = ", ".join(["?" for _ in columns])
                for log in audit_logs:
                    original_cursor.execute(
                        f"INSERT INTO update_audit_log ({', '.join(columns)}) VALUES ({placeholders})",
                        tuple(log)
                    )
                print(f"[REPROCESS] Copied {len(audit_logs)} audit log entries")

            # 7. Re-chain and copy database_access_log entries
            # Get last hash from original database
            original_cursor.execute(
                "SELECT log_hash FROM database_access_log ORDER BY log_id DESC LIMIT 1"
            )
            last_hash_row = original_cursor.fetchone()
            previous_hash = last_hash_row['log_hash'] if last_hash_row else "genesis_hash_empty_log"

            # Get access logs from reprocess database (skip genesis entry)
            reprocess_cursor.execute(
                "SELECT * FROM database_access_log WHERE operation != 'isolated_database_creation' ORDER BY log_id"
            )
            access_logs = reprocess_cursor.fetchall()

            if access_logs:
                for log in access_logs:
                    log_dict = dict(log)

                    # Recalculate hash with new previous_hash for chain integrity
                    new_hash = calculate_access_log_hash(
                        log_dict['timestamp'],
                        log_dict['operation'],
                        json.loads(log_dict['user_context']) if log_dict.get('user_context') else {},
                        "",
                        json.loads(log_dict['details']) if log_dict.get('details') else {},
                        previous_hash
                    )

                    # Insert with recalculated hash
                    original_cursor.execute(
                        """
                        INSERT INTO database_access_log (
                            timestamp, operation, user_context, details,
                            log_hash, previous_hash, auth0_user_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            log_dict['timestamp'],
                            log_dict['operation'],
                            log_dict['user_context'],
                            log_dict['details'],
                            new_hash,
                            previous_hash,
                            log_dict.get('auth0_user_id')
                        )
                    )
                    previous_hash = new_hash

                print(f"[REPROCESS] Copied {len(access_logs)} access log entries (re-chained)")

            # 8. Copy electronic_signatures if any
            reprocess_cursor.execute("SELECT * FROM electronic_signatures")
            signatures = reprocess_cursor.fetchall()

            if signatures:
                columns = [desc[0] for desc in reprocess_cursor.description]
                placeholders = ", ".join(["?" for _ in columns])
                for sig in signatures:
                    original_cursor.execute(
                        f"INSERT INTO electronic_signatures ({', '.join(columns)}) VALUES ({placeholders})",
                        tuple(sig)
                    )
                print(f"[REPROCESS] Copied {len(signatures)} electronic signatures")

            # 9. Copy authenticated_users if any
            reprocess_cursor.execute("SELECT * FROM authenticated_users")
            users = reprocess_cursor.fetchall()

            if users:
                for user in users:
                    user_dict = dict(user)
                    auth0_user_id = user_dict['auth0_user_id']

                    # Check if user already exists
                    original_cursor.execute(
                        "SELECT auth0_user_id FROM authenticated_users WHERE auth0_user_id = ?",
                        (auth0_user_id,)
                    )
                    if original_cursor.fetchone():
                        continue  # Skip existing users

                    columns = [desc[0] for desc in reprocess_cursor.description]
                    placeholders = ", ".join(["?" for _ in columns])
                    original_cursor.execute(
                        f"INSERT INTO authenticated_users ({', '.join(columns)}) VALUES ({placeholders})",
                        tuple(user)
                    )
                print(f"[REPROCESS] Copied {len(users)} authenticated users")

            # 10. Commit and close
            original_conn.commit()
            original_conn.close()
            reprocess_conn.close()

            # 11. Update backup manifest with run IDs
            if manifest_path and manifest_path.exists():
                try:
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)

                    manifest['original_run_id'] = original_run_id
                    manifest['superseded_by_run_id'] = reprocess_run_id

                    with open(manifest_path, 'w', encoding='utf-8') as f:
                        json.dump(manifest, f, indent=2)

                    print(f"[REPROCESS] Updated backup manifest with run IDs")
                except Exception as e:
                    print(f"[REPROCESS] Warning: Failed to update manifest: {e}")

            print(f"[REPROCESS] Database merge successful")
            return original_run_id, reprocess_run_id

        except Exception as e:
            print(f"[REPROCESS] Error merging databases: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _start_reprocess(self, stem: str, task_path: Path, raw_path: Path, payload: dict, timestamp: str) -> None:
        """Start reprocessing in background with simple status dialog."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton
        import shutil
        import json

        # Store reprocess info for post-processing
        original_task_root = self.task_root

        # Use provided timestamp for consistent folder naming
        reprocess_folder_name = f"{stem}_{timestamp}"
        print(f"[REPROCESS] Using folder name: {reprocess_folder_name}")

        # Create nested reprocess directory structure
        reprocess_dir = original_task_root / "reprocess"
        reprocess_dir.mkdir(exist_ok=True)
        output_dir = reprocess_dir

        # Backup original files to exports/backups/ before reprocessing
        exports_dir = original_task_root / "exports"
        backups_dir = exports_dir / "backups"
        backups_dir.mkdir(exist_ok=True)

        # Backup epoch file
        comp_file = exports_dir / f"{stem}_comp_epo.fif"
        backed_up_files = []

        if comp_file.exists():
            backup_file = backups_dir / f"{stem}_comp_epo_{timestamp}.fif"
            shutil.copy2(comp_file, backup_file)
            backed_up_files.append({
                "filename": comp_file.name,
                "original_path": str(comp_file.relative_to(original_task_root)),
                "backup_path": str(backup_file.relative_to(original_task_root)),
                "size_bytes": comp_file.stat().st_size
            })
            print(f"[REPROCESS] Backed up {comp_file.name} to {backup_file}")

        # Backup database file
        db_file = original_task_root / "run_database.db"
        if db_file.exists():
            db_backup_file = backups_dir / f"run_database_{timestamp}.db"
            shutil.copy2(db_file, db_backup_file)
            backed_up_files.append({
                "filename": db_file.name,
                "original_path": str(db_file.relative_to(original_task_root)),
                "backup_path": str(db_backup_file.relative_to(original_task_root)),
                "size_bytes": db_file.stat().st_size
            })
            print(f"[REPROCESS] Backed up {db_file.name} to {db_backup_file}")

        # Create backup manifest for provenance tracking
        if backed_up_files:
            manifest_path = backups_dir / f"{stem}_backup_manifest_{timestamp}.json"
            manifest = {
                "backup_timestamp": datetime.now().isoformat(),
                "backup_reason": "manual_override_reprocess",
                "original_run_id": None,  # Will be populated after merge
                "superseded_by_run_id": None,  # Will be populated after merge
                "backed_up_files": backed_up_files,
                "manual_overrides": payload.get("modifications", {}),
                "reprocess_task_file": str(task_path.relative_to(original_task_root)),
                "reprocess_task_hash": payload.get("task_file_hash", ""),
                "fix_type": payload.get("fix_type", "unknown")
            }

            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)

            print(f"[REPROCESS] Created backup manifest: {manifest_path.name}")

        # Create simple non-modal dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Reprocessing")
        dialog.setModal(False)  # Non-blocking
        dialog.resize(350, 120)

        layout = QVBoxLayout()
        label = QLabel(f"Reprocessing {stem}...\n\nThis may take several minutes.\nYou can continue using the GUI.")
        label.setWordWrap(True)
        layout.addWidget(label)

        cancel_btn = QPushButton("Cancel")
        layout.addWidget(cancel_btn)
        dialog.setLayout(layout)

        # Start process
        process = QProcess(self)

        # Capture output for error reporting
        output_lines = []
        process.readyReadStandardOutput.connect(
            lambda: output_lines.append(process.readAllStandardOutput().data().decode('utf-8', errors='ignore'))
        )
        process.readyReadStandardError.connect(
            lambda: output_lines.append(process.readAllStandardError().data().decode('utf-8', errors='ignore'))
        )

        # Pass reprocess info to completion handler
        reprocess_info = {
            'stem': stem,
            'original_task_root': original_task_root,
            'reprocess_folder_name': reprocess_folder_name,
            'output_dir': output_dir,
            'manifest_path': manifest_path if backed_up_files else None,
            'timestamp': timestamp,
            'backups_dir': backups_dir
        }

        process.finished.connect(
            lambda code, status: self._on_reprocess_done(code, dialog, output_lines, reprocess_info)
        )

        cancel_btn.clicked.connect(lambda: (process.kill(), dialog.close()))

        cmd_args = [
            "process",
            "--task-file",
            str(task_path),
            "--file",
            str(raw_path)
        ]

        if output_dir:
            cmd_args.extend(["--output", str(output_dir)])

        print(f"[REPROCESS] Starting: autocleaneeg-pipeline {' '.join(cmd_args)}")
        print(f"[REPROCESS] Reprocess folder: reprocess/{reprocess_folder_name}")
        process.start("autocleaneeg-pipeline", cmd_args)

        dialog.show()

    def _on_reprocess_done(
        self,
        exit_code: int,
        dialog,
        output_lines: list,
        reprocess_info: dict
    ) -> None:
        """Handle reprocess completion and copy results to original folder."""
        import shutil

        print(f"[REPROCESS DEBUG] _on_reprocess_done called with exit_code={exit_code}")
        dialog.close()

        stem = reprocess_info['stem']
        original_task_root = reprocess_info['original_task_root']
        reprocess_folder_name = reprocess_info['reprocess_folder_name']
        output_dir = reprocess_info['output_dir']

        print(f"[REPROCESS DEBUG] stem={stem}, reprocess_folder_name={reprocess_folder_name}")
        print(f"[REPROCESS DEBUG] original_task_root={original_task_root}")

        if exit_code == 0:
            print(f"[REPROCESS DEBUG] Process completed successfully, starting file copy...")
            # Copy reprocessed files from temp folder to original folder
            try:
                reprocess_folder = (original_task_root / "reprocess" / reprocess_folder_name).resolve()

                if not reprocess_folder.exists():
                    QMessageBox.warning(
                        self,
                        "Reprocess Warning",
                        f"Reprocess completed but output folder not found:\n{reprocess_folder}\n\n"
                        f"Files may be in a different location."
                    )
                    return

                # Copy exports folder contents
                reprocess_exports = reprocess_folder / "exports"
                original_exports = original_task_root / "exports"

                if reprocess_exports.exists():
                    for file_path in reprocess_exports.glob("*"):
                        if file_path.is_file():
                            # Skip GUI state files (not processing artifacts)
                            if file_path.name == "autoclean_exclusion_decisions.json":
                                print(f"[REPROCESS] Skipped GUI state file: {file_path.name}")
                                continue

                            dest_path = original_exports / file_path.name
                            shutil.copy2(file_path, dest_path)
                            print(f"[REPROCESS] Copied {file_path.name} to original exports")

                # Copy reports folder contents (preserving structure)
                reprocess_reports = reprocess_folder / "reports"
                original_reports = original_task_root / "reports"

                if reprocess_reports.exists():
                    for item in reprocess_reports.rglob("*"):
                        if item.is_file():
                            rel_path = item.relative_to(reprocess_reports)
                            dest_path = original_reports / rel_path
                            dest_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(item, dest_path)
                            print(f"[REPROCESS] Copied {rel_path} to original reports")

                # Copy ICA folder contents
                reprocess_ica = reprocess_folder / "ica"
                original_ica = original_task_root / "ica"

                if reprocess_ica.exists():
                    for file_path in reprocess_ica.glob("*"):
                        if file_path.is_file():
                            dest_path = original_ica / file_path.name
                            shutil.copy2(file_path, dest_path)
                            print(f"[REPROCESS] Copied {file_path.name} to original ica")

                # Merge databases to consolidate run records
                original_db_path = original_task_root / "run_database.db"
                reprocess_db_path = reprocess_folder / "run_database.db"
                manifest_path = reprocess_info.get('manifest_path')

                if original_db_path.exists() and reprocess_db_path.exists():
                    original_run_id, reprocess_run_id = self._merge_reprocess_database(
                        original_db_path,
                        reprocess_db_path,
                        stem,
                        manifest_path
                    )

                    if original_run_id and reprocess_run_id:
                        print(f"[REPROCESS] Database merge completed successfully")
                        print(f"[REPROCESS]   Original run: {original_run_id}")
                        print(f"[REPROCESS]   Reprocess run: {reprocess_run_id}")
                    else:
                        print(f"[REPROCESS] Warning: Database merge failed - see logs above")
                else:
                    print(f"[REPROCESS] Warning: Database files not found, skipping merge")

                QMessageBox.information(
                    self,
                    "Success",
                    f"Reprocessed {stem} successfully!\n\n"
                    f"Results copied to original task folder.\n"
                    f"Original files backed up to exports/backups/\n\n"
                    f"Temp folder can be deleted: reprocess/{reprocess_folder_name}"
                )
                self.refreshFileTree()

            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Copy Error",
                    f"Reprocessing completed but failed to copy results:\n\n{str(e)}\n\n"
                    f"You can manually copy from:\n{reprocess_folder}"
                )
        else:
            # Show error with captured output
            output = ''.join(output_lines) if output_lines else "No output captured"
            error_msg = f"Reprocessing failed with exit code {exit_code}.\n\nOutput:\n{output[-1000:]}"
            QMessageBox.warning(self, "Reprocessing Failed", error_msg)

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
        
        # Get the key for this item to check epoch information
        key = item.data(0, Qt.UserRole + 1)
        epoch_info = ""
        if key and key in self.decisions:
            record = self.decisions[key]
            bad_count = record.get("bad_epochs_count", 0)
            if bad_count > 0:
                epoch_info = f" ({bad_count} bad epochs)"
        
        display = base_label
        if status and status != "UNSET":
            display = f"{base_label} [{meta['label']}]{epoch_info}"
        elif epoch_info:
            display = f"{base_label}{epoch_info}"
        
        item.setText(0, display)
        
        # Set background color based on status and epoch information
        if status and status != "UNSET":
            color = QColor(meta["color"])
            color.setAlpha(60)
            item.setBackground(0, color)
        elif epoch_info:
            # Light orange/yellow for files with bad epochs but no decision
            color = QColor("#f39c12")
            color.setAlpha(40)
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
    def _parse_metadata_json(self, json_path: Path) -> dict[str, list]:
        """Parse metadata JSON and extract bad channels and rejected ICA components.

        Args:
            json_path: Path to the JSON metadata file

        Returns:
            Dict with 'bad_channels' and 'rejected_ica' keys
        """
        result = {"bad_channels": [], "rejected_ica": []}

        if not json_path or not json_path.exists():
            return result

        try:
            data = json.loads(json_path.read_text())
            metadata_section = data.get("metadata", {})

            # Extract unified channel removals (preferred)
            channel_removals = metadata_section.get("channel_removals", [])
            if channel_removals and isinstance(channel_removals, list):
                result["channel_removals"] = channel_removals
                # Build bad_channels list from removals for backward compatibility
                result["bad_channels"] = [r["channel"] for r in channel_removals]
            else:
                # Fallback to legacy bad channels extraction
                bad_channels = metadata_section.get("step_clean_bad_channels", {}).get("bads", [])
                if isinstance(bad_channels, list):
                    result["bad_channels"] = bad_channels

            # Extract rejected ICA components
            ica_rejection = metadata_section.get("step_apply_ica_component_rejection", {})
            rejected_comps = ica_rejection.get("ica", {}).get("final_excluded_indices", [])
            if isinstance(rejected_comps, list):
                result["rejected_ica"] = rejected_comps

        except Exception as e:
            print(f"Warning: Could not parse metadata JSON {json_path}: {e}")

        return result

    def _refresh_related_list(self, file_path: Path) -> None:
        if self.related_list is None:
            return
        self.related_list.clear()

        json_path = None  # Track JSON path for parsing

        for asset_type, asset_path, exists in self._gather_related_files(file_path):
            if exists and asset_path:
                # File exists - show with checkmark and green color
                display_name = f"✓ {asset_type}: {asset_path.name}"
                item = QListWidgetItem(display_name)
                item.setForeground(QColor("#27ae60"))  # Green
                item.setToolTip(str(asset_path))
                item.setData(Qt.UserRole, str(asset_path))
            else:
                # File missing - show with X and gray color
                display_name = f"✗ {asset_type}: not found"
                item = QListWidgetItem(display_name)
                item.setForeground(QColor("#95a5a6"))  # Gray
                item.setToolTip(f"Expected: {asset_path}" if asset_path else "Path could not be resolved")
                item.setData(Qt.UserRole, "")  # No path to open

            self.related_list.addItem(item)

            # Track JSON path for parsing metadata
            if asset_type == "Metadata (JSON)" and exists and asset_path:
                json_path = asset_path

        # Parse and display metadata if JSON exists
        if json_path:
            metadata = self._parse_metadata_json(json_path)

            # Add separator
            separator = QListWidgetItem("─────────────────")
            separator.setForeground(QColor("#95a5a6"))
            separator.setData(Qt.UserRole, "")  # Non-clickable
            self.related_list.addItem(separator)

            # Display bad channels with removal reasons (enhanced)
            channel_removals = metadata.get("channel_removals", [])
            if channel_removals:
                # Enhanced display: group by removal reason
                grouped = _group_channel_removals(channel_removals)
                total_count = len(metadata.get("bad_channels", []))

                # Header showing total
                header_item = QListWidgetItem(f"Bad Channels ({total_count}):")
                header_item.setForeground(QColor("#e67e22"))  # Orange
                header_item.setData(Qt.UserRole, "")  # Non-clickable
                self.related_list.addItem(header_item)

                # Display each group with color-coded reason
                for reason, channels in grouped.items():
                    label, color = _get_removal_reason_display(reason)
                    channels_str = ", ".join(channels)
                    reason_item = QListWidgetItem(f"  [{label}] {channels_str}")
                    reason_item.setForeground(QColor(color))
                    reason_item.setData(Qt.UserRole, "")  # Non-clickable
                    self.related_list.addItem(reason_item)
            elif metadata["bad_channels"]:
                # Fallback: legacy flat display
                channels_str = ", ".join(metadata["bad_channels"])
                bad_ch_item = QListWidgetItem(f"Bad Channels: {channels_str}")
                bad_ch_item.setForeground(QColor("#e67e22"))  # Orange
                bad_ch_item.setData(Qt.UserRole, "")  # Non-clickable
                self.related_list.addItem(bad_ch_item)
            else:
                # No bad channels
                bad_ch_item = QListWidgetItem("Bad Channels: None")
                bad_ch_item.setForeground(QColor("#95a5a6"))  # Gray
                bad_ch_item.setData(Qt.UserRole, "")  # Non-clickable
                self.related_list.addItem(bad_ch_item)

            # Display rejected ICA components
            if metadata["rejected_ica"]:
                ica_str = ", ".join(map(str, metadata["rejected_ica"]))
                ica_item = QListWidgetItem(f"Rejected ICA: [{ica_str}]")
                ica_item.setForeground(QColor("#e67e22"))  # Orange
            else:
                ica_item = QListWidgetItem("Rejected ICA: None")
                ica_item.setForeground(QColor("#95a5a6"))  # Gray
            ica_item.setData(Qt.UserRole, "")  # Non-clickable
            self.related_list.addItem(ica_item)

    def _open_related_item(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.UserRole)
        if not data or data == "":
            # File is missing, show tooltip info instead
            QMessageBox.warning(
                self,
                "File Not Found",
                f"This file is not available.\n\n{item.toolTip()}"
            )
            return

        file_path = Path(str(data))
        if not file_path.exists():
            QMessageBox.warning(
                self,
                "File Not Found",
                f"File no longer exists:\n{file_path}"
            )
            return

        _open_path(file_path)

    def _gather_related_files(self, file_path: Path) -> list[tuple[str, Optional[Path], bool]]:
        """Gather related files using asset resolution system.

        Returns:
            List of tuples: (asset_type, file_path, exists)
        """
        results: list[tuple[str, Optional[Path], bool]] = []

        # Strip suffixes to get normalized stem
        stem = strip_suffixes(file_path.stem, config=self.config)

        # Resolve ICA file
        if self.task_root:
            ica_path = self.task_root / "ica" / f"{stem}-ica.fif"
            results.append(("ICA Components", ica_path, ica_path.exists()))

        # Resolve JSON metadata - construct path from task root
        if self.task_root:
            json_path = self.task_root / "reports" / "run_reports" / f"{stem}_autoclean_metadata.json"
            results.append(("Metadata (JSON)", json_path, json_path.exists()))

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

    def _navigate_up(self) -> None:
        """Navigate to the previous file in the tree using up arrow key."""
        if not hasattr(self, "file_tree") or self.file_tree is None:
            return
        
        current_item = self.file_tree.currentItem()
        if current_item is None:
            # If no current item, select the first item
            if self.file_tree.topLevelItemCount() > 0:
                self.file_tree.setCurrentItem(self.file_tree.topLevelItem(0))
            return
        
        current_index = self.file_tree.indexOfTopLevelItem(current_item)
        if current_index > 0:
            # Move to previous item
            prev_item = self.file_tree.topLevelItem(current_index - 1)
            if prev_item is not None:
                self.file_tree.setCurrentItem(prev_item)

    def _navigate_down(self) -> None:
        """Navigate to the next file in the tree using down arrow key."""
        if not hasattr(self, "file_tree") or self.file_tree is None:
            return
        
        current_item = self.file_tree.currentItem()
        if current_item is None:
            # If no current item, select the first item
            if self.file_tree.topLevelItemCount() > 0:
                self.file_tree.setCurrentItem(self.file_tree.topLevelItem(0))
            return
        
        current_index = self.file_tree.indexOfTopLevelItem(current_item)
        if current_index < self.file_tree.topLevelItemCount() - 1:
            # Move to next item
            next_item = self.file_tree.topLevelItem(current_index + 1)
            if next_item is not None:
                self.file_tree.setCurrentItem(next_item)

    # ------------------------------------------------------------------
    # QA Export functionality
    # ------------------------------------------------------------------
    def _calculate_export_hash(self, file_key: str) -> str:
        """Calculate fast hash using bad_epoch_indices from CSV metadata.

        This avoids reading large .set files by hashing only the metadata
        that defines which epochs should be dropped.

        Args:
            file_key: The CSV record key for the file

        Returns:
            SHA256 hash of the export configuration
        """
        record = self.decisions.get(file_key, {})
        if not isinstance(record, dict):
            record = {}
        metadata = {
            'bad_epoch_indices': record.get('bad_epoch_indices', ''),
            'total_epochs': record.get('total_epochs', 0),
            'file_key': file_key
        }
        hash_str = json.dumps(metadata, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()

    def _batch_export_to_qa(self) -> None:
        """Export cleaned .set files (bad epochs removed) to qa/ folder.

        Only exports files where bad_epochs_count > 0 (modified files).
        Uses hash checking to avoid redundant writes.
        Progress is shown via QProgressDialog.
        """
        from PyQt6.QtWidgets import QProgressDialog, QMessageBox
        from datetime import datetime
        import mne

        # Find all files with bad epochs marked
        files_to_export = [
            (key, record) for key, record in self.decisions.items()
            if isinstance(record, dict) and record.get('bad_epochs_count', 0) > 0
        ]

        print(f"[QA EXPORT] Found {len(files_to_export)} files to export")
        for key, record in files_to_export:
            print(f"[QA EXPORT]   {key}: type={type(record)}, bad_epochs={record.get('bad_epochs_count', 'N/A') if isinstance(record, dict) else 'NOT DICT'}")

        if not files_to_export:
            QMessageBox.information(
                self,
                "No Files to Export",
                "No files have bad epochs marked for export."
            )
            return

        # Create qa/ directory if it doesn't exist
        qa_dir = self.task_root / "qa" if self.task_root else None
        if not qa_dir:
            QMessageBox.warning(
                self,
                "Task Root Not Found",
                "Cannot determine task root directory for QA exports."
            )
            return

        qa_dir.mkdir(exist_ok=True)

        # Create progress dialog
        progress = QProgressDialog(
            "Exporting cleaned files to QA...",
            "Cancel",
            0,
            len(files_to_export),
            self
        )
        progress.setWindowTitle("Batch Export to QA")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        exported_count = 0
        skipped_count = 0
        error_count = 0

        for idx, (file_key, record) in enumerate(files_to_export):
            if progress.wasCanceled():
                break

            progress.setValue(idx)

            # Debug: verify record type
            if not isinstance(record, dict):
                print(f"[QA EXPORT] ERROR: record is {type(record)} not dict for {file_key}")
                error_count += 1
                continue

            relative_path = record.get('relative_path', '')
            progress.setLabelText(f"Processing {relative_path}...")

            try:
                # Calculate current hash
                current_hash = self._calculate_export_hash(file_key)

                # Check if already exported with same hash
                existing_hash = record.get('qa_export_hash', '')
                if existing_hash == current_hash:
                    skipped_count += 1
                    continue

                # Load original file from exports/
                source_file = self.exports_dir / relative_path
                if not source_file.exists():
                    print(f"[QA EXPORT] Source file not found: {source_file}")
                    error_count += 1
                    continue

                # Load epochs
                epochs = mne.read_epochs_eeglab(str(source_file), verbose=False)

                # Get bad epoch indices from CSV
                bad_indices_str = record.get('bad_epoch_indices', '')
                if bad_indices_str:
                    bad_indices = [int(idx.strip()) for idx in bad_indices_str.split(',') if idx.strip()]

                    # Map back to selection indices (in case some epochs were already dropped)
                    bad_selection_indices = []
                    for bad_num in bad_indices:
                        if bad_num in epochs.selection:
                            sel_idx = epochs.selection.tolist().index(bad_num)
                            bad_selection_indices.append(sel_idx)

                    # Drop bad epochs
                    if bad_selection_indices:
                        epochs.drop(bad_selection_indices, reason='USER', verbose=False)

                # Save to qa/ with same filename using MNE's native save
                dest_file = qa_dir / source_file.name
                epochs.save(str(dest_file), overwrite=True, verbose=False)

                # Update CSV record with export metadata
                record['qa_export_hash'] = current_hash
                record['qa_export_timestamp'] = datetime.now().isoformat()
                record['qa_export_path'] = f"qa/{source_file.name}"

                exported_count += 1

            except Exception as e:
                import traceback
                print(f"[QA EXPORT] Error exporting {file_key}: {e}")
                print(f"[QA EXPORT] Traceback: {traceback.format_exc()}")
                error_count += 1

        progress.setValue(len(files_to_export))

        # Save updated CSV
        self._commit_decisions()

        # Create unified QA preprocessing log
        qa_log_path = self._create_qa_preprocessing_log()

        # Show summary
        summary = f"Export complete:\n\n"
        summary += f"Exported: {exported_count}\n"
        summary += f"Skipped (unchanged): {skipped_count}\n"
        if error_count > 0:
            summary += f"Errors: {error_count}\n"
        if qa_log_path:
            summary += f"\n✓ QA preprocessing log created"

        QMessageBox.information(self, "Export Complete", summary)

    def _create_qa_preprocessing_log(self) -> Optional[Path]:
        """Create unified QA preprocessing log merging auto and manual metrics.

        Combines preprocessing_log.csv with exclusion decisions to create
        a comprehensive qa_preprocessing_log.csv in the qa/ folder.

        Returns:
            Path to generated QA log, or None if preprocessing_log unavailable
        """
        import pandas as pd

        # Check if we have preprocessing log
        if self.preprocessing_log_df is None or self.preprocessing_log_df.empty:
            print("[QA LOG] No preprocessing log available, skipping QA log generation")
            return None

        # Check if qa directory exists
        qa_dir = self.task_root / "qa" if self.task_root else None
        if not qa_dir or not qa_dir.exists():
            print("[QA LOG] QA directory not found, skipping QA log generation")
            return None

        print(f"[QA LOG] Creating QA preprocessing log from {len(self.preprocessing_log_df)} records")

        # Start with copy of preprocessing log
        qa_log = self.preprocessing_log_df.copy()

        # Get key column from config
        config = _load_config()
        key_column = config.get("logfile", {}).get("key_column", "subj_basename") if config else "subj_basename"

        if key_column not in qa_log.columns:
            print(f"[QA LOG] Warning: key column '{key_column}' not found in preprocessing log")
            key_column = qa_log.columns[0]  # Fallback to first column

        # Create mapping from entry to manual review data
        manual_data = {}
        for entry, record in self.decisions.items():
            if not isinstance(record, dict):
                continue

            # Strip suffixes to get subj_basename
            subj_basename = strip_suffixes(entry, config=config)

            manual_data[subj_basename] = {
                'qa_status': record.get('status', ''),
                'manual_bad_epochs': record.get('bad_epochs_count', 0),
                'manual_bad_epoch_indices': record.get('bad_epoch_indices', ''),
                'manual_review_timestamp': record.get('last_updated', ''),
                'manual_review_notes': record.get('notes', ''),
                'qa_exported': record.get('qa_export_timestamp', ''),
            }

        # Add new QA columns
        qa_log['qa_status'] = qa_log[key_column].map(lambda x: manual_data.get(x, {}).get('qa_status', ''))
        qa_log['manual_bad_epochs'] = qa_log[key_column].map(lambda x: manual_data.get(x, {}).get('manual_bad_epochs', 0))
        qa_log['manual_bad_epoch_indices'] = qa_log[key_column].map(lambda x: manual_data.get(x, {}).get('manual_bad_epoch_indices', ''))
        qa_log['manual_review_timestamp'] = qa_log[key_column].map(lambda x: manual_data.get(x, {}).get('manual_review_timestamp', ''))
        qa_log['manual_review_notes'] = qa_log[key_column].map(lambda x: manual_data.get(x, {}).get('manual_review_notes', ''))
        qa_log['qa_exported'] = qa_log[key_column].map(lambda x: manual_data.get(x, {}).get('qa_exported', ''))

        # Update original epoch_badtrials to include manual bad epochs
        if 'epoch_badtrials' in qa_log.columns:
            qa_log['epoch_badtrials'] = qa_log['epoch_badtrials'].fillna(0) + qa_log['manual_bad_epochs'].fillna(0)

            # Recalculate epoch_percent with updated bad epochs count
            if 'epoch_trials' in qa_log.columns:
                qa_log['epoch_percent'] = (
                    (qa_log['epoch_trials'] - qa_log['epoch_badtrials']) / qa_log['epoch_trials']
                ).fillna(1.0)
        else:
            print("[QA LOG] Warning: 'epoch_badtrials' column not found, cannot update metrics")

        # Save to qa/ folder
        qa_log_path = qa_dir / "qa_preprocessing_log.csv"
        qa_log.to_csv(qa_log_path, index=False)

        print(f"[QA LOG] Created QA preprocessing log: {qa_log_path}")
        print(f"[QA LOG]   - {len(qa_log)} total records")
        print(f"[QA LOG]   - {len([k for k in manual_data if manual_data[k]['qa_status']])} manually reviewed")

        return qa_log_path

    def _open_qa_folder(self) -> None:
        """Open the QA folder in the system file browser."""
        if not self.task_root:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "QA Folder Not Found",
                "Cannot open QA folder: task root directory not set."
            )
            return

        qa_dir = self.task_root / "qa"
        if not qa_dir.exists():
            from PyQt6.QtWidgets import QMessageBox
            # Create the qa directory if it doesn't exist
            qa_dir.mkdir(exist_ok=True)
            print(f"[QA FOLDER] Created QA directory: {qa_dir}")

        _open_path(qa_dir)
        print(f"[QA FOLDER] Opened QA folder: {qa_dir}")


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
