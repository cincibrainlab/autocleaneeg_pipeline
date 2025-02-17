import sys
from pathlib import Path

# Add the src directory to Python path if package is not installed
src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import os
import subprocess
import webbrowser
from pathlib import Path

import fitz
import matplotlib.pyplot as plt
import mne
import scipy.io as sio
from dotenv import load_dotenv
from PyQt5.Qt import *
from PyQt5.QtCore import QAbstractItemModel, QModelIndex, Qt, pyqtRemoveInputHook
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QTreeView,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from autoclean.step_functions.io import save_epochs_to_set
from autoclean.utils.database import get_run_record
from autoclean.utils.logging import message

pyqtRemoveInputHook()


plt.style.use("default")
mne.viz.set_browser_backend("qt")


class JsonTreeModel(QAbstractItemModel):
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

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        item = index.internalPointer()
        return item.key if index.column() == 0 else item.value

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return ["Key", "Value"][section]
        return None


load_dotenv()


class FileSelector(QWidget):
    def __init__(self, autoclean_dir):
        super().__init__()
        self.current_dir = autoclean_dir
        self.modified_files = set()
        self.current_run_id = None
        self.current_run_record = None
        self.current_run_record_window = None
        self.plot_widget = None
        self.current_epochs = None  # Store the currently loaded epochs

        self.initUI()

        if self.current_dir:
            self.loadFiles()
            self.updateStatusBar()

    def initUI(self):
        # Main splitter to divide the interface horizontally
        self.splitter = QSplitter(Qt.Horizontal, self)

        # Left container (directory controls + file tree + buttons)
        left_container = QWidget()
        self.left_layout = QVBoxLayout()
        left_container.setLayout(self.left_layout)

        self.select_dir_btn = QPushButton("Select Directory")
        self.select_dir_btn.clicked.connect(self.selectDirectory)
        self.left_layout.addWidget(self.select_dir_btn)

        self.open_folder_btn = QPushButton("Open Current Folder")
        self.open_folder_btn.clicked.connect(self.openCurrentFolder)
        self.left_layout.addWidget(self.open_folder_btn)

        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabel("Files")
        self.file_tree.itemClicked.connect(self.onFileSelect)
        self.left_layout.addWidget(self.file_tree)

        self.plot_btn = QPushButton("Review Selected File")
        self.plot_btn.clicked.connect(self.plotFile)
        self.plot_btn.setEnabled(False)
        self.left_layout.addWidget(self.plot_btn)

        # Close Plot button
        self.close_plot_btn = QPushButton("Close Review Plot")
        self.close_plot_btn.clicked.connect(self.closePlot)
        self.close_plot_btn.setEnabled(False)
        self.left_layout.addWidget(self.close_plot_btn)

        # New "Save Edits" button
        self.save_edits_btn = QPushButton("Save Edits")
        self.save_edits_btn.setEnabled(False)
        self.left_layout.addWidget(self.save_edits_btn)

        self.view_record_btn = QPushButton("View Run Record")
        self.view_record_btn.clicked.connect(self.viewRunRecord)
        self.view_record_btn.setEnabled(False)
        self.left_layout.addWidget(self.view_record_btn)

        self.exit_btn = QPushButton("Exit")
        self.exit_btn.clicked.connect(self.close)
        self.left_layout.addWidget(self.exit_btn)

        # Right container (for the plot widget)
        self.right_container = QWidget()
        self.right_layout = QVBoxLayout()

        # Create and style instruction label
        self.instruction_widget = QWidget()
        instruction_layout = QVBoxLayout()

        title_label = QLabel("Manual Epoch Rejection Instructions")
        title_label.setStyleSheet(
            """
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 20px;
            }
        """
        )
        title_label.setAlignment(Qt.AlignCenter)

        instructions = QLabel(
            """
            <html>
            <style>
                p { margin: 10px 0; }
                .step { color: #2980b9; font-weight: bold; }
                .key { color: #27ae60; font-weight: bold; }
                .note { color: #c0392b; }
            </style>
            <body>
            <p><span class='step'>1.</span> Select a directory containing .set files using the <span class='key'>Select Directory</span> button</p>
            <p><span class='step'>2.</span> Navigate through the file tree and select a .set file to review</p>
            <p><span class='step'>3.</span> Click <span class='key'>Review Selected File</span> to load the EEG epochs</p>
            <p><span class='step'>4.</span> In the epoch viewer:</p>
            <ul>
                <li>Use <span class='key'>←/→</span> keys to navigate between epochs</li>
                <li>Press <span class='key'>Space</span> to mark/unmark bad epochs</li>
                <li>Click <span class='key'>Save Edits</span> when finished to save your changes</li>
                <li>After confirmation, edited files will be saved to the <span class='key'>postedit</span> directory</li>
            </ul>
            <p><span class='step'>5.</span> Click <span class='key'>View Run Record</span> to see:</p>
            <ul>
                <li>Processing history and parameters</li>
                <li>Channel rejection plots</li>
                <li>ICA component plots</li>
                <li>Power spectral density plots</li>
                <li>Artifact detection statistics</li>
            </ul>
            <p><span class='note'>Note: Modified files will be marked with an asterisk (*) in red</span></p>
            </body>
            </html>
        """
        )
        instructions.setStyleSheet(
            """
            QLabel {
                font-size: 14px;
                color: #34495e;
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
            }
        """
        )
        instructions.setWordWrap(True)
        instructions.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        instruction_layout.addWidget(title_label)
        instruction_layout.addWidget(instructions)
        instruction_layout.addStretch()
        self.instruction_widget.setLayout(instruction_layout)

        self.right_layout.addWidget(self.instruction_widget)
        self.right_container.setLayout(self.right_layout)

        # Add the two containers to the splitter
        self.splitter.addWidget(left_container)
        self.splitter.addWidget(self.right_container)

        # Set initial sizes, left smaller, right larger
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)

        # Create status bar
        self.status_bar = QStatusBar()
        self.status_bar.setFixedHeight(25)

        # Main layout for the entire window
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.splitter)
        main_layout.addWidget(self.status_bar)
        # Remove margins around the status bar
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)

        self.setGeometry(300, 300, 1200, 600)  # Wider default width
        self.setWindowTitle("Autoclean 1.0 - Manual Epoch Rejection")

    def openCurrentFolder(self):
        if self.current_dir:
            if sys.platform == "darwin":
                subprocess.run(["open", self.current_dir])
            elif sys.platform == "linux":
                subprocess.run(["xdg-open", self.current_dir])
            else:
                os.startfile(self.current_dir)

    def updateStatusBar(self):
        if self.current_dir:
            self.status_bar.showMessage(f"Current directory: {self.current_dir}")
        else:
            self.status_bar.showMessage("No directory selected")

    def render_pdf_page(pdf_path, page_num=0, zoom=1.0):
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        q_img = QImage(
            pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGBA8888
        )
        return QPixmap.fromImage(q_img)

    def loadFiles(self):
        self.file_tree.clear()
        if self.current_dir is not None:
            root = QTreeWidgetItem(self.file_tree, [os.path.basename(self.current_dir)])
            self.populateTree(root, self.current_dir)
            root.setExpanded(True)

            # Expand the first folder if it exists
            if root.childCount() > 0:
                first_child = root.child(0)
                first_child.setExpanded(True)  # Expand the first folder
                self.file_tree.expandItem(
                    first_child
                )  # Ensure the first folder is expanded

    def populateTree(self, parent, path):
        # Directories
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                folder = QTreeWidgetItem(parent, [item])
                folder.setIcon(0, self.style().standardIcon(self.style().SP_DirIcon))
                self.populateTree(folder, item_path)
        # Files
        for item in os.listdir(path):
            if item.endswith(".set"):
                file_item = QTreeWidgetItem(parent, [item])
                file_item.setIcon(
                    0, self.style().standardIcon(self.style().SP_FileIcon)
                )
                if item in self.modified_files:
                    file_item.setText(0, f"{item} *")
                    file_item.setForeground(0, Qt.red)

    def selectDirectory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            self.current_dir = dir_path
            self.loadFiles()
            self.updateStatusBar()

    def getRunId(self, file_path):
        EEG = sio.loadmat(file_path)
        return str(EEG["etc"]["run_id"][0][0][0])

    def onFileSelect(self, item):
        if item.text(0).endswith(".set") or item.text(0).endswith(".set *"):
            self.selected_file = item.text(0).replace(" *", "")
            self.plot_btn.setEnabled(True)
            path_parts = []
            current = item
            while current is not None:
                path_parts.insert(0, current.text(0).replace(" *", ""))
                current = current.parent()
            self.selected_file_path = os.path.join(self.current_dir, *path_parts[1:])
            try:
                self.current_run_id = self.getRunId(self.selected_file_path)
                self.current_run_record = get_run_record(self.current_run_id)
                self.view_record_btn.setEnabled(True)
            except Exception:
                self.view_record_btn.setEnabled(False)
        else:
            self.plot_btn.setEnabled(False)
            self.view_record_btn.setEnabled(False)

    def viewRunRecord(self):
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QPixmap
        from PyQt5.QtWidgets import (
            QComboBox,
            QHBoxLayout,
            QLabel,
            QPushButton,
        )

        original_filename = self.current_run_record["metadata"]["step_import"][
            "unprocessedFile"
        ]

        if hasattr(self, "current_run_id") and self.current_run_record:
            try:
                self.current_run_record_window = QWidget()
                self.current_run_record_window.setWindowTitle(
                    f"File: {original_filename} - Run Record - {self.current_run_id}"
                )
                self.current_run_record_window.resize(1000, 800)

                layout = QVBoxLayout()

                # Create splitter for tree and artifact view
                splitter = QSplitter(Qt.Horizontal)

                # Left side - JSON tree in scroll area
                scroll_tree = QScrollArea()
                tree_view = QTreeView()
                model = JsonTreeModel(self.current_run_record)
                tree_view.setModel(model)
                tree_view.setAlternatingRowColors(True)
                tree_view.setHeaderHidden(False)
                # tree_view.expandAll()
                scroll_tree.setWidget(tree_view)
                scroll_tree.setWidgetResizable(True)
                splitter.addWidget(scroll_tree)

                # Right side - Artifact reports
                artifact_widget = QWidget()
                artifact_layout = QVBoxLayout()

                # Dropdown for PNG/PDF file selection
                file_dropdown = QComboBox()

                # Add zoom controls
                zoom_widget = QWidget()
                zoom_layout = QHBoxLayout()
                zoom_in_btn = QPushButton("+")
                zoom_out_btn = QPushButton("-")
                zoom_reset_btn = QPushButton("Reset")
                zoom_fit_btn = QPushButton("Fit")
                open_folder_btn = QPushButton("Open Folder")
                zoom_layout.addWidget(zoom_in_btn)
                zoom_layout.addWidget(zoom_out_btn)
                zoom_layout.addWidget(zoom_reset_btn)
                zoom_layout.addWidget(zoom_fit_btn)
                zoom_layout.addWidget(open_folder_btn)
                zoom_widget.setLayout(zoom_layout)

                # Get paths
                subject_id = self.current_run_record["metadata"][
                    "step_convert_to_bids"
                ]["bids_subject"]
                session = self.current_run_record["metadata"]["step_convert_to_bids"][
                    "bids_session"
                ]
                task = self.current_run_record["metadata"]["step_convert_to_bids"][
                    "bids_task"
                ]
                run = self.current_run_record["metadata"]["step_convert_to_bids"][
                    "bids_run"
                ]
                bids_root = Path(
                    self.current_run_record["metadata"]["step_prepare_directories"][
                        "bids"
                    ]
                )
                derivatives_stem = "pylossless"

                derivatives_dir = Path(
                    bids_root,
                    "derivatives",
                    derivatives_stem,
                    "sub-" + subject_id,
                    "eeg",
                )

                def open_derivatives_folder():
                    folder_path = str(derivatives_dir)
                    if sys.platform == "darwin":  # macOS
                        subprocess.run(["open", folder_path])
                    elif sys.platform == "win32":  # Windows
                        os.startfile(folder_path)
                    else:  # Linux
                        subprocess.run(["xdg-open", folder_path])

                open_folder_btn.clicked.connect(open_derivatives_folder)

                # Get all PNG and PDF files in derivatives directory
                image_files = list(derivatives_dir.glob("*.png")) + list(
                    derivatives_dir.glob("*.pdf")
                )

                if image_files:
                    # Add PNG/PDF filenames to dropdown
                    file_dropdown.addItems([f.name for f in image_files])

                    def update_image(index):
                        """Load and display the selected image or PDF file."""
                        try:
                            # Get the selected file path
                            file_path = str(image_files[index])
                            print(f"Loading document from: {file_path}")

                            if file_path.lower().endswith(".pdf"):
                                # Open PDF in system's default browser
                                webbrowser.open(f"file://{file_path}")
                                print(f"Opened PDF in browser: {file_path}")
                                return

                            # Clear any previously displayed content for images
                            for i in reversed(range(artifact_layout.count())):
                                widget = artifact_layout.itemAt(i).widget()
                                if isinstance(widget, (QLabel, QScrollArea)):
                                    widget.deleteLater()

                            if file_path.lower().endswith(".png"):
                                # Set up scroll area and label for PNG
                                scroll = QScrollArea()
                                label = QLabel()

                                # Load PNG at full resolution
                                label.original_pixmap = QPixmap(file_path)

                                # Enable mouse tracking for better interaction
                                scroll.setWidgetResizable(True)
                                scroll.setHorizontalScrollBarPolicy(
                                    Qt.ScrollBarAsNeeded
                                )
                                scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

                                # Set minimum size for scroll area
                                scroll.setMinimumSize(400, 400)

                                scroll.setWidget(label)
                                artifact_layout.addWidget(scroll)
                                print(
                                    f"Successfully loaded PNG at full resolution: {file_path}"
                                )

                                # Calculate fit scale
                                available_width = scroll.width() - 20
                                available_height = scroll.height() - 20
                                width_ratio = (
                                    available_width / label.original_pixmap.width()
                                )
                                height_ratio = (
                                    available_height / label.original_pixmap.height()
                                )
                                scale = min(width_ratio, height_ratio)

                                # Scale image to fit by default
                                scaled_pixmap = label.original_pixmap.scaled(
                                    int(label.original_pixmap.width() * scale),
                                    int(label.original_pixmap.height() * scale),
                                    Qt.KeepAspectRatio,
                                    Qt.SmoothTransformation,
                                )
                                label.setPixmap(scaled_pixmap)
                            else:
                                print(f"Unsupported file type: {file_path}")
                                return

                        except Exception as e:
                            print(f"Error loading document: {str(e)}")

                    def zoom_in():
                        # For images, zoom using original high-res pixmap
                        scroll = artifact_layout.itemAt(2).widget()
                        label = scroll.widget()
                        if hasattr(label, "original_pixmap"):
                            current_scale = (
                                label.pixmap().width() / label.original_pixmap.width()
                            )
                            new_scale = current_scale * 1.2
                            scaled_pixmap = label.original_pixmap.scaled(
                                int(label.original_pixmap.width() * new_scale),
                                int(label.original_pixmap.height() * new_scale),
                                Qt.KeepAspectRatio,
                                Qt.SmoothTransformation,
                            )
                            label.setPixmap(scaled_pixmap)

                    def zoom_out():
                        # For images, zoom using original high-res pixmap
                        scroll = artifact_layout.itemAt(2).widget()
                        label = scroll.widget()
                        if hasattr(label, "original_pixmap"):
                            current_scale = (
                                label.pixmap().width() / label.original_pixmap.width()
                            )
                            new_scale = current_scale / 1.2
                            scaled_pixmap = label.original_pixmap.scaled(
                                int(label.original_pixmap.width() * new_scale),
                                int(label.original_pixmap.height() * new_scale),
                                Qt.KeepAspectRatio,
                                Qt.SmoothTransformation,
                            )
                            label.setPixmap(scaled_pixmap)

                    def zoom_reset():
                        # Reset to original size for images
                        scroll = artifact_layout.itemAt(2).widget()
                        label = scroll.widget()
                        if hasattr(label, "original_pixmap"):
                            label.setPixmap(label.original_pixmap)

                    def zoom_fit():
                        message("info", "Zooming to fit")

                        if artifact_layout.count() > 2:
                            scroll = artifact_layout.itemAt(2).widget()
                            if not scroll:
                                message("info", "No scroll area found")
                                return

                            label = scroll.widget()
                            if not hasattr(label, "original_pixmap"):
                                message("info", "No original pixmap found")
                                return
                        else:
                            message("info", "No scroll area found")
                            return

                        # Get the available space in the scroll area
                        available_width = (
                            scroll.width() - 20
                        )  # Account for scrollbar width
                        available_height = (
                            scroll.height() - 20
                        )  # Account for scrollbar height

                        # Calculate scaling factors
                        width_ratio = available_width / label.original_pixmap.width()
                        height_ratio = available_height / label.original_pixmap.height()

                        # Use the smaller ratio to ensure the image fits both dimensions
                        scale = min(width_ratio, height_ratio)

                        # Scale the image
                        scaled_pixmap = label.original_pixmap.scaled(
                            int(label.original_pixmap.width() * scale),
                            int(label.original_pixmap.height() * scale),
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation,
                        )
                        label.setPixmap(scaled_pixmap)

                    # Add resize event handler to auto-fit when window is resized
                    def on_resize(event):
                        zoom_fit()
                        QWidget.resizeEvent(self.current_run_record_window, event)

                    self.current_run_record_window.resizeEvent = on_resize

                    # Connect signals
                    file_dropdown.currentIndexChanged.connect(update_image)
                    zoom_in_btn.clicked.connect(zoom_in)
                    zoom_out_btn.clicked.connect(zoom_out)
                    zoom_reset_btn.clicked.connect(zoom_reset)
                    zoom_fit_btn.clicked.connect(zoom_fit)

                    # Add widgets to layout
                    artifact_layout.addWidget(file_dropdown)
                    artifact_layout.addWidget(zoom_widget)
                    artifact_widget.setLayout(artifact_layout)

                    splitter.addWidget(artifact_widget)
                    layout.addWidget(splitter)

                    self.current_run_record_window.setLayout(layout)
                    self.current_run_record_window.show()

                    # Initialize with first image
                    update_image(0)

                else:
                    QMessageBox.warning(
                        self,
                        "Warning",
                        "No PNG or PDF files found in derivatives directory",
                    )

            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Error retrieving run record: {str(e)}"
                )
        else:
            QMessageBox.warning(self, "Warning", "No run record found for this ID")

    def closePlot(self):
        """Close the plot and show instructions"""
        if self.plot_widget is not None:
            self.plot_widget.close()
            self.plot_widget.deleteLater()
            self.plot_widget = None
            self.close_plot_btn.setEnabled(False)
            self.save_edits_btn.setEnabled(False)
            self.instruction_widget.show()
            QApplication.processEvents()

    def plotFile(self):
        if hasattr(self, "selected_file_path"):
            try:
                print("INFO", "Plotting file")

                # Check if this is a RAW file
                is_raw = "_raw.set" in self.selected_file_path.lower()

                if is_raw:
                    # Load as raw EEG
                    raw = mne.io.read_raw_eeglab(self.selected_file_path, preload=True)
                    self.current_raw = raw.copy()
                    self.save_edits_btn.setEnabled(
                        False
                    )  # Disable saving for raw files
                else:
                    # Load as epochs
                    epochs = mne.read_epochs_eeglab(self.selected_file_path)
                    self.current_epochs = epochs.copy()
                    self.save_edits_btn.setEnabled(True)

                # Hide instructions
                self.instruction_widget.hide()

                # Remove old plot widget if present
                if self.plot_widget is not None:
                    print("INFO", "Removing old plot widget")
                    self.right_layout.removeWidget(self.plot_widget)
                    self.plot_widget.close()
                    self.plot_widget = None

                if not is_raw:
                    self.original_epoch_count = len(self.current_epochs)
                    print("INFO", "Original epoch count:", self.original_epoch_count)

                    def close_plot():
                        message("info", "Closing plot after save.")
                        message("info", f"Epoch number: {len(self.current_epochs)}")
                        message("info", "Manually marked epochs for removal:")
                        message("info", "=" * 50)
                        bad_epochs = sorted(self.plot_widget.mne.bad_epochs)
                        if bad_epochs:
                            message("info", f"Total epochs marked: {len(bad_epochs)}")
                            message("info", f"Epoch indices: {bad_epochs}")
                        else:
                            message("info", "No epochs were marked for removal")
                        message("info", "=" * 50)

                        run_record = get_run_record(self.current_run_id)
                        autoclean_dict = {
                            "run_id": self.current_run_id,
                            "stage_files": run_record["metadata"]["entrypoint"][
                                "stage_files"
                            ],
                            "stage_dir": Path(
                                run_record["metadata"]["step_prepare_directories"][
                                    "stage"
                                ]
                            ),
                            "unprocessed_file": run_record["unprocessed_file"],
                        }
                        reply = QMessageBox.question(
                            self,
                            "Confirm Save",
                            "Are you sure you want to save these changes?",
                            QMessageBox.Yes | QMessageBox.No,
                        )

                        if reply == QMessageBox.Yes:
                            message("info", "Saving epochs to file...")
                            self.current_epochs.drop(self.plot_widget.mne.bad_epochs)
                            save_epochs_to_set(
                                self.current_epochs, autoclean_dict, stage="post_edit"
                            )
                        else:
                            message("info", "Save cancelled by user")

                        self.closePlot()

                    self.save_edits_btn.clicked.connect(close_plot)

                    # Create the plot widget for epochs
                    self.plot_widget = self.current_epochs.plot(
                        n_epochs=10, show=False, block=True, picks="all", events=True
                    )
                else:
                    # Create the plot widget for raw data
                    self.plot_widget = self.current_raw.plot(
                        show=False, block=True, scalings="auto"
                    )

                # Embed the plot in our GUI
                self.right_layout.addWidget(self.plot_widget)
                self.plot_widget.show()
                self.close_plot_btn.setEnabled(True)

                print("INFO", "Plot widget created and embedded in GUI")
                if not is_raw:
                    print("INFO", f"Initial epoch count: {len(self.current_epochs)}")

            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Error loading/plotting file: {str(e)}"
                )
                print(f"Error in plotFile: {str(e)}")
                self.instruction_widget.show()  # Show instructions if plot fails


def run_autoclean_review(autoclean_dir):
    app = QApplication(sys.argv)
    app.setStyleSheet("")
    window = FileSelector(autoclean_dir)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_autoclean_review(Path("/tmp"))
