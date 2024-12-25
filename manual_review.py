# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mne",
#     "rich", 
#     "numpy",
#     "python-dotenv",
#     "openneuro-py",
#     "pyyaml",
#     "schema",
#     "mne-bids",
#     "pandas",
#     "pathlib",
#     "pybv",
#     "torch",
#     "pyprep",
#     "eeglabio",
#     "autoreject",
#     "python-ulid",
#     "pylossless @ /Users/ernie/Documents/GitHub/EegServer/pylossless",
#     "textual",
#     "textual-dev",
#     "asyncio",
#     "mplcairo",
#     "unqlite",
#     "PyQt5",
#     "matplotlib",
#     "mne-qt-browser",
#     "scipy",
#     "pyjsonviewer"
# ]
# ///


import sys
import os
import subprocess
import mne
import json
import scipy.io as sio
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QFileDialog,
    QVBoxLayout, QTreeWidget, QTreeWidgetItem, QStatusBar, QMessageBox,
    QTreeView, QSplitter
)
from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from autoclean_pipeline_v2 import save_epochs_to_set, get_run_record
import pyjsonviewer

from PyQt5.QtCore import pyqtRemoveInputHook
from pdb import set_trace

pyqtRemoveInputHook()


plt.style.use('default')
mne.viz.set_browser_backend('qt')

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
        row = parent_item.parent.children.index(parent_item) if parent_item.parent else 0
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
    def __init__(self):
        super().__init__()
        self.current_dir = os.getenv('AUTOCLEAN_DIR', None)
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

        self.select_dir_btn = QPushButton('Select Directory')
        self.select_dir_btn.clicked.connect(self.selectDirectory)
        self.left_layout.addWidget(self.select_dir_btn)

        self.open_folder_btn = QPushButton('Open Current Folder')
        self.open_folder_btn.clicked.connect(self.openCurrentFolder)
        self.left_layout.addWidget(self.open_folder_btn)

        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabel('Files')
        self.file_tree.itemClicked.connect(self.onFileSelect)
        self.left_layout.addWidget(self.file_tree)

        self.plot_btn = QPushButton('Review Selected File')
        self.plot_btn.clicked.connect(self.plotFile)
        self.plot_btn.setEnabled(False)
        self.left_layout.addWidget(self.plot_btn)

        # New "Save Edits" button
        self.save_edits_btn = QPushButton('Save Edits')
        self.save_edits_btn.clicked.connect(self.saveEdits)
        self.save_edits_btn.setEnabled(False)
        self.left_layout.addWidget(self.save_edits_btn)

        self.view_record_btn = QPushButton('View Run Record')
        self.view_record_btn.clicked.connect(self.viewRunRecord)
        self.view_record_btn.setEnabled(False)
        self.left_layout.addWidget(self.view_record_btn)

        self.exit_btn = QPushButton('Exit')
        self.exit_btn.clicked.connect(self.close)
        self.left_layout.addWidget(self.exit_btn)

        self.status_bar = QStatusBar()
        self.left_layout.addWidget(self.status_bar)

        # Right container (for the plot widget)
        self.right_container = QWidget()
        self.right_layout = QVBoxLayout()
        self.right_container.setLayout(self.right_layout)

        # Add the two containers to the splitter
        self.splitter.addWidget(left_container)
        self.splitter.addWidget(self.right_container)

        # Set initial sizes, left smaller, right larger
        self.splitter.setStretchFactor(0, 0)  
        self.splitter.setStretchFactor(1, 1)

        # Main layout for the entire window
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.splitter)
        self.setLayout(main_layout)

        self.setGeometry(300, 300, 1200, 600)  # Wider default width
        self.setWindowTitle('Manual Epoch Rejection')

    def openCurrentFolder(self):
        if self.current_dir:
            if sys.platform == 'darwin':
                subprocess.run(['open', self.current_dir])
            elif sys.platform == 'linux':
                subprocess.run(['xdg-open', self.current_dir])
            else:
                os.startfile(self.current_dir)

    def updateStatusBar(self):
        if self.current_dir:
            self.status_bar.showMessage(f"Current directory: {self.current_dir}")
        else:
            self.status_bar.showMessage("No directory selected")

    def loadFiles(self):
        self.file_tree.clear()
        if self.current_dir is not None:
            root = QTreeWidgetItem(self.file_tree, [os.path.basename(self.current_dir)])
            self.populateTree(root, self.current_dir)
            root.setExpanded(True)

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
            if item.endswith('.set'):
                file_item = QTreeWidgetItem(parent, [item])
                file_item.setIcon(0, self.style().standardIcon(self.style().SP_FileIcon))
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
        return str(EEG['etc']['run_id'][0][0][0])

    def onFileSelect(self, item):
        if item.text(0).endswith('.set') or item.text(0).endswith('.set *'):
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
        if hasattr(self, 'current_run_id') and self.current_run_record:
            try:
                self.current_run_record_window = QWidget()
                self.current_run_record_window.setWindowTitle(f"Run Record - {self.current_run_id}")
                self.current_run_record_window.setGeometry(400, 400, 800, 600)

                layout = QVBoxLayout()
                tree_view = QTreeView()
                model = JsonTreeModel(self.current_run_record)
                tree_view.setModel(model)
                tree_view.setAlternatingRowColors(True)
                tree_view.setHeaderHidden(False)
                tree_view.expandAll()
                layout.addWidget(tree_view)
                self.current_run_record_window.setLayout(layout)
                self.current_run_record_window.show()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error retrieving run record: {str(e)}")
        else:
                QMessageBox.warning(self, "Warning", "No run record found for this ID")

    def plotFile(self):
        if hasattr(self, 'selected_file_path'):
            try:
                epochs = mne.read_epochs_eeglab(self.selected_file_path)

                # Remove old plot widget if present
                if self.plot_widget is not None:
                    self.right_layout.removeWidget(self.plot_widget)
                    self.plot_widget.close()
                    self.plot_widget = None

                # Store original epochs
                self.current_epochs = epochs.copy()  # Make sure we have a copy
                self.original_epoch_count = len(epochs)

                # Store the original events array
                self.original_events = self.current_epochs.events.copy()
                # breakpoint()

                # Create the plot widget embedded in our GUI
                self.plot_widget = self.current_epochs.plot(
                    n_epochs=len(epochs),
                    show=False,  # Don't show in separate window
                    block=True,  # Don't block
                    picks='all'
                )
                # Add close handler to the plot widget
                def handle_close():
                    print("Plot closed, checking for dropped epochs...")
                    self.current_epochs.drop_bad()
                    if hasattr(self, 'current_epochs'):
                        # Compare original and updated events to find dropped epochs
                        original_event_ids = set(self.original_events[:, 0])
                        updated_event_ids = set(self.current_epochs.events[:, 0])
                        dropped_ids = original_event_ids - updated_event_ids

                        if dropped_ids:
                            print(f"Epochs marked as bad: {list(dropped_ids)}")
                            print(f"Epochs dropped. New count: {len(self.current_epochs)}")
                        else:
                            print("No epochs were dropped.")

                self.plot_widget.destroyed.connect(handle_close)

                # Embed the plot in our GUI
                self.right_layout.addWidget(self.plot_widget)
                self.plot_widget.show()

                # Enable save button and store reference for access during save
                self.save_edits_btn.setEnabled(True)

                print("Plot widget created and embedded in GUI")
                print(f"Initial epoch count: {len(self.current_epochs)}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading/plotting file: {str(e)}")
                print(f"Error in plotFile: {str(e)}")
    def saveEdits(self):
        if not hasattr(self, 'current_epochs'):
            QMessageBox.warning(self, "Warning", "No epochs loaded to save.")
            return

        try:
            print("\nStarting save process...")
            print(f"Original epoch count: {self.original_epoch_count}")

            # Find dropped epochs by comparing event IDs
            original_event_ids = set(self.original_events[:, 0])
            updated_event_ids = set(self.current_epochs.events[:, 0])
            dropped_ids = original_event_ids - updated_event_ids

            # Check if any epochs were dropped
            if not dropped_ids:
                response = QMessageBox.question(
                    self,
                    "No Changes Detected",
                    "No epochs were dropped. Would you like to:\n\n" +
                    "1. Save anyway (Yes)\n" +
                    "2. Cancel and try marking epochs again (No)",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if response == QMessageBox.No:
                    return

            print(f"Applying {len(dropped_ids)} drops...")

            # Get the indices of the dropped epochs from the original events array
            dropped_indices = [
                i for i, event_id in enumerate(self.original_events[:, 0])
                if event_id in dropped_ids
            ]

            # Create a copy of the original epochs and drop the marked epochs
            epochs_to_save = self.current_epochs.copy()
            if dropped_indices:
                epochs_to_save.drop(dropped_indices, reason='manual')

            final_count = len(epochs_to_save)
            print(f"Final epoch count: {final_count}")

            # Only proceed with save if epochs were actually dropped
            # or user confirmed they want to save anyway
            run_record = get_run_record(self.current_run_id)
            autoclean_dict = {
                'run_id': self.current_run_id,
                'stage_files': run_record['metadata']['entrypoint']['stage_files'],
                'stage_dir': Path(run_record['metadata']['step_prepare_directories']['stage']),
                'unprocessed_file': run_record['unprocessed_file']
            }

            print("Saving epochs to file...")
            save_epochs_to_set(epochs_to_save, autoclean_dict, stage='post_edit')
            self.modified_files.add(self.selected_file)
            self.loadFiles()

            # Show detailed results
            dropped_count = self.original_epoch_count - final_count
            message = (
                f"Save completed successfully.\n\n"
                f"Original epochs: {self.original_epoch_count}\n"
                f"Epochs dropped: {dropped_count}\n"
                f"Final epochs: {final_count}"
            )

            QMessageBox.information(self, "Save Complete", message)
            self.save_edits_btn.setEnabled(False)
            if self.plot_widget:
                self.plot_widget.deleteLater()
            print("Save process completed")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving edits: {str(e)}")
            import traceback
            print(f"Error during save: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet("")
    window = FileSelector()
    window.show()
    sys.exit(app.exec_())
