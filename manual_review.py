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
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QFileDialog,
                            QVBoxLayout, QTreeWidget, QTreeWidgetItem, QStatusBar, QMessageBox,
                            QTextEdit, QLabel, QLineEdit, QComboBox, QCheckBox, QTreeView)
from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex
import qdarkstyle
import numpy as np
from dotenv import load_dotenv
from autoclean_pipeline_v2 import save_epochs_to_set, get_run_record

import scipy.io as sio
import pyjsonviewer


# Force light mode for matplotlib
import matplotlib.pyplot as plt
plt.style.use('default')

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

        if not parent.isValid():
            parent_item = self._root
        else:
            parent_item = parent.internalPointer()

        if row < len(parent_item.children):
            return self.createIndex(row, column, parent_item.children[row])
        return QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()

        child_item = index.internalPointer()
        parent_item = child_item.parent

        if parent_item is self._root:
            return QModelIndex()

        if parent_item is None:
            return QModelIndex()

        if parent_item.parent is None:
            return QModelIndex()

        row = parent_item.parent.children.index(parent_item)
        return self.createIndex(row, 0, parent_item)

    def rowCount(self, parent=QModelIndex()):
        if parent.column() > 0:
            return 0

        if not parent.isValid():
            parent_item = self._root
        else:
            parent_item = parent.internalPointer()

        return len(parent_item.children)

    def columnCount(self, parent=QModelIndex()):
        return 2

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None

        item = index.internalPointer()

        if index.column() == 0:
            return item.key
        elif index.column() == 1:
            return item.value

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
        self.initUI()
        if self.current_dir:
            self.loadFiles()
            self.updateStatusBar()
        self.current_run_id = None
        self.current_run_record = None
        self.current_run_record_window = None

    def initUI(self):
        layout = QVBoxLayout()

        # Create button to select directory
        self.select_dir_btn = QPushButton('Select Directory')
        self.select_dir_btn.clicked.connect(self.selectDirectory)
        layout.addWidget(self.select_dir_btn)

        # Add open folder button
        self.open_folder_btn = QPushButton('Open Current Folder')
        self.open_folder_btn.clicked.connect(self.openCurrentFolder)
        layout.addWidget(self.open_folder_btn)

        # Create tree widget to show directory structure
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabel('Files')
        self.file_tree.itemClicked.connect(self.onFileSelect)
        layout.addWidget(self.file_tree)

        # Add plot button
        self.plot_btn = QPushButton('Review Selected File')
        self.plot_btn.clicked.connect(self.plotFile)
        self.plot_btn.setEnabled(False)
        layout.addWidget(self.plot_btn)

        # Add view record button
        self.view_record_btn = QPushButton('View Run Record')
        self.view_record_btn.clicked.connect(self.viewRunRecord)
        self.view_record_btn.setEnabled(False)
        layout.addWidget(self.view_record_btn)

        # Add exit button
        self.exit_btn = QPushButton('Exit')
        self.exit_btn.clicked.connect(self.close)
        layout.addWidget(self.exit_btn)

        # Add status bar
        self.status_bar = QStatusBar()
        layout.addWidget(self.status_bar)

        self.setLayout(layout)
        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle('Manual Epoch Rejection')

    def openCurrentFolder(self):
        if self.current_dir:
            if sys.platform == 'darwin':  # macOS
                subprocess.run(['open', self.current_dir])
            elif sys.platform == 'linux':  # Linux
                subprocess.run(['xdg-open', self.current_dir])
            else:  # Windows
                os.startfile(self.current_dir)

    def updateStatusBar(self):
        if self.current_dir:
            self.status_bar.showMessage(f"Current directory: {self.current_dir}")
        else:
            self.status_bar.showMessage("No directory selected")

    def loadFiles(self):
        # Clear previous items
        self.file_tree.clear()
        
        # Create root item
        root = QTreeWidgetItem(self.file_tree, [os.path.basename(self.current_dir)])
        self.populateTree(root, self.current_dir)
        
        # Expand root by default
        root.setExpanded(True)

    def populateTree(self, parent, path):
        # First add all subdirectories
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                folder = QTreeWidgetItem(parent, [item])
                folder.setIcon(0, self.style().standardIcon(self.style().SP_DirIcon))
                self.populateTree(folder, item_path)
        
        # Then add all .set files
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
        # Only enable plot button if a .set file is selected
        if item.text(0).endswith('.set') or item.text(0).endswith('.set *'):
            self.selected_file = item.text(0).replace(" *", "")
            self.plot_btn.setEnabled(True)
            # Store the full path by traversing up the tree
            path_parts = []
            current = item
            while current is not None:
                path_parts.insert(0, current.text(0).replace(" *", ""))
                current = current.parent()
            # Join with the current_dir to get absolute path
            self.selected_file_path = os.path.join(self.current_dir, *path_parts[1:])
            
            # Load epochs to check for run_id
            try:
                self.view_record_btn.setEnabled(True)
                self.current_run_id = self.getRunId(self.selected_file_path)
                self.current_run_record = get_run_record(self.current_run_id)
                print(f"Debug: Current run ID: {self.current_run_id}")
                print(f"Debug: Current run record: {self.current_run_record}")
            except Exception as e:
                print(f"Error checking run_id: {e}")
                self.view_record_btn.setEnabled(False)
        else:
            self.plot_btn.setEnabled(False)
            self.view_record_btn.setEnabled(False)
            
    def viewRunRecord(self):
        if hasattr(self, 'current_run_id'):
            try:
                if self.current_run_record:
                    # Create a new window to display the JSON
                    self.current_run_record_window = QWidget()
                    self.current_run_record_window.setWindowTitle(f"Run Record - {self.current_run_id}")
                    self.current_run_record_window.setGeometry(400, 400, 800, 600)
                    
                    layout = QVBoxLayout()
                    
                    # Create QTreeView and set up model
                    tree_view = QTreeView()
                    model = JsonTreeModel(self.current_run_record)
                    tree_view.setModel(model)
                    tree_view.setAlternatingRowColors(True)
                    tree_view.setHeaderHidden(False)
                    tree_view.expandAll()
                    
                    layout.addWidget(tree_view)
                    
                    self.current_run_record_window.setLayout(layout)
                    self.current_run_record_window.show()
                else:
                    QMessageBox.warning(self, "Warning", "No run record found for this ID")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error retrieving run record: {str(e)}")

    def plotFile(self):
        if hasattr(self, 'selected_file_path'):
            print("Debug: plotFile() triggered.")
            print(f"Debug: Selected file path: {self.selected_file_path}")

            try:
                print("Debug: Attempting to load epochs...")
                epochs = mne.read_epochs_eeglab(self.selected_file_path)
                print("Debug: Epochs successfully loaded.")
                print(f"Debug: Loaded {len(epochs)} epochs.")

                print(f"Debug: Run ID: {epochs.info['description']}")

                print("Debug: Launching MNE QtBrowser for epoch review.")
                epochs.plot(block=True)  # User reviews epochs here
                print("Debug: MNE QtBrowser closed.")

                # Apply changes made in the browser
                print("Debug: Applying drop_bad() to reflect GUI changes.")
                epochs.drop_bad()
                print("Debug: drop_bad() applied. Current number of epochs:", len(epochs))
                # Automatically save the updated epochs with EDIT_ prefix
                run_record = get_run_record(self.current_run_id)
                dir_path = os.path.dirname(self.selected_file_path)
                base_name = os.path.basename(self.selected_file_path)
                new_name = 'EDIT_' + base_name
                save_path = os.path.join(dir_path, new_name)

                print(f"Debug: Saving updated epochs to {save_path}")
                save_epochs_to_set(epochs, autoclean_dict=run_record, output_path=dir_path)
                print("Debug: Save completed.")

                self.modified_files.add(self.selected_file)
                print("Debug: File marked as modified. Reloading file tree.")
                self.loadFiles()

            except Exception as e:
                print(f"Debug: Exception occurred: {e}")
                QMessageBox.critical(self, "Error", f"Error loading/plotting file: {str(e)}")


app = QApplication(sys.argv)
app.setStyleSheet("")  # Use default light theme
window = FileSelector()
window.show()
sys.exit(app.exec_())