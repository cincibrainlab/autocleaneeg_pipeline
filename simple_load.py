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


import mne

epochs = mne.read_epochs_eeglab('/Users/ernie/Downloads/autoclean/rest_eyesopen/stage/_postcleaneeg/0006_rest_postcleaneeg_epo.set')

epochs.plot(block=True)