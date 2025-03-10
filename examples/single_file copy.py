from pathlib import Path
from autoclean import Pipeline
import mne

# Define paths - modify these to match your system
EXAMPLE_OUTPUT_DIR = Path("/Users/ernie/Data/autoclean_v3")  # Where processed data will be stored
CONFIG_FILE = Path("/Users/ernie/Documents/GitHub/autoclean_pipeline/configs/autoclean_config_rest_4k.yaml")  # Path to config relative to working directory OR absolute path

"""Example of processing a single EEG file."""
# Create pipeline instance
pipeline = Pipeline(
    autoclean_dir=EXAMPLE_OUTPUT_DIR,
    autoclean_config=CONFIG_FILE,
    verbose='INFO' # Set to 'DEBUG' for more detailed logging
)

# Example file path - modify this to point to your EEG file
file_path = Path("/Users/ernie/Data/results/spg302_qeeg/140101_C1D1BL_EO.set")


# Process the file
pipeline.process_file(
    file_path=file_path,
    task="rest_eyesopen_grael4k",  # Choose appropriate task
)