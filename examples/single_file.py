from pathlib import Path

from autoclean import Pipeline

# Define paths - modify these to match your system
EXAMPLE_OUTPUT_DIR = Path("Path to desired output directory")  # Where processed data will be stored
CONFIG_FILE = Path("configs/autoclean_config.yaml")  # Path to config relative to working directory OR absolute path

"""Example of processing a single EEG file."""
# Create pipeline instance
pipeline = Pipeline(
    autoclean_dir=EXAMPLE_OUTPUT_DIR,
    autoclean_config=CONFIG_FILE,
    verbose='INFO' # Set to 'DEBUG' for more detailed logging
)

# Example file path - modify this to point to your EEG file
file_path = Path("C:/Users/example/Documents/DATA/n141_resting/raw/0079_rest.raw")

# Process the file
pipeline.process_file(
    file_path=file_path,
    task="RestingEyesOpen",  # Choose appropriate task
)