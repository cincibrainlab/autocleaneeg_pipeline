from pathlib import Path

from autoclean import Pipeline

# Define paths - modify these to match your system
EXAMPLE_OUTPUT_DIR = Path("/mnt/srv2/robots/aud_assr40/test")  # Where processed data will be stored
CONFIG_FILE = Path("/mnt/srv2/robots/aud_assr40/configs/autoclean_config.yaml")  # Path to config relative to working directory OR absolute path

"""Example of processing a single EEG file."""
# Create pipeline instance
pipeline = Pipeline(
    autoclean_dir=EXAMPLE_OUTPUT_DIR,
    autoclean_config=CONFIG_FILE,
    verbose='INFO' # Set to 'DEBUG' for more detailed logging
)

# Example file path - modify this to point to your EEG file
file_path = Path("/mnt/srv2/robots/aud_assr40/input/allego_8__uid1205-17-34-38_data.set")

# Process the file
pipeline.process_file(
    file_path=file_path,
    task="MouseXdatAssr",  # Choose appropriate task
)