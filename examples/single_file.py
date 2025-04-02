from pathlib import Path
from autoclean import Pipeline
import mne

# Define paths - modify these to match your system
EXAMPLE_OUTPUT_DIR = Path("C:/Users/Gam9LG/Documents/Autoclean/VEP")  # Where processed data will be stored
CONFIG_FILE = Path("configs/autoclean_config_hbcd.yaml")  # Path to config relative to working directory OR absolute path

"""Example of processing a single EEG file."""
# Create pipeline instance
pipeline = Pipeline(
    autoclean_dir=EXAMPLE_OUTPUT_DIR,
    autoclean_config=CONFIG_FILE,
    verbose='INFO' # Set to 'DEBUG' for more detailed logging
)

# Example file path - modify this to point to your EEG file
file_path = Path("C:/Users/Gam9LG/Documents/HBCD_exampleFiles/CHCCH0014_V04/CHCCH0014_256983_V04_VEP.mff")


# Process the file
pipeline.process_file(
    file_path=file_path,
    task="HBCD_VEP",  # Choose appropriate task
)