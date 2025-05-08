from pathlib import Path
from autoclean import Pipeline

# Define paths - modify these to match your system
EXAMPLE_OUTPUT_DIR = Path("C:/Users/Gam9LG/Documents/AutocleanDev/")  # Where processed data will be stored
CONFIG_FILE = Path("C:/Users/Gam9LG/Downloads/autoclean_files/config.yaml")  # Path to config relative to working directory OR absolute path

"""Example of processing a single EEG file."""
# Create pipeline instance
pipeline = Pipeline(
    autoclean_dir=EXAMPLE_OUTPUT_DIR,
    autoclean_config=CONFIG_FILE,
    verbose='INFO' # Set to 'DEBUG' for more detailed logging
)

# Example file path - modify this to point to your EEG file
# file_path = Path("C:/Users/Gam9LG/Documents/DATA/hbcd_mmn/sub-896714_ses-V03_task-MMN_acq-eeg_eeg.set")
# file_path = Path("C:/Users/Gam9LG/Documents/HBCD_exampleFiles/CHCCH0014_V04/CHCCH0014_256983_V04_MMN.mff")
# file_path = Path("C:/Users/Gam9LG/Documents/HBCD_exampleFiles/CHCCH0014_V04/CHCCH0014_256983_V04_VEP.mff")
file_path = Path("C:/Users/Gam9LG/Documents/DATA/n141_resting/raw/0199_rest.raw")


# Process the file
pipeline.process_file(
    file_path=file_path,
    task="TestingRest",  # Choose appropriate task
)