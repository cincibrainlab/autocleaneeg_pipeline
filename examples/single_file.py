from pathlib import Path
from autoclean import Pipeline

# Define paths - modify these to match your system
EXAMPLE_OUTPUT_DIR = Path("C:/Users/Gam9LG/Documents/AutocleanTesting")  # Where processed data will be stored

"""Example of processing a single EEG file."""
# Create pipeline instance
pipeline = Pipeline(autoclean_dir=EXAMPLE_OUTPUT_DIR)

pipeline.add_task("examples/resting_eyes_open.py")

# Example file path - modify this to point to your EEG file
file_path = Path("C:/Users/Gam9LG/Documents/DATA/rest_eyesopen/0418_rest.raw")

# Process the file
pipeline.process_file(
    file_path=file_path,
    task="RestingEyesOpen",  # Choose appropriate task
)
