from pathlib import Path
from autoclean import Pipeline

# Define paths - modify these to match your system
EXAMPLE_OUTPUT_DIR = Path("C:/Users/Gam9LG/Documents/AutocleanTesting")  # Where processed data will be stored

"""Example of processing a single EEG file."""
# Create pipeline instance
pipeline = Pipeline(output_dir=EXAMPLE_OUTPUT_DIR, verbose='INFO')

# Example file path - modify this to point to your EEG file
directory_path = Path("C:/Users/Gam9LG/Documents/DATA/rest_eyesopen")

# Process the file
pipeline.process_directoCanry_async(
    directory_path=directory_path,
    task="RestingEyesOpen",  # Choose appropriate task
)
