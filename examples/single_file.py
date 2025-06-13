import asyncio
from pathlib import Path
from autoclean import Pipeline

"""Example of processing a single EEG file."""
# Create pipeline instance
pipeline = Pipeline(verbose='INFO')

# Example file path - modify this to point to your EEG file
file_path = Path("C:/Users/Gam9LG/Documents/DATA/rest_eyesopen/2502_rest.raw")

# Process the file
pipeline.process_file(
    file_path=file_path,
    task="RestingEyesOpen",  # Choose appropriate task
)
