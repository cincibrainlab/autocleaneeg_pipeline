from autoclean import Pipeline

"""Example of processing a single EEG file."""
# Create pipeline instance
pipeline = Pipeline(verbose='INFO')

# Process the file
pipeline.process_file(
    file_path="C:/Users/Gam9LG/Documents/DATA/rest_eyesopen/2502_rest.raw",
    task="RestingEyesOpen",  # Choose appropriate task
)
