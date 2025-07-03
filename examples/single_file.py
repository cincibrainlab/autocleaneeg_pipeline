from autoclean import Pipeline

"""Example of processing a single EEG file."""
# Create pipeline instance
pipeline = Pipeline(verbose='INFO')

# Process the file
pipeline.process_file(
    file_path="C:/Users/Gam9LG/Documents/DATA/p300/170104_C7D1BL_P300.set",
    task="p300_grael4k",  # Choose appropriate task
)
