from autoclean import Pipeline
"""Example of processing a single EEG file."""

# raw = read_raw_eeglab("C:/Users/Gam9LG/Documents/Autoclean-EEG/output/Test_Merge_07-22-2025/bids/derivatives/autoclean-v2.1.0/intermediate/04_trim/0199_rest_trim_raw.set", preload=True)

# Create pipeline instance
pipeline = Pipeline(verbose='INFO')

# Process the file
pipeline.process_file(
    file_path="C:/Users/Gam9LG/Documents/DATA/stat_learning/1037_slstructured.raw",
    task="Statistical_Learning",  # Choose appropriate task
)
