from autoclean import Pipeline

# Input and output paths
INPUT_DIR = "/Users/sueo8x/Documents/TestEegData"
OUTPUT_DIR = "/Users/sueo8x/Documents/TestOutput"

# Initialize pipeline
pipeline = Pipeline(
    autoclean_dir=OUTPUT_DIR,
    autoclean_config="configs/autoclean_config.yaml"
)

# Process a single file
pipeline.process_file(
    file_path=f"{INPUT_DIR}/resting_eyes_open.raw",
    task="RestingEyesOpen"
)


