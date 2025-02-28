import asyncio
from pathlib import Path

from autoclean import Pipeline

async def main():
    # Define paths - modify these to match your system
    EXAMPLE_OUTPUT_DIR = Path("/srv2/RAWDATA/Autoclean_Example_Chirp/output")  # Where processed data will be stored
    CONFIG_FILE = Path("configs/autoclean_config.yaml")  # Path to config relative to working directory OR absolute path

    """Example of batch processing multiple EEG files asynchronously."""
    # Create pipeline instance
    pipeline = Pipeline(
        autoclean_dir=EXAMPLE_OUTPUT_DIR,
        autoclean_config=CONFIG_FILE,
        verbose='INFO'
    )

    # Example INPUT directory path - modify this to point to your EEG files
    directory = Path("/srv2/RAWDATA/Autoclean_Example_Chirp/input")

    # Process all files in directory
    await pipeline.process_directory_async(
        directory=directory,
        task="ChirpDefault",  # Choose appropriate task
        sub_directories=False, # Optional: process files in subdirectories
        pattern="*.raw", # Optional: specify a pattern to filter files (use "*.extention" for all files of that extension)
        max_concurrent=3 # Optional: specify the maximum number of concurrent processes
    )

if __name__ == "__main__":
    asyncio.run(main())

