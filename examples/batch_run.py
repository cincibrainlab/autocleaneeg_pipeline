import asyncio
from pathlib import Path

from autoclean import Pipeline

async def main():
    # Define paths - modify these to match your system
    EXAMPLE_OUTPUT_DIR = Path("C:/Users/Gam9LG/Documents/AutocleanDev2/")  # Where processed data will be stored
    CONFIG_FILE = Path("C:/Users/Gam9LG/Downloads/autoclean_files/config.yaml")  # Path to config relative to working directory OR absolute path

    """Example of batch processing multiple EEG files asynchronously."""
    # Create pipeline instance
    pipeline = Pipeline(
        autoclean_dir=EXAMPLE_OUTPUT_DIR,
        autoclean_config=CONFIG_FILE,
        verbose='HEADER'
    )

    # Example INPUT directory path - modify this to point to your EEG files
    directory = Path("C:/Users/Gam9LG/Documents/DATA/ICA_Files")

    # Process all files in directory
    await pipeline.process_directory_async(
        directory=directory,
        task="TestingRest",  # Choose appropriate task
        sub_directories=False, # Optional: process files in subdirectories
        pattern="*.set", # Optional: specify a pattern to filter files (use "*.extention" for all files of that extension)
        max_concurrent=2 # Optional: specify the maximum number of concurrent processes
    )

if __name__ == "__main__":
    asyncio.run(main())

