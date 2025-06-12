import asyncio
from pathlib import Path

from autoclean import Pipeline

EXAMPLE_OUTPUT_DIR = Path("path/to/output/directory")  # Where processed data will be stored

async def batch_run():
    """Example of batch processing multiple EEG files asynchronously."""
    # Create pipeline instance
    pipeline = Pipeline(
        output_dir=EXAMPLE_OUTPUT_DIR,
        verbose='HEADER'
    )
    # Example INPUT directory path - modify this to point to your EEG files
    directory = Path("path/to/input/directory")

    # Process all files in directory
    await pipeline.process_directory_async(
        directory=directory,
        task="RestingEyesOpen",  # Choose appropriate task
        sub_directories=False, # Optional: process files in subfolders
        pattern="*.set", # Optional: specify a pattern to filter files (use "*.extention" for all files of that extension)
        max_concurrent=3 # Optional: specify the maximum number of concurrent files to process
    )

def single_file_run():
    pipeline = Pipeline(
        output_dir=EXAMPLE_OUTPUT_DIR,
        verbose='HEADER'
    )
    file_path = Path("path/to/input/file")

    pipeline.process_file(
        file_path=file_path,
        task="RestingEyesOpen",  # Choose appropriate task
    )
    
if __name__ == "__main__":
    #Batch run example
    asyncio.run(batch_run())

    #Single file run example
    single_file_run()

