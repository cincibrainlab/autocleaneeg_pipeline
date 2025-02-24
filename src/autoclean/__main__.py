"""Main entry point for the autoclean package."""
import sys
from pathlib import Path
from autoclean.core.pipeline import Pipeline

# Fixed container paths
DATA_DIR = "/data"
CONFIG_DIR = "/app/configs"
OUTPUT_DIR = "/app/output"

def main():
    """Main entry point for the autoclean package."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AutoClean EEG Processing Pipeline')
    parser.add_argument('--task', type=str, required=True,
                      help='Task to run (e.g., RestingEyesOpen)')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to data file or directory')
    parser.add_argument('--config', type=str, default='/app/configs/autoclean_config.yaml',
                      help='Path to config file')
    parser.add_argument('--output', type=str, default='/app/output',
                      help='Output directory')
    
    args = parser.parse_args()
    
    print(f"Starting AutoClean Pipeline...")
    print(f"Task: {args.task}")
    print(f"Using data from: {DATA_DIR}")
    print(f"Using config from: {CONFIG_DIR}")
    print(f"Output will be written to: {OUTPUT_DIR}")
    
    # Initialize pipeline with fixed paths
    pipeline = Pipeline(
        autoclean_dir=OUTPUT_DIR,
        autoclean_config=f"{CONFIG_DIR}/autoclean_config.yaml"
    )
    
    # Check if input is file or directory
    input_path = Path(DATA_DIR)
    if input_path.is_file():
        print(f"Processing single file: {input_path}")
        pipeline.process_file(
            file_path=str(input_path),
            task=args.task
        )
    else:
        print(f"Processing all files in directory: {input_path}")
        pipeline.process_directory(
            directory=str(input_path),
            task=args.task
        )

if __name__ == '__main__':
    sys.exit(main()) 