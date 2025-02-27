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
    
    print(f"Starting AutoClean Pipeline Container...")
    print(f"Task: {args.task}")
    print(f"Using data from: {DATA_DIR}")
    print(f"Using config from: {CONFIG_DIR}")
    print(f"Output will be written to: {OUTPUT_DIR}")
    
    # Initialize pipeline with fixed paths
    pipeline = Pipeline(
        autoclean_dir=OUTPUT_DIR,
        autoclean_config=f"{CONFIG_DIR}/autoclean_config.yaml"
    )

    # Print all arguments for debugging
    print("\nArguments:")
    print(f"  --task: {args.task}")
    print(f"  --data: {args.data}")
    print(f"  --config: {args.config}")
    print(f"  --output: {args.output}")
    print()
    
    # Check if input is file or directory
    input_path = Path(args.data)
    print(f"Input path: {input_path}")
    
    # Always look in DATA_DIR since that's where docker mounts the data
    full_path = Path(DATA_DIR) / input_path.name
    print(f"Full path: {full_path}")

    # list files in DATA_DIR
    print(f"Files in {DATA_DIR}:")
    for file in Path(DATA_DIR).glob('*'):
        print(file)
    
    if full_path.is_file():  # Just check if it's a file, no extension validation
        print(f"Processing single file: {full_path}")
        if not full_path.exists():
            print(f"Error: File not found in mounted directory: {full_path}")
            return 1
        pipeline.process_file(
            file_path=str(full_path),
            task=args.task
        )
    else:
        print(f"Processing all files in directory: {DATA_DIR}")
        pipeline.process_directory(
            directory=DATA_DIR,
            task=args.task
        )

if __name__ == '__main__':
    sys.exit(main()) 