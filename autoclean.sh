#!/bin/bash

# =============================================================================
# AutoClean EEG Processing Pipeline Shell Script
# =============================================================================
#
# INSTALLATION:
# 1. Save this file as 'autoclean' in /usr/local/bin
#    sudo cp autoclean.sh /usr/local/bin/autoclean
# 2. Make it executable:
#    chmod +x /usr/local/bin/autoclean
#
# QUICK START:
# 1. Basic usage:
#    autoclean -DataPath "/path/to/data" -Task "RestingEyesOpen" -ConfigPath "/path/to/config.yaml"
#
# 2. View help:
#    autoclean --help
#
# REQUIREMENTS:
# - Docker and docker-compose must be installed and running
# - Appropriate permissions to execute Docker commands
# =============================================================================

# Debug mode flag
DEBUG=false

# Function to print debug information
debug_log() {
    if [ "$DEBUG" = true ]; then
        echo "DEBUG: $1" >&2
    fi
}

# Convert Windows paths to WSL paths
convert_path() {
    local path=$1
    debug_log "Converting path: $path"
    # Check if it's a Windows-style path
    if [[ $path == [A-Za-z]:\\* ]]; then
        # Convert Windows path to WSL path
        drive=${path:0:1}
        path=${path:2}
        local converted_path="/mnt/${drive,,}${path//\\//}"
        debug_log "Converted Windows path to WSL path: $converted_path"
        echo "$converted_path"
    else
        debug_log "Path appears to be Unix-style, no conversion needed"
        echo "$path"
    fi
}

show_help() {
    cat << 'EOF'
Starts containerized autoclean pipeline.
Usage: autoclean -DataPath <path> -Task <task> -ConfigPath <config> [-OutputPath <path>] [-WorkDir <path>] [-Debug]

Required:
  -DataPath <path>    Directory containing raw EEG data or file path to single data file
  -Task <task>        Task type (Defined in src/autoclean/tasks)
  -ConfigPath <path>  Path to configuration YAML file

Optional:
  -OutputPath <path>  Output directory (default: ./output)
  -WorkDir <path>     Working directory for the autoclean pipeline (default: current directory)
  -Debug              Enable verbose debug output
  --help              Show this help message

Example:
  autoclean -DataPath "/data/raw" -Task "RestingEyesOpen" -ConfigPath "/configs/autoclean_config.yaml"
EOF
}

main() {
    debug_log "Starting autoclean with arguments: $@"
    
    # Check for help flag
    if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
        debug_log "Help flag detected, showing help message"
        show_help
        exit 0
    fi

    # Check if required arguments are provided
    if [ "$#" -lt 3 ]; then
        echo "Error: Missing required arguments"
        echo "Use 'autoclean --help' for usage information"
        exit 1
    fi

    # Parse arguments
    local data_path=""
    local task=""
    local config_path=""
    local output_path="./output"
    local work_dir=$(pwd)  # Default to current directory

    debug_log "Parsing command line arguments"
    while [[ $# -gt 0 ]]; do
        debug_log "Processing argument: $1"
        case $1 in
            -DataPath)
                debug_log "Found -DataPath argument with value: $2"
                data_path=$(convert_path "$2")
                debug_log "Converted data_path: $data_path"
                shift 2
                ;;
            -Task)
                debug_log "Found -Task argument with value: $2"
                task="$2"
                shift 2
                ;;
            -ConfigPath)
                debug_log "Found -ConfigPath argument with value: $2"
                config_path=$(convert_path "$2")
                debug_log "Converted config_path: $config_path"
                shift 2
                ;;
            -OutputPath)
                debug_log "Found -OutputPath argument with value: $2"
                output_path=$(convert_path "$2")
                debug_log "Converted output_path: $output_path"
                shift 2
                ;;
            -WorkDir)
                debug_log "Found -WorkDir argument with value: $2"
                work_dir=$(convert_path "$2")
                debug_log "Converted work_dir: $work_dir"
                shift 2
                ;;
            -Debug)
                debug_log "Debug mode enabled"
                DEBUG=true
                shift
                ;;
            *)
                echo "Error: Unknown parameter: $1"
                echo "Use 'autoclean --help' for usage information"
                exit 1
                ;;
        esac
    done

    # Validate required parameters
    debug_log "Validating required parameters"
    if [ -z "$data_path" ] || [ -z "$task" ] || [ -z "$config_path" ]; then
        echo "Error: Missing required parameters"
        debug_log "Missing parameters: data_path=$data_path, task=$task, config_path=$config_path"
        echo "Use 'autoclean --help' for usage information"
        exit 1
    fi

    # Validate work directory
    debug_log "Validating work directory: $work_dir"
    if [ ! -d "$work_dir" ]; then
        echo "Error: Working directory does not exist: $work_dir"
        exit 1
    fi

    debug_log "Checking if data_path is a file or directory: $data_path"
    if [ -f "$data_path" ]; then
        # If data_path is a file, use its parent directory for mounting
        debug_log "data_path is a file"
        echo "DataPath is a file. Mounting parent directory: $(dirname "$data_path")"
        local data_file=$(basename "$data_path")
        debug_log "data_file basename: $data_file"
        export EEG_DATA_PATH=$(dirname "$data_path")
        debug_log "EEG_DATA_PATH set to: $EEG_DATA_PATH"
    elif [ -d "$data_path" ]; then
        # If data_path is a directory, use it directly
        debug_log "data_path is a directory"
        export EEG_DATA_PATH="$data_path"
        debug_log "EEG_DATA_PATH set to: $EEG_DATA_PATH"
    else
        echo "Error: Data path does not exist: $data_path"
        debug_log "Data path validation failed"
        exit 1
    fi

    debug_log "Checking if config directory exists: $config_path"
    if [ ! -d "$config_path" ]; then
        echo "Error: Config directory does not exist: $config_path"
        debug_log "Config directory validation failed"
        exit 1
    fi

    # Create output directory if it doesn't exist
    debug_log "Creating output directory if it doesn't exist: $output_path"
    mkdir -p "$output_path"
    debug_log "mkdir exit code: $?"

    # Display information
    echo "Using data from: $EEG_DATA_PATH"
    echo "Using configs from: $config_path"
    echo "Task: $task"
    echo "Output will be written to: $output_path"
    echo "Working directory: $work_dir"
    if [ "$DEBUG" = true ]; then
        echo "Debug mode: ENABLED"
    fi

    # Get config filename
    debug_log "Extracting config filename and directory"
    local config_file=$(basename "$config_path")
    local config_dir=$(dirname "$config_path")
    debug_log "config_file: $config_file, config_dir: $config_dir"

    # Set remaining environment variables for docker-compose
    debug_log "Setting environment variables for docker-compose"
    export CONFIG_PATH="$config_dir"
    export OUTPUT_PATH="$output_path"
    debug_log "CONFIG_PATH=$CONFIG_PATH, OUTPUT_PATH=$OUTPUT_PATH"

    # Change to the working directory
    debug_log "Changing to working directory: $work_dir"
    cd "$work_dir"
    debug_log "Current directory after cd: $(pwd)"

    # Run using docker-compose
    echo "Starting docker-compose..."
    if [ -n "$data_file" ]; then
        # For single file - pass just the filename since parent dir is mounted as /data
        debug_log "Processing single file mode with data_file: $data_file"
        echo "Processing single file: $data_file"
        debug_log "Running docker-compose command: docker-compose run autoclean --task \"$task\" --data \"$data_file\" --config \"$config_file\" --output \"$output_path\""
        docker-compose run autoclean --task "$task" --data "$data_file" --config "$config_file" --output "$output_path"
        debug_log "docker-compose exit code: $?"
    else
        # For directory
        debug_log "Processing directory mode with data_path: $data_path"
        echo "Processing all files in directory: $data_path"
        debug_log "Running docker-compose command: docker-compose run autoclean --task \"$task\" --data \"$data_path\" --config \"$config_file\" --output \"$output_path\""
        docker-compose run autoclean --task "$task" --data "$data_path" --config "$config_file" --output "$output_path"
        debug_log "docker-compose exit code: $?"
    fi
    
    debug_log "Autoclean processing completed"
}

# Execute main function with all arguments
debug_log "Script started with arguments: $@"
main "$@" 