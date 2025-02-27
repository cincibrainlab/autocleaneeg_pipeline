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

# Convert Windows paths to Unix-compatible paths
convert_path() {
    local path=$1
    debug_log "Converting path: $path"
    # Check if it's a Windows-style path
    if [[ $path == *\\* ]]; then
        # Replace backslashes with forward slashes
        local converted_path="${path//\\//}"
        debug_log "Converted Windows path to Unix path: $converted_path"
        echo "$converted_path"
    else
        debug_log "Path appears to be Unix-style, no conversion needed"
        echo "$path"
    fi
}

show_help() {
    cat << 'EOF'
Starts containerized autoclean pipeline.
Usage: autoclean -DataPath <path> -Task <task> -ConfigPath <config> [-OutputPath <path>] [-WorkDir <path>] [-BindMount] [-Debug]

Required:
  -DataPath <path>    Directory containing raw EEG data or file path to single data file
  -Task <task>        Task type (Defined in src/autoclean/tasks)
  -ConfigPath <path>  Path to configuration YAML file

Optional:
  -OutputPath <path>  Output directory (default: ./output)
  -WorkDir <path>     Working directory for the autoclean pipeline (default: current directory)
  -BindMount          Enable bind mount of configs to container
  -Debug              Enable verbose debug output
  --help              Show this help message

Example:
  autoclean -DataPath "/data/raw" -Task "RestingEyesOpen" -ConfigPath "/configs/autoclean_config.yaml" -BindMount
EOF
}

main() {
    debug_log "Starting autoclean with arguments: $@"
    
    if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
        debug_log "Help flag detected, showing help message"
        show_help
        exit 0
    fi

    if [ "$#" -lt 3 ]; then
        echo "Error: Missing required arguments"
        echo "Use 'autoclean --help' for usage information"
        exit 1
    fi

    local data_path=""
    local task=""
    local config_path=""
    local output_path="./output"
    local work_dir=$(pwd)

    debug_log "Parsing command line arguments"
    while [[ $# -gt 0 ]]; do
        debug_log "Processing argument: $1"
        case $1 in
            -DataPath) data_path=$(convert_path "$2"); shift 2 ;;
            -Task) task="$2"; shift 2 ;;
            -ConfigPath) config_path=$(convert_path "$2"); shift 2 ;;
            -OutputPath) output_path=$(convert_path "$2"); shift 2 ;;
            -WorkDir) work_dir=$(convert_path "$2"); shift 2 ;;
            -Debug) DEBUG=true; shift ;;
            *) echo "Error: Unknown parameter: $1"; echo "Use 'autoclean --help' for usage information"; exit 1 ;;
        esac
    done
    
    debug_log "AUTOCLEAN SCRIPT ARGS"
    debug_log "data_path: $data_path"
    debug_log "task: $task"
    debug_log "config_path: $config_path"
    debug_log "output_path: $output_path"
    debug_log "work_dir: $work_dir"
    debug_log "DEBUG: $DEBUG"

    debug_log "Validating required parameters"
    if [ -z "$data_path" ] || [ -z "$task" ] || [ -z "$config_path" ]; then
        echo "Error: Missing required parameters"
        debug_log "Missing parameters: data_path=$data_path, task=$task, config_path=$config_path"
        echo "Use 'autoclean --help' for usage information"
        exit 1
    fi

    debug_log "Validating work directory: $work_dir"
    if [ ! -d "$work_dir" ]; then
        echo "Warning: Working directory does not exist: $work_dir"
        echo "Continuing with execution..."
    fi

    # Check if we should skip path validation (for Docker-in-Docker operations)
    if [ -n "$AUTOCLEAN_SKIP_PATH_VALIDATION" ]; then
        debug_log "AUTOCLEAN_SKIP_PATH_VALIDATION is set, skipping path validation for host paths"
        
        # For file paths, extract the filename for later use
        if [[ "$data_path" == */* ]]; then
            local data_file=$(basename "$data_path")
            debug_log "Extracted filename from path: $data_file"
        fi
        
        # Set environment variables directly without validation
        export EEG_DATA_PATH=$(dirname "$data_path")
        debug_log "EEG_DATA_PATH set to: $EEG_DATA_PATH (without validation)"
        
        # Create output directory if it doesn't exist
        debug_log "Creating output directory if it doesn't exist: $output_path"
        mkdir -p "$output_path"
        debug_log "mkdir exit code: $?"
    else
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
    fi

    echo "Using data from: $EEG_DATA_PATH"
    echo "Using configs from: $config_path"
    echo "Task: $task"
    echo "Output will be written to: $output_path"
    echo "Working directory: $work_dir"
    if [ "$DEBUG" = true ]; then
        echo "Debug mode: ENABLED"
    fi

    debug_log "Extracting config filename and directory"
    local config_file=$(basename "$config_path")
    local config_dir=$(dirname "$config_path")

    debug_log "Setting environment variables for docker-compose"
    export CONFIG_PATH="$config_path"
    export OUTPUT_PATH="$output_path"
    debug_log "CONFIG_PATH=$CONFIG_PATH, OUTPUT_PATH=$OUTPUT_PATH"

    debug_log "Changing to working directory: $work_dir"
    cd "$work_dir"
    debug_log "Current directory after cd: $(pwd)"
    
    # Check if docker-compose.yml exists in the current directory
    if [ -f "docker-compose.yml" ]; then
        echo "Found docker-compose.yml in current directory"
    else
        echo "ERROR: docker-compose.yml not found in current directory: $(pwd)"
        echo "Directory contents:"
        ls -la
    fi
    
    # Check Docker status
    echo "Checking Docker status:"
    if docker info > /dev/null 2>&1; then
        echo "Docker is running"
    else
        echo "WARNING: Docker may not be running properly"
    fi
    
    # Print environment variables for debugging
    echo "Environment variables for docker-compose:"
    echo "EEG_DATA_PATH=$EEG_DATA_PATH"
    echo "CONFIG_PATH=$CONFIG_PATH"
    echo "OUTPUT_PATH=$OUTPUT_PATH"
    


    if [ -n "$data_file" ]; then
        echo "Processing single file: $data_file"
        debug_log "Running docker-compose command: docker-compose run --rm autoclean --task \"$task\" --data \"$data_file\" --config \"$config_file\" --output \"$output_path\""
        
        # Run with verbose output
        docker-compose run --rm autoclean --task "$task" --data "$data_file" --config "$config_file" --output "$output_path"
        DOCKER_EXIT_CODE=$?
        debug_log "docker-compose exit code: $DOCKER_EXIT_CODE"
        
        if [ $DOCKER_EXIT_CODE -ne 0 ]; then
            echo "ERROR: docker-compose command failed with exit code $DOCKER_EXIT_CODE"
            echo "Checking for running containers:"
            docker ps
            echo "Checking for stopped containers:"
            docker ps -a
        fi
    else
        # For directory
        echo "Processing all files in directory: $data_path"       
        debug_log "Running command: docker-compose run autoclean --task \"$task\" --data \"$data_path\" --config \"$config_file\" --output \"$output_path\""
 
        # Run with verbose output
        docker-compose run --rm autoclean --task "$task" --data "$data_path" --config "$config_file" --output "$output_path"
        DOCKER_EXIT_CODE=$?
        debug_log "docker-compose exit code: $DOCKER_EXIT_CODE"
        
        if [ $DOCKER_EXIT_CODE -ne 0 ]; then
            echo "ERROR: docker-compose command failed with exit code $DOCKER_EXIT_CODE"
            echo "Checking for running containers:"
            docker ps
            echo "Checking for stopped containers:"
            docker ps -a
        fi
    fi
    
    debug_log "Autoclean processing completed"
}

# Execute main function with all arguments
debug_log "Script started with arguments: $@"
main "$@"