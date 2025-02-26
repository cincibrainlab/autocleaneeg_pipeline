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

# Convert Windows paths to WSL paths
convert_path() {
    local path=$1
    # Check if it's a Windows-style path
    if [[ $path == [A-Za-z]:\\* ]]; then
        # Convert Windows path to WSL path
        drive=${path:0:1}
        path=${path:2}
        echo "/mnt/${drive,,}${path//\\//}"
    else
        echo "$path"
    fi
}

show_help() {
    cat << 'EOF'
Starts containerized autoclean pipeline.
Usage: autoclean -DataPath <path> -Task <task> -ConfigPath <config> [-OutputPath <path>]

Required:
  -DataPath <path>    Directory containing raw EEG data or file path to single data file
  -Task <task>        Task type (Defined in src/autoclean/tasks)
  -ConfigPath <path>  Path to configuration YAML file

Optional:
  -OutputPath <path>  Output directory (default: ./output)
  --help             Show this help message

Example:
  autoclean -DataPath "/data/raw" -Task "RestingEyesOpen" -ConfigPath "/configs/autoclean_config.yaml"
EOF
}

main() {
    # Check for help flag
    if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
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

    while [[ $# -gt 0 ]]; do
        case $1 in
            -DataPath)
                data_path=$(convert_path "$2")
                shift 2
                ;;
            -Task)
                task="$2"
                shift 2
                ;;
            -ConfigPath)
                config_path=$(convert_path "$2")
                shift 2
                ;;
            -OutputPath)
                output_path=$(convert_path "$2")
                shift 2
                ;;
            *)
                echo "Error: Unknown parameter: $1"
                echo "Use 'autoclean --help' for usage information"
                exit 1
                ;;
        esac
    done

    # Validate required parameters
    if [ -z "$data_path" ] || [ -z "$task" ] || [ -z "$config_path" ]; then
        echo "Error: Missing required parameters"
        echo "Use 'autoclean --help' for usage information"
        exit 1
    fi

    if [ -f "$data_path" ]; then
        # If data_path is a file, use its parent directory for mounting
        echo "DataPath is a file. Mounting parent directory: $(dirname "$data_path")"
        local data_file=$(basename "$data_path")
        export EEG_DATA_PATH=$(dirname "$data_path")
    elif [ -d "$data_path" ]; then
        # If data_path is a directory, use it directly
        export EEG_DATA_PATH="$data_path"
    else
        echo "Error: Data path does not exist: $data_path"
        exit 1
    fi


    if [ ! -f "$config_path" ]; then
        echo "Error: Config path does not exist: $config_path"
        exit 1
    fi

    # Create output directory if it doesn't exist
    mkdir -p "$output_path"

    # Display information
    echo "Using data from: $EEG_DATA_PATH"
    echo "Using configs from: $config_path"
    echo "Task: $task"
    echo "Output will be written to: $output_path"

    # Get config filename
    local config_file=$(basename "$config_path")
    local config_dir=$(dirname "$config_path")

    # Set remaining environment variables for docker-compose
    export CONFIG_PATH="$config_dir"
    export OUTPUT_PATH="$output_path"

    # Run using docker-compose
    echo "Starting docker-compose..."
    if [ -n "$data_file" ]; then
        # For single file - pass just the filename since parent dir is mounted as /data
        echo "Processing single file: $data_file"
        docker-compose run autoclean --task "$task" --data "$data_file" --config "$config_file" --output "$output_path"
    else
        # For directory
        echo "Processing all files in directory: $data_path"
        docker-compose run autoclean --task "$task" --data "$data_path" --config "$config_file" --output "$output_path"
    fi
}

# Execute main function with all arguments
main "$@" 