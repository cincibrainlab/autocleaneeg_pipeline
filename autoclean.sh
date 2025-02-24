#!/bin/bash

# =============================================================================
# AutoClean EEG Processing Pipeline Shell Script
# =============================================================================
#
# INSTALLATION:
# 1. Save this file as 'autoclean.sh'
# 2. Make it executable:
#    chmod +x autoclean.sh
# 3. Source it in your shell:
#    source autoclean.sh
#    or add to your ~/.bashrc:
#    echo "source /path/to/autoclean.sh" >> ~/.bashrc
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

autoclean() {
    # Check for help flag
    if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
        autoclean_help
        return 0
    }

    # Check if required arguments are provided
    if [ "$#" -lt 3 ]; then
        echo "Error: Missing required arguments"
        echo "Use 'autoclean --help' for usage information"
        return 1
    }

    # Parse arguments
    data_path=""
    task=""
    config_path=""
    output_path="./output"

    while [[ $# -gt 0 ]]; do
        case $1 in
            -DataPath)
                data_path="$2"
                shift 2
                ;;
            -Task)
                task="$2"
                shift 2
                ;;
            -ConfigPath)
                config_path="$2"
                shift 2
                ;;
            -OutputPath)
                output_path="$2"
                shift 2
                ;;
            *)
                echo "Error: Unknown parameter: $1"
                echo "Use 'autoclean --help' for usage information"
                return 1
                ;;
        esac
    done

    # Validate required parameters
    if [ -z "$data_path" ] || [ -z "$task" ] || [ -z "$config_path" ]; then
        echo "Error: Missing required parameters"
        echo "Use 'autoclean --help' for usage information"
        return 1
    fi

    # Check if paths exist
    if [ ! -d "$data_path" ]; then
        echo "Error: Data path does not exist: $data_path"
        return 1
    fi

    if [ ! -f "$config_path" ]; then
        echo "Error: Config path does not exist: $config_path"
        return 1
    fi

    # Create output directory if it doesn't exist
    mkdir -p "$output_path"

    # Display information
    echo "Using data from: $data_path"
    echo "Using configs from: $config_path"
    echo "Task: $task"
    echo "Output will be written to: $output_path"

    # Get config filename
    config_file=$(basename "$config_path")
    config_dir=$(dirname "$config_path")

    # Set environment variables for docker-compose
    export EEG_DATA_PATH="$data_path"
    export CONFIG_PATH="$config_dir"
    export OUTPUT_PATH="$output_path"

    # Run using docker-compose
    docker-compose run --rm autoclean --task "$task" --data "$data_path" --config "$config_file" --output "$output_path"
}

autoclean_help() {
    cat << 'EOF'
NAME
    autoclean - Automated EEG Processing Pipeline

SYNOPSIS
    autoclean -DataPath <path> -Task <task_name> -ConfigPath <config_path> [-OutputPath <output_path>]
    autoclean --help

DESCRIPTION
    Processes EEG data using a containerized pipeline with specified configurations.

REQUIRED ARGUMENTS
    -DataPath <path>
        Path to the directory containing raw EEG data files.
        Must be an existing directory.

    -Task <task_name>
        Name of the processing task to execute.
        Available tasks: RestingEyesOpen, ASSR, ChirpDefault, HBCD_MMN, MouseXDatResting

    -ConfigPath <config_path>
        Path to the configuration YAML file.
        Must be an existing file.

OPTIONAL ARGUMENTS
    -OutputPath <output_path>
        Directory where processed data will be saved.
        Defaults to "./output"
        Will be created if it doesn't exist.

    --help
        Display this help message and exit.

EXAMPLES
    # Process resting state data with default output location
    autoclean -DataPath "/data/eeg/raw" -Task "RestingEyesOpen" -ConfigPath "/configs/autoclean_config.yaml"

    # Process with custom output location
    autoclean -DataPath "/data/eeg/raw" -Task "ASSR" -ConfigPath "/configs/autoclean_config.yaml" -OutputPath "/results/assr_output"

    # Display help
    autoclean --help
EOF
}

# Make the functions available in the shell
export -f autoclean
export -f autoclean_help 