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
    if [[ $path == [A-Za-z]:\\* ]]; then
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
    local bind_mount=false

    debug_log "Parsing command line arguments"
    while [[ $# -gt 0 ]]; do
        debug_log "Processing argument: $1"
        case $1 in
            -DataPath) data_path=$(convert_path "$2"); shift 2 ;;
            -Task) task="$2"; shift 2 ;;
            -ConfigPath) config_path=$(convert_path "$2"); shift 2 ;;
            -OutputPath) output_path=$(convert_path "$2"); shift 2 ;;
            -WorkDir) work_dir=$(convert_path "$2"); shift 2 ;;
            -BindMount) bind_mount=true; shift ;;
            -Debug) DEBUG=true; shift ;;
            *) echo "Error: Unknown parameter: $1"; echo "Use 'autoclean --help' for usage information"; exit 1 ;;
        esac
    done

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

    debug_log "Checking if data_path is a file or directory: $data_path"
    if [ -f "$data_path" ]; then
        echo "DataPath is a file. Mounting parent directory: $(dirname "$data_path")"
        local data_file=$(basename "$data_path")
        export EEG_DATA_PATH=$(dirname "$data_path")
    elif [ -d "$data_path" ]; then
        export EEG_DATA_PATH="$data_path"
    else
        echo "Warning: Data path does not exist: $data_path"
        echo "Continuing with execution..."
        export EEG_DATA_PATH="$data_path"
    fi

    debug_log "Checking if config directory exists: $config_path"
    if [ ! -d "$config_path" ]; then
        echo "Warning: Config directory does not exist: $config_path"
        echo "Continuing with execution..."
    fi

    debug_log "Creating output directory if it doesn't exist: $output_path"
    mkdir -p "$output_path"

    echo "Using data from: $EEG_DATA_PATH"
    echo "Using configs from: $config_path"
    echo "Task: $task"
    echo "Output will be written to: $output_path"
    echo "Working directory: $work_dir"
    if [ "$DEBUG" = true ]; then
        echo "Debug mode: ENABLED"
    fi
    if [ "$bind_mount" = true ]; then
        echo "Bind mount enabled: Mapping configs to container"
    fi

    debug_log "Extracting config filename and directory"
    local config_file=$(basename "$config_path")
    local config_dir=$(dirname "$config_path")

    debug_log "Setting environment variables for docker-compose"
    export CONFIG_PATH="$config_path"
    export OUTPUT_PATH="$output_path"
    debug_log "CONFIG_PATH=$CONFIG_PATH, OUTPUT_PATH=$OUTPUT_PATH"

    debug_log "Changing to working directory: $work_dir"
    cd "$work_dir" || echo "Warning: Failed to change to working directory: $work_dir"

    # Prepare docker-compose command with optional bind mount
    echo "Starting docker-compose..."
    local compose_cmd="docker-compose run --no-deps"
    if [ "$bind_mount" = true ]; then
        local host_config_dir="/mnt/srv2/robots/aud_assr2/configs"  # Corrected host path
        debug_log "Using host config directory: $host_config_dir"
        ls -la "$config_path" >&2 || echo "Warning: Cannot access config path: $config_path"
        ls -la "$host_config_dir" >&2 || echo "Warning: Host config directory not accessible"
        compose_cmd="$compose_cmd -v $host_config_dir:/app/configs"
        echo "Bind mount command: $compose_cmd"
    fi

    if [ -n "$data_file" ]; then
        echo "Processing single file: $data_file"
        debug_log "Running docker-compose command: $compose_cmd autoclean --task \"$task\" --data \"$data_file\" --config \"/app/configs/autoclean_config.yaml\" --output \"$output_path\""
        $compose_cmd autoclean --task "$task" --data "$data_file" --config "/app/configs/autoclean_config.yaml" --output "$output_path" &
        local pid=$!
        local container_id=$(docker ps -lq)  # Get container ID right after start
        debug_log "Container ID: $container_id"
        wait $pid
        local exit_code=$?
        debug_log "Docker-compose exit code: $exit_code"
        if [ "$bind_mount" = true ]; then
            debug_log "Checking host config directory after run: $host_config_dir"
            ls -la "$host_config_dir" >&2 || echo "Warning: Cannot access host config directory after run"
        fi
        if docker ps -q | grep -q "$container_id"; then
            debug_log "Inspecting container mounts"
            docker inspect "$container_id" --format '{{ .Mounts }}' >&2 || echo "Warning: Failed to inspect container mounts"
        else
            debug_log "Container $container_id is not running, skipping mount inspection"
        fi
        docker-compose run --rm autoclean ls -la /app/configs >&2 || echo "Warning: Failed to list /app/configs"
        exit $exit_code
    else
        echo "Processing all files in directory: $data_path"
        debug_log "Running docker-compose command: $compose_cmd autoclean --task \"$task\" --data \"$data_path\" --config \"/app/configs/autoclean_config.yaml\" --output \"$output_path\""
        $compose_cmd autoclean --task "$task" --data "$data_path" --config "/app/configs/autoclean_config.yaml" --output "$output_path"
        local exit_code=$?
        debug_log "docker-compose exit code: $exit_code"
        exit $exit_code
    fi
}

debug_log "Script started with arguments: $@"
main "$@"