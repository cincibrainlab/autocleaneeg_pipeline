#!/bin/bash

# Help function
show_help() {
    echo "Usage: autoclean -t TASK -d DATA_PATH [-c CONFIG_PATH]"
    echo
    echo "Required arguments:"
    echo "  -t TASK        Task to run (e.g., RestingEyesOpen)"
    echo "  -d DATA_PATH   Path to data file or directory"
    echo
    echo "Optional arguments:"
    echo "  -c CONFIG_PATH Path to config directory (default: ./configs)"
    echo "  -h            Show this help message"
    exit 1
}

# Default config path
CONFIG_PATH="./configs"

# Parse arguments
while getopts "t:d:c:h" opt; do
    case $opt in
        t) TASK="$OPTARG" ;;
        d) DATA_PATH="$OPTARG" ;;
        c) CONFIG_PATH="$OPTARG" ;;
        h) show_help ;;
        ?) show_help ;;
    esac
done

# Check required arguments
if [ -z "$TASK" ] || [ -z "$DATA_PATH" ]; then
    echo "Error: Task (-t) and Data Path (-d) are required"
    show_help
fi

# Validate paths
if [ ! -e "$DATA_PATH" ]; then
    echo "Error: Data path does not exist: $DATA_PATH"
    exit 1
fi

if [ ! -d "$CONFIG_PATH" ]; then
    echo "Error: Config path does not exist or is not a directory: $CONFIG_PATH"
    exit 1
fi

# Print info
echo "Using data from: $DATA_PATH"
echo "Using configs from: $CONFIG_PATH"
echo "Task: $TASK"

# Run docker command
docker run -it --rm \
    -v "$DATA_PATH":/data \
    -v "$CONFIG_PATH":/app/configs \
    autoclean:latest --task "$TASK" 