# AutoClean EEG Pipeline function
autoclean() {
    # Check if we have at least 2 arguments
    if [ $# -lt 2 ]; then
        echo "Usage: autoclean DATA_PATH TASK [CONFIG_PATH]"
        echo "Example: autoclean /path/to/data RestingEyesOpen"
        return 1
    fi

    local DATA_PATH="$1"
    local TASK="$2"
    local CONFIG_PATH="${3:-./configs}"  # Use third argument or default to ./configs

    # Validate paths
    if [ ! -e "$DATA_PATH" ]; then
        echo "Error: Data path does not exist: $DATA_PATH"
        return 1
    fi

    if [ ! -d "$CONFIG_PATH" ]; then
        echo "Error: Config path does not exist or is not a directory: $CONFIG_PATH"
        return 1
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
} 