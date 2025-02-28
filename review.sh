#!/bin/bash

# Convert Windows paths to Unix-compatible paths
convert_path() {
    local path=$1
    # Check if it's a Windows-style path
    if [[ $path == *\\* ]]; then
        # Replace backslashes with forward slashes
        echo "${path//\\//}"
    else
        echo "$path"
    fi
}

# Get output directory from environment variable or command line
OUTPUT_DIR="${OUTPUT_PATH:-./output}"
PORT=6080

CONTAINER_NAME="review"
if [ -n "$BOT_NAME" ]; then
    CONTAINER_NAME="review-$BOT_NAME"
fi

# Parse command line arguments
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 [OUTPUT_DIRECTORY]"
    echo ""
    echo "You can specify the output directory as an argument or use the OUTPUT_PATH environment variable."
    echo "Example:"
    echo "  $0 /path/to/eeg/data"
    echo "  OUTPUT_PATH=/path/to/eeg/data $0"
    exit 0
fi

# If first argument is provided, use it as output directory
if [ -n "$1" ]; then
    OUTPUT_DIR="$1"
fi

# Convert path if needed
OUTPUT_DIR=$(convert_path "$OUTPUT_DIR")



# Start the review container
echo "Starting autoclean review container..."
echo "Output directory: $OUTPUT_DIR"
echo "noVNC will be available at: http://localhost:$PORT"

# Run the container
docker compose -f - up --remove-orphans <<EOF
services:
  review:
    container_name: ${CONTAINER_NAME}
    build:
      context: .
      dockerfile: Dockerfile.review
    image: autoclean-review:latest
    volumes:
      - ${OUTPUT_DIR}:/app/output
    environment:
      - PYTHONUNBUFFERED=1
      - QT_QPA_PLATFORM=xcb
    ports:
      - ${PORT}:6080     # noVNC web interface
EOF

