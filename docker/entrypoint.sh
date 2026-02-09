#!/bin/bash
set -e

# Default mode if not specified
MODE=${MODE:-live}
echo "Starting AutocleanEEG Serve in ${MODE} mode..."

# Determine config file name based on mode
CONFIG_FILE="serve-${MODE}.yaml"

# Initialize workspace if config provided
if [ -f /config/serve-config.yaml ]; then
    echo "Initializing workspace from /config/serve-config.yaml..."

    # Create workspace structure
    mkdir -p /workspace/deploy /workspace/runtimes/${MODE}

    # Copy config with mode-appropriate name
    cp /config/serve-config.yaml /workspace/${CONFIG_FILE}

    # Link output directory
    ln -sf /output /workspace/output

    echo "Workspace initialized for ${MODE} mode."
else
    echo "WARNING: No config found at /config/serve-config.yaml"
    echo "Container will start but processing requires configuration."
fi

# Export MODE for supervisord to use
export MODE

# Start supervisord
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
