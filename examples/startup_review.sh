#!/bin/bash

# This script runs when the container starts
# It's placed in /custom-cont-init.d/ and will be executed automatically

# Ensure proper permissions for the output directory
chown -R abc:abc /app/output

# Create desktop shortcut with proper permissions
chown abc:abc /home/abc/Desktop/autoclean-review.desktop

# Set up any required environment variables
echo "export PYTHONPATH=/app:$PYTHONPATH" >> /home/abc/.bashrc
echo "export PATH=/app/venv/bin:$PATH" >> /home/abc/.bashrc

# Activate virtual environment in bashrc
echo "source /app/venv/bin/activate" >> /home/abc/.bashrc 