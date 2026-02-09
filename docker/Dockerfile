# AutocleanEEG Serve - Docker Container
# Single-command deployment for EEG processing automation
#
# Build: docker build -t autoclean/serve:latest -f docker/Dockerfile .
# Run:   docker run -d -p 4000:4000 -v ~/data:/data -v ~/config:/config autoclean/serve:latest

FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    redis-server \
    supervisor \
    curl \
    git \
    build-essential \
    libgl1 \
    libegl1 \
    libglib2.0-0 \
    libxkbcommon0 \
    libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

# Set environment variables
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    MODE=live \
    WORKERS=2 \
    API_PORT=4000 \
    LOG_LEVEL=INFO

# Create workspace structure
RUN mkdir -p /workspace /data /config /output /var/log/supervisor

# Copy package files for installation
WORKDIR /app
COPY pyproject.toml README.md LICENSE.md ./
COPY src ./src
COPY configs ./configs

# Install autocleaneeg-pipeline with uv (significantly faster than pip)
RUN uv pip install --system . \
    && uv cache clean

# Copy Docker-specific files
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose API port
EXPOSE 4000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:4000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
