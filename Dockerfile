# Use Python 3.10 as base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/app/venv \
    PATH="/app/venv/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Install uv and create venv
RUN pip install uv && \
    uv venv /app/venv

# Copy project files
COPY pyproject.toml LICENSE ./
COPY README.md ./
COPY src/autoclean ./src/autoclean
COPY configs ./configs

# Install Python dependencies using uv
RUN uv pip install -e .

# Create directory for mounting data
RUN mkdir -p /data

# Set default command
ENTRYPOINT ["autoclean"]
CMD ["--help"] 