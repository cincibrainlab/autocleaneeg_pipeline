# AutoClean EEG

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A modular framework for automated EEG data processing, built on MNE-Python and PyLossless.

## Features

- Automated EEG preprocessing pipeline
- Support for multiple EEG paradigms (ASSR, Chirp, MMN, Resting State)
- BIDS-compatible data organization
- Extensive quality control and reporting
- Modular task-based architecture
- Database-backed processing tracking

## Installation

```bash
pip install autoclean-eeg
```

For development installation:

```bash
git clone https://github.com/yourusername/autoclean-eeg.git
cd autoclean-eeg
pip install -e ".[dev]"
```

## Quick Start

```python
from autoclean import Pipeline

# Initialize pipeline
pipeline = Pipeline(
    autoclean_dir="/path/to/output",
    autoclean_config="config.yaml"
)

# Process a single file
pipeline.process_file(
    file_path="/path/to/data.raw",
    task="rest_eyesopen"
)

# Process multiple files
pipeline.process_directory(
    directory="/path/to/data",
    task="rest_eyesopen",
    pattern="*.raw"
)
```

## Documentation

Full documentation is available at [https://autoclean-eeg.readthedocs.io/](https://autoclean-eeg.readthedocs.io/)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{autoclean_eeg,
  author = {Gammoh, Gavin, Pedapati, Ernest, and Grace },
  title = {AutoClean EEG: Automated EEG Processing Pipeline},
  year = {2024},
  publisher = {GitHub},
  url = {[https://github.com/yourusername/autoclean-eeg](https://github.com/cincibrainlab/autoclean_pipeline/)}
}
```

## Acknowledgments

- Cincinnati Children's Hospital Medical Center
- Built with [MNE-Python](https://mne.tools/) and [PyLossless](https://github.com/lina-usc/pylossless) 

## Docker Usage

The pipeline can be run in a containerized environment using Docker. This ensures consistent execution across different systems and isolates the pipeline dependencies.

### Building the Docker Image

```bash
# Build the image
docker build -t autoclean:latest .

# Or using docker-compose
docker-compose build
```

### Running the Pipeline

There are two ways to run the pipeline:

1. Using docker-compose (recommended):
```bash
# Run with default configuration
docker-compose up

# Run with specific task and parameters
docker-compose run --rm autoclean --task resting_eyes_open --input /data/input --output /data/output
```

2. Using docker directly:
```bash
docker run -it --rm \
  --shm-size=2g \
  -v $(pwd)/data:/data \
  -v $(pwd)/configs:/app/configs \
  autoclean:latest --task resting_eyes_open --input /data/input --output /data/output
```

### Data Mounting

- Mount your input data directory to `/data` inside the container
- Configuration files should be mounted to `/app/configs`
- Output will be written to the mounted data directory

### Resource Configuration

The default configuration in docker-compose.yml allocates:
- 2GB shared memory
- 4-8GB RAM
- These can be adjusted in the docker-compose.yml file

### GUI Support (if needed)

For GUI support on Linux systems:
```bash
xhost +local:docker
docker-compose run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix autoclean
```
