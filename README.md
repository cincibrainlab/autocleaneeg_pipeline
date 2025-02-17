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
  url = {https://github.com/yourusername/autoclean-eeg}
}
```

## Acknowledgments

- Cincinnati Children's Hospital Medical Center
- Built with [MNE-Python](https://mne.tools/) and [PyLossless](https://github.com/lina-usc/pylossless) 
