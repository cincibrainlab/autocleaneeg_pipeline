# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoClean EEG is a modular framework for automated EEG data processing built on MNE-Python. It supports multiple EEG paradigms (ASSR, Chirp, MMN, Resting State) with BIDS-compatible data organization and database-backed processing tracking.

## Pipeline Architecture Understanding

I now have comprehensive knowledge of the AutoClean pipeline's architecture, including:

- The core design principles of a modular, extensible EEG processing framework
- How tasks are dynamically constructed using mixins
- The plugin system for handling different EEG formats and event processing
- The configuration management through YAML-based hierarchical settings
- The typical research workflow from setup to production processing
- Performance considerations and bottlenecks in EEG data processing
- Challenges in handling diverse EEG datasets, especially complex cases like pediatric data

Key insights about the pipeline's implementation:
- Uses a "Lego Block" approach for task composition
- Supports dynamic mixin discovery and combination
- Provides extensive customization options
- Focuses on modularity and extensibility
- Handles multiple EEG paradigms and file formats

Specific implementation patterns learned:
- Task classes inherit from base Task and automatically incorporate mixins
- Configuration is highly flexible and stage-aware
- Emphasis on type hints, code quality, and scientific computing best practices
- Docker and cross-platform support are key design considerations

## Development Commands

### Code Quality
```bash
# Format code
black src/autoclean/
isort src/autoclean/

# Type checking
mypy src/autoclean/

# Linting
ruff check src/autoclean/

# Testing with coverage
pytest --cov=autoclean

# Run specific test suites
pytest tests/unit/                    # Unit tests only
pytest tests/integration/            # Integration tests only  
pytest tests/unit/ -k "test_pipeline" # Specific test patterns
```

### Build and Installation
```bash
# Development installation
pip install -e .

# With GUI dependencies
pip install -e ".[gui]"

# Build package
python -m build
```

### Documentation
```bash
# Build docs (from docs/ directory)
cd docs
make html

# Clean docs
make clean
```

### Docker Development
```bash
# Build and run pipeline
docker-compose up autoclean

# Run review GUI
docker-compose up review

# Shell access
docker-compose run autoclean bash
```

### CI/CD Pipeline
```bash
# The project has comprehensive GitHub Actions CI workflows:

# Main CI Pipeline (.github/workflows/ci.yml):
# - Matrix testing: Python 3.10-3.12 across Ubuntu/macOS/Windows
# - Code quality: black, isort, ruff, mypy 
# - Security: bandit, pip-audit
# - Testing: pytest with coverage reporting
# - Build verification and package validation
# - Integration tests on key platforms

# Additional workflows:
# - Performance benchmarking (.github/workflows/benchmark.yml)
# - Automated dependency updates via Dependabot
# - Coverage reporting via Codecov
# - Auto-merge for trusted dependency updates

# Test execution approach:
# - 164 unit tests with 119 passing (32 failing due to import issues)
# - Mock-heavy strategy to avoid computational bottlenecks
# - Synthetic EEG data generation for realistic testing
# - Fast CI execution targeting <15 minute runs
```

[... rest of the existing file content remains the same ...]