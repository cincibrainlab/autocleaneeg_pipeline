# AutoClean EEG Testing Infrastructure

This directory contains the complete testing infrastructure for AutoClean EEG, designed to support robust CI/CD workflows.

## Testing Philosophy

### Goal: Coverage of Real Behavior, Not Coverage of Lines

**Test the contract, not the implementation**
Each mixin/function has a job — ICA removes artifacts, audit logs are tamper-proof, epoching produces correct epoch counts. Tests should verify those promises hold, not just that code runs without crashing.

**Mock at the boundary, not inside the system**
EEG data (MNE objects), file I/O, and databases are expensive. Mock those at the outer edge, but let the actual logic inside each function run. The goal is catching real bugs, not achieving a green coverage number with hollow mocks.

**Tests as documentation**
Each test should read like a spec: "given a raw EEG with 3 bad channels, when `clean_bad_channels` runs, the result has fewer channels." Future developers should understand the expected behavior from the test alone.

**Prioritize failure modes**
For audit/database, the most important tests aren't the happy path — they're: does the hash chain break when tampered? Does the SQL trigger block modification of completed runs? What happens when ICA fails to converge?

**No redundant tests**
One well-designed test that exercises real logic is worth more than 10 that call a function and assert it didn't raise. Don't write multiple tests that all verify the same shallow behavior.

**One reason to fail per test**
Each test should have a single intent. A test that checks 5 unrelated things fails for ambiguous reasons and is hard to fix. If a test needs a name with "and" in it, split it.

**Know where the boundaries are**
"Mock at the boundary" only works if everyone agrees where the boundaries are. For this codebase:
- **MNE objects** (Raw, Epochs, ICA) — create minimal synthetic instances, don't mock internals
- **Filesystem** — use `tmp_path` fixtures, never real project paths
- **SQLite/database** — use in-memory databases (`:memory:`) in tests
- **GitHub/network** (block updates) — always mock, no network calls in tests

**Test names are specs**
A test name should describe the scenario and the expected outcome, not just the method being called.
- `test_ica` — bad
- `test_ica_removes_ocular_artifacts_from_frontal_channels` — good

If you can't write a descriptive name, the test's intent isn't clear enough yet.

### What to Avoid
- Tests that mock so much they only test the mock
- Asserting `result is not None` as the only check
- Duplicating what existing tests already cover well
- Inflating coverage numbers without catching real bugs

---

## Test Structure

```
tests/
├── conftest.py                 # Pytest configuration and shared fixtures
├── pytest.ini                 # Test configuration and markers
├── test_data_setup.py         # Script to generate synthetic test data
├── unit/                      # Unit tests for individual components
│   ├── core/                  # Tests for core Pipeline and Task classes
│   ├── mixins/                # Tests for mixin functionality
│   ├── plugins/               # Tests for plugin system
│   └── utils/                 # Tests for utilities
├── integration/               # End-to-end integration tests
├── fixtures/                  # Test data and utilities
│   ├── data/                  # Synthetic EEG test data files
│   ├── configs/               # Test configuration files
│   ├── synthetic_data.py      # Synthetic data generation utilities
│   └── test_utils.py          # Core testing utilities and base classes
└── README.md                  # This file
```

## Quick Start

### Generate Test Data
```bash
cd /path/to/autoclean_pipeline
PYTHONPATH=src python3 tests/test_data_setup.py
```

### Run Tests
```bash
# All tests
python3 -m pytest

# Unit tests only
python3 -m pytest tests/unit/

# Integration tests only
python3 -m pytest tests/integration/

# Specific test file
python3 -m pytest tests/unit/test_synthetic_data.py -v

# Tests with coverage
python3 -m pytest --cov=src/autoclean --cov-report=html
```

## Test Data

### Synthetic Data Files
The test infrastructure includes realistic synthetic EEG data:

- **resting_129ch_raw.fif**: 129-channel GSN-HydroCel resting state data
- **resting_128ch_raw.fif**: 128-channel GSN-HydroCel resting state data  
- **chirp_129ch_raw.fif**: Chirp stimulus paradigm with events
- **mmn_129ch_raw.fif**: MMN paradigm with standard/deviant events
- **resting_1020_raw.fif**: 32-channel standard 10-20 montage data
- **bad_channels_raw.fif**: Data with many bad channels (for QC testing)

### Corrupted Data Samples
- **extremely_noisy.fif**: High-amplitude noise (500 μV)
- **many_bad_channels.fif**: Most channels marked as bad
- **too_short.fif**: Very short duration (1 second)

### Test Configurations
- **test_basic_config.yaml**: Standard processing with ICA disabled for speed
- **test_minimal_config.yaml**: Minimal processing for fastest tests
- **test_comprehensive_config.yaml**: Full processing pipeline test

## Key Features

### Synthetic Data Generation
- **Realistic EEG signals**: Alpha, beta, theta rhythms with appropriate spatial distribution
- **Artifacts**: Eye blinks, muscle activity, and common noise patterns
- **Multiple montages**: GSN-HydroCel (128, 129), standard 10-20, MEA30
- **Event patterns**: Resting state, chirp, MMN, ASSR paradigms
- **Quality variations**: Normal data and problematic samples for QC testing

### Testing Utilities

#### EEGAssertions
```python
from tests.fixtures.test_utils import EEGAssertions

# Assert Raw object properties
EEGAssertions.assert_raw_properties(raw, expected_sfreq=250, expected_n_channels=128)

# Assert data quality
EEGAssertions.assert_data_quality(raw, max_amplitude=200e-6, min_good_channels=100)

# Assert processing outputs
EEGAssertions.assert_processing_outputs(output_dir, ["*.fif", "*.json", "*.html"])
```

#### Mock Operations
```python
from tests.fixtures.test_utils import MockOperations

# Mock heavy operations for faster testing
mock_ica = MockOperations.mock_ica_fit(raw, n_components=15)
cleaned_epochs = MockOperations.mock_autoreject_fit(epochs)
bad_channels = MockOperations.mock_ransac_channels(raw, n_bad=5)
```

#### BaseTestCase
```python
from tests.fixtures.test_utils import BaseTestCase

class TestMyComponent(BaseTestCase):
    def test_something(self):
        # Automatic temp directory setup/cleanup
        pipeline = self.create_test_pipeline()
        # Test with pipeline...
```

### Pytest Configuration

#### Markers
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests  
- `@pytest.mark.slow`: Tests taking >10 seconds
- `@pytest.mark.requires_data`: Tests needing synthetic data files
- `@pytest.mark.synthetic`: Tests using only synthetic data

#### Coverage
Tests target >70% code coverage with HTML reports generated in `htmlcov/`.

## CI Integration

### GitHub Actions Compatibility
- Runs on Ubuntu, macOS, Windows
- Python 3.10, 3.11, 3.12 support
- Dependency caching for scientific packages
- Parallel test execution
- Coverage reporting

### Test Data Management
- Synthetic data files (~80MB total) generated on-demand
- No large files committed to git
- Corrupted samples for error condition testing
- Platform-agnostic file paths

## Development Workflow

### Adding New Tests

1. **Unit Tests**: Place in appropriate `tests/unit/` subdirectory
2. **Integration Tests**: Place in `tests/integration/`
3. **New Fixtures**: Add to `tests/fixtures/`
4. **Test Data**: Extend `synthetic_data.py` if needed

### Test Requirements

- Use descriptive test names and docstrings
- Include both positive and negative test cases
- Mock heavy operations (ICA, RANSAC) for speed
- Test error conditions and edge cases
- Follow existing patterns and utilities

### Performance Guidelines

- Unit tests should run in <2 seconds each
- Use `@pytest.mark.slow` for tests >10 seconds
- Mock computational bottlenecks
- Use minimal synthetic data duration (10-30 seconds)

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH=src` is set
2. **Missing Test Data**: Run `tests/test_data_setup.py`
3. **MNE Warnings**: Filtered in pytest.ini, check for new warning types
4. **Memory Issues**: Reduce synthetic data duration or use mocks

### Debug Mode
```bash
# Verbose output
python3 -m pytest -v -s

# Stop on first failure  
python3 -m pytest -x

# Run specific test with debugging
python3 -m pytest tests/unit/test_synthetic_data.py::test_name -v -s
```