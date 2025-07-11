# Development Tests

This directory contains specialized tests used during development of major features. These tests are separate from the main test suite and focus on specific functionality that may require manual verification or special setup.

## Test Files

### `test_database_schema.py`
Tests database schema migrations and encrypted outputs functionality. Useful for:
- Validating database schema changes
- Testing encrypted output storage
- Verifying database integrity

### `test_encryption.py` 
Core encryption functionality tests for Part 11 compliance mode. Covers:
- Encryption manager functionality
- Output type classification
- File encryption and storage
- Compression and decompression

### `test_pipeline_integration.py`
Integration tests for the pipeline with encryption features. Tests:
- Pipeline initialization in compliance mode
- Output routing functionality
- End-to-end encryption workflow
- Routed method integration

## Usage

These tests can be run individually during development:

```bash
# Run database schema tests
python tests/development/test_database_schema.py

# Run encryption tests
python tests/development/test_encryption.py

# Run pipeline integration tests
python tests/development/test_pipeline_integration.py
```

## Notes

- These tests may require specific environment setup (auth0 credentials, etc.)
- They are not part of the automated CI/CD pipeline
- Use these for manual verification during feature development
- Consider integrating successful tests into the main test suite when appropriate