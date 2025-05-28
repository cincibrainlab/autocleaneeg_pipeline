# Local Development Guide

This guide helps you set up and run code quality checks locally before committing to the AutoClean EEG Pipeline repository.

## 🚀 Quick Start

### 1. Install Development Tools
```bash
# Install all development tools at once
python scripts/install_dev_tools.py

# OR install manually
pip install black isort ruff mypy pytest pytest-cov
```

### 2. Run Code Quality Checks
```bash
# Check all code quality issues
python scripts/check_code_quality.py

# Auto-fix issues where possible
python scripts/check_code_quality.py --fix
```

### 3. Alternative: Use Makefile Commands
```bash
# Check code quality
make check

# Auto-format code
make format

# Run linting
make lint

# Run all checks (like CI)
make ci-check
```

## 📋 Available Commands

### Code Quality Scripts

#### `scripts/check_code_quality.py`
Main script for running code quality checks locally.

```bash
# Basic usage
python scripts/check_code_quality.py                    # Check all issues
python scripts/check_code_quality.py --fix              # Auto-fix issues
python scripts/check_code_quality.py --quiet            # Run silently
python scripts/check_code_quality.py --src tests/       # Check specific directory

# Run individual checks
python scripts/check_code_quality.py --check black      # Just formatting
python scripts/check_code_quality.py --check isort      # Just import sorting  
python scripts/check_code_quality.py --check ruff       # Just linting
python scripts/check_code_quality.py --check mypy       # Just type checking
```

#### `scripts/install_dev_tools.py`
Installs all necessary development tools.

```bash
python scripts/install_dev_tools.py
```

### Makefile Commands

The Makefile provides convenient shortcuts for common development tasks:

```bash
# Setup
make install-dev        # Install development tools
make install            # Install package in dev mode
make dev-setup          # Complete development setup

# Code Quality
make check              # Run all quality checks
make check-fix          # Run checks and auto-fix
make format             # Auto-format code (black + isort)
make format-check       # Check formatting without changes
make lint               # Run linting (ruff + mypy)

# Testing
make test               # Run unit tests
make test-cov           # Run tests with coverage
make test-integration   # Run integration tests
make test-perf          # Run performance benchmarks
make test-all           # Run all tests

# CI Simulation
make ci-check           # Run same checks as CI
make validate           # Validate code is ready for CI

# Utilities
make clean              # Clean temporary files
make all                # Run format, lint, and test
make fix-all            # Auto-fix all possible issues
```

## 🔧 Individual Tools

### Black (Code Formatting)
```bash
# Check formatting
black --check --diff src/autoclean/

# Auto-format
black src/autoclean/
```

### isort (Import Sorting)
```bash
# Check import sorting
isort --check-only --diff src/autoclean/

# Auto-sort imports
isort src/autoclean/
```

### Ruff (Fast Linting)
```bash
# Check for issues
ruff check src/autoclean/

# Auto-fix issues
ruff check --fix src/autoclean/
```

### mypy (Type Checking)
```bash
# Type check
mypy src/autoclean/ --ignore-missing-imports
```

## 🪝 Optional Pre-commit Hooks

Pre-commit hooks are **optional** (as requested). To use them:

```bash
# Install pre-commit (if not already installed)
pip install pre-commit

# Install the hooks (this makes them run automatically on commit)
pre-commit install

# Run hooks manually on all files
pre-commit run --all-files

# Skip hooks for a specific commit (if needed)
git commit -m "message" --no-verify
```

To disable pre-commit hooks completely:
```bash
pre-commit uninstall
```

## 🎯 Recommended Workflow

### Before Making Changes
```bash
# 1. Set up development environment (one time)
make dev-setup

# 2. Check current code quality
make check
```

### During Development
```bash
# Quick format and lint check
make quick-check

# Or run individual tools as needed
black src/autoclean/
isort src/autoclean/
ruff check --fix src/autoclean/
```

### Before Committing
```bash
# Option 1: Full CI simulation
make ci-check

# Option 2: Just code quality
make check-fix

# Option 3: Manual check and fix
python scripts/check_code_quality.py --fix
```

## 📊 Understanding Output

### Check Script Output
```
🚀 Running Local Code Quality Checks
==================================================
Source directory: src/autoclean
Fix mode: OFF

🔍 Black code formatting...
   ✅ Black code formatting passed

🔍 isort import sorting...
   ❌ isort import sorting failed
   Output:
   ERROR: src/autoclean/core/pipeline.py Imports are incorrectly sorted...

==================================================
📋 CODE QUALITY SUMMARY
==================================================
✅ PASS Black Formatting
❌ FAIL Import Sorting
✅ PASS Ruff Linting
✅ PASS Type Checking

Overall: 3/4 checks passed
⚠️  Some checks failed. Run with --fix to auto-correct issues.
💡 Tip: Use 'python scripts/check_code_quality.py --fix' to automatically fix most issues
```

### What Each Tool Checks

- **Black**: Code formatting (line length, quotes, spacing)
- **isort**: Import statement organization and sorting
- **Ruff**: Code linting (unused imports, style issues, potential bugs)
- **mypy**: Type checking and type hint validation

## 🚨 Common Issues and Fixes

### Import Sorting Issues
```bash
# Fix automatically
isort src/autoclean/

# Or use the check script
python scripts/check_code_quality.py --fix --check isort
```

### Code Formatting Issues
```bash
# Fix automatically
black src/autoclean/

# Or use the check script
python scripts/check_code_quality.py --fix --check black
```

### Linting Issues
```bash
# Fix automatically where possible
ruff check --fix src/autoclean/

# Manual fixes required for some issues
# Check the output for specific guidance
```

### Type Checking Issues
```bash
# Type issues usually require manual fixes
# Add type hints or ignore specific lines:
# type: ignore[error-type]
```

## 🔄 Integration with CI

The local tools use the **exact same configuration** as the CI pipeline:

- Same tool versions
- Same configuration files (pyproject.toml, .pre-commit-config.yaml)
- Same command-line arguments

This ensures that if your code passes local checks, it will pass CI checks.

## 💡 Tips for Efficient Development

1. **Use `--fix` frequently**: Most issues can be auto-corrected
2. **Run checks before big commits**: Catch issues early
3. **Use the Makefile**: Convenient shortcuts for common tasks
4. **Set up your editor**: Configure your IDE to run these tools automatically
5. **Check specific files**: Use `--src` to check only modified files

### Editor Integration Examples

**VS Code settings.json**:
```json
{
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "editor.formatOnSave": true
}
```

**PyCharm**: Enable Black, isort, and mypy in Settings → Tools → External Tools

## 🎉 Summary

With these tools, you can:
- ✅ **Catch issues early** before committing
- ✅ **Auto-fix** most formatting and style issues  
- ✅ **Match CI requirements** exactly
- ✅ **Work efficiently** with convenient commands
- ✅ **Stay optional** - no mandatory pre-commit hooks

Run `make help` to see all available commands!