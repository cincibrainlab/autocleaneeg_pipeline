# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2025-06-17

### Added
- Modular mixin refactoring implementing standalone functions pattern
- Comprehensive standalone functions for EEG processing operations
- Enhanced separation of concerns between mixins and core processing logic
- Improved code maintainability and testability

### Changed
- **BREAKING**: Refactored mixin architecture to use standalone functions as underlying implementation
- Mixins now act as thin wrappers around standalone functions rather than containing processing logic
- Enhanced error handling and validation in processing functions
- Improved documentation and type hints throughout codebase

### Fixed
- Resolved mixin inheritance conflicts with sophisticated MRO detection
- Fixed bad channel detection consistency between original and refactored implementations
- Corrected dictionary key mapping issues in channel processing
- Enhanced pipeline result reproducibility

### Removed
- Performance monitoring workflows and benchmarking infrastructure
- Redundant code patterns replaced by standalone function approach

### Technical Improvements
- Applied comprehensive code quality fixes (black, isort, ruff)
- Enhanced test coverage for refactored components
- Improved module organization and import structure
- Better separation between pipeline logic and processing algorithms

## [2.0.4] - 2024-12-XX

### Fixed
- Package configuration and config loading for deployment
- Executable name consistency for deployment

## [2.0.1] - 2024-12-XX

### Fixed
- Executable name to autoclean-eeg for deployment consistency

## [2.0.0] - 2024-12-XX

### Added
- Major API redesign with simplified Pipeline initialization
- Python task files with embedded configuration (replacing YAML dependencies)
- Dynamic mixin discovery and combination system
- Enhanced plugin architecture for EEG formats and event processors
- Workspace management with interactive setup wizard
- Cross-platform compatibility improvements

### Changed
- **BREAKING**: Pipeline initialization API (`autoclean_dir` → `output_dir`)
- **BREAKING**: Task configuration moved from YAML to Python files
- Simplified workflow with "drop and go" task approach
- Enhanced CLI interface and commands

### Removed
- YAML configuration file dependencies for basic operation
- Complex setup requirements for simple use cases