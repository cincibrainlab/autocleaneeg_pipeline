# AutoClean Pipeline Examples

This directory contains practical examples demonstrating how to use the AutoClean Pipeline for EEG data processing. These examples are designed to help researchers and developers understand the capabilities and flexibility of the pipeline.

## Available Examples

### Basic Usage
- **[basic_usage.py](basic_usage.py)**: Simple demonstration of loading and processing a single EEG file using a predefined task.

### Advanced Features
- **[custom_step_function.py](custom_step_function.py)**: Demonstrates how to create and use custom step functions to extend the pipeline's capabilities.
- **[custom_task.py](custom_task.py)**: Shows how to create a custom task for processing Event-Related Potential (ERP) data.
- **[batch_processing.py](batch_processing.py)**: Illustrates how to process multiple EEG files in parallel with error handling and progress tracking.
- **[advanced_configuration.py](advanced_configuration.py)**: Examples of working with configuration files, environment variables, and configuration merging.

### Specific EEG Paradigms
- **[Chirp_HumanData.py](Chirp_HumanData.py)**: Example of processing auditory chirp paradigm data.
- **[start_review.py](start_review.py)**: How to start the review GUI for interactive data inspection.

## Running the Examples

Most examples can be run directly from the command line. For instance:

```bash
# Basic usage
python examples/basic_usage.py

# Custom task
python examples/custom_task.py

# Batch processing
python examples/batch_processing.py --simulated --n-simulated 3

# Advanced configuration
python examples/advanced_configuration.py
```

## Prerequisites

To run these examples, you'll need to have the AutoClean Pipeline installed, along with its dependencies. The examples are designed to work with simulated data if real EEG files are not available.

## Example Outputs

Most examples will generate output files in a designated directory (usually `./output` or a similar path). These outputs typically include:

- Processed EEG data files (in `.fif` format)
- Power spectral density plots (`.png`)
- ICA component plots (`.png`)
- HTML/PDF reports summarizing the analysis
- CSV files with summary metrics

## Creating Your Own Examples

Feel free to use these examples as templates for your own EEG processing workflows. The modular nature of the AutoClean Pipeline makes it easy to adapt to specific experimental paradigms and research questions.

For more in-depth documentation, refer to the [user guide](../docs/user-guide) and [API reference](../docs/api-reference) in the docs directory.

 