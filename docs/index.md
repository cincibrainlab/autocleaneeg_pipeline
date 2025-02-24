# AutoClean Pipeline

<div class="grid cards" markdown>

-   :zap: __Lightning Fast Processing__

    ---

    Process EEG data with unprecedented speed using our optimized pipeline

    [:octicons-arrow-right-24: Quick Start](getting-started/quick-start.md)

-   :material-pipe: __Modular Architecture__

    ---
    
    Build custom processing pipelines with plug-and-play components

    [:octicons-arrow-right-24: Pipeline Overview](user-guide/pipeline-overview.md)

-   :material-function: __Extensive Step Functions__

    ---
    
    Rich library of pre-built processing steps for common EEG operations

    [:octicons-arrow-right-24: Step Functions](user-guide/step-functions.md)

-   :material-cog: __Flexible Configuration__

    ---
    
    Easily customize processing parameters through YAML configuration files

    [:octicons-arrow-right-24: Configuration](getting-started/configuration.md)

</div>

## About AutoClean Pipeline

AutoClean Pipeline is a powerful, modular framework for automated EEG data processing. Built on top of MNE and PyLossless, it provides researchers with a flexible and efficient way to process EEG data while maintaining the highest standards of data quality.

```python
from autoclean import Pipeline

# Initialize the pipeline
pipeline = Pipeline(
    task="resting_eyes_open",
    data_path="path/to/data"
)

# Run the pipeline
pipeline.run()
```

## Key Features

- **:octicons-zap-24: High Performance**: Optimized for speed and efficiency
- **:octicons-plug-24: Modular Design**: Easy to extend and customize
- **:octicons-tools-24: Rich Toolset**: Comprehensive set of processing functions
- **:octicons-checklist-24: Quality Control**: Built-in validation and reporting
- **:octicons-sync-24: Reproducible**: Version-controlled configurations
- **:octicons-database-24: BIDS Compatible**: Standardized data organization

## Getting Started

Get up and running with AutoClean Pipeline in minutes:

=== "Installation"

    ```bash
    pip install autoclean-pipeline
    ```

=== "Configuration"

    ```yaml
    task: resting_eyes_open
    parameters:
      sampling_rate: 1000
      line_noise: 60
    ```

=== "Usage"

    ```python
    from autoclean import Pipeline
    
    pipeline = Pipeline.from_config("config.yaml")
    pipeline.run()
    ```

## Project Status

AutoClean Pipeline is actively maintained and used in production environments. We follow semantic versioning and maintain a detailed changelog of all updates.

[:octicons-arrow-right-24: View Changelog](changelog.md){ .md-button }
