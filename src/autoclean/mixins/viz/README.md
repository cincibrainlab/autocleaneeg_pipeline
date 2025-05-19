# Reporting Mixins for AutoClean

This directory contains a set of mixins that provide reporting and visualization functionality for EEG data in the AutoClean pipeline. These mixins are designed to be used with Task classes to generate comprehensive reports, visualizations, and summaries of EEG processing results.

## Overview

The reporting functionality has been refactored from the original `reports.py` file into a set of specialized mixins, following the same pattern as the `SignalProcessingMixin` classes. This provides several advantages:

1. **Modularity**: Each type of reporting functionality is encapsulated in its own mixin
2. **Maintainability**: Smaller, more focused files are easier to understand and maintain
3. **Extensibility**: New reporting functionality can be added by creating new mixins
4. **Composition**: Task classes can selectively include the reporting functionality they need

## Architecture

### Base Mixin

- `base.py`: Defines the `ReportingMixin` base class that provides common utility methods for all reporting mixins

### Specialized Mixins

- `visualization.py`: Provides methods for generating visualizations of EEG data, such as raw vs. cleaned data overlays and bad channel plots
- `ica.py`: Provides methods for generating visualizations and reports of ICA components
- `reports.py`: Provides methods for generating comprehensive reports of processing runs, including PDF reports and processing logs

### Main Mixin

- `main.py`: Combines all specialized mixins into a single comprehensive `ReportingMixin` class

## Configuration Integration

The reporting mixins respect configuration toggles from `autoclean_config.yaml`, similar to the `SignalProcessingMixin`. Each method checks if its corresponding step is enabled in the configuration using the `_check_step_enabled(step_name)` method.

Example configuration settings for reporting:

```yaml
tasks:
  rest_eyesopen:
    settings:
      report_generation_step:
        enabled: true
        value: {}
      log_update_step:
        enabled: true
        value: {}
      json_summary_step:
        enabled: true
        value: {}
```

## Usage

The reporting mixins are designed to be used within Task classes. The main `ReportingMixin` class is already included in the base `Task` class, so all reporting functionality is available to any task that inherits from `Task`.

### Example Usage

```python
from autoclean.core.task import Task

class MyEEGTask(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def run(self, *args, **kwargs):
        # ... process EEG data ...
        
        # Generate visualizations
        self.plot_raw_vs_cleaned_overlay(raw_original, raw_cleaned, pipeline, autoclean_dict)
        self.plot_bad_channels_with_topography(raw_original, raw_cleaned, pipeline, autoclean_dict)
        self.psd_topo_figure(raw_original, raw_cleaned, pipeline, autoclean_dict)
        
        # Generate ICA reports
        self.plot_ica_components(ica, epochs, autoclean_dict, pipeline)
        
        # Generate comprehensive report
        self.generate_report(raw_original, raw_cleaned, pipeline, autoclean_dict)
```

### Backwards Compatibility

For backwards compatibility with existing code that calls functions from the original `reports.py`, wrapper methods are provided that map to the new mixin methods. For example, `step_psd_topo_figure` is a wrapper around `psd_topo_figure`.

## Available Methods

### Visualization Methods

- `plot_raw_vs_cleaned_overlay`: Plot raw data channels over the full duration, overlaying the original and cleaned data
- `plot_bad_channels_with_topography`: Plot bad channels with a topographical map and time series overlays
- `psd_topo_figure`: Generate and save a combined PSD and topographical map figure
- `step_psd_topo_figure`: Wrapper around `psd_topo_figure` for backwards compatibility
- `generate_mmn_erp`: Analyze MMN data and create a PDF report with ERPs and topographies

### ICA Reporting Methods

- `plot_ica_components`: Generate plots of ICA components, including time courses, topographies, and spectra
- `plot_ica_sources`: Plot ICA sources and overlay EOG channels
- `generate_ica_report`: Generate a comprehensive report of ICA components

### Report Generation Methods

- `generate_report`: Generate a comprehensive report of the EEG processing pipeline
- `update_processing_log`: Update the processing log with information about the current processing run
- `generate_json_summary`: Generate a JSON summary of the processing run

## Configuration

Reporting methods respect configuration settings in `autoclean_config.yaml`. The following configuration options are available:

```yaml
tasks:
  rest_eyesopen:
    settings:
      # Visualization settings
      plot_raw_vs_cleaned_step:
        enabled: true
        value: {}
      bad_channel_report_step:
        enabled: true
        value: {}
      psd_topo_step:
        enabled: true
        value: {}
      mmn_erp_step:
        enabled: true
        value: {}
      
      # ICA reporting settings
      ica_plot_components_step:
        enabled: true
        value: {}
      ica_plot_sources_step:
        enabled: true
        value: {}
      ica_report_step:
        enabled: true
        value: {}
      
      # Report generation settings
      report_generation_step:
        enabled: true
        value: {}
      log_update_step:
        enabled: true
        value: {}
      json_summary_step:
        enabled: true
        value: {}
```

## Future Development

Additional reporting functionality can be added by creating new mixins or extending existing ones. Possible future enhancements include:

1. Interactive HTML reports with plotly or bokeh
2. More specialized reports for specific EEG paradigms (P300, SSVEP, etc.)
3. Statistical analysis reports
4. Machine learning model performance reports
5. Group-level analysis reports

The main `ReportingMixin` is included in the `Task` base class, which means that all task classes automatically inherit the reporting functionality. For example:

```python

```

## Components

### Visualization

- `plot_raw_vs_cleaned_overlay`: Plot raw data channels over the full duration, overlaying original and cleaned data
- `plot_bad_channels_with_topography`: Plot bad channels with topographical maps and time series overlays
- `psd_topo_figure`: Generate and save combined PSD and topographical map figures
- `generate_mmn_erp`: Analyze MMN data and create a PDF report with ERPs and topographies

### ICA Reporting

- `plot_ica_full`: Plot ICA components over the full duration with their labels and probabilities
- `generate_ica_reports`: Generate comprehensive ICA reports
- `_plot_ica_components`: Internal method to plot ICA components with labels and save reports

### Report Generation

- `create_run_report`: Create a scientific report in PDF format using ReportLab
- `update_task_processing_log`: Update the task-specific processing log CSV file
- `create_json_summary`: Create a comprehensive JSON summary of the processing run
