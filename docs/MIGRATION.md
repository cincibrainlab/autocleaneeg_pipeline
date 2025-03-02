# Migration from `reports.py` to Reporting Mixins

This document outlines the migration strategy from the standalone `autoclean.step_functions.reports` module to the new object-oriented `autoclean.mixins.reporting` package.

## Overview

The reporting functionality in AutoClean has been refactored from a set of standalone functions into a collection of mixins that can be used with Task classes. This provides several benefits:

1. Better integration with the Task class
2. Improved configuration handling
3. More consistent API
4. Better documentation with mkdocs compatibility

## Completed Migration Steps

The following steps have been completed:

1. Created reporting mixins package with specialized mixins:
   - `ReportingMixin` (base class)
   - `VisualizationMixin`
   - `ICAReportingMixin`
   - `ReportGenerationMixin`

2. Updated all task classes to use the new mixins:
   - Removed imports from `autoclean.step_functions.reports`
   - Updated report generation methods to use mixin methods
   - Fixed method signatures and parameters

3. Updated docstrings for improved mkdocs compatibility

4. Marked original `reports.py` module as deprecated

## Remaining Migration Steps

The following functions from `reports.py` are still in use and need to be migrated to the mixins:

1. Functions used in `pipeline.py`:
   - `create_json_summary`
   - `create_run_report`
   - `update_task_processing_log`

2. Functions used in `continuous.py`:
   - `plot_bad_channels_with_topography`

## Migration Plan

1. Add these functions to the appropriate mixins (likely `ReportGenerationMixin`)
2. Update the calling code to use the mixins
3. Once all functions are migrated, remove the original `reports.py` module

## Mapping from Old to New Functions

| Old Function (reports.py) | New Method (Mixin) |
|---------------------------|-------------------|
| `step_plot_raw_vs_cleaned_overlay` | `VisualizationMixin.plot_raw_vs_cleaned_overlay` |
| `step_plot_ica_full` | `ICAReportingMixin.plot_ica_full` |
| `step_generate_ica_reports` | `ICAReportingMixin.plot_ica_components` |
| `step_psd_topo_figure` | `VisualizationMixin.psd_topo_figure` |
| `plot_bad_channels_with_topography` | (To be migrated) |
| `create_json_summary` | (To be migrated) |
| `create_run_report` | (To be migrated) |
| `update_task_processing_log` | (To be migrated) |
| `generate_mmn_erp` | (To be migrated) |

## Using the New Mixins

Instead of importing reporting functions from `autoclean.step_functions.reports`, you can now access them directly from any Task instance:

```python
# Old approach
from autoclean.step_functions.reports import step_plot_raw_vs_cleaned_overlay

class MyTask(Task):
    def process(self):
        # Process data
        step_plot_raw_vs_cleaned_overlay(
            self.pipeline.raw, self.cleaned_raw, self.pipeline, self.config
        )
```

```python
# New approach
class MyTask(Task):  # Task automatically includes ReportingMixin
    def process(self):
        # Process data
        self.plot_raw_vs_cleaned_overlay(
            self.pipeline.raw, self.cleaned_raw, self.pipeline, self.config
        )
```

The new approach is more concise and better integrated with the Task class.
