"""Reporting mixins for autoclean tasks.

This package provides reporting functionality for the AutoClean pipeline through
a set of specialized mixins that can be used with Task classes. The mixins are
designed to generate visualizations, reports, and summaries from EEG data
processing results.

Module Structure:

- `base.py`: Base reporting mixin with common utility methods
- `visualization.py`: Mixin for generating EEG data visualizations
- `ica.py`: Mixin for ICA component visualizations and reports
- `reports.py`: Mixin for generating comprehensive reports
- `main.py`: Combined mixin that provides all reporting functionality

The main `ReportingMixin` class combines all specialized mixins into a single interface,
making it easy to integrate reporting functionality into task classes.

Example:
    ```python
    from autoclean.core.task import Task
    
    # Task class automatically includes ReportingMixin via inheritance
    class MyEEGTask(Task):
        def process(self, raw, pipeline, autoclean_dict):
            # Process the data
            raw_cleaned = self.apply_preprocessing(raw)
            
            # Use reporting methods
            self.plot_raw_vs_cleaned_overlay(raw, raw_cleaned, pipeline, autoclean_dict)
            self.generate_report(raw, raw_cleaned, pipeline, autoclean_dict)
    ```

Configuration:
    All reporting methods respect configuration settings in `autoclean_config.yaml`,
    checking if their corresponding steps are enabled before execution.
"""

from autoclean.mixins.reporting.main import ReportingMixin
from autoclean.mixins.reporting.visualization import VisualizationMixin
from autoclean.mixins.reporting.ica import ICAReportingMixin
from autoclean.mixins.reporting.reports import ReportGenerationMixin

__all__ = ["ReportingMixin", "VisualizationMixin", "ICAReportingMixin", "ReportGenerationMixin"]
