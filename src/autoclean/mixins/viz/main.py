"""Main reporting mixin that combines all specialized reporting mixins.

This module provides a comprehensive reporting mixin that combines all specialized
reporting mixins into a single class. This allows for easy integration of all
reporting functionality into task classes.

The main ReportingMixin class inherits from all specialized reporting mixins,
providing a unified interface for generating visualizations, reports, and summaries
from EEG data processing results.

Example:
    ```python
    from autoclean.core.task import Task
    
    # Task class automatically includes ReportingMixin via inheritance
    class MyEEGTask(Task):
        def process(self, raw, pipeline, autoclean_dict):
            # Process the data
            raw_cleaned = self.apply_preprocessing(raw)
            
            # Use reporting methods from different mixins in a unified way
            self.plot_raw_vs_cleaned_overlay(raw, raw_cleaned, pipeline, autoclean_dict)
            self.plot_ica_components(ica, raw, autoclean_dict, pipeline)
            self.generate_report(raw, raw_cleaned, pipeline, autoclean_dict)
    ```
"""

from autoclean.mixins.viz.base import BaseVizMixin
from autoclean.mixins.viz.visualization import VisualizationMixin
from autoclean.mixins.viz.ica import ICAReportingMixin


class ReportingMixin(VisualizationMixin, ICAReportingMixin):
    """Main mixin class that combines all reporting functionality.
    
    This class inherits from all specialized reporting mixins, providing a unified
    interface for generating visualizations, reports, and summaries from EEG data
    processing results.
    
    The ReportingMixin provides the following capabilities:
    
    1. EEG data visualizations:
       - Raw vs. cleaned data overlays
       - Bad channel reports with topographies
       - PSD and topographical maps
       - MMN ERP analysis and visualization
       
    2. ICA visualizations and reports:
       - Component property plots
       - Rejected component reports
       - Component activation plots
       
    """
    
    pass
