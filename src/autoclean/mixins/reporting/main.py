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

from autoclean.mixins.reporting.base import ReportingMixin as BaseReportingMixin
from autoclean.mixins.reporting.visualization import VisualizationMixin
from autoclean.mixins.reporting.ica import ICAReportingMixin
from autoclean.mixins.reporting.reports import ReportGenerationMixin

class ReportingMixin(
    BaseReportingMixin,
    VisualizationMixin,
    ICAReportingMixin,
    ReportGenerationMixin
):
    """Main mixin class that combines all reporting functionality.
    
    This class inherits from all specialized reporting mixins to provide
    a comprehensive set of reporting methods for EEG data processing results.
    
    The ReportingMixin provides the following capabilities:
    
    1. EEG data visualizations:
       - Raw vs. cleaned data overlays
       - Bad channel reports with topographies
       - PSD and topographical maps
       - MMN ERP analysis and visualization
    
    2. ICA component visualizations and reports:
       - Full-duration component activations
       - Component properties and topographies
       - Component rejection documentation
    
    3. Comprehensive reporting:
       - PDF summary reports
       - Processing log updates
       - JSON summaries for machine-readable output
    
    This combined mixin follows the same pattern as the SignalProcessingMixin,
    providing a clean interface for task classes to access all reporting functionality.
    All methods respect configuration settings in `autoclean_config.yaml`.
    
    Note:
        This class is automatically included in the base `Task` class through
        multiple inheritance, so any task that inherits from `Task` will have
        access to all reporting methods.
    """
    pass
