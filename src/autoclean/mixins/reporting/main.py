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

class ReportingMixin(BaseReportingMixin, VisualizationMixin, ICAReportingMixin):
    """Main mixin class that combines all reporting functionality.
    
    This class uses composition to integrate functionality from all specialized
    reporting mixins, providing a unified interface for generating visualizations,
    reports, and summaries from EEG data processing results.
    
    The ReportingMixin provides the following capabilities through delegation:
    
    1. EEG data visualizations:
       - Raw vs. cleaned data overlays
       - Bad channel reports with topographies
       - PSD and topographical maps
       - MMN ERP analysis and visualization
       
    2. ICA visualizations and reports:
       - Component property plots
       - Rejected component reports
       - Component activation plots
       
    3. Report generation:
       - Processing summary reports (PDF)
       - Log updates
       - JSON summaries
    """
    
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
        
    #     # Initialize all mixins using composition instead of inheritance
    #     self.base_mixin = BaseReportingMixin()
    #     self.viz_mixin = VisualizationMixin()
    #     self.ica_mixin = ICAReportingMixin()
    #     self.report_mixin = ReportGenerationMixin()
        
    # # Delegate methods to the appropriate mixins
    # def __getattr__(self, name):
    #     # Check each mixin for the attribute
    #     for mixin in [self.viz_mixin, self.ica_mixin, self.report_mixin, self.base_mixin]:
    #         if hasattr(mixin, name):
    #             return getattr(mixin, name)
        
    #     # If not found, raise AttributeError
    #     raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    pass
