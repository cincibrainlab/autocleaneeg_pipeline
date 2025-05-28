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

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Dict, List, Type

# Get the current directory
_current_dir = Path(__file__).parent

# Collect all python files in the viz directory
_mixin_modules = {
    name: importlib.import_module(f"{__package__}.{name}")
    for finder, name, ispkg in pkgutil.iter_modules([str(_current_dir)])
    if not name.startswith("_")  # Skip private modules
}

# Initialize collections
__all__: List[str] = []
mixin_registry: Dict[str, Type] = {}

# Dynamically import all Mixin classes from each module
for module_name, module in _mixin_modules.items():
    # Get all classes with "Mixin" in the name to avoid including helper or utility classes
    mixin_classes = {
        name: obj
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if "Mixin" in name
        and obj.__module__
        == module.__name__  # Only include directly defined classes, not imported ones
    }

    # Add to __all__
    __all__.extend(mixin_classes.keys())

    # Add to mixin registry
    mixin_registry.update(mixin_classes)

    # Add classes to the current namespace
    globals().update(mixin_classes)
