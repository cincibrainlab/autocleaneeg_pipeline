"""Mixin classes for autoclean tasks.

This package contains mixin classes that provide common functionality
that can be shared across different task types.
"""

from autoclean.mixins.signal_processing.REGISTRY import SignalProcessingMixin
from autoclean.mixins.viz.REGISTRY import ReportingMixin

__all__ = [
    "SignalProcessingMixin",
    "ReportingMixin",
]

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Dict, List, Type, Any

# Get the directory containing this file
_current_dir = Path(__file__).parent

# Initialize collections
__all__: List[str] = []
mixin_categories: Dict[str, Any] = {}

# Discover all subdirectories that have an __init__.py (indicating a package)
for finder, name, ispkg in pkgutil.iter_modules([str(_current_dir)]):
    if ispkg and not name.startswith("_"):  # Only include packages and skip private ones
        try:
            # Try to import the main module from this category (e.g., signal_processing.REGISTRY)
            module_path = f"{__package__}.{name}.REGISTRY"
            main_module = importlib.import_module(module_path)
            
            # Identify primary mixin classes from the main module
            # These are usually composite mixins that combine functionality
            main_mixins = {
                obj_name: obj
                for obj_name, obj in inspect.getmembers(main_module, inspect.isclass)
                if "Mixin" in obj_name and obj.__module__ == main_module.__name__
            }
            
            # Add discovered mixins to globals and __all__
            for mixin_name, mixin_class in main_mixins.items():
                globals()[mixin_name] = mixin_class
                __all__.append(mixin_name)
                
            # Store the category for reference
            mixin_categories[name] = main_mixins
            
        except (ImportError, ModuleNotFoundError) as e:
            # If no main.py exists, try importing directly from the category's __init__.py
            try:
                category_module = importlib.import_module(f"{__package__}.{name}")
                
                # Get exported classes from the category's __all__
                if hasattr(category_module, "__all__"):
                    for exported_name in category_module.__all__:
                        if hasattr(category_module, exported_name):
                            exported_obj = getattr(category_module, exported_name)
                            if inspect.isclass(exported_obj) and "Mixin" in exported_name:
                                globals()[exported_name] = exported_obj
                                __all__.append(exported_name)
                                
                mixin_categories[name] = category_module
                
            except ImportError:
                # If we can't import either, skip this category
                print(f"Warning: Could not import mixin category '{name}'")

# Ensure we always have at least the main mixins
if not __all__:
    # Fall back to hard-coded imports if auto-discovery failed
    from autoclean.mixins.signal_processing.REGISTRY import SignalProcessingMixin
    from autoclean.mixins.viz.REGISTRY import ReportingMixin
    
    __all__ = ["SignalProcessingMixin", "ReportingMixin"]
    globals()["SignalProcessingMixin"] = SignalProcessingMixin
    globals()["ReportingMixin"] = ReportingMixin
