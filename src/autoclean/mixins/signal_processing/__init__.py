"""Signal processing mixin classes for autoclean tasks.

This package contains mixin classes that provide signal processing functionality
that can be shared across different task types.
"""

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Dict, List, Type

# Get the current directory
_current_dir = Path(__file__).parent

# Collect all python files in the signal_processing directory
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
