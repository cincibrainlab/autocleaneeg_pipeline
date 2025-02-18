# ./src/autoclean/tasks/__init__.py
"""Task definitions for the autoclean pipeline."""

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Dict, List, Type

from autoclean.core.task import Task

# Get the current directory
_current_dir = Path(__file__).parent

# Collect all python files in the tasks directory
_task_modules = {
    name: importlib.import_module(f"{__package__}.{name}")
    for finder, name, ispkg in pkgutil.iter_modules([str(_current_dir)])
    if not name.startswith("_")  # Skip private modules
}

# Initialize collections
__all__: List[str] = []
task_registry: Dict[str, Type[Task]] = {}

# Dynamically import all Task classes from each module
for module_name, module in _task_modules.items():
    # Get all classes that inherit from Task
    task_classes = {
        name.lower(): obj  # Convert to lowercase for consistent lookup
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, Task) and obj != Task  # Exclude the base Task class
    }

    # Add to __all__
    __all__.extend(task_classes.keys())

    # Add to task registry
    task_registry.update(task_classes)

    # Add classes to the current namespace
    globals().update(task_classes)
