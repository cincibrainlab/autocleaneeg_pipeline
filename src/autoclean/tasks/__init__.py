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

# Collect all approved python files in the tasks directory (non-pending)
_approved_modules = {
    name: importlib.import_module(f"{__package__}.{name}")
    for finder, name, ispkg in pkgutil.iter_modules([str(_current_dir)])
    if not name.startswith("_")  # Skip private modules
}

_pending_dir = _current_dir / "pending_approval"
_pending_modules = {}
if _pending_dir.exists():
    for finder, name, ispkg in pkgutil.iter_modules([str(_pending_dir)]):
        if name.startswith("_"):
            continue
        module = importlib.import_module(f"{__package__}.pending_approval.{name}")
        _pending_modules[f"pending_approval.{name}"] = module

# Initialize collections
__all__: List[str] = []
task_registry: Dict[str, Type[Task]] = {}

# Load approved tasks first
for module_name, module in _approved_modules.items():
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

# Load pending_approval tasks — do NOT overwrite already-registered approved tasks
for module_name, module in _pending_modules.items():
    # Get all classes that inherit from Task
    task_classes = {
        name.lower(): obj  # Convert to lowercase for consistent lookup
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, Task) and obj != Task  # Exclude the base Task class
    }

    for task_name, task_class in task_classes.items():
        if task_name not in task_registry:
            # Only register pending tasks that don't conflict with approved tasks
            __all__.append(task_name)
            task_registry[task_name] = task_class
            globals()[task_name] = task_class
