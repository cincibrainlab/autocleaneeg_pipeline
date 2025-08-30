"""Utilities for safely discovering and loading AutoClean tasks."""

import importlib.util
import inspect
import pkgutil
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Type

from autoclean.core.task import Task

# Optional dependencies - may not be available in all contexts
try:
    from autoclean.utils.user_config import user_config

    USER_CONFIG_AVAILABLE = True
except ImportError:
    USER_CONFIG_AVAILABLE = False

    # Mock user_config for when it's not available
    class MockUserConfig:
        @property
        def tasks_dir(self):
            return Path.home() / ".autoclean" / "tasks"

    user_config = MockUserConfig()

# Import logging utilities for warnings
try:
    from autoclean.utils.logging import message

    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False


class DiscoveredTask(NamedTuple):
    """Represents a successfully discovered task."""

    name: str
    description: str
    source: str
    class_obj: Optional[Type[Task]] = None


class InvalidTaskFile(NamedTuple):
    """Represents a file that failed to load as a task."""

    source: str
    error: str


class SkippedTaskFile(NamedTuple):
    """Represents a file that was intentionally skipped during discovery."""

    source: str
    reason: str


class TaskOverride(NamedTuple):
    """Represents a workspace task that overrides a built-in task."""

    task_name: str
    workspace_source: str
    builtin_source: str
    description: str


def _extract_task_description(task_class: Type[Task]) -> str:
    """Extract a clean description from a task's docstring."""
    if not task_class.__doc__:
        return "No description available"

    # Get first non-empty line from docstring
    lines = task_class.__doc__.strip().split("\n")
    for line in lines:
        cleaned = line.strip()
        if cleaned:
            return cleaned

    return "No description available"


def _is_valid_task_class(obj: type, module_name: str) -> bool:
    """Check if an object is a valid Task subclass."""
    try:
        # Must be a class
        if not inspect.isclass(obj):
            return False

        # Must subclass Task but not be Task itself
        if not issubclass(obj, Task) or obj is Task:
            return False

        # Avoid importing tasks from other modules (prevents duplicates)
        if hasattr(obj, "__module__") and obj.__module__ != module_name:
            return False

        return True
    except Exception:
        return False


def _discover_builtin_tasks() -> Tuple[List[DiscoveredTask], List[InvalidTaskFile]]:
    """Discover built-in tasks from the autoclean.tasks package."""
    valid_tasks: List[DiscoveredTask] = []
    invalid_files: List[InvalidTaskFile] = []

    try:
        import autoclean.tasks
    except ImportError as e:
        invalid_files.append(
            InvalidTaskFile(
                source="autoclean.tasks", error=f"Failed to import tasks package: {e}"
            )
        )
        return valid_tasks, invalid_files

    for module_info in pkgutil.iter_modules(autoclean.tasks.__path__):
        # Skip private modules and templates
        if module_info.name.startswith("_") or module_info.name == "TEMPLATE":
            continue

        full_module_name = f"autoclean.tasks.{module_info.name}"

        try:
            module = importlib.import_module(full_module_name)

            for name, obj in inspect.getmembers(module):
                if _is_valid_task_class(obj, full_module_name):
                    valid_tasks.append(
                        DiscoveredTask(
                            name=obj.__name__,
                            description=_extract_task_description(obj),
                            source=inspect.getfile(obj),
                            class_obj=obj,
                        )
                    )

        except Exception as e:
            error_msg = str(e)
            if isinstance(e, SyntaxError):
                error_msg = f"Syntax error at line {e.lineno}: {e.msg}"

            invalid_files.append(
                InvalidTaskFile(
                    source=f"{full_module_name}.py",
                    error=f"{type(e).__name__}: {error_msg}",
                )
            )

    return valid_tasks, invalid_files


def _discover_custom_tasks() -> (
    Tuple[List[DiscoveredTask], List[InvalidTaskFile], List[SkippedTaskFile]]
):
    """Discover custom tasks from user configuration directory."""
    valid_tasks: List[DiscoveredTask] = []
    invalid_files: List[InvalidTaskFile] = []
    skipped_files: List[SkippedTaskFile] = []

    if not USER_CONFIG_AVAILABLE:
        invalid_files.append(
            InvalidTaskFile(
                source="user_config",
                error="Failed to import user config: user_config module not available",
            )
        )
        return valid_tasks, invalid_files, skipped_files

    # Check if tasks directory exists
    if not user_config.tasks_dir.exists():
        return valid_tasks, invalid_files, skipped_files

    for task_file in user_config.tasks_dir.glob("*.py"):
        # Skip private files, templates, and test fixtures
        if task_file.name.startswith("_"):
            skipped_files.append(
                SkippedTaskFile(
                    source=str(task_file), reason="Private file (starts with '_')"
                )
            )
            continue
        elif "template" in task_file.name.lower():
            skipped_files.append(
                SkippedTaskFile(
                    source=str(task_file),
                    reason="Template file (contains 'template' in name)",
                )
            )
            continue
        elif "test" in task_file.name.lower():
            skipped_files.append(
                SkippedTaskFile(
                    source=str(task_file),
                    reason="Test file (contains 'test' in name) - rename to remove 'test' if you want it loaded",
                )
            )
            continue
        elif task_file.name in [
            "bad_import_task.py",
            "bad_syntax_task.py",
            "good_task.py",
        ]:
            skipped_files.append(
                SkippedTaskFile(source=str(task_file), reason="Test fixture file")
            )
            continue

        try:
            # Create a unique module name to avoid conflicts
            module_name = f"custom_task_{task_file.stem}_{id(task_file)}"

            spec = importlib.util.spec_from_file_location(module_name, task_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load spec for {task_file}")

            module = importlib.util.module_from_spec(spec)

            # Add to sys.modules temporarily to handle relative imports
            sys.modules[module_name] = module

            try:
                spec.loader.exec_module(module)

                task_found = False
                for name, obj in inspect.getmembers(module):
                    if _is_valid_task_class(obj, module_name):
                        valid_tasks.append(
                            DiscoveredTask(
                                name=obj.__name__,
                                description=_extract_task_description(obj),
                                source=str(task_file),
                                class_obj=obj,
                            )
                        )
                        task_found = True

                if not task_found:
                    invalid_files.append(
                        InvalidTaskFile(
                            source=str(task_file),
                            error="No valid Task subclass found in file",
                        )
                    )

            finally:
                # Clean up sys.modules
                sys.modules.pop(module_name, None)

        except SyntaxError as e:
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            if e.text:
                error_msg += f" ('{e.text.strip()}')"

            invalid_files.append(
                InvalidTaskFile(source=str(task_file), error=error_msg)
            )

        except Exception as e:
            error_msg = str(e)
            # Provide more helpful error messages for common issues
            if isinstance(e, ModuleNotFoundError):
                error_msg = f"Missing dependency: {e.name}"
            elif isinstance(e, AttributeError) and "module" in str(e):
                error_msg = f"Import error: {e}"

            invalid_files.append(
                InvalidTaskFile(
                    source=str(task_file), error=f"{type(e).__name__}: {error_msg}"
                )
            )

    return valid_tasks, invalid_files, skipped_files


def safe_discover_tasks() -> (
    Tuple[List[DiscoveredTask], List[InvalidTaskFile], List[SkippedTaskFile]]
):
    """Safely discover all built-in and custom tasks with workspace priority.

    Workspace tasks automatically override built-in tasks with the same name.
    This allows users to safely customize built-in tasks without modifying
    the package installation.

    Returns:
        A tuple containing three lists:
        - A list of DiscoveredTask objects for valid tasks
        - A list of InvalidTaskFile objects for files that failed to load
        - A list of SkippedTaskFile objects for files that were intentionally skipped
    """
    all_valid_tasks: List[DiscoveredTask] = []
    all_invalid_files: List[InvalidTaskFile] = []
    all_skipped_files: List[SkippedTaskFile] = []

    # Discover custom tasks FIRST (higher priority)
    custom_tasks, custom_errors, custom_skipped = _discover_custom_tasks()
    all_valid_tasks.extend(custom_tasks)
    all_invalid_files.extend(custom_errors)
    all_skipped_files.extend(custom_skipped)

    # Discover built-in tasks SECOND (lower priority)
    builtin_tasks, builtin_errors = _discover_builtin_tasks()
    all_valid_tasks.extend(builtin_tasks)
    all_invalid_files.extend(builtin_errors)

    # Handle duplicates with workspace priority (workspace tasks override built-in)
    seen_names: Dict[str, DiscoveredTask] = {}
    unique_tasks = []
    override_info = []

    for task in all_valid_tasks:
        if task.name not in seen_names:
            seen_names[task.name] = task
            unique_tasks.append(task)
        else:
            # Found duplicate - workspace task overrides built-in task
            existing_task = seen_names[task.name]
            existing_source = Path(existing_task.source).name
            override_source = Path(task.source).name

            # Determine if this is a workspace override of a built-in task
            if "tasks" in existing_task.source and "autoclean.tasks" in task.source:
                # Workspace task is already loaded, built-in task is being skipped
                override_info.append(
                    InvalidTaskFile(
                        source=f"override: {existing_source}",
                        error=f"Workspace task '{task.name}' overrides built-in task from package",
                    )
                )
            elif "autoclean.tasks" in existing_task.source and "tasks" in task.source:
                # This shouldn't happen with our discovery order, but handle it
                # Replace built-in with workspace task
                seen_names[task.name] = task
                unique_tasks = [t for t in unique_tasks if t.name != task.name]
                unique_tasks.append(task)
                override_info.append(
                    InvalidTaskFile(
                        source=f"override: {override_source}",
                        error=f"Workspace task '{task.name}' overrides built-in task from package",
                    )
                )
            else:
                # True duplicate within same source type - still a warning
                override_info.append(
                    InvalidTaskFile(
                        source=override_source,
                        error=f"Duplicate task name '{task.name}' (already defined in {existing_source})",
                    )
                )

    # Add override info to invalid files list for reporting
    all_invalid_files.extend(override_info)

    return unique_tasks, all_invalid_files, all_skipped_files


def extract_config_from_task(task_name: str, config_key: str) -> Optional[str]:
    """Extract a configuration value from a task if it exists.

    Args:
        task_name: Name of the task to check
        config_key: The configuration key to extract (e.g., 'dataset_name', 'input_path')

    Returns:
        Configuration value if found in task config, None otherwise
    """
    try:
        # Get all valid tasks
        valid_tasks, _, _ = safe_discover_tasks()

        # Find the task by name (case-insensitive)
        task_obj = None
        for task in valid_tasks:
            if task.name.lower() == task_name.lower():
                task_obj = task
                break

        if not task_obj:
            return None

        # Import the task module to access its config
        module_name = f"temp_task_{task_obj.source.replace('/', '_').replace('.', '_')}"
        spec = importlib.util.spec_from_file_location(module_name, task_obj.source)

        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)

            # Look for config dictionary in the module
            if hasattr(module, "config") and isinstance(module.config, dict):
                return module.config.get(config_key)

        finally:
            # Clean up
            sys.modules.pop(module_name, None)

    except Exception:
        # If anything fails, just return None
        pass

    return None


def get_task_overrides() -> List[TaskOverride]:
    """Get information about workspace tasks that override built-in tasks.

    Returns:
        List of TaskOverride objects describing which workspace tasks
        override which built-in tasks.
    """
    overrides = []

    # Get all built-in and custom tasks separately
    builtin_tasks, _ = _discover_builtin_tasks()
    custom_tasks, _, _ = _discover_custom_tasks()

    # Create lookup for built-in tasks
    builtin_by_name = {task.name: task for task in builtin_tasks}

    # Check for overrides
    for custom_task in custom_tasks:
        if custom_task.name in builtin_by_name:
            builtin_task = builtin_by_name[custom_task.name]
            overrides.append(
                TaskOverride(
                    task_name=custom_task.name,
                    workspace_source=custom_task.source,
                    builtin_source=builtin_task.source,
                    description=custom_task.description,
                )
            )

    return overrides


def get_task_by_name(task_name: str) -> Optional[Type[Task]]:
    """Get a task class by its name.

    Args:
        task_name: The name of the task to retrieve

    Returns:
        The Task class if found, None otherwise
    """
    valid_tasks, _, _ = safe_discover_tasks()

    for task in valid_tasks:
        if task.name == task_name and task.class_obj:
            return task.class_obj

    return None
