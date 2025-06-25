"""Autoclean Mixins Package.

This package dynamically discovers and provides mixin classes for the Task base class.
Any class ending with 'Mixin' in a .py file within a subdirectory of this package
(or BaseMixin in base.py) will be collected and made available for inheritance by Task.
"""

import importlib
import inspect
import pkgutil
import types
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

# --- Foundational BaseMixin Import ---
# BaseMixin is considered essential and is loaded directly.
# It will be the first mixin in the inheritance list for Task (after ABC).
try:
    from .base import BaseMixin

    _BASE_MIXIN_CLASS: Type[Any] = BaseMixin
    _base_mixin_found = True
except ImportError:
    print(
        "CRITICAL ERROR: autoclean.mixins.base.BaseMixin could not be imported. "
        "Task functionality will be severely impaired. A placeholder will be used."
    )

    class BaseMixinPlaceholder:  # Define a placeholder to prevent crashes
        pass

    _BASE_MIXIN_CLASS = BaseMixinPlaceholder
    _base_mixin_found = False


# --- Helper function to check for method collisions ---
def _warn_on_method_collisions(mixins_tuple: Tuple[Type[Any], ...]) -> None:
    """Checks for method name collisions among the discovered mixins and warns the user."""
    method_definitions: Dict[str, List[Tuple[str, str]]] = (
        {}
    )  # method_name: [(class_name, module_name), ...]
    collision_found = False

    for mixin_cls in mixins_tuple:
        # Iterate through the class's own __dict__ to find methods defined directly in it.
        for name, member in mixin_cls.__dict__.items():
            # Consider it a method if it's callable, not a dunder, and is a function, classmethod, or staticmethod.
            if callable(member) and not name.startswith("__"):
                is_relevant_callable = False
                if isinstance(member, (types.FunctionType, classmethod, staticmethod)):
                    is_relevant_callable = True

                if is_relevant_callable:
                    class_info = (mixin_cls.__name__, mixin_cls.__module__)
                    if name in method_definitions:
                        # Only add if this specific class_info isn't already listed for this method name
                        if class_info not in method_definitions[name]:
                            method_definitions[name].append(class_info)
                    else:
                        method_definitions[name] = [class_info]

    for method_name, defining_classes in method_definitions.items():
        if len(defining_classes) > 1:
            collision_found = True
            # Sort by module then class name for consistent warning output
            defining_classes.sort()
            class_details_str = "; ".join(
                [
                    f"{cls_name} (from {mod_name})"
                    for cls_name, mod_name in defining_classes
                ]
            )
            print(
                f"WARNING: Method '{method_name}' is defined in multiple mixin classes: {class_details_str}."
            )

            # Determine which class's method will likely take precedence based on the input mixins_tuple order
            winner_info = None
            min_index = float("inf")

            for cls_name_conflict, mod_name_conflict in defining_classes:
                for i, discovered_mixin_cls in enumerate(mixins_tuple):
                    if (
                        discovered_mixin_cls.__name__ == cls_name_conflict
                        and discovered_mixin_cls.__module__ == mod_name_conflict
                    ):
                        if i < min_index:
                            min_index = i
                            winner_info = (cls_name_conflict, mod_name_conflict)
                        break

            if winner_info:
                # This warning refers to precedence within the _all_individual_mixins_tuple,
                # which helps understand overlaps before any consolidation attempt.
                print(
                    f"         Among these, the implementation from '{winner_info[0]} (from {winner_info[1]})' appears earliest in the initial discovered list."
                )
            else:
                print(
                    f"         Could not determine precedence for '{method_name}' among listed classes based on initial discovery order."
                )

    if collision_found:
        print(
            "INFO: Please review these method overlaps. If these mixins are combined, the Method Resolution Order (MRO) "
            "of the combined class will ultimately determine which implementation is used."
        )


# --- Dynamic Discovery of Other Mixin Classes ---
_discovered_other_mixins: List[Type[Any]] = []
_current_package_path = Path(__file__).parent
_current_package_name = __name__  # Should be 'autoclean.mixins'

# Iterate over all items (modules and sub-packages) in the current 'mixins' directory
for module_info in pkgutil.iter_modules([str(_current_package_path)]):
    # Skip the 'base.py' module as BaseMixin is handled separately
    if module_info.name == "base":
        continue

    # Construct the full import path for the module or sub-package
    full_item_name = f"{_current_package_name}.{module_info.name}"

    if module_info.ispkg:
        # This item is a sub-package (e.g., 'signal_processing', 'viz')
        try:
            sub_package_module = importlib.import_module(full_item_name)
            # Iterate through modules (.py files) within this sub-package
            for sub_module_info in pkgutil.iter_modules(sub_package_module.__path__):
                # Skip private modules (e.g., __pycache__ or _internal.py)
                if sub_module_info.name.startswith("_"):
                    continue

                full_sub_module_name = f"{full_item_name}.{sub_module_info.name}"
                try:
                    module = importlib.import_module(full_sub_module_name)
                    for class_name, class_obj in inspect.getmembers(
                        module, inspect.isclass
                    ):
                        # Criteria for a discoverable mixin:
                        # 1. Defined in the currently inspected module (not imported from elsewhere).
                        # 2. Class name ends with "Mixin".
                        # 3. It's not the BaseMixin class itself (already handled).
                        if (
                            class_obj.__module__ == full_sub_module_name
                            and class_name.endswith("Mixin")
                            and class_obj is not _BASE_MIXIN_CLASS
                        ):
                            if class_obj not in _discovered_other_mixins:
                                _discovered_other_mixins.append(class_obj)
                except ImportError as e:
                    print("=" * 80)
                    print("🚨 CRITICAL MIXIN IMPORT ERROR 🚨")
                    print(
                        f"Could not import module '{full_sub_module_name}' for mixin discovery"
                    )
                    print(f"Error: {e}")
                    print("STOPPING EXECUTION - Fix import error before continuing!")
                    print(
                        "Check for syntax errors or missing dependencies in the mixin file."
                    )
                    print("=" * 80)
                    raise SystemExit(
                        f"Critical mixin import error: {full_sub_module_name} - {e}"
                    )
                except Exception as e:
                    print("=" * 80)
                    print("🚨 CRITICAL MIXIN ERROR 🚨")
                    print(f"Error inspecting module '{full_sub_module_name}': {e}")
                    print("STOPPING EXECUTION - Fix mixin error before continuing!")
                    print("=" * 80)
                    raise SystemExit(
                        f"Critical mixin error: {full_sub_module_name} - {e}"
                    )
        except ImportError as e:
            print("=" * 80)
            print("🚨 CRITICAL SUB-PACKAGE IMPORT ERROR 🚨")
            print(
                f"Could not import sub-package '{full_item_name}' for mixin discovery"
            )
            print(f"Error: {e}")
            print("STOPPING EXECUTION - Fix import error before continuing!")
            print("=" * 80)
            raise SystemExit(
                f"Critical sub-package import error: {full_item_name} - {e}"
            )
        except SyntaxError as e:
            print("=" * 80)
            print("🚨 CRITICAL SYNTAX ERROR IN MIXIN SUB-PACKAGE 🚨")
            print(f"Sub-package: {full_item_name}")
            print(f"File: {e.filename}")
            print(f"Line {e.lineno}: {e.text}")
            print(f"Error: {e}")
            print("STOPPING EXECUTION - Fix syntax error before continuing!")
            print("=" * 80)
            raise SystemExit(
                f"Critical syntax error in sub-package: {e.filename}, line {e.lineno}"
            )
        except Exception as e:
            print("=" * 80)
            print("🚨 CRITICAL SUB-PACKAGE ERROR 🚨")
            print(f"Error processing sub-package '{full_item_name}': {e}")
            print("STOPPING EXECUTION - Fix error before continuing!")
            import traceback

            print(f"Full traceback: {traceback.format_exc()}")
            print("=" * 80)
            raise SystemExit(f"Critical sub-package error: {full_item_name} - {e}")
    else:
        # This item is a module directly under 'mixins' (e.g., mixins/some_other_mixins.py)
        try:
            module = importlib.import_module(full_item_name)
            for class_name, class_obj in inspect.getmembers(module, inspect.isclass):
                if (
                    class_obj.__module__ == full_item_name
                    and class_name.endswith("Mixin")
                    and class_obj is not _BASE_MIXIN_CLASS
                ):
                    if class_obj not in _discovered_other_mixins:
                        _discovered_other_mixins.append(class_obj)
        except ImportError as e:
            print("=" * 80)
            print("🚨 CRITICAL MODULE IMPORT ERROR 🚨")
            print(f"Could not import module '{full_item_name}' for mixin discovery")
            print(f"Error: {e}")
            print("STOPPING EXECUTION - Fix import error before continuing!")
            print("=" * 80)
            raise SystemExit(f"Critical module import error: {full_item_name} - {e}")
        except SyntaxError as e:
            print("=" * 80)
            print("🚨 CRITICAL SYNTAX ERROR IN MIXIN MODULE 🚨")
            print(f"Module: {full_item_name}")
            print(f"File: {e.filename}")
            print(f"Line {e.lineno}: {e.text}")
            print(f"Error: {e}")
            print("STOPPING EXECUTION - Fix syntax error before continuing!")
            print("=" * 80)
            raise SystemExit(
                f"Critical syntax error in module: {e.filename}, line {e.lineno}"
            )
        except Exception as e:
            print("=" * 80)
            print("🚨 CRITICAL MODULE ERROR 🚨")
            print(f"Error inspecting module '{full_item_name}': {e}")
            print("STOPPING EXECUTION - Fix error before continuing!")
            import traceback

            print(f"Full traceback: {traceback.format_exc()}")
            print("=" * 80)
            raise SystemExit(f"Critical module error: {full_item_name} - {e}")

# --- Assemble the Final Tuple of Mixins for Task Inheritance ---

# Sort the discovered other mixins alphabetically by class name for a consistent MRO.
_discovered_other_mixins.sort(key=lambda cls: cls.__name__)

# The final list of mixins starts with BaseMixin (if found and real), then the others.
_final_mixins_list: List[Type[Any]] = []
if _base_mixin_found:  # Add the real BaseMixin if it was successfully imported
    _final_mixins_list.append(_BASE_MIXIN_CLASS)

for mixin_cls in _discovered_other_mixins:
    if mixin_cls not in _final_mixins_list:
        _final_mixins_list.append(mixin_cls)

# This is the full list of individual mixins (BaseMixin + others sorted alphabetically)
_all_individual_mixins_tuple: Tuple[Type[Any], ...] = tuple(_final_mixins_list)

# Run the collision check on the individual mixins *before* attempting to combine them.
_warn_on_method_collisions(_all_individual_mixins_tuple)

# --- Combine all discovered mixins into a single effective mixin for Task ---
_EFFECTIVE_MIXIN_FOR_TASK: Type[
    Any
]  # This will be the single class Task inherits (apart from ABC)

try:
    if not _all_individual_mixins_tuple:
        # Case: No mixins found at all (e.g., if BaseMixin import failed and no others discovered)
        class _EmptyMixinPlaceholder(object):  # Must inherit from object
            pass

        _EFFECTIVE_MIXIN_FOR_TASK = _EmptyMixinPlaceholder
        print(
            "WARNING: No mixins (not even BaseMixin placeholder) were loaded. "
            "Task will use a minimal placeholder and may lack functionality."
        )
    elif len(_all_individual_mixins_tuple) == 1:
        # Case: Only one mixin discovered (e.g., only BaseMixin or its placeholder)
        _EFFECTIVE_MIXIN_FOR_TASK = _all_individual_mixins_tuple[0]
    else:
        # Case: Multiple mixins discovered. Attempt to combine them into a single class.
        # The MRO error, if it occurs, will happen here.
        _EFFECTIVE_MIXIN_FOR_TASK = type(
            "CombinedAutocleanMixins",  # Name of the new dynamically created class
            _all_individual_mixins_tuple,  # Tuple of base classes to inherit from
            {},  # Empty dictionary for class attributes (can add some if needed)
        )
    # If we reached here, _EFFECTIVE_MIXIN_FOR_TASK is a valid class:
    # either the original single mixin, the successfully combined one, or an empty placeholder.

except TypeError as e:
    # This block executes if type(...) creation failed for multiple mixins due to MRO conflict.
    print("-" * 80)
    print("CRITICAL MRO ERROR IN AUTOCLEAN MIXIN SYSTEM:")
    print(
        "Could not create a combined class from all discovered mixins due to a Method Resolution Order (MRO) conflict."
    )
    print(
        "This usually means that the inheritance relationships *between* the mixin classes themselves are incompatible when combined alphabetically."
    )
    print(f"Python error details: {e}")
    print(
        "\nDiscovered individual mixins that were attempted to be combined (BaseMixin first, then others alphabetically):"
    )
    for m_idx, m_cls in enumerate(_all_individual_mixins_tuple):
        print(f"  {m_idx+1}. {m_cls.__name__} (from module: {m_cls.__module__})")
    print("\nPlease inspect the inheritance hierarchies of these mixins.")
    print(
        "Look for conflicting inheritance patterns, e.g., MixinA needing (X then Y) and MixinB needing (Y then X) where X, Y are common bases."
    )
    print(
        "The application will use a placeholder for Task mixins, which will likely cause further errors or missing functionality."
    )
    print("-" * 80)

    class _MroErrorPlaceholderMixin(object):  # Must inherit from object
        _AUTOCLEAN_MIXIN_MRO_ERROR_OCCURRED = True

        # Optional: Add a method to make it obvious in use
        def __getattr__(self, name: str) -> Any:
            if not name.startswith(
                "__"
            ):  # Avoid interfering with special methods too much
                print(
                    f"WARNING: Attribute '{name}' accessed on _MroErrorPlaceholderMixin. Functionality is missing due to MRO failure."
                )
            raise AttributeError(
                f"'{name}' not found; Task mixins failed to load due to MRO error."
            )

        def __init_subclass__(cls, **kwargs):
            print(
                f"WARNING: Class {cls.__name__} is inheriting from _MroErrorPlaceholderMixin due to an earlier MRO failure in autoclean.mixins setup."
            )
            super().__init_subclass__(**kwargs)

    _EFFECTIVE_MIXIN_FOR_TASK = _MroErrorPlaceholderMixin

# DISCOVERED_MIXINS will now be a tuple containing the SINGLE effective class for Task to inherit (plus ABC).
DISCOVERED_MIXINS: Tuple[Type[Any], ...] = (_EFFECTIVE_MIXIN_FOR_TASK,)

# --- Exports for the Package ---
__all__ = ["DISCOVERED_MIXINS"]
if (
    _base_mixin_found
):  # If BaseMixin was real and successfully imported, make it available for direct import.
    __all__.append("BaseMixin")

# Optional: Print the name of the class that Task will actually use from mixins.
# print(f"[autoclean.mixins] Effective class in DISCOVERED_MIXINS for Task inheritance: {_EFFECTIVE_MIXIN_FOR_TASK.__name__}")
