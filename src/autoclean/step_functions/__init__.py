"""Processing step functions."""

import inspect
from . import continuous, epochs, io, reports


# Automatically collect all functions that start with 'step_' from each module
def _collect_step_functions(module):
    return [
        name
        for name, obj in inspect.getmembers(module)
        if inspect.isfunction(obj) and name.startswith("step_")
    ]


__all__ = (
    _collect_step_functions(continuous)
    + _collect_step_functions(epochs)
    + _collect_step_functions(io)
    + _collect_step_functions(reports)
)

# Import all step functions for direct access
from .continuous import *
from .epochs import *
from .io import *
from .reports import *
