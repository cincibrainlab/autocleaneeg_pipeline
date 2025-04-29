# src/autoclean/plugins/formats/__init__.py
"""Format registrations for AutoClean.

This package contains modules that register file formats for AutoClean.
"""

# Import the core formats to ensure they are registered
try:
    from . import core_formats
except ImportError:
    pass
