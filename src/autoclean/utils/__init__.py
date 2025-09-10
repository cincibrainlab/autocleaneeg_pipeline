"""Utility functions and helpers.

This package provides various utility modules for AutoClean.
Import specific modules directly instead of using package-level imports
for better performance:

    from autoclean.utils.config import load_config
    from autoclean.utils.logging import message
    etc.

Package-level imports have been removed to prevent eager loading
of heavy modules like database (2.5s import time).
"""

# Removed eager imports to prevent 2.5s+ startup delays
# Import specific modules directly: from autoclean.utils.module import function

__all__ = [
    # Note: Package-level imports removed for performance
    # Import from specific modules instead:
    # from autoclean.utils.bids import step_convert_to_bids
    # from autoclean.utils.config import load_config  
    # from autoclean.utils.database import get_run_record
    # from autoclean.utils.file_system import step_prepare_directories
    # from autoclean.utils.logging import message
    # from autoclean.utils.montage import VALID_MONTAGES
]
