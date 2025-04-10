# src/autoclean/utils/logging.py
"""Logging utilities for the autoclean package."""

import logging
import os
import sys
import warnings
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from loguru import logger

# Remove default handler
logger.remove()

# Define custom levels with specific order
# Standard levels are already defined:
# - DEBUG(10)
# - INFO(20)
# - SUCCESS(25) - Built into loguru
# - WARNING(30)
# - ERROR(40)
# - CRITICAL(50)

# Only define our custom levels
logger.level("HEADER", no=28, color="<blue>", icon="🧠")     # Between SUCCESS and WARNING
logger.level("VALUES", no=5, color="<cyan>", icon="➤")       # Keep lowest level for debug values


# Create a custom warning handler that redirects to loguru
class WarningToLogger:
    def __init__(self):
        self._last_warning = None

    def __call__(self, message, category, filename, lineno, file=None, line=None):
        # Skip duplicate warnings
        warning_key = (str(message), category, filename, lineno)
        if warning_key == self._last_warning:
            return
        self._last_warning = warning_key

        # Format the warning message
        warning_message = f"{category.__name__}: {str(message)}"
        logger.warning(warning_message)


# Set up the warning handler
warning_handler = WarningToLogger()
warnings.showwarning = warning_handler


class LogLevel(str, Enum):
    """Enum for log levels matching MNE's logging levels.

    These levels correspond to Python's standard logging levels:
    - VALUES = 5 (custom debug values)
    - DEBUG = 10
    - INFO = 20
    - SUCCESS = 25 (built into loguru)
    - HEADER = 28 (custom)
    - WARNING = 30
    - ERROR = 40
    - CRITICAL = 50
    """

    VALUES = "VALUES"      # Custom level for debug values (5)
    DEBUG = "DEBUG"        # Standard debug (10)
    INFO = "INFO"         # Standard info (20)
    SUCCESS = "SUCCESS"   # Built-in loguru success level (25)
    HEADER = "HEADER"     # Custom header level (28)
    WARNING = "WARNING"   # Standard warning (30)
    ERROR = "ERROR"       # Standard error (40)
    CRITICAL = "CRITICAL" # Standard critical (50)

    @classmethod
    def from_value(cls, value: Union[str, int, bool, None]) -> "LogLevel":
        """Convert various input types to LogLevel.

        Args:
            value: Input value that can be:
                - str: One of DEBUG, INFO, WARNING, ERROR, or CRITICAL
                - int: Standard Python logging level (10, 20, 30, 40, 50)
                - bool: True for INFO, False for WARNING
                - None: Use MNE_LOGGING_LEVEL env var or default to INFO

        Returns:
            LogLevel: The corresponding log level
        """
        if value is None:
            # Check environment variable first
            env_level = os.getenv("MNE_LOGGING_LEVEL", "INFO")
            return cls.from_value(env_level)

        if isinstance(value, bool):
            return cls.INFO if value else cls.WARNING

        if isinstance(value, int):
            # Map Python's standard logging levels
            level_map = {
                5: cls.VALUES,  # Custom debug values level
                logging.DEBUG: cls.DEBUG,  # 10
                logging.INFO: cls.INFO,  # 20
                logging.WARNING: cls.WARNING,  # 30
                logging.ERROR: cls.ERROR,  # 40
                logging.CRITICAL: cls.CRITICAL,  # 50
            }
            # Find the closest level that's less than or equal to the input
            valid_levels = sorted(level_map.keys())
            for level in reversed(valid_levels):
                if value >= level:
                    return level_map[level]
            return cls.VALUES

        if isinstance(value, str):
            try:
                return cls(value.upper())
            except ValueError:
                return cls.INFO

        return cls.INFO  # Default fallback


class MessageType(str, Enum):
    """Enum for message types with their corresponding log levels and symbols."""

    HEADER = "header"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    VALUES = "values"
    DEBUG = "debug"


def message(level: str, text: str, **kwargs) -> None:
    """
    Enhanced logging function with support for lazy evaluation and context.

    Parameters
    ----------
    level : str
        Log level ('debug', 'info', 'warning', etc.)
    text : str
        Message text to log
    **kwargs
        Additional context variables for formatting
    """
    # Convert level to proper case
    level = level.upper()

    # Handle expensive computations lazily
    if kwargs:
        logger.opt(lazy=True).log(level, text, **kwargs)
    else:
        logger.log(level, text)


def configure_logger(
    verbose: Optional[Union[bool, str, int, LogLevel]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    task: Optional[str] = None
) -> str:
    """
    Configure the logger based on verbosity level and output directory.

    Parameters
    ----------
    verbose : bool, str, int, LogLevel, optional
        Controls logging verbosity. Can be:
        - bool: True is the same as 'INFO', False is the same as 'WARNING'
        - str: One of 'DEBUG', 'INFO', 'HEADER', WARNING', 'ERROR', or 'CRITICAL'
        - int: Standard Python logging level (10=DEBUG, 20=INFO, etc.)
        - LogLevel enum: Direct log level specification
        - None: Reads MNE_LOGGING_LEVEL environment variable, defaults to INFO
    output_dir : str or Path, optional
        Directory where task outputs will be stored
    task : str, optional
        Name of the current task. If provided, logs will be stored in task's debug directory

    Returns
    -------
    str
        Appropriate MNE verbosity level ('DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL')
    """
    logger.remove()

    # Convert input to LogLevel using our new conversion method
    level = LogLevel.from_value(verbose)

    # Map our custom levels to appropriate MNE levels
    mne_level_map = {
        LogLevel.VALUES: "DEBUG",    # Our lowest level maps to DEBUG
        LogLevel.DEBUG: "DEBUG",
        LogLevel.HEADER: "WARNING",  # Headers are for UI, use WARNING for MNE
        LogLevel.INFO: "INFO",
        LogLevel.SUCCESS: "INFO",    # Success messages map to INFO
        LogLevel.WARNING: "WARNING",
        LogLevel.ERROR: "ERROR",
        LogLevel.CRITICAL: "CRITICAL"
    }

    # Set up log directory
    if output_dir is not None and task is not None:
        # Create logs directory in the task's debug directory
        log_dir = Path(output_dir) / task / "logs"
    else:
        # Fallback to current working directory if no task-specific path
        log_dir = Path(os.getcwd()) / "logs"
    
    # Create logs directory
    log_dir.mkdir(parents=True, exist_ok=True)

    # File handler with rotation and retention
    logger.add(
        str(log_dir / "autoclean_{time}.log"),
        rotation="1 day",  # Rotate daily
        retention="1 week",  # Keep logs for 1 week
        compression="zip",  # Compress rotated logs
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        backtrace=True,  # Better exception logging
        diagnose=True,  # Show variable values in tracebacks
        enqueue=True,  # Thread-safe logging
        colorize=True,  # No colors in file
        catch=True,  # Catch errors within handlers
    )

    # Console handler with colors and context
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
        backtrace=True,
        diagnose=True,
        catch=True,
    )

    # Return appropriate MNE verbosity level
    return mne_level_map[level]


# Initialize with default settings (will check MNE_LOGGING_LEVEL env var)
configure_logger()
