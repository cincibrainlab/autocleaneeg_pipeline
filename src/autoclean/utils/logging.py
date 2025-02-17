# src/autoclean/utils/logging.py
"""Logging utilities for the autoclean package."""

import os
import logging
import sys
import warnings
from loguru import logger
from typing import Optional, Union
from enum import Enum

# Remove default handler
logger.remove()

# Add only our custom levels (DEBUG and SUCCESS are built-in)
logger.level("HEADER", no=25, color="<blue>", icon="🧠")
logger.level("VALUES", no=5, color="<cyan>", icon="➤")


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
    - DEBUG = 10
    - INFO = 20
    - WARNING = 30
    - ERROR = 40
    - CRITICAL = 50

    Plus custom levels:
    - VALUES = 5
    - HEADER = 25
    """

    VALUES = "VALUES"  # Custom level for debug values
    DEBUG = "DEBUG"
    HEADER = "HEADER"  # Custom level for section headers
    INFO = "INFO"
    SUCCESS = "SUCCESS"  # Using built-in Loguru SUCCESS level
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

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

    Args:
        level: Log level ('debug', 'info', 'warning', etc.)
        text: Message text to log
        **kwargs: Additional context variables for formatting
    """
    # Convert level to proper case
    level = level.upper()

    # Handle expensive computations lazily
    if kwargs:
        logger.opt(lazy=True).log(level, text, **kwargs)
    else:
        logger.log(level, text)


def configure_logger(verbose: Optional[Union[bool, str, int, LogLevel]] = None) -> None:
    """
    Configure the logger based on verbosity level, following MNE's conventions.

    Args:
        verbose: Controls logging verbosity. Can be:
            - bool: True is the same as 'INFO', False is the same as 'WARNING'
            - str: One of 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'
            - int: Standard Python logging level (10=DEBUG, 20=INFO, etc.)
            - LogLevel enum: Direct log level specification
            - None: Reads MNE_LOGGING_LEVEL environment variable, defaults to INFO
    """
    logger.remove()

    # Convert input to LogLevel using our new conversion method
    level = LogLevel.from_value(verbose)

    # File handler with rotation and retention
    logger.add(
        "logs/autoclean_{time}.log",
        rotation="1 day",  # Rotate daily
        retention="1 week",  # Keep logs for 1 week
        compression="zip",  # Compress rotated logs
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        backtrace=True,  # Better exception logging
        diagnose=True,  # Show variable values in tracebacks
        enqueue=True,  # Thread-safe logging
        colorize=False,  # No colors in file
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


# Initialize with default settings (will check MNE_LOGGING_LEVEL env var)
configure_logger()
