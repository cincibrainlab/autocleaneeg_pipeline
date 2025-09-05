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

from autoclean import __version__
from autoclean.utils.user_config import UserConfigManager

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
logger.level("HEADER", no=28, color="<blue>", icon="🧠")  # Between SUCCESS and WARNING


# Create a custom warning handler that redirects to loguru
class WarningToLogger:
    """Custom warning handler that redirects warnings to loguru."""

    def __init__(self):
        """Initialize the warning handler."""
        self._last_warning = None

    def __call__(
        self, warning_message, category, filename, lineno, file=None, line=None
    ):
        """Call the warning handler."""
        # Skip duplicate warnings
        warning_key = (str(warning_message), category, filename, lineno)
        if warning_key == self._last_warning:
            return
        self._last_warning = warning_key

        # Format the warning message
        warning_message = f"{category.__name__}: {str(warning_message)}"
        logger.warning(warning_message)


# Set up the warning handler
warning_handler = WarningToLogger()
warnings.showwarning = warning_handler


class LogLevel(str, Enum):
    """Enum for log levels matching MNE's logging levels.

    These levels correspond to Python's standard logging levels plus custom levels.

    .. rubric:: Standard Levels

    - DEBUG = 10
    - INFO = 20
    - WARNING = 30
    - ERROR = 40
    - CRITICAL = 50

    .. rubric:: Custom Levels

    - HEADER = 28 (Custom header level)
    - SUCCESS = 25 (Built-in Loguru success level)

    .. note::
        This enum is for internal use only and should not be directly accessed.
        Use the message() function instead.

    """

    # Hide these values from documentation
    #: Standard debug (10)
    DEBUG = "DEBUG"
    #: Standard info (20)
    INFO = "INFO"
    #: Built-in loguru success level (25)
    SUCCESS = "SUCCESS"
    #: Custom header level (28)
    HEADER = "HEADER"
    #: Standard warning (30)
    WARNING = "WARNING"
    #: Standard error (40)
    ERROR = "ERROR"
    #: Standard critical (50)
    CRITICAL = "CRITICAL"

    @classmethod
    def from_value(cls, value: Union[str, int, bool, None]) -> "LogLevel":
        """Convert various input types to LogLevel.

        Parameters
        ----------
        value : Union[str, int, bool, None]
            Input value that can be:
                - **str**: One of DEBUG, INFO, WARNING, ERROR, or CRITICAL
                - **int**: Standard Python logging level (10, 20, 30, 40, 50)
                - **bool**: True for INFO, False for WARNING
                - **None**: Use MNE_LOGGING_LEVEL env var or default to INFO

        Returns
        -------
        LogLevel : LogLevel
            The corresponding log level
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

        if isinstance(value, str):
            try:
                return cls(value.upper())
            except ValueError:
                return cls.INFO

        return cls.INFO  # Default fallback


class MessageType(str, Enum):
    """Enum for message types with their corresponding log levels and symbols."""

    ERROR = "error"
    WARNING = "warning"
    HEADER = "header"
    SUCCESS = "success"
    INFO = "info"
    DEBUG = "debug"


def message(level: str, text: str, **kwargs) -> None:
    """
    Enhanced logging function with support for lazy evaluation and context.
    Outputs to the console and the log file.

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
    task: Optional[str] = None,
    logs_dir: Optional[Union[str, Path]] = None,
) -> str:
    """
    Configure the logger based on verbosity level and output directory.

    Parameters
    ----------
    verbose : bool, str, int, LogLevel, optional
        Controls logging verbosity. Can be:

        - **bool**: True is the same as 'INFO', False is the same as 'WARNING'
        - **str**: One of 'DEBUG', 'INFO', 'HEADER', WARNING', 'ERROR', or 'CRITICAL'
        - **int**: Standard Python logging level (10=DEBUG, 20=INFO, etc.)
        - **LogLevel enum**: Direct log level specification
        - **None**: Reads MNE_LOGGING_LEVEL environment variable, defaults to INFO
    output_dir : str or Path, optional
        Directory where task outputs will be stored (legacy parameter, prefer logs_dir)
    task : str, optional
        Name of the current task (legacy parameter, used for fallback directory structure)
    logs_dir : str or Path, optional
        Exact path to the logs directory. If provided, this takes precedence over
        output_dir and task parameters. If not specified, logs are written to the
        AutoClean workspace (e.g., ``~/Documents/Autoclean-EEG/logs``).

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
        LogLevel.DEBUG: "DEBUG",
        LogLevel.INFO: "INFO",
        LogLevel.SUCCESS: "INFO",  # Success messages map to INFO
        LogLevel.HEADER: "WARNING",  # Headers are for UI, uses WARNING for MNE
        LogLevel.WARNING: "WARNING",
        LogLevel.ERROR: "ERROR",
        LogLevel.CRITICAL: "CRITICAL",
    }

    # Set up log directory using correct structure
    if logs_dir is not None:
        # Use the exact logs directory provided (preferred method)
        log_dir = Path(logs_dir)
    elif output_dir is not None and task is not None:
        # Legacy fallback: try to reconstruct path (may not work with dataset names)
        log_dir = (
            Path(output_dir)
            / task
            / "bids"
            / "derivatives"
            / f"autoclean-v{__version__}"
            / "logs"
        )
    else:
        # Default to the AutoClean workspace if no task-specific path is provided
        log_dir = UserConfigManager().config_dir / "logs"

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
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",  # pylint: disable=line-too-long
        colorize=True,
        backtrace=True,
        diagnose=True,
        catch=True,
    )

    # Return appropriate MNE verbosity level
    return mne_level_map[level]


# Initialize with default settings (will check MNE_LOGGING_LEVEL env var)
configure_logger()
