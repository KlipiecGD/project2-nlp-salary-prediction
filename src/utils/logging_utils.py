import logging
import sys

logger = logging.getLogger(__name__)


def setup_logging(
    log_file: str = None,
    level: int = logging.INFO,
    log_to_console: bool = True,
    detailed: bool = False,
) -> None:
    """
    Configures logging with optional file and console output.
    The log file is cleared on each run.

    Args:
        log_file: str, The path to the log file. If None, no file logging is performed.
        level: int, The minimum logging level to capture (e.g., logging.INFO, logging.DEBUG).
        log_to_console: bool, If True, logs will also be printed to the console.
        detailed: bool, If True, use a detailed log format.
    """
    # Always clear existing handlers at the start to prevent duplicates.
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set logger level
    logger.setLevel(level)

    # Add console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = (
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            if detailed
            else logging.Formatter("%(levelname)s - %(message)s")
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler with 'w' mode to clear the file on each run
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w")  # 'w' mode to overwrite
        formatter = (
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            if detailed
            else logging.Formatter("%(levelname)s - %(message)s")
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
