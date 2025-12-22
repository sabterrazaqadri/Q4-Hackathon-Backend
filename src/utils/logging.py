import logging
import sys
from typing import Optional


def setup_logging(level: Optional[str] = "INFO") -> logging.Logger:
    """
    Set up logging configuration for the application and return a logger instance
    """
    # Determine the logging level
    log_level = getattr(logging, level.upper()) if level else logging.INFO

    # Create a custom formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a handler that writes to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(formatter)

    # Get the root logger and configure it
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(handler)

    # Set specific loggers to WARNING level to reduce verbosity
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Return a logger for the application
    return logging.getLogger("rag_backend")