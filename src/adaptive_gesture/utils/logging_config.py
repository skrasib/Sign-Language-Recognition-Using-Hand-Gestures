from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(
    log_directory: str | Path,
    level: int = logging.INFO,
) -> Path:
    """Configure console + rotating-file logging once for the application.

    The log directory is created automatically. Repeated calls are safe and do
    not attach duplicate handlers. The returned path is the active log file.
    """

    log_directory = Path(log_directory)
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / "adaptive_gesture.log"

    root_logger = logging.getLogger()
    if getattr(root_logger, "_adaptive_gesture_configured", False):
        return log_path

    root_logger.setLevel(level)
    formatter = logging.Formatter(DEFAULT_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger._adaptive_gesture_configured = True

    return log_path
