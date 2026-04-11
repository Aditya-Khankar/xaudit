"""Structured logging for cognidrift.

Console handler: WARNING+ by default, DEBUG with --debug flag.
File handler: only created when output_dir is specified.
"""

import logging
from pathlib import Path


def setup_logger(output_dir: str | None = None, debug: bool = False) -> logging.Logger:
    """Configure cognidrift logger.

    Args:
        output_dir: If provided, writes cognidrift.log to this directory.
        debug: If True, console shows DEBUG level messages.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("cognidrift")
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug else logging.WARNING)
    console_handler.setFormatter(
        logging.Formatter("%(levelname)s: %(message)s")
    )
    logger.addHandler(console_handler)

    if output_dir:
        log_path = Path(output_dir) / "cognidrift.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    return logger
