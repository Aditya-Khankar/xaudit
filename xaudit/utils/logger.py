import logging
from pathlib import Path
from rich.logging import RichHandler


def setup_logger(output_dir: str | None = None, debug: bool = False) -> logging.Logger:
    """Configure XAudit logger using Rich for premium console output.

    Args:
        output_dir: If provided, writes xaudit.log (standard text) to this directory.
        debug: If True, console shows DEBUG level messages with Rich styling.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("xaudit")
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # Console Handler (Rich)
    rich_handler = RichHandler(
        level=logging.DEBUG if debug else logging.WARNING,
        console=None,  # Uses default rich console
        show_path=False,
        markup=True,
    )
    logger.addHandler(rich_handler)

    # File Handler (Standard Text)
    if output_dir:
        log_path = Path(output_dir) / "xaudit.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    return logger
