#!/usr/bin/env python3
"""
TASNI Logging Configuration with structlog
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import structlog
from structlog.dev import ConsoleRenderer
from structlog.processors import TimeStamper, add_log_level
from structlog.stdlib import ProcessorFormatter


def make_logger(
    name: str = "tasni",
    level: str = "INFO",
    log_file: Path | None = None,
) -> structlog.stdlib.BoundLogger:
    """
    Create structured logger with processors.
    """
    timestamp_processor = TimeStamper(fmt="iso", key="timestamp")
    level_processor = add_log_level

    processors = [
        timestamp_processor,
        level_processor,
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(pad_level=0),
    ]

    formatter = ProcessorFormatter(
        processor=processors,
        foreign_pre_chain=[structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
    )

    logger = structlog.get_logger("tasni." + name)

    # Console handler
    console_handler = structlog.stdlib.RendererFormatter(
        renderer=ConsoleRenderer(pad_level=0),
        wrapper_class=structlog.stdlib.BoundLogger,
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(console_handler)
    logger.addHandler(ch)
    logger.setLevel(level.upper())

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            ProcessorFormatter(
                foreign_pre_chain=[
                    structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
                ],
                processor=structlog.processors.JSONRenderer(),
            )
        )
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get TASNI structured logger by name."""
    return structlog.get_logger(f"tasni.{name}")


class LogMixin:
    """Mixin class that adds structured logging to any class."""

    @property
    def log(self) -> structlog.stdlib.BoundLogger:
        return get_logger(self.__class__.__name__.lower())


def init_script_logging(
    script_name: str, log_dir: Path | None = None
) -> structlog.stdlib.BoundLogger:
    """Initialize structured logging for a script."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = None
    if log_dir:
        log_dir = Path(log_dir)
        log_file = log_dir / f"{script_name}_{timestamp}.log"

    return make_logger(name=script_name, log_file=log_file)
