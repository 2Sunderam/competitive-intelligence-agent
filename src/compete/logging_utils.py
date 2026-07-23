from __future__ import annotations

import logging
import sys
from typing import Any


_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger("compete")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"compete.{name}")


def log_step(logger: logging.Logger, step: str, **fields: Any) -> None:
    parts = [f"step={step}"]
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    logger.info(" | ".join(parts))
