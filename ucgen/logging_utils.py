from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator


def setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("ucgen").setLevel(lvl)


@contextmanager
def time_block(logger: logging.Logger, title: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000
        logger.info("%s done in %.1f ms", title, dt)
