"""One place to configure logging, so every module prints the same format.

The level comes from the LOG_LEVEL environment variable (default INFO), so
development can show DEBUG messages while production stays quieter.
"""

import logging
import os


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
