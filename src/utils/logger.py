"""
Logging Module
Records all trading activity for review and debugging

A black box flight recorder:
- Every trade is recorded
- Every error is logged
- Can review everything later
"""

import logging
import os
from datetime import datetime


def setup_logger(name: str = "trading_bot"):
    """
    Set up a logger that writes to both console and file.

    Two outputs,
    - Console: See what's happening now
    - File: Review later for analysis
    """

    # Create logs folder if it doesn't exist
    log_dir = "data/logs"
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console handler (see in terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # File handler (save to file)
    log_filename = f"{log_dir}/trading_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    # Format for logs
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger