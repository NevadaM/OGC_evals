import logging
import sys
import os
from datetime import datetime

def setup_logger(name: str = "ogc_eval", log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger that writes to both console and a timestamped file.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent adding handlers multiple times if function called repeatedly
    if logger.handlers:
        return logger

    # File Handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Writing to {log_file}")
    
    return logger

def get_module_logger(module_name: str) -> logging.Logger:
    """
    Returns a logger for a specific module, inheriting the configuration from the root logger.
    """
    return logging.getLogger(f"ogc_eval.{module_name}")
