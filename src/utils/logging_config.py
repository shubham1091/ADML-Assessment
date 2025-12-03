import logging
import os
from datetime import datetime

# Create logs directory
LOG_DIR = "Data/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Create log file with timestamp
log_filename = os.path.join(LOG_DIR, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Configure logging for root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create file handler
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add handlers to root logger
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

def get_logger(name):
    """Get a logger instance"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
