"""
Enhanced logging configuration for Fraud Detection System
"""

import logging
import os
from datetime import datetime
from pathlib import Path

# Create logs directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Configure logging
def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create file handler
    log_file = LOG_DIR / f"fraud_detection_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

def get_logger(name):
    """Get a logger instance"""
    return logging.getLogger(name)

# Application loggers
model_logger = get_logger('fraud_detection.model')
data_logger = get_logger('fraud_detection.data')
api_logger = get_logger('fraud_detection.api')
dashboard_logger = get_logger('fraud_detection.dashboard')

# Setup logging on import
setup_logging()
