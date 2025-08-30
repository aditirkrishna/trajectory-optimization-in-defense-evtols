"""
Logging configuration for the perception layer.

This module sets up comprehensive logging for the perception layer with
proper formatting, file rotation, and different log levels for different
components.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional
import sys


def setup_logging(logging_config: Dict[str, Any], log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging for the perception layer.
    
    Args:
        logging_config: Logging configuration from config file
        log_level: Log level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_file = Path(logging_config["file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler (simple format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=parse_size(logging_config["max_file_size"]),
        backupCount=logging_config["backup_count"],
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # File gets all levels
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Create perception layer logger
    perception_logger = logging.getLogger("perception_layer")
    perception_logger.info("Logging system initialized")
    perception_logger.info(f"Log level: {log_level}")
    perception_logger.info(f"Log file: {log_file}")
    
    return perception_logger


def parse_size(size_str: str) -> int:
    """
    Parse size string (e.g., "10MB", "1GB") to bytes.
    
    Args:
        size_str: Size string with unit
    
    Returns:
        Size in bytes
    """
    size_str = size_str.upper()
    if size_str.endswith('KB'):
        return int(float(size_str[:-2]) * 1024)
    elif size_str.endswith('MB'):
        return int(float(size_str[:-2]) * 1024 * 1024)
    elif size_str.endswith('GB'):
        return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
    else:
        return int(size_str)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific component.
    
    Args:
        name: Logger name (e.g., "preprocessing", "fusion")
    
    Returns:
        Logger instance
    """
    return logging.getLogger(f"perception_layer.{name}")


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls with parameters.
    
    Args:
        logger: Logger instance to use
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} returned {type(result).__name__}")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed with error: {e}")
                raise
        return wrapper
    return decorator


def log_execution_time(logger: logging.Logger):
    """
    Decorator to log function execution time.
    
    Args:
        logger: Logger instance to use
    """
    import time
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.debug(f"Starting {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"{func.__name__} completed in {execution_time:.2f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {execution_time:.2f}s: {e}")
                raise
        return wrapper
    return decorator


class ProgressLogger:
    """
    Logger for progress tracking during long operations.
    """
    
    def __init__(self, logger: logging.Logger, operation_name: str, total_steps: int):
        """
        Initialize progress logger.
        
        Args:
            logger: Logger instance
            operation_name: Name of the operation
            total_steps: Total number of steps
        """
        self.logger = logger
        self.operation_name = operation_name
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = None
        self.last_log_time = 0
        
    def start(self):
        """Start the operation."""
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation_name} ({self.total_steps} steps)")
        
    def step(self, step_name: str = None):
        """Increment progress."""
        self.current_step += 1
        current_time = time.time()
        
        # Log every 10% or every 30 seconds
        progress = self.current_step / self.total_steps
        if (progress * 100) % 10 < (1 / self.total_steps * 100) or (current_time - self.last_log_time) > 30:
            elapsed = current_time - self.start_time
            eta = (elapsed / self.current_step) * (self.total_steps - self.current_step)
            
            step_info = f" - {step_name}" if step_name else ""
            self.logger.info(f"{self.operation_name}: {self.current_step}/{self.total_steps} "
                           f"({progress:.1%}){step_info} - ETA: {eta:.1f}s")
            self.last_log_time = current_time
    
    def finish(self):
        """Finish the operation."""
        total_time = time.time() - self.start_time
        self.logger.info(f"Completed {self.operation_name} in {total_time:.2f}s")


# Import time for ProgressLogger
import time
