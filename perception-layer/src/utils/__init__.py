"""
Utility functions and classes for the perception layer.

This package provides common utilities used across the perception layer
including coordinate transformations, data validation, and helper functions.
"""

from .config import Config, get_config
from .logging import (
    setup_logging, 
    get_logger, 
    log_function_call, 
    log_execution_time, 
    ProgressLogger
)
from .coordinates import (
    transform_coordinates,
    calculate_distance,
    calculate_bearing,
    get_utm_zone
)
from .validation import (
    validate_raster,
    validate_vector,
    validate_coordinates,
    validate_time_range
)
from .file_utils import (
    ensure_directory,
    get_file_info,
    create_temp_file,
    cleanup_temp_files
)

__all__ = [
    # Configuration
    "Config",
    "get_config",
    
    # Logging
    "setup_logging",
    "get_logger", 
    "log_function_call",
    "log_execution_time",
    "ProgressLogger",
    
    # Coordinates
    "transform_coordinates",
    "calculate_distance",
    "calculate_bearing", 
    "get_utm_zone",
    
    # Validation
    "validate_raster",
    "validate_vector",
    "validate_coordinates",
    "validate_time_range",
    
    # File utilities
    "ensure_directory",
    "get_file_info",
    "create_temp_file",
    "cleanup_temp_files",
]

