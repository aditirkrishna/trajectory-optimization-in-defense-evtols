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
    get_utm_zone,
    validate_coordinates,
    create_bounding_box,
    interpolate_coordinates,
    get_crs_info,
    convert_units,
    wgs84_to_utm,
    utm_to_wgs84,
    haversine_distance,
    geodesic_distance,
    CoordinateError
)
from .validation import (
    validate_raster,
    validate_vector,
    validate_coordinates,
    validate_time_range,
    validate_file_path,
    validate_config,
    # Enhanced geospatial validation functions
    validate_raster_metadata,
    validate_raster_data_quality,
    validate_vector_geometry,
    validate_spatial_consistency,
    validate_temporal_data,
    validate_perception_config,
    validate_performance_metrics,
    ValidationError
)
from .file_utils import (
    ensure_directory,
    get_file_info,
    create_temp_file,
    cleanup_temp_files,
    cleanup_temp_directories,
    copy_file,
    move_file,
    list_files,
    get_file_size_mb,
    save_json,
    load_json,
    # Enhanced geospatial file utilities
    get_geospatial_file_info,
    compress_file,
    decompress_file,
    batch_process_files,
    validate_file_integrity,
    create_file_manifest,
    optimize_geospatial_file,
    get_supported_formats,
    FileUtilsError
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
    "validate_coordinates",
    "create_bounding_box",
    "interpolate_coordinates",
    "get_crs_info",
    "convert_units",
    "wgs84_to_utm",
    "utm_to_wgs84",
    "haversine_distance",
    "geodesic_distance",
    "CoordinateError",
    
    # Validation
    "validate_raster",
    "validate_vector",
    "validate_coordinates",
    "validate_time_range",
    "validate_file_path",
    "validate_config",
    # Enhanced geospatial validation
    "validate_raster_metadata",
    "validate_raster_data_quality",
    "validate_vector_geometry",
    "validate_spatial_consistency",
    "validate_temporal_data",
    "validate_perception_config",
    "validate_performance_metrics",
    "ValidationError",
    
    # File utilities
    "ensure_directory",
    "get_file_info",
    "create_temp_file",
    "cleanup_temp_files",
    "cleanup_temp_directories",
    "copy_file",
    "move_file",
    "list_files",
    "get_file_size_mb",
    "save_json",
    "load_json",
    # Enhanced geospatial file utilities
    "get_geospatial_file_info",
    "compress_file",
    "decompress_file",
    "batch_process_files",
    "validate_file_integrity",
    "create_file_manifest",
    "optimize_geospatial_file",
    "get_supported_formats",
    "FileUtilsError",
]

