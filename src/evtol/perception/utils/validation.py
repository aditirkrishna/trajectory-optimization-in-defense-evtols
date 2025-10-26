"""
Enhanced validation utilities for the perception layer.

This module provides comprehensive validation functions for geospatial data including:
- Raster data validation (metadata, CRS, bounds, resolution)
- Vector data validation (geometry, topology, self-intersections)
- Multi-band raster validation
- GeoTIFF-specific validation
- Data quality validation (statistics, outliers, ranges)
- Spatial consistency checks
- Temporal data validation
- Configuration validation
- Performance validation
"""

import logging
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
from datetime import datetime, timedelta
import warnings

# Optional imports - will be imported when needed
try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.errors import RasterioError
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

try:
    from shapely.geometry import shape, Point, LineString, Polygon
    from shapely.validation import make_valid
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation-related errors."""
    pass


def validate_raster(
    data: np.ndarray,
    expected_shape: Optional[tuple] = None,
    expected_dtype: Optional[np.dtype] = None,
    check_nan: bool = True,
    check_infinite: bool = True
) -> bool:
    """
    Validate raster data array.
    
    Args:
        data: Raster data array
        expected_shape: Expected shape of the array
        expected_dtype: Expected data type
        check_nan: Whether to check for NaN values
        check_infinite: Whether to check for infinite values
        
    Returns:
        True if validation passes
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        # Check if data is numpy array
        if not isinstance(data, np.ndarray):
            raise ValidationError("Data must be a numpy array")
        
        # Check shape
        if expected_shape and data.shape != expected_shape:
            raise ValidationError(f"Expected shape {expected_shape}, got {data.shape}")
        
        # Check dtype
        if expected_dtype and data.dtype != expected_dtype:
            raise ValidationError(f"Expected dtype {expected_dtype}, got {data.dtype}")
        
        # Check for NaN values
        if check_nan and np.any(np.isnan(data)):
            logger.warning("Raster contains NaN values")
        
        # Check for infinite values
        if check_infinite and np.any(np.isinf(data)):
            logger.warning("Raster contains infinite values")
        
        return True
        
    except Exception as e:
        logger.error(f"Raster validation failed: {e}")
        raise ValidationError(f"Raster validation failed: {e}")


def validate_raster_data(
    data: np.ndarray,
    expected_shape: Optional[tuple] = None,
    expected_dtype: Optional[np.dtype] = None,
    check_nan: bool = True,
    check_infinite: bool = True
) -> bool:
    """
    Alias for validate_raster for backward compatibility.
    
    Args:
        data: Raster data array
        expected_shape: Expected shape of the array
        expected_dtype: Expected data type
        check_nan: Whether to check for NaN values
        check_infinite: Whether to check for infinite values
        
    Returns:
        True if validation passes
        
    Raises:
        ValidationError: If validation fails
    """
    return validate_raster(data, expected_shape, expected_dtype, check_nan, check_infinite)


def validate_vector(
    data: Dict[str, Any],
    required_fields: Optional[list] = None,
    geometry_type: Optional[str] = None
) -> bool:
    """
    Validate vector data (GeoJSON-like structure).
    
    Args:
        data: Vector data dictionary
        required_fields: List of required fields
        geometry_type: Expected geometry type
        
    Returns:
        True if validation passes
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        # Check if data is dictionary
        if not isinstance(data, dict):
            raise ValidationError("Vector data must be a dictionary")
        
        # Check required fields
        if required_fields:
            for field in required_fields:
                if field not in data:
                    raise ValidationError(f"Required field '{field}' not found")
        
        # Check geometry type if specified
        if geometry_type and 'type' in data:
            if data['type'] != geometry_type:
                raise ValidationError(f"Expected geometry type '{geometry_type}', got '{data['type']}'")
        
        return True
        
    except Exception as e:
        logger.error(f"Vector validation failed: {e}")
        raise ValidationError(f"Vector validation failed: {e}")


def validate_coordinates(
    lat: float, 
    lon: float, 
    alt: Optional[float] = None,
    check_range: bool = True
) -> bool:
    """
    Validate coordinate values and ranges.
    
    Args:
        lat: Latitude (degrees)
        lon: Longitude (degrees)
        alt: Optional altitude (meters)
        check_range: Whether to check coordinate ranges
        
    Returns:
        True if coordinates are valid
        
    Raises:
        ValidationError: If coordinates are invalid
    """
    try:
        # Check for NaN or infinite values
        if not (np.isfinite(lat) and np.isfinite(lon)):
            raise ValidationError("Coordinates contain NaN or infinite values")
            
        if alt is not None and not np.isfinite(alt):
            raise ValidationError("Altitude contains NaN or infinite values")
            
        if not check_range:
            return True
            
        # Check latitude range (-90 to 90)
        if not (-90 <= lat <= 90):
            raise ValidationError(f"Latitude {lat} is outside valid range [-90, 90]")
            
        # Check longitude range (-180 to 180)
        if not (-180 <= lon <= 180):
            raise ValidationError(f"Longitude {lon} is outside valid range [-180, 180]")
            
        # Check altitude range (reasonable bounds)
        if alt is not None:
            if not (-1000 <= alt <= 50000):  # -1km to 50km
                logger.warning(f"Altitude {alt} is outside typical range [-1000, 50000] meters")
                
        return True
        
    except Exception as e:
        logger.error(f"Coordinate validation failed: {e}")
        raise ValidationError(f"Coordinate validation failed: {e}")


def validate_time_range(
    start_time: Union[datetime, str],
    end_time: Union[datetime, str],
    max_duration: Optional[timedelta] = None
) -> bool:
    """
    Validate time range.
    
    Args:
        start_time: Start time (datetime or ISO string)
        end_time: End time (datetime or ISO string)
        max_duration: Maximum allowed duration
        
    Returns:
        True if time range is valid
        
    Raises:
        ValidationError: If time range is invalid
    """
    try:
        # Convert strings to datetime if needed
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        # Check if start_time is before end_time
        if start_time >= end_time:
            raise ValidationError("Start time must be before end time")
        
        # Check duration if specified
        if max_duration:
            duration = end_time - start_time
            if duration > max_duration:
                raise ValidationError(f"Duration {duration} exceeds maximum {max_duration}")
        
        return True
        
    except Exception as e:
        logger.error(f"Time range validation failed: {e}")
        raise ValidationError(f"Time range validation failed: {e}")


def validate_file_path(
    file_path: str,
    check_exists: bool = True,
    check_readable: bool = True,
    check_writable: bool = False
) -> bool:
    """
    Validate file path.
    
    Args:
        file_path: Path to file
        check_exists: Whether to check if file exists
        check_readable: Whether to check if file is readable
        check_writable: Whether to check if file is writable
        
    Returns:
        True if file path is valid
        
    Raises:
        ValidationError: If file path is invalid
    """
    try:
        import os
        
        # Check if file exists
        if check_exists and not os.path.exists(file_path):
            raise ValidationError(f"File does not exist: {file_path}")
        
        # Check if file is readable
        if check_readable and not os.access(file_path, os.R_OK):
            raise ValidationError(f"File is not readable: {file_path}")
        
        # Check if file is writable
        if check_writable and not os.access(file_path, os.W_OK):
            raise ValidationError(f"File is not writable: {file_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"File path validation failed: {e}")
        raise ValidationError(f"File path validation failed: {e}")


def validate_config(
    config: Dict[str, Any],
    required_keys: Optional[list] = None,
    value_types: Optional[Dict[str, type]] = None
) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        value_types: Dictionary mapping keys to expected types
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValidationError: If configuration is invalid
    """
    try:
        # Check if config is dictionary
        if not isinstance(config, dict):
            raise ValidationError("Configuration must be a dictionary")
        
        # Check required keys
        if required_keys:
            for key in required_keys:
                if key not in config:
                    raise ValidationError(f"Required key '{key}' not found in configuration")
        
        # Check value types
        if value_types:
            for key, expected_type in value_types.items():
                if key in config and not isinstance(config[key], expected_type):
                    raise ValidationError(f"Key '{key}' must be of type {expected_type}, got {type(config[key])}")
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise ValidationError(f"Configuration validation failed: {e}")


# Enhanced Geospatial Validation Functions

def validate_raster_metadata(
    raster_path: str,
    expected_crs: Optional[str] = None,
    expected_bounds: Optional[Tuple[float, float, float, float]] = None,
    expected_resolution: Optional[Tuple[float, float]] = None,
    expected_dtype: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate raster metadata including CRS, bounds, resolution, and data type.
    
    Args:
        raster_path: Path to raster file
        expected_crs: Expected CRS (e.g., "EPSG:32632")
        expected_bounds: Expected bounds (minx, miny, maxx, maxy)
        expected_resolution: Expected resolution (x_res, y_res)
        expected_dtype: Expected data type
        
    Returns:
        Dictionary with validation results and metadata
        
    Raises:
        ValidationError: If validation fails
    """
    if not RASTERIO_AVAILABLE:
        raise ValidationError("rasterio is required for raster metadata validation. Install with: pip install rasterio")
    
    try:
        with rasterio.open(raster_path) as src:
            # Get metadata
            crs = src.crs
            bounds = src.bounds
            resolution = src.res
            dtype = src.dtypes[0]
            shape = src.shape
            count = src.count
            
            validation_results = {
                'valid': True,
                'crs': str(crs),
                'bounds': bounds,
                'resolution': resolution,
                'dtype': dtype,
                'shape': shape,
                'count': count,
                'errors': [],
                'warnings': []
            }
            
            # Validate CRS
            if expected_crs:
                if str(crs) != expected_crs:
                    validation_results['errors'].append(f"CRS mismatch: expected {expected_crs}, got {crs}")
                    validation_results['valid'] = False
            
            # Validate bounds
            if expected_bounds:
                if not np.allclose(bounds, expected_bounds, atol=1e-6):
                    validation_results['errors'].append(f"Bounds mismatch: expected {expected_bounds}, got {bounds}")
                    validation_results['valid'] = False
            
            # Validate resolution
            if expected_resolution:
                if not np.allclose(resolution, expected_resolution, atol=1e-6):
                    validation_results['errors'].append(f"Resolution mismatch: expected {expected_resolution}, got {resolution}")
                    validation_results['valid'] = False
            
            # Validate dtype
            if expected_dtype:
                if dtype != expected_dtype:
                    validation_results['errors'].append(f"Data type mismatch: expected {expected_dtype}, got {dtype}")
                    validation_results['valid'] = False
            
            if not validation_results['valid']:
                raise ValidationError(f"Raster metadata validation failed: {validation_results['errors']}")
            
            return validation_results
            
    except Exception as e:
        logger.error(f"Raster metadata validation failed for {raster_path}: {e}")
        raise ValidationError(f"Raster metadata validation failed: {e}")


def validate_raster_data_quality(
    raster_path: str,
    check_nodata: bool = True,
    check_outliers: bool = True,
    outlier_threshold: float = 3.0,
    expected_range: Optional[Tuple[float, float]] = None
) -> Dict[str, Any]:
    """
    Validate raster data quality including nodata values, outliers, and data ranges.
    
    Args:
        raster_path: Path to raster file
        check_nodata: Whether to check for nodata values
        check_outliers: Whether to check for outliers
        outlier_threshold: Standard deviations for outlier detection
        expected_range: Expected data range (min, max)
        
    Returns:
        Dictionary with quality validation results
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        with rasterio.open(raster_path) as src:
            # Read data
            data = src.read(1)  # Read first band
            
            quality_results = {
                'valid': True,
                'nodata_count': 0,
                'outlier_count': 0,
                'data_range': (float(np.nanmin(data)), float(np.nanmax(data))),
                'data_mean': float(np.nanmean(data)),
                'data_std': float(np.nanstd(data)),
                'errors': [],
                'warnings': []
            }
            
            # Check nodata values
            if check_nodata:
                if src.nodata is not None:
                    nodata_mask = (data == src.nodata)
                    quality_results['nodata_count'] = int(np.sum(nodata_mask))
                    if quality_results['nodata_count'] > 0:
                        quality_results['warnings'].append(f"Found {quality_results['nodata_count']} nodata values")
            
            # Check for NaN values
            nan_count = int(np.sum(np.isnan(data)))
            if nan_count > 0:
                quality_results['warnings'].append(f"Found {nan_count} NaN values")
            
            # Check outliers
            if check_outliers:
                valid_data = data[~np.isnan(data)]
                if len(valid_data) > 0:
                    mean_val = np.mean(valid_data)
                    std_val = np.std(valid_data)
                    outlier_mask = np.abs(valid_data - mean_val) > outlier_threshold * std_val
                    quality_results['outlier_count'] = int(np.sum(outlier_mask))
                    if quality_results['outlier_count'] > 0:
                        quality_results['warnings'].append(f"Found {quality_results['outlier_count']} outliers")
            
            # Check expected range
            if expected_range:
                min_val, max_val = expected_range
                if quality_results['data_range'][0] < min_val or quality_results['data_range'][1] > max_val:
                    quality_results['errors'].append(f"Data range {quality_results['data_range']} outside expected range {expected_range}")
                    quality_results['valid'] = False
            
            if not quality_results['valid']:
                raise ValidationError(f"Data quality validation failed: {quality_results['errors']}")
            
            return quality_results
            
    except Exception as e:
        logger.error(f"Data quality validation failed for {raster_path}: {e}")
        raise ValidationError(f"Data quality validation failed: {e}")


def validate_vector_geometry(
    vector_path: str,
    check_topology: bool = True,
    check_self_intersections: bool = True,
    repair_geometry: bool = False
) -> Dict[str, Any]:
    """
    Validate vector geometry including topology and self-intersections.
    
    Args:
        vector_path: Path to vector file
        check_topology: Whether to check topology
        check_self_intersections: Whether to check for self-intersections
        repair_geometry: Whether to attempt geometry repair
        
    Returns:
        Dictionary with geometry validation results
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        # Read vector data
        gdf = gpd.read_file(vector_path)
        
        geometry_results = {
            'valid': True,
            'total_features': len(gdf),
            'valid_features': 0,
            'invalid_features': 0,
            'self_intersections': 0,
            'errors': [],
            'warnings': []
        }
        
        for idx, row in gdf.iterrows():
            geom = row.geometry
            
            # Check if geometry is valid
            if geom is None:
                geometry_results['invalid_features'] += 1
                geometry_results['errors'].append(f"Feature {idx}: Null geometry")
                continue
            
            # Check for self-intersections
            if check_self_intersections and geom.geom_type in ['Polygon', 'MultiPolygon']:
                if not geom.is_simple:
                    geometry_results['self_intersections'] += 1
                    geometry_results['warnings'].append(f"Feature {idx}: Self-intersection detected")
                    
                    if repair_geometry:
                        try:
                            repaired_geom = make_valid(geom)
                            gdf.at[idx, 'geometry'] = repaired_geom
                            geometry_results['warnings'].append(f"Feature {idx}: Geometry repaired")
                        except Exception as e:
                            geometry_results['errors'].append(f"Feature {idx}: Could not repair geometry: {e}")
            
            # Check topology
            if check_topology:
                if not geom.is_valid:
                    geometry_results['invalid_features'] += 1
                    geometry_results['errors'].append(f"Feature {idx}: Invalid topology")
                else:
                    geometry_results['valid_features'] += 1
        
        # Check overall validity
        if geometry_results['invalid_features'] > 0:
            geometry_results['valid'] = False
            geometry_results['errors'].append(f"{geometry_results['invalid_features']} features have invalid geometry")
        
        if not geometry_results['valid']:
            raise ValidationError(f"Geometry validation failed: {geometry_results['errors']}")
        
        return geometry_results
        
    except Exception as e:
        logger.error(f"Geometry validation failed for {vector_path}: {e}")
        raise ValidationError(f"Geometry validation failed: {e}")


def validate_spatial_consistency(
    raster_path: str,
    vector_path: str,
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Validate spatial consistency between raster and vector data.
    
    Args:
        raster_path: Path to raster file
        vector_path: Path to vector file
        tolerance: Tolerance for spatial comparison
        
    Returns:
        Dictionary with spatial consistency validation results
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        # Read raster bounds
        with rasterio.open(raster_path) as src:
            raster_bounds = src.bounds
            raster_crs = src.crs
        
        # Read vector bounds
        gdf = gpd.read_file(vector_path)
        vector_bounds = gdf.total_bounds
        vector_crs = gdf.crs
        
        consistency_results = {
            'valid': True,
            'raster_bounds': raster_bounds,
            'vector_bounds': vector_bounds,
            'raster_crs': str(raster_crs),
            'vector_crs': str(vector_crs),
            'errors': [],
            'warnings': []
        }
        
        # Check CRS consistency
        if str(raster_crs) != str(vector_crs):
            consistency_results['errors'].append(f"CRS mismatch: raster={raster_crs}, vector={vector_crs}")
            consistency_results['valid'] = False
        
        # Check bounds overlap
        if not (
            raster_bounds[0] <= vector_bounds[2] and
            raster_bounds[2] >= vector_bounds[0] and
            raster_bounds[1] <= vector_bounds[3] and
            raster_bounds[3] >= vector_bounds[1]
        ):
            consistency_results['errors'].append("No spatial overlap between raster and vector data")
            consistency_results['valid'] = False
        
        # Check bounds containment (optional)
        if (
            raster_bounds[0] <= vector_bounds[0] and
            raster_bounds[1] <= vector_bounds[1] and
            raster_bounds[2] >= vector_bounds[2] and
            raster_bounds[3] >= vector_bounds[3]
        ):
            consistency_results['warnings'].append("Vector data is completely contained within raster bounds")
        
        if not consistency_results['valid']:
            raise ValidationError(f"Spatial consistency validation failed: {consistency_results['errors']}")
        
        return consistency_results
        
    except Exception as e:
        logger.error(f"Spatial consistency validation failed: {e}")
        raise ValidationError(f"Spatial consistency validation failed: {e}")


def validate_temporal_data(
    data: Union[pd.DataFrame, pd.Series],
    time_column: str,
    expected_frequency: Optional[str] = None,
    check_gaps: bool = True,
    max_gap_hours: Optional[float] = None
) -> Dict[str, Any]:
    """
    Validate temporal data for consistency and completeness.
    
    Args:
        data: Temporal data (DataFrame or Series)
        time_column: Name of time column
        expected_frequency: Expected frequency (e.g., '1H', '15T')
        check_gaps: Whether to check for temporal gaps
        max_gap_hours: Maximum allowed gap in hours
        
    Returns:
        Dictionary with temporal validation results
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        if isinstance(data, pd.Series):
            time_series = data
        else:
            time_series = data[time_column]
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(time_series):
            time_series = pd.to_datetime(time_series)
        
        temporal_results = {
            'valid': True,
            'start_time': time_series.min(),
            'end_time': time_series.max(),
            'duration': time_series.max() - time_series.min(),
            'total_points': len(time_series),
            'gaps': [],
            'errors': [],
            'warnings': []
        }
        
        # Sort by time
        time_series = time_series.sort_values()
        
        # Check for gaps
        if check_gaps and len(time_series) > 1:
            time_diffs = time_series.diff().dropna()
            
            if expected_frequency:
                expected_timedelta = pd.Timedelta(expected_frequency)
                irregular_intervals = time_diffs[time_diffs != expected_timedelta]
                if len(irregular_intervals) > 0:
                    temporal_results['warnings'].append(f"Found {len(irregular_intervals)} irregular intervals")
            
            if max_gap_hours:
                large_gaps = time_diffs[time_diffs > pd.Timedelta(hours=max_gap_hours)]
                if len(large_gaps) > 0:
                    temporal_results['gaps'] = large_gaps.tolist()
                    temporal_results['warnings'].append(f"Found {len(large_gaps)} gaps larger than {max_gap_hours} hours")
        
        # Check for duplicate timestamps
        duplicates = time_series.duplicated()
        if duplicates.any():
            temporal_results['warnings'].append(f"Found {duplicates.sum()} duplicate timestamps")
        
        # Check for future dates
        future_dates = time_series > pd.Timestamp.now()
        if future_dates.any():
            temporal_results['warnings'].append(f"Found {future_dates.sum()} future timestamps")
        
        return temporal_results
        
    except Exception as e:
        logger.error(f"Temporal data validation failed: {e}")
        raise ValidationError(f"Temporal data validation failed: {e}")


def validate_perception_config(
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate perception layer configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with configuration validation results
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        config_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Required sections
        required_sections = [
            'coordinate_system', 'spatial', 'processing', 
            'terrain', 'urban', 'atmosphere', 'threats'
        ]
        
        for section in required_sections:
            if section not in config:
                config_results['errors'].append(f"Missing required section: {section}")
                config_results['valid'] = False
        
        # Validate coordinate system
        if 'coordinate_system' in config:
            cs_config = config['coordinate_system']
            if 'working_crs' not in cs_config:
                config_results['errors'].append("Missing working_crs in coordinate_system")
                config_results['valid'] = False
            
            # Validate CRS format
            try:
                CRS.from_string(cs_config.get('working_crs', ''))
            except Exception:
                config_results['errors'].append("Invalid working_crs format")
                config_results['valid'] = False
        
        # Validate spatial parameters
        if 'spatial' in config:
            spatial_config = config['spatial']
            if 'base_resolution' not in spatial_config:
                config_results['errors'].append("Missing base_resolution in spatial")
                config_results['valid'] = False
            
            if 'tile_size' not in spatial_config:
                config_results['errors'].append("Missing tile_size in spatial")
                config_results['valid'] = False
            
            # Check reasonable values
            if spatial_config.get('base_resolution', 0) <= 0:
                config_results['errors'].append("base_resolution must be positive")
                config_results['valid'] = False
            
            if spatial_config.get('tile_size', 0) <= 0:
                config_results['errors'].append("tile_size must be positive")
                config_results['valid'] = False
        
        # Validate altitude bands
        if 'spatial' in config and 'altitude_bands' in config['spatial']:
            altitude_bands = config['spatial']['altitude_bands']
            if not isinstance(altitude_bands, list) or len(altitude_bands) < 2:
                config_results['errors'].append("altitude_bands must be a list with at least 2 values")
                config_results['valid'] = False
            
            # Check if bands are in ascending order
            if altitude_bands != sorted(altitude_bands):
                config_results['errors'].append("altitude_bands must be in ascending order")
                config_results['valid'] = False
        
        if not config_results['valid']:
            raise ValidationError(f"Configuration validation failed: {config_results['errors']}")
        
        return config_results
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise ValidationError(f"Configuration validation failed: {e}")


def validate_performance_metrics(
    file_size_mb: float,
    processing_time_seconds: float,
    memory_usage_mb: float,
    max_file_size_mb: Optional[float] = None,
    max_processing_time_seconds: Optional[float] = None,
    max_memory_usage_mb: Optional[float] = None
) -> Dict[str, Any]:
    """
    Validate performance metrics against thresholds.
    
    Args:
        file_size_mb: File size in MB
        processing_time_seconds: Processing time in seconds
        memory_usage_mb: Memory usage in MB
        max_file_size_mb: Maximum allowed file size
        max_processing_time_seconds: Maximum allowed processing time
        max_memory_usage_mb: Maximum allowed memory usage
        
    Returns:
        Dictionary with performance validation results
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        performance_results = {
            'valid': True,
            'file_size_mb': file_size_mb,
            'processing_time_seconds': processing_time_seconds,
            'memory_usage_mb': memory_usage_mb,
            'errors': [],
            'warnings': []
        }
        
        # Check file size
        if max_file_size_mb and file_size_mb > max_file_size_mb:
            performance_results['errors'].append(f"File size {file_size_mb:.2f} MB exceeds limit {max_file_size_mb} MB")
            performance_results['valid'] = False
        
        # Check processing time
        if max_processing_time_seconds and processing_time_seconds > max_processing_time_seconds:
            performance_results['errors'].append(f"Processing time {processing_time_seconds:.2f}s exceeds limit {max_processing_time_seconds}s")
            performance_results['valid'] = False
        
        # Check memory usage
        if max_memory_usage_mb and memory_usage_mb > max_memory_usage_mb:
            performance_results['errors'].append(f"Memory usage {memory_usage_mb:.2f} MB exceeds limit {max_memory_usage_mb} MB")
            performance_results['valid'] = False
        
        # Add warnings for high usage
        if max_file_size_mb and file_size_mb > 0.8 * max_file_size_mb:
            performance_results['warnings'].append(f"File size {file_size_mb:.2f} MB is approaching limit {max_file_size_mb} MB")
        
        if max_processing_time_seconds and processing_time_seconds > 0.8 * max_processing_time_seconds:
            performance_results['warnings'].append(f"Processing time {processing_time_seconds:.2f}s is approaching limit {max_processing_time_seconds}s")
        
        if max_memory_usage_mb and memory_usage_mb > 0.8 * max_memory_usage_mb:
            performance_results['warnings'].append(f"Memory usage {memory_usage_mb:.2f} MB is approaching limit {max_memory_usage_mb} MB")
        
        if not performance_results['valid']:
            raise ValidationError(f"Performance validation failed: {performance_results['errors']}")
        
        return performance_results
        
    except Exception as e:
        logger.error(f"Performance validation failed: {e}")
        raise ValidationError(f"Performance validation failed: {e}")
