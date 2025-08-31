"""
Clearance Analysis Module

This module provides functions for computing clearance-related features:
- Vertical clearance from ground to flight altitudes
- Obstacle height computation (DSM - DTM)
- Landing zone identification
- Corridor clearance analysis
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path

# Optional imports for geospatial operations
try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Import utilities with absolute paths
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.validation import validate_raster_data, validate_coordinates
from utils.coordinates import transform_coordinates

logger = logging.getLogger(__name__)


class ClearanceAnalysisError(Exception):
    """Exception raised for clearance analysis errors."""
    pass


def compute_clearance(
    dem: np.ndarray,
    flight_altitude: float,
    safety_margin: float = 10.0,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Compute vertical clearance from ground to flight altitude.
    
    Args:
        dem: Digital Elevation Model as numpy array
        flight_altitude: Flight altitude above ground level (meters)
        safety_margin: Safety margin to add to clearance (meters)
        output_path: Optional path to save clearance raster
        
    Returns:
        Clearance array (positive = sufficient clearance, negative = insufficient)
        
    Raises:
        ClearanceAnalysisError: If computation fails
    """
    try:
        validate_raster_data(dem)
        
        # Compute clearance: flight_altitude - (dem - min_dem)
        min_elevation = np.nanmin(dem)
        clearance = flight_altitude - (dem - min_elevation) - safety_margin
        
        if output_path and RASTERIO_AVAILABLE:
            _save_clearance_raster(clearance, output_path)
            
        logger.info(f"Computed clearance for {flight_altitude}m flight altitude")
        return clearance
        
    except Exception as e:
        raise ClearanceAnalysisError(f"Failed to compute clearance: {str(e)}")


def compute_obstacle_height(
    dsm: np.ndarray,
    dtm: np.ndarray,
    min_height_threshold: float = 2.0,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Compute obstacle height as DSM - DTM.
    
    Args:
        dsm: Digital Surface Model (includes buildings, trees, etc.)
        dtm: Digital Terrain Model (ground only)
        min_height_threshold: Minimum height to consider as obstacle (meters)
        output_path: Optional path to save obstacle height raster
        
    Returns:
        Obstacle height array (DSM - DTM)
        
    Raises:
        ClearanceAnalysisError: If computation fails
    """
    try:
        validate_raster_data(dsm)
        validate_raster_data(dtm)
        
        if dsm.shape != dtm.shape:
            raise ClearanceAnalysisError("DSM and DTM must have the same shape")
            
        # Compute obstacle height
        obstacle_height = dsm - dtm
        
        # Apply threshold
        obstacle_height = np.where(obstacle_height >= min_height_threshold, 
                                 obstacle_height, 0)
        
        if output_path and RASTERIO_AVAILABLE:
            _save_obstacle_height_raster(obstacle_height, output_path)
            
        logger.info(f"Computed obstacle heights with {min_height_threshold}m threshold")
        return obstacle_height
        
    except Exception as e:
        raise ClearanceAnalysisError(f"Failed to compute obstacle height: {str(e)}")


def compute_landing_zones(
    dem: np.ndarray,
    slope_threshold: float = 15.0,
    roughness_threshold: float = 2.0,
    min_area: int = 100,
    pixel_size: float = 1.0,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Identify potential landing zones based on terrain characteristics.
    
    Args:
        dem: Digital Elevation Model as numpy array
        slope_threshold: Maximum acceptable slope in degrees
        roughness_threshold: Maximum acceptable roughness in meters
        min_area: Minimum area for landing zone in square meters
        pixel_size: Pixel size in meters
        output_path: Optional path to save landing zone raster
        
    Returns:
        Dictionary containing landing zone data and statistics
        
    Raises:
        ClearanceAnalysisError: If computation fails
    """
    try:
        validate_raster_data(dem)
        
        # Import terrain analysis functions
        from .terrain_analysis import compute_slope, compute_roughness
        
        # Compute terrain features
        slope = compute_slope(dem, pixel_size, method="horn")
        roughness = compute_roughness(dem, window_size=5, roughness_type="std")
        
        # Create landing zone mask
        slope_mask = slope <= slope_threshold
        roughness_mask = roughness <= roughness_threshold
        
        # Combine masks
        landing_mask = slope_mask & roughness_mask
        
        # Remove small areas
        if SCIPY_AVAILABLE:
            labeled_mask, num_features = ndimage.label(landing_mask)
            for i in range(1, num_features + 1):
                area = np.sum(labeled_mask == i) * (pixel_size ** 2)
                if area < min_area:
                    landing_mask[labeled_mask == i] = False
        
        # Compute statistics
        total_area = np.sum(landing_mask) * (pixel_size ** 2)
        num_zones = np.sum(landing_mask)
        
        # Find zone centroids
        if SCIPY_AVAILABLE and num_zones > 0:
            labeled_mask, num_features = ndimage.label(landing_mask)
            centroids = []
            for i in range(1, num_features + 1):
                coords = np.where(labeled_mask == i)
                centroid_y = np.mean(coords[0])
                centroid_x = np.mean(coords[1])
                centroids.append((centroid_y, centroid_x))
        else:
            centroids = []
        
        result = {
            'landing_mask': landing_mask,
            'total_area': total_area,
            'num_zones': num_zones,
            'centroids': centroids,
            'slope': slope,
            'roughness': roughness,
            'slope_threshold': slope_threshold,
            'roughness_threshold': roughness_threshold
        }
        
        if output_path and RASTERIO_AVAILABLE:
            _save_landing_zones_raster(landing_mask, output_path)
            
        logger.info(f"Identified {num_zones} landing zones with {total_area:.1f}m² total area")
        return result
        
    except Exception as e:
        raise ClearanceAnalysisError(f"Failed to compute landing zones: {str(e)}")


def compute_corridor_clearance(
    dem: np.ndarray,
    corridor_path: List[Tuple[float, float]],
    corridor_width: float = 50.0,
    flight_altitude: float = 100.0,
    safety_margin: float = 10.0,
    pixel_size: float = 1.0,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compute clearance along a flight corridor.
    
    Args:
        dem: Digital Elevation Model as numpy array
        corridor_path: List of (x, y) coordinates defining corridor path
        corridor_width: Width of corridor in meters
        flight_altitude: Flight altitude above ground level (meters)
        safety_margin: Safety margin to add to clearance (meters)
        pixel_size: Pixel size in meters
        output_path: Optional path to save corridor clearance raster
        
    Returns:
        Dictionary containing corridor clearance data
        
    Raises:
        ClearanceAnalysisError: If computation fails
    """
    try:
        validate_raster_data(dem)
        
        if len(corridor_path) < 2:
            raise ClearanceAnalysisError("Corridor path must have at least 2 points")
            
        # Create corridor mask
        corridor_mask = _create_corridor_mask(
            dem.shape, corridor_path, corridor_width, pixel_size
        )
        
        # Compute clearance within corridor
        min_elevation = np.nanmin(dem)
        clearance = flight_altitude - (dem - min_elevation) - safety_margin
        
        # Apply corridor mask
        corridor_clearance = np.where(corridor_mask, clearance, np.nan)
        
        # Compute statistics
        valid_clearance = corridor_clearance[~np.isnan(corridor_clearance)]
        
        if len(valid_clearance) > 0:
            min_clearance = np.min(valid_clearance)
            max_clearance = np.max(valid_clearance)
            mean_clearance = np.mean(valid_clearance)
            clearance_std = np.std(valid_clearance)
            
            # Identify critical points (low clearance)
            critical_threshold = 5.0  # meters
            critical_points = valid_clearance < critical_threshold
            num_critical = np.sum(critical_points)
        else:
            min_clearance = max_clearance = mean_clearance = clearance_std = np.nan
            num_critical = 0
        
        result = {
            'corridor_clearance': corridor_clearance,
            'corridor_mask': corridor_mask,
            'min_clearance': min_clearance,
            'max_clearance': max_clearance,
            'mean_clearance': mean_clearance,
            'clearance_std': clearance_std,
            'num_critical_points': num_critical,
            'corridor_length': _compute_path_length(corridor_path, pixel_size),
            'corridor_width': corridor_width
        }
        
        if output_path and RASTERIO_AVAILABLE:
            _save_corridor_clearance_raster(corridor_clearance, output_path)
            
        logger.info(f"Computed corridor clearance: min={min_clearance:.1f}m, "
                   f"mean={mean_clearance:.1f}m, critical_points={num_critical}")
        return result
        
    except Exception as e:
        raise ClearanceAnalysisError(f"Failed to compute corridor clearance: {str(e)}")


# Internal helper functions

def _create_corridor_mask(
    dem_shape: Tuple[int, int],
    corridor_path: List[Tuple[float, float]],
    corridor_width: float,
    pixel_size: float
) -> np.ndarray:
    """Create a mask for the flight corridor."""
    mask = np.zeros(dem_shape, dtype=bool)
    
    # Convert corridor width to pixels
    width_pixels = int(corridor_width / pixel_size)
    
    # For each segment in the corridor path
    for i in range(len(corridor_path) - 1):
        start = corridor_path[i]
        end = corridor_path[i + 1]
        
        # Convert to pixel coordinates
        start_pixel = (int(start[1] / pixel_size), int(start[0] / pixel_size))
        end_pixel = (int(end[1] / pixel_size), int(end[0] / pixel_size))
        
        # Create line segment
        line_coords = _get_line_coordinates(start_pixel, end_pixel)
        
        # Expand line to corridor width
        for y, x in line_coords:
            if 0 <= y < dem_shape[0] and 0 <= x < dem_shape[1]:
                # Create rectangular corridor around line
                y_min = max(0, y - width_pixels // 2)
                y_max = min(dem_shape[0], y + width_pixels // 2 + 1)
                x_min = max(0, x - width_pixels // 2)
                x_max = min(dem_shape[1], x + width_pixels // 2 + 1)
                
                mask[y_min:y_max, x_min:x_max] = True
    
    return mask


def _get_line_coordinates(
    start: Tuple[int, int],
    end: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """Get all pixel coordinates along a line using Bresenham's algorithm."""
    x0, y0 = start
    x1, y1 = end
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    
    if dx > dy:
        return _get_line_coordinates_x(x0, y0, x1, y1)
    else:
        return _get_line_coordinates_y(x0, y0, x1, y1)


def _get_line_coordinates_x(
    x0: int, y0: int, x1: int, y1: int
) -> List[Tuple[int, int]]:
    """Get line coordinates when dx > dy."""
    coords = []
    dx = x1 - x0
    dy = y1 - y0
    
    step_y = 1 if dy >= 0 else -1
    dy = abs(dy)
    
    error = 2 * dy - dx
    y = y0
    
    for x in range(x0, x1 + 1):
        coords.append((y, x))
        if error > 0:
            y += step_y
            error -= 2 * dx
        error += 2 * dy
    
    return coords


def _get_line_coordinates_y(
    x0: int, y0: int, x1: int, y1: int
) -> List[Tuple[int, int]]:
    """Get line coordinates when dy > dx."""
    coords = []
    dx = x1 - x0
    dy = y1 - y0
    
    step_x = 1 if dx >= 0 else -1
    dx = abs(dx)
    
    error = 2 * dx - dy
    x = x0
    
    for y in range(y0, y1 + 1):
        coords.append((y, x))
        if error > 0:
            x += step_x
            error -= 2 * dy
        error += 2 * dx
    
    return coords


def _compute_path_length(
    path: List[Tuple[float, float]],
    pixel_size: float
) -> float:
    """Compute the total length of a path."""
    total_length = 0.0
    
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]
        
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        segment_length = np.sqrt(dx**2 + dy**2) * pixel_size
        total_length += segment_length
    
    return total_length


def _save_clearance_raster(clearance: np.ndarray, output_path: str):
    """Save clearance raster to file."""
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=clearance.shape[0],
        width=clearance.shape[1],
        count=1,
        dtype=clearance.dtype
    ) as dst:
        dst.write(clearance, 1)


def _save_obstacle_height_raster(obstacle_height: np.ndarray, output_path: str):
    """Save obstacle height raster to file."""
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=obstacle_height.shape[0],
        width=obstacle_height.shape[1],
        count=1,
        dtype=obstacle_height.dtype
    ) as dst:
        dst.write(obstacle_height, 1)


def _save_landing_zones_raster(landing_mask: np.ndarray, output_path: str):
    """Save landing zones raster to file."""
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=landing_mask.shape[0],
        width=landing_mask.shape[1],
        count=1,
        dtype=landing_mask.dtype
    ) as dst:
        dst.write(landing_mask.astype(np.uint8), 1)


def _save_corridor_clearance_raster(corridor_clearance: np.ndarray, output_path: str):
    """Save corridor clearance raster to file."""
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=corridor_clearance.shape[0],
        width=corridor_clearance.shape[1],
        count=1,
        dtype=corridor_clearance.dtype
    ) as dst:
        dst.write(corridor_clearance, 1)
