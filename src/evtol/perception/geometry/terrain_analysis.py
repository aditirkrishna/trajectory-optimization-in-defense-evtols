"""
Terrain Analysis Module

This module provides functions for computing terrain features from DEM data:
- Slope computation (central differences, Horn's method)
- Aspect computation
- Curvature computation
- Roughness computation (standard deviation in windows)
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path

# Optional imports for geospatial operations
try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_origin
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

from utils.validation import validate_raster_data
from utils.coordinates import get_crs_info

logger = logging.getLogger(__name__)


class TerrainAnalysisError(Exception):
    """Exception raised for terrain analysis errors."""
    pass


def compute_slope(
    dem: np.ndarray,
    pixel_size: Union[float, Tuple[float, float]] = 1.0,
    method: str = "horn",
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Compute slope from DEM using specified method.
    
    Args:
        dem: Digital Elevation Model as numpy array
        pixel_size: Pixel size in meters (single value or (dx, dy))
        method: Slope computation method ("horn", "central_diff", "sobel")
        output_path: Optional path to save slope raster
        
    Returns:
        Slope array in degrees (0-90)
        
    Raises:
        TerrainAnalysisError: If computation fails
    """
    try:
        validate_raster_data(dem)
        
        if isinstance(pixel_size, (int, float)):
            dx = dy = pixel_size
        else:
            dx, dy = pixel_size
            
        if method == "horn":
            slope = _compute_slope_horn(dem, dx, dy)
        elif method == "central_diff":
            slope = _compute_slope_central_diff(dem, dx, dy)
        elif method == "sobel":
            slope = _compute_slope_sobel(dem, dx, dy)
        else:
            raise TerrainAnalysisError(f"Unknown slope method: {method}")
            
        # Convert to degrees and clip to valid range
        slope_degrees = np.degrees(slope)
        slope_degrees = np.clip(slope_degrees, 0, 90)
        
        if output_path and RASTERIO_AVAILABLE:
            _save_slope_raster(slope_degrees, output_path)
            
        logger.info(f"Computed slope using {method} method")
        return slope_degrees
        
    except Exception as e:
        raise TerrainAnalysisError(f"Failed to compute slope: {str(e)}")


def compute_aspect(
    dem: np.ndarray,
    pixel_size: Union[float, Tuple[float, float]] = 1.0,
    method: str = "horn",
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Compute aspect from DEM using specified method.
    
    Args:
        dem: Digital Elevation Model as numpy array
        pixel_size: Pixel size in meters (single value or (dx, dy))
        method: Aspect computation method ("horn", "central_diff", "sobel")
        output_path: Optional path to save aspect raster
        
    Returns:
        Aspect array in degrees (0-360, 0=N, 90=E, 180=S, 270=W)
        
    Raises:
        TerrainAnalysisError: If computation fails
    """
    try:
        validate_raster_data(dem)
        
        if isinstance(pixel_size, (int, float)):
            dx = dy = pixel_size
        else:
            dx, dy = pixel_size
            
        if method == "horn":
            aspect = _compute_aspect_horn(dem, dx, dy)
        elif method == "central_diff":
            aspect = _compute_aspect_central_diff(dem, dx, dy)
        elif method == "sobel":
            aspect = _compute_aspect_sobel(dem, dx, dy)
        else:
            raise TerrainAnalysisError(f"Unknown aspect method: {method}")
            
        # Convert to degrees and normalize to 0-360
        aspect_degrees = np.degrees(aspect)
        aspect_degrees = np.where(aspect_degrees < 0, aspect_degrees + 360, aspect_degrees)
        
        if output_path and RASTERIO_AVAILABLE:
            _save_aspect_raster(aspect_degrees, output_path)
            
        logger.info(f"Computed aspect using {method} method")
        return aspect_degrees
        
    except Exception as e:
        raise TerrainAnalysisError(f"Failed to compute aspect: {str(e)}")


def compute_curvature(
    dem: np.ndarray,
    pixel_size: Union[float, Tuple[float, float]] = 1.0,
    curvature_type: str = "total",
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Compute curvature from DEM.
    
    Args:
        dem: Digital Elevation Model as numpy array
        pixel_size: Pixel size in meters (single value or (dx, dy))
        curvature_type: Type of curvature ("total", "profile", "plan")
        output_path: Optional path to save curvature raster
        
    Returns:
        Curvature array (positive = convex, negative = concave)
        
    Raises:
        TerrainAnalysisError: If computation fails
    """
    try:
        validate_raster_data(dem)
        
        if isinstance(pixel_size, (int, float)):
            dx = dy = pixel_size
        else:
            dx, dy = pixel_size
            
        if curvature_type == "total":
            curvature = _compute_total_curvature(dem, dx, dy)
        elif curvature_type == "profile":
            curvature = _compute_profile_curvature(dem, dx, dy)
        elif curvature_type == "plan":
            curvature = _compute_plan_curvature(dem, dx, dy)
        else:
            raise TerrainAnalysisError(f"Unknown curvature type: {curvature_type}")
            
        if output_path and RASTERIO_AVAILABLE:
            _save_curvature_raster(curvature, output_path)
            
        logger.info(f"Computed {curvature_type} curvature")
        return curvature
        
    except Exception as e:
        raise TerrainAnalysisError(f"Failed to compute curvature: {str(e)}")


def compute_roughness(
    dem: np.ndarray,
    window_size: int = 3,
    roughness_type: str = "std",
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Compute terrain roughness using sliding window.
    
    Args:
        dem: Digital Elevation Model as numpy array
        window_size: Size of sliding window (must be odd)
        roughness_type: Type of roughness ("std", "range", "iqr")
        output_path: Optional path to save roughness raster
        
    Returns:
        Roughness array
        
    Raises:
        TerrainAnalysisError: If computation fails
    """
    try:
        validate_raster_data(dem)
        
        if window_size % 2 == 0:
            window_size += 1  # Ensure odd window size
            
        if roughness_type == "std":
            roughness = _compute_roughness_std(dem, window_size)
        elif roughness_type == "range":
            roughness = _compute_roughness_range(dem, window_size)
        elif roughness_type == "iqr":
            roughness = _compute_roughness_iqr(dem, window_size)
        else:
            raise TerrainAnalysisError(f"Unknown roughness type: {roughness_type}")
            
        if output_path and RASTERIO_AVAILABLE:
            _save_roughness_raster(roughness, output_path)
            
        logger.info(f"Computed {roughness_type} roughness with {window_size}x{window_size} window")
        return roughness
        
    except Exception as e:
        raise TerrainAnalysisError(f"Failed to compute roughness: {str(e)}")


def compute_terrain_features(
    dem: np.ndarray,
    pixel_size: Union[float, Tuple[float, float]] = 1.0,
    window_size: int = 3,
    output_dir: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Compute all terrain features from DEM.
    
    Args:
        dem: Digital Elevation Model as numpy array
        pixel_size: Pixel size in meters
        window_size: Window size for roughness computation
        output_dir: Optional directory to save all feature rasters
        
    Returns:
        Dictionary containing all terrain features
        
    Raises:
        TerrainAnalysisError: If computation fails
    """
    try:
        features = {}
        
        # Compute basic terrain features
        features['slope'] = compute_slope(dem, pixel_size, method="horn")
        features['aspect'] = compute_aspect(dem, pixel_size, method="horn")
        features['curvature'] = compute_curvature(dem, pixel_size, curvature_type="total")
        features['roughness'] = compute_roughness(dem, window_size, roughness_type="std")
        
        # Save outputs if directory provided
        if output_dir and RASTERIO_AVAILABLE:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            for name, data in features.items():
                file_path = output_path / f"{name}.tif"
                _save_terrain_raster(data, file_path, name)
                
        logger.info("Computed all terrain features")
        return features
        
    except Exception as e:
        raise TerrainAnalysisError(f"Failed to compute terrain features: {str(e)}")


# Internal helper functions

def _compute_slope_horn(dem: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Compute slope using Horn's method."""
    # Horn's method uses 3x3 window with weights
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / (8 * dx)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / (8 * dy)
    
    dz_dx = ndimage.convolve(dem, kernel_x, mode='nearest')
    dz_dy = ndimage.convolve(dem, kernel_y, mode='nearest')
    
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    return slope


def _compute_slope_central_diff(dem: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Compute slope using central differences."""
    dz_dx = np.gradient(dem, axis=1) / dx
    dz_dy = np.gradient(dem, axis=0) / dy
    
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    return slope


def _compute_slope_sobel(dem: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Compute slope using Sobel operators."""
    sobel_x = ndimage.sobel(dem, axis=1) / dx
    sobel_y = ndimage.sobel(dem, axis=0) / dy
    
    slope = np.arctan(np.sqrt(sobel_x**2 + sobel_y**2))
    return slope


def _compute_aspect_horn(dem: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Compute aspect using Horn's method."""
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / (8 * dx)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / (8 * dy)
    
    dz_dx = ndimage.convolve(dem, kernel_x, mode='nearest')
    dz_dy = ndimage.convolve(dem, kernel_y, mode='nearest')
    
    aspect = np.arctan2(dz_dy, dz_dx)
    return aspect


def _compute_aspect_central_diff(dem: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Compute aspect using central differences."""
    dz_dx = np.gradient(dem, axis=1) / dx
    dz_dy = np.gradient(dem, axis=0) / dy
    
    aspect = np.arctan2(dz_dy, dz_dx)
    return aspect


def _compute_aspect_sobel(dem: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Compute aspect using Sobel operators."""
    sobel_x = ndimage.sobel(dem, axis=1) / dx
    sobel_y = ndimage.sobel(dem, axis=0) / dy
    
    aspect = np.arctan2(sobel_y, sobel_x)
    return aspect


def _compute_total_curvature(dem: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Compute total curvature."""
    # Second derivatives
    d2z_dx2 = np.gradient(np.gradient(dem, axis=1), axis=1) / (dx**2)
    d2z_dy2 = np.gradient(np.gradient(dem, axis=0), axis=0) / (dy**2)
    d2z_dxdy = np.gradient(np.gradient(dem, axis=1), axis=0) / (dx * dy)
    
    # First derivatives
    dz_dx = np.gradient(dem, axis=1) / dx
    dz_dy = np.gradient(dem, axis=0) / dy
    
    # Total curvature formula
    p = dz_dx**2 + dz_dy**2
    q = p + 1
    
    curvature = (d2z_dx2 * dz_dy**2 - 2 * d2z_dxdy * dz_dx * dz_dy + d2z_dy2 * dz_dx**2) / (q**1.5)
    return curvature


def _compute_profile_curvature(dem: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Compute profile curvature."""
    # Second derivatives
    d2z_dx2 = np.gradient(np.gradient(dem, axis=1), axis=1) / (dx**2)
    d2z_dy2 = np.gradient(np.gradient(dem, axis=0), axis=0) / (dy**2)
    d2z_dxdy = np.gradient(np.gradient(dem, axis=1), axis=0) / (dx * dy)
    
    # First derivatives
    dz_dx = np.gradient(dem, axis=1) / dx
    dz_dy = np.gradient(dem, axis=0) / dy
    
    # Profile curvature formula
    p = dz_dx**2 + dz_dy**2
    q = p + 1
    
    curvature = (d2z_dx2 * dz_dx**2 + 2 * d2z_dxdy * dz_dx * dz_dy + d2z_dy2 * dz_dy**2) / (p * q**0.5)
    return curvature


def _compute_plan_curvature(dem: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Compute plan curvature."""
    # Second derivatives
    d2z_dx2 = np.gradient(np.gradient(dem, axis=1), axis=1) / (dx**2)
    d2z_dy2 = np.gradient(np.gradient(dem, axis=0), axis=0) / (dy**2)
    d2z_dxdy = np.gradient(np.gradient(dem, axis=1), axis=0) / (dx * dy)
    
    # First derivatives
    dz_dx = np.gradient(dem, axis=1) / dx
    dz_dy = np.gradient(dem, axis=0) / dy
    
    # Plan curvature formula
    p = dz_dx**2 + dz_dy**2
    q = p + 1
    
    curvature = (d2z_dx2 * dz_dy**2 - 2 * d2z_dxdy * dz_dx * dz_dy + d2z_dy2 * dz_dx**2) / (p**1.5)
    return curvature


def _compute_roughness_std(dem: np.ndarray, window_size: int) -> np.ndarray:
    """Compute roughness as standard deviation in window."""
    return ndimage.generic_filter(dem, np.std, size=window_size)


def _compute_roughness_range(dem: np.ndarray, window_size: int) -> np.ndarray:
    """Compute roughness as range in window."""
    def range_func(x):
        return np.max(x) - np.min(x)
    return ndimage.generic_filter(dem, range_func, size=window_size)


def _compute_roughness_iqr(dem: np.ndarray, window_size: int) -> np.ndarray:
    """Compute roughness as interquartile range in window."""
    def iqr_func(x):
        return np.percentile(x, 75) - np.percentile(x, 25)
    return ndimage.generic_filter(dem, iqr_func, size=window_size)


def _save_slope_raster(slope: np.ndarray, output_path: str):
    """Save slope raster to file."""
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=slope.shape[0],
        width=slope.shape[1],
        count=1,
        dtype=slope.dtype
    ) as dst:
        dst.write(slope, 1)


def _save_aspect_raster(aspect: np.ndarray, output_path: str):
    """Save aspect raster to file."""
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=aspect.shape[0],
        width=aspect.shape[1],
        count=1,
        dtype=aspect.dtype
    ) as dst:
        dst.write(aspect, 1)


def _save_curvature_raster(curvature: np.ndarray, output_path: str):
    """Save curvature raster to file."""
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=curvature.shape[0],
        width=curvature.shape[1],
        count=1,
        dtype=curvature.dtype
    ) as dst:
        dst.write(curvature, 1)


def _save_roughness_raster(roughness: np.ndarray, output_path: str):
    """Save roughness raster to file."""
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=roughness.shape[0],
        width=roughness.shape[1],
        count=1,
        dtype=roughness.dtype
    ) as dst:
        dst.write(roughness, 1)


def _save_terrain_raster(data: np.ndarray, output_path: Path, feature_name: str):
    """Save terrain feature raster to file."""
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype
    ) as dst:
        dst.write(data, 1)
        dst.update_tags(feature_name=feature_name)
