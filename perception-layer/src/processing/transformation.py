"""
Data transformation utilities for the perception layer.

This module provides functions for transforming geospatial data including:
- Coordinate system reprojection
- Data resampling and interpolation
- Spatial clipping and masking
- Data normalization and scaling
"""

import logging
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np

# Optional imports for geospatial data handling
try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.mask import mask
    from rasterio.features import geometry_mask
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import geopandas as gpd
    from shapely.geometry import box, Point, LineString, Polygon
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

# Import utilities with absolute paths
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.coordinates import transform_coordinates
from utils.validation import validate_coordinates

logger = logging.getLogger(__name__)


class DataTransformationError(Exception):
    """Custom exception for data transformation errors."""
    pass


def reproject_raster(
    raster_data: Dict[str, Any],
    target_crs: str,
    target_resolution: Optional[Tuple[float, float]] = None,
    resampling_method: str = 'bilinear'
) -> Dict[str, Any]:
    """
    Reproject raster data to a different coordinate reference system.
    
    Args:
        raster_data: Dictionary containing raster data (from load_raster_data)
        target_crs: Target coordinate reference system
        target_resolution: Target resolution (x_res, y_res) in target CRS units
        resampling_method: Resampling method ('nearest', 'bilinear', 'cubic', etc.)
        
    Returns:
        Dictionary containing reprojected raster data
        
    Raises:
        DataTransformationError: If reprojection fails
    """
    try:
        if not RASTERIO_AVAILABLE:
            raise DataTransformationError("rasterio is not available")
        
        data = raster_data['data']
        metadata = raster_data['metadata']
        source_crs = metadata['crs']
        
        # Check if reprojection is needed
        if str(source_crs) == target_crs:
            logger.info(f"Raster already in target CRS: {target_crs}")
            return raster_data
        
        # Determine target resolution
        if target_resolution is None:
            target_resolution = metadata['res']
        
        # Calculate transform
        transform, width, height = calculate_default_transform(
            source_crs, target_crs,
            metadata['shape'][1], metadata['shape'][0],
            metadata['bounds'],
            resolution=target_resolution
        )
        
        # Get resampling method
        resampling = getattr(Resampling, resampling_method, Resampling.bilinear)
        
        # Reproject data
        reprojected_data = np.empty((data.shape[0], height, width), dtype=data.dtype)
        
        for i in range(data.shape[0]):
            reproject(
                source=data[i],
                destination=reprojected_data[i],
                src_transform=metadata['transform'],
                src_crs=source_crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=resampling,
                nodata=metadata.get('nodata')
            )
        
        # Update metadata
        new_metadata = metadata.copy()
        new_metadata.update({
            'crs': target_crs,
            'transform': transform,
            'shape': (height, width),
            'res': target_resolution,
            'bounds': rasterio.transform.array_bounds(height, width, transform)
        })
        
        result = {
            'data': reprojected_data,
            'metadata': new_metadata,
            'bands': raster_data['bands'],
            'window': raster_data['window'],
            'file_path': raster_data['file_path'],
            'transformation_info': {
                'source_crs': source_crs,
                'target_crs': target_crs,
                'resampling_method': resampling_method,
                'target_resolution': target_resolution
            }
        }
        
        logger.info(f"Reprojected raster from {source_crs} to {target_crs}: shape={reprojected_data.shape}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to reproject raster: {e}")
        raise DataTransformationError(f"Failed to reproject raster: {e}")


def reproject_vector(
    vector_data: Dict[str, Any],
    target_crs: str
) -> Dict[str, Any]:
    """
    Reproject vector data to a different coordinate reference system.
    
    Args:
        vector_data: Dictionary containing vector data (from load_vector_data)
        target_crs: Target coordinate reference system
        
    Returns:
        Dictionary containing reprojected vector data
        
    Raises:
        DataTransformationError: If reprojection fails
    """
    try:
        if not GEOPANDAS_AVAILABLE:
            raise DataTransformationError("geopandas is not available")
        
        gdf = vector_data['data']
        source_crs = str(gdf.crs)
        
        # Check if reprojection is needed
        if source_crs == target_crs:
            logger.info(f"Vector already in target CRS: {target_crs}")
            return vector_data
        
        # Reproject
        reprojected_gdf = gdf.to_crs(target_crs)
        
        # Update metadata
        new_metadata = vector_data['metadata'].copy()
        new_metadata.update({
            'crs': target_crs,
            'bounds': reprojected_gdf.total_bounds.tolist(),
            'area': reprojected_gdf.geometry.area.sum() if len(reprojected_gdf) > 0 else 0.0
        })
        
        result = {
            'data': reprojected_gdf,
            'metadata': new_metadata,
            'file_path': vector_data['file_path'],
            'bbox': vector_data['bbox'],
            'target_crs': target_crs,
            'transformation_info': {
                'source_crs': source_crs,
                'target_crs': target_crs
            }
        }
        
        logger.info(f"Reprojected vector from {source_crs} to {target_crs}: {len(reprojected_gdf)} features")
        return result
        
    except Exception as e:
        logger.error(f"Failed to reproject vector: {e}")
        raise DataTransformationError(f"Failed to reproject vector: {e}")


def resample_raster(
    raster_data: Dict[str, Any],
    target_resolution: Tuple[float, float],
    resampling_method: str = 'bilinear'
) -> Dict[str, Any]:
    """
    Resample raster data to a different resolution.
    
    Args:
        raster_data: Dictionary containing raster data
        target_resolution: Target resolution (x_res, y_res)
        resampling_method: Resampling method ('nearest', 'bilinear', 'cubic', etc.)
        
    Returns:
        Dictionary containing resampled raster data
        
    Raises:
        DataTransformationError: If resampling fails
    """
    try:
        if not RASTERIO_AVAILABLE:
            raise DataTransformationError("rasterio is not available")
        
        # Use reproject_raster with same CRS but different resolution
        return reproject_raster(
            raster_data,
            target_crs=raster_data['metadata']['crs'],
            target_resolution=target_resolution,
            resampling_method=resampling_method
        )
        
    except Exception as e:
        logger.error(f"Failed to resample raster: {e}")
        raise DataTransformationError(f"Failed to resample raster: {e}")


def clip_to_bounds(
    data: Dict[str, Any],
    bounds: Tuple[float, float, float, float],
    data_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Clip data to specified bounds.
    
    Args:
        data: Dictionary containing data (raster or vector)
        bounds: Bounding box (minx, miny, maxx, maxy)
        data_type: Type of data ('raster' or 'vector')
        
    Returns:
        Dictionary containing clipped data
        
    Raises:
        DataTransformationError: If clipping fails
    """
    try:
        # Determine data type if not specified
        if data_type is None:
            if 'data' in data and hasattr(data['data'], 'geometry'):
                data_type = 'vector'
            else:
                data_type = 'raster'
        
        if data_type == 'raster':
            return clip_raster_to_bounds(data, bounds)
        elif data_type == 'vector':
            return clip_vector_to_bounds(data, bounds)
        else:
            raise DataTransformationError(f"Unsupported data type: {data_type}")
        
    except Exception as e:
        logger.error(f"Failed to clip data to bounds: {e}")
        raise DataTransformationError(f"Failed to clip data to bounds: {e}")


def clip_raster_to_bounds(
    raster_data: Dict[str, Any],
    bounds: Tuple[float, float, float, float]
) -> Dict[str, Any]:
    """
    Clip raster data to specified bounds.
    
    Args:
        raster_data: Dictionary containing raster data
        bounds: Bounding box (minx, miny, maxx, maxy)
        
    Returns:
        Dictionary containing clipped raster data
        
    Raises:
        DataTransformationError: If clipping fails
    """
    try:
        if not RASTERIO_AVAILABLE:
            raise DataTransformationError("rasterio is not available")
        
        # Create geometry from bounds
        minx, miny, maxx, maxy = bounds
        geometry = box(minx, miny, maxx, maxy)
        
        # Create mask
        mask = geometry_mask(
            [geometry],
            out_shape=raster_data['metadata']['shape'],
            transform=raster_data['metadata']['transform'],
            invert=True
        )
        
        # Apply mask to data
        data = raster_data['data']
        masked_data = data.copy()
        masked_data[:, ~mask] = raster_data['metadata'].get('nodata', 0)
        
        # Calculate new bounds
        new_bounds = bounds
        
        result = {
            'data': masked_data,
            'metadata': raster_data['metadata'].copy(),
            'bands': raster_data['bands'],
            'window': raster_data['window'],
            'file_path': raster_data['file_path'],
            'clipping_info': {
                'original_bounds': raster_data['metadata']['bounds'],
                'clipped_bounds': new_bounds,
                'mask_shape': mask.shape
            }
        }
        
        logger.info(f"Clipped raster to bounds: {bounds}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to clip raster to bounds: {e}")
        raise DataTransformationError(f"Failed to clip raster to bounds: {e}")


def clip_vector_to_bounds(
    vector_data: Dict[str, Any],
    bounds: Tuple[float, float, float, float]
) -> Dict[str, Any]:
    """
    Clip vector data to specified bounds.
    
    Args:
        vector_data: Dictionary containing vector data
        bounds: Bounding box (minx, miny, maxx, maxy)
        
    Returns:
        Dictionary containing clipped vector data
        
    Raises:
        DataTransformationError: If clipping fails
    """
    try:
        if not GEOPANDAS_AVAILABLE:
            raise DataTransformationError("geopandas is not available")
        
        gdf = vector_data['data']
        
        # Create bounding box geometry
        minx, miny, maxx, maxy = bounds
        bbox_geom = box(minx, miny, maxx, maxy)
        
        # Clip geometries
        clipped_gdf = gdf[gdf.geometry.intersects(bbox_geom)].copy()
        clipped_gdf.geometry = clipped_gdf.geometry.intersection(bbox_geom)
        
        # Update metadata
        new_metadata = vector_data['metadata'].copy()
        new_metadata.update({
            'bounds': clipped_gdf.total_bounds.tolist(),
            'feature_count': len(clipped_gdf),
            'area': clipped_gdf.geometry.area.sum() if len(clipped_gdf) > 0 else 0.0
        })
        
        result = {
            'data': clipped_gdf,
            'metadata': new_metadata,
            'file_path': vector_data['file_path'],
            'bbox': vector_data['bbox'],
            'target_crs': vector_data['target_crs'],
            'clipping_info': {
                'original_bounds': vector_data['metadata']['bounds'],
                'clipped_bounds': bounds,
                'original_feature_count': vector_data['metadata']['feature_count'],
                'clipped_feature_count': len(clipped_gdf)
            }
        }
        
        logger.info(f"Clipped vector to bounds: {bounds}, {len(clipped_gdf)} features remaining")
        return result
        
    except Exception as e:
        logger.error(f"Failed to clip vector to bounds: {e}")
        raise DataTransformationError(f"Failed to clip vector to bounds: {e}")


def normalize_data(
    data: np.ndarray,
    method: str = 'minmax',
    range_min: float = 0.0,
    range_max: float = 1.0,
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize data using various methods.
    
    Args:
        data: Input data array
        method: Normalization method ('minmax', 'zscore', 'robust', 'log')
        range_min: Minimum value for minmax normalization
        range_max: Maximum value for minmax normalization
        **kwargs: Additional parameters for specific methods
        
    Returns:
        Tuple of (normalized_data, normalization_info)
        
    Raises:
        DataTransformationError: If normalization fails
    """
    try:
        # Handle NaN values
        data_clean = data.copy()
        nan_mask = np.isnan(data_clean)
        
        if method == 'minmax':
            # Min-max normalization
            data_min = np.nanmin(data_clean)
            data_max = np.nanmax(data_clean)
            
            if data_max > data_min:
                normalized = (data_clean - data_min) / (data_max - data_min)
                normalized = normalized * (range_max - range_min) + range_min
            else:
                normalized = np.full_like(data_clean, range_min)
            
            info = {
                'method': 'minmax',
                'original_min': data_min,
                'original_max': data_max,
                'target_min': range_min,
                'target_max': range_max
            }
        
        elif method == 'zscore':
            # Z-score normalization
            data_mean = np.nanmean(data_clean)
            data_std = np.nanstd(data_clean)
            
            if data_std > 0:
                normalized = (data_clean - data_mean) / data_std
            else:
                normalized = np.zeros_like(data_clean)
            
            info = {
                'method': 'zscore',
                'original_mean': data_mean,
                'original_std': data_std
            }
        
        elif method == 'robust':
            # Robust normalization using percentiles
            q25 = np.nanpercentile(data_clean, 25)
            q75 = np.nanpercentile(data_clean, 75)
            iqr = q75 - q25
            
            if iqr > 0:
                normalized = (data_clean - q25) / iqr
            else:
                normalized = np.zeros_like(data_clean)
            
            info = {
                'method': 'robust',
                'q25': q25,
                'q75': q75,
                'iqr': iqr
            }
        
        elif method == 'log':
            # Log normalization
            data_min = np.nanmin(data_clean)
            if data_min <= 0:
                # Add offset to make all values positive
                offset = abs(data_min) + 1e-10
                data_clean = data_clean + offset
                info = {'method': 'log', 'offset': offset}
            else:
                info = {'method': 'log', 'offset': 0}
            
            normalized = np.log(data_clean)
        
        else:
            raise DataTransformationError(f"Unsupported normalization method: {method}")
        
        # Restore NaN values
        normalized[nan_mask] = np.nan
        
        logger.info(f"Normalized data using {method} method: shape={data.shape}")
        return normalized, info
        
    except Exception as e:
        logger.error(f"Failed to normalize data: {e}")
        raise DataTransformationError(f"Failed to normalize data: {e}")


def apply_transformation_pipeline(
    data: Dict[str, Any],
    transformations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Apply a series of transformations to data.
    
    Args:
        data: Input data dictionary
        transformations: List of transformation dictionaries with 'type' and parameters
        
    Returns:
        Dictionary containing transformed data
        
    Raises:
        DataTransformationError: If transformation pipeline fails
    """
    try:
        result = data.copy()
        
        for i, transform_config in enumerate(transformations):
            transform_type = transform_config['type']
            logger.info(f"Applying transformation {i+1}/{len(transformations)}: {transform_type}")
            
            if transform_type == 'reproject':
                if 'data' in result and hasattr(result['data'], 'geometry'):
                    result = reproject_vector(result, transform_config['target_crs'])
                else:
                    result = reproject_raster(
                        result,
                        transform_config['target_crs'],
                        transform_config.get('target_resolution'),
                        transform_config.get('resampling_method', 'bilinear')
                    )
            
            elif transform_type == 'resample':
                result = resample_raster(
                    result,
                    transform_config['target_resolution'],
                    transform_config.get('resampling_method', 'bilinear')
                )
            
            elif transform_type == 'clip':
                result = clip_to_bounds(
                    result,
                    transform_config['bounds'],
                    transform_config.get('data_type')
                )
            
            elif transform_type == 'normalize':
                if 'data' in result and isinstance(result['data'], np.ndarray):
                    normalized_data, norm_info = normalize_data(
                        result['data'],
                        transform_config.get('method', 'minmax'),
                        transform_config.get('range_min', 0.0),
                        transform_config.get('range_max', 1.0)
                    )
                    result['data'] = normalized_data
                    result['normalization_info'] = norm_info
            
            else:
                raise DataTransformationError(f"Unknown transformation type: {transform_type}")
        
        logger.info(f"Applied {len(transformations)} transformations successfully")
        return result
        
    except Exception as e:
        logger.error(f"Failed to apply transformation pipeline: {e}")
        raise DataTransformationError(f"Failed to apply transformation pipeline: {e}")
