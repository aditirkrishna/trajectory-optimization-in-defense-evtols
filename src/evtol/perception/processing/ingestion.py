"""
Data ingestion utilities for the perception layer.

This module provides functions for loading and ingesting various geospatial data formats
including raster, vector, NetCDF, and point cloud data.
"""

import logging
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
from pathlib import Path

# Optional imports for geospatial data handling
try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString, Polygon
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

try:
    import netCDF4 as nc
    NETCDF4_AVAILABLE = True
except ImportError:
    NETCDF4_AVAILABLE = False

try:
    import laspy
    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False

# Import utilities with absolute paths
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.validation import validate_file_path
from utils.file_utils import get_geospatial_file_info

logger = logging.getLogger(__name__)


class DataIngestionError(Exception):
    """Custom exception for data ingestion errors."""
    pass


def load_raster_data(
    file_path: str,
    bands: Optional[List[int]] = None,
    window: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    nodata_value: Optional[float] = None
) -> Dict[str, Any]:
    """
    Load raster data from various formats (GeoTIFF, etc.).
    
    Args:
        file_path: Path to the raster file
        bands: List of band indices to load (None for all bands)
        window: Window to read ((row_start, row_stop), (col_start, col_stop))
        nodata_value: Value to use for nodata pixels
        
    Returns:
        Dictionary containing raster data and metadata
        
    Raises:
        DataIngestionError: If loading fails
    """
    try:
        if not RASTERIO_AVAILABLE:
            raise DataIngestionError("rasterio is not available")
        
        validate_file_path(file_path)
        
        with rasterio.open(file_path) as src:
            # Get metadata
            metadata = {
                'crs': str(src.crs),
                'transform': src.transform,
                'bounds': src.bounds,
                'shape': src.shape,
                'count': src.count,
                'dtype': src.dtypes[0],
                'nodata': src.nodata,
                'res': src.res
            }
            
            # Determine bands to read
            if bands is None:
                bands = list(range(1, src.count + 1))
            
            # Read data
            if window:
                data = src.read(bands, window=window)
            else:
                data = src.read(bands)
            
            # Handle nodata values
            if nodata_value is not None:
                if src.nodata is not None:
                    data = np.where(data == src.nodata, nodata_value, data)
            
            # Convert to float32 for consistency
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            
            result = {
                'data': data,
                'metadata': metadata,
                'bands': bands,
                'window': window,
                'file_path': file_path
            }
            
            logger.info(f"Loaded raster data from {file_path}: shape={data.shape}, bands={len(bands)}")
            return result
            
    except Exception as e:
        logger.error(f"Failed to load raster data from {file_path}: {e}")
        raise DataIngestionError(f"Failed to load raster data: {e}")


def load_vector_data(
    file_path: str,
    columns: Optional[List[str]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    crs: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load vector data from various formats (Shapefile, GeoJSON, etc.).
    
    Args:
        file_path: Path to the vector file
        columns: List of columns to load (None for all columns)
        bbox: Bounding box filter (minx, miny, maxx, maxy)
        crs: Target CRS for reprojection (None to keep original)
        
    Returns:
        Dictionary containing vector data and metadata
        
    Raises:
        DataIngestionError: If loading fails
    """
    try:
        if not GEOPANDAS_AVAILABLE:
            raise DataIngestionError("geopandas is not available")
        
        validate_file_path(file_path)
        
        # Load data
        gdf = gpd.read_file(file_path, bbox=bbox, columns=columns)
        
        # Reproject if requested
        if crs and str(gdf.crs) != crs:
            gdf = gdf.to_crs(crs)
            logger.info(f"Reprojected vector data from {gdf.crs} to {crs}")
        
        # Get metadata
        metadata = {
            'crs': str(gdf.crs),
            'bounds': gdf.total_bounds.tolist(),
            'feature_count': len(gdf),
            'geometry_types': gdf.geometry.geom_type.unique().tolist(),
            'columns': gdf.columns.tolist(),
            'area': gdf.geometry.area.sum() if len(gdf) > 0 else 0.0
        }
        
        result = {
            'data': gdf,
            'metadata': metadata,
            'file_path': file_path,
            'bbox': bbox,
            'target_crs': crs
        }
        
        logger.info(f"Loaded vector data from {file_path}: {len(gdf)} features, {len(gdf.columns)} columns")
        return result
        
    except Exception as e:
        logger.error(f"Failed to load vector data from {file_path}: {e}")
        raise DataIngestionError(f"Failed to load vector data: {e}")


def load_netcdf_data(
    file_path: str,
    variables: Optional[List[str]] = None,
    time_slice: Optional[Union[int, slice]] = None,
    spatial_slice: Optional[Dict[str, slice]] = None
) -> Dict[str, Any]:
    """
    Load NetCDF data from .nc files.
    
    Args:
        file_path: Path to the NetCDF file
        variables: List of variables to load (None for all)
        time_slice: Time dimension slice
        spatial_slice: Spatial dimension slices
        
    Returns:
        Dictionary containing NetCDF data and metadata
        
    Raises:
        DataIngestionError: If loading fails
    """
    try:
        if not NETCDF4_AVAILABLE:
            raise DataIngestionError("netCDF4 is not available")
        
        validate_file_path(file_path)
        
        with nc.Dataset(file_path, 'r') as dataset:
            # Get metadata
            metadata = {
                'dimensions': dict(dataset.dimensions),
                'variables': list(dataset.variables.keys()),
                'attributes': dict(dataset.__dict__),
                'file_path': file_path
            }
            
            # Determine variables to load
            if variables is None:
                variables = list(dataset.variables.keys())
            
            # Load data
            data = {}
            for var_name in variables:
                if var_name in dataset.variables:
                    var = dataset.variables[var_name]
                    
                    # Apply slices
                    slices = []
                    for dim_name in var.dimensions:
                        if dim_name in metadata['dimensions']:
                            dim_size = metadata['dimensions'][dim_name]
                            
                            if dim_name == 'time' and time_slice is not None:
                                slices.append(time_slice)
                            elif spatial_slice and dim_name in spatial_slice:
                                slices.append(spatial_slice[dim_name])
                            else:
                                slices.append(slice(None))
                    
                    # Load variable data
                    if slices:
                        var_data = var[slices]
                    else:
                        var_data = var[:]
                    
                    data[var_name] = {
                        'data': var_data,
                        'dimensions': var.dimensions,
                        'attributes': dict(var.__dict__)
                    }
            
            result = {
                'data': data,
                'metadata': metadata,
                'variables': variables,
                'time_slice': time_slice,
                'spatial_slice': spatial_slice
            }
            
            logger.info(f"Loaded NetCDF data from {file_path}: {len(variables)} variables")
            return result
            
    except Exception as e:
        logger.error(f"Failed to load NetCDF data from {file_path}: {e}")
        raise DataIngestionError(f"Failed to load NetCDF data: {e}")


def load_point_cloud_data(
    file_path: str,
    fields: Optional[List[str]] = None,
    bounds: Optional[Tuple[float, float, float, float, float, float]] = None
) -> Dict[str, Any]:
    """
    Load point cloud data from LAS/LAZ files.
    
    Args:
        file_path: Path to the point cloud file
        fields: List of fields to load (None for all)
        bounds: Bounding box filter (minx, miny, minz, maxx, maxy, maxz)
        
    Returns:
        Dictionary containing point cloud data and metadata
        
    Raises:
        DataIngestionError: If loading fails
    """
    try:
        if not LASPY_AVAILABLE:
            raise DataIngestionError("laspy is not available")
        
        validate_file_path(file_path)
        
        with laspy.read(file_path) as las:
            # Get metadata
            metadata = {
                'point_count': las.header.point_count,
                'point_format': las.header.point_format,
                'point_length': las.header.point_length,
                'bounds': las.header.bounds,
                'scales': las.header.scales,
                'offsets': las.header.offsets,
                'crs': str(las.header.crs) if las.header.crs else None
            }
            
            # Determine fields to load
            if fields is None:
                fields = list(las.point_format.dimension_names)
            
            # Load data
            data = {}
            for field in fields:
                if hasattr(las, field):
                    field_data = getattr(las, field)
                    data[field] = field_data
            
            # Apply bounds filter if specified
            if bounds:
                minx, miny, minz, maxx, maxy, maxz = bounds
                mask = (
                    (data['X'] >= minx) & (data['X'] <= maxx) &
                    (data['Y'] >= miny) & (data['Y'] <= maxy) &
                    (data['Z'] >= minz) & (data['Z'] <= maxz)
                )
                
                for field in data:
                    data[field] = data[field][mask]
                
                metadata['filtered_point_count'] = len(data['X'])
            
            result = {
                'data': data,
                'metadata': metadata,
                'fields': fields,
                'bounds': bounds,
                'file_path': file_path
            }
            
            logger.info(f"Loaded point cloud data from {file_path}: {len(data.get('X', []))} points")
            return result
            
    except Exception as e:
        logger.error(f"Failed to load point cloud data from {file_path}: {e}")
        raise DataIngestionError(f"Failed to load point cloud data: {e}")


def get_data_info(file_path: str) -> Dict[str, Any]:
    """
    Get comprehensive information about a data file.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        Dictionary containing data file information
        
    Raises:
        DataIngestionError: If info retrieval fails
    """
    try:
        # Get basic file info
        file_info = get_geospatial_file_info(file_path)
        
        # Add data-specific information
        data_info = {
            **file_info,
            'supported_for_loading': False,
            'loading_functions': []
        }
        
        file_ext = file_info['extension'].lower()
        
        # Check what loading functions are available
        if file_ext in ['.tif', '.tiff', '.geotiff'] and RASTERIO_AVAILABLE:
            data_info['supported_for_loading'] = True
            data_info['loading_functions'].append('load_raster_data')
        
        elif file_ext in ['.shp', '.geojson', '.gpkg', '.kml', '.kmz'] and GEOPANDAS_AVAILABLE:
            data_info['supported_for_loading'] = True
            data_info['loading_functions'].append('load_vector_data')
        
        elif file_ext in ['.nc', '.netcdf'] and NETCDF4_AVAILABLE:
            data_info['supported_for_loading'] = True
            data_info['loading_functions'].append('load_netcdf_data')
        
        elif file_ext in ['.las', '.laz'] and LASPY_AVAILABLE:
            data_info['supported_for_loading'] = True
            data_info['loading_functions'].append('load_point_cloud_data')
        
        return data_info
        
    except Exception as e:
        logger.error(f"Failed to get data info for {file_path}: {e}")
        raise DataIngestionError(f"Failed to get data info: {e}")


def batch_load_data(
    file_paths: List[str],
    data_type: Optional[str] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Batch load multiple data files.
    
    Args:
        file_paths: List of file paths to load
        data_type: Type of data to load ('raster', 'vector', 'netcdf', 'point_cloud')
        **kwargs: Additional arguments for loading functions
        
    Returns:
        List of loaded data dictionaries
        
    Raises:
        DataIngestionError: If batch loading fails
    """
    try:
        results = []
        
        for i, file_path in enumerate(file_paths, 1):
            try:
                logger.info(f"Loading data file {i}/{len(file_paths)}: {file_path}")
                
                # Determine loading function based on file type or data_type
                if data_type == 'raster' or file_path.lower().endswith(('.tif', '.tiff', '.geotiff')):
                    if RASTERIO_AVAILABLE:
                        result = load_raster_data(file_path, **kwargs)
                    else:
                        raise DataIngestionError("rasterio not available for raster loading")
                
                elif data_type == 'vector' or file_path.lower().endswith(('.shp', '.geojson', '.gpkg')):
                    if GEOPANDAS_AVAILABLE:
                        result = load_vector_data(file_path, **kwargs)
                    else:
                        raise DataIngestionError("geopandas not available for vector loading")
                
                elif data_type == 'netcdf' or file_path.lower().endswith(('.nc', '.netcdf')):
                    if NETCDF4_AVAILABLE:
                        result = load_netcdf_data(file_path, **kwargs)
                    else:
                        raise DataIngestionError("netCDF4 not available for NetCDF loading")
                
                elif data_type == 'point_cloud' or file_path.lower().endswith(('.las', '.laz')):
                    if LASPY_AVAILABLE:
                        result = load_point_cloud_data(file_path, **kwargs)
                    else:
                        raise DataIngestionError("laspy not available for point cloud loading")
                
                else:
                    raise DataIngestionError(f"Unknown data type for file: {file_path}")
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                results.append({
                    'error': str(e),
                    'file_path': file_path,
                    'success': False
                })
        
        logger.info(f"Batch loading completed: {len([r for r in results if 'error' not in r])}/{len(file_paths)} successful")
        return results
        
    except Exception as e:
        logger.error(f"Failed to batch load data: {e}")
        raise DataIngestionError(f"Failed to batch load data: {e}")
