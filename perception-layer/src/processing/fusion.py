"""
Data fusion utilities for the perception layer.

This module provides functions for combining and fusing multiple geospatial datasets
into unified representations for the planner.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np

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


class DataFusionError(Exception):
    """Custom exception for data fusion errors."""
    pass


def fuse_raster_layers(
    raster_layers: List[Dict[str, Any]],
    fusion_method: str = 'stack',
    target_crs: Optional[str] = None,
    target_resolution: Optional[Tuple[float, float]] = None,
    weights: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Fuse multiple raster layers into a unified representation.
    
    Args:
        raster_layers: List of raster data dictionaries
        fusion_method: Fusion method ('stack', 'weighted_average', 'max', 'min')
        target_crs: Target coordinate reference system
        target_resolution: Target resolution (x_res, y_res)
        weights: Weights for weighted fusion methods
        
    Returns:
        Dictionary containing fused raster data
        
    Raises:
        DataFusionError: If fusion fails
    """
    try:
        if not RASTERIO_AVAILABLE:
            raise DataFusionError("rasterio is not available")
        
        if len(raster_layers) == 0:
            raise DataFusionError("No raster layers provided")
        
        # Determine target CRS and resolution
        if target_crs is None:
            target_crs = raster_layers[0]['metadata']['crs']
        
        if target_resolution is None:
            target_resolution = raster_layers[0]['metadata']['res']
        
        # Reproject all layers to target CRS and resolution
        aligned_layers = []
        for layer in raster_layers:
            if str(layer['metadata']['crs']) != target_crs:
                from .transformation import reproject_raster
                layer = reproject_raster(layer, target_crs, target_resolution)
            aligned_layers.append(layer)
        
        # Ensure all layers have the same shape
        reference_shape = aligned_layers[0]['data'].shape[1:]
        aligned_data = []
        
        for layer in aligned_layers:
            if layer['data'].shape[1:] != reference_shape:
                from .transformation import resample_raster
                layer = resample_raster(layer, target_resolution)
            aligned_data.append(layer['data'])
        
        # Perform fusion
        if fusion_method == 'stack':
            # Stack layers along band dimension
            fused_data = np.concatenate(aligned_data, axis=0)
            
        elif fusion_method == 'weighted_average':
            if weights is None:
                weights = [1.0 / len(aligned_data)] * len(aligned_data)
            
            if len(weights) != len(aligned_data):
                raise DataFusionError("Number of weights must match number of layers")
            
            # Weighted average of first bands
            fused_data = np.zeros_like(aligned_data[0])
            for i, (data, weight) in enumerate(zip(aligned_data, weights)):
                fused_data += data[0] * weight  # Use first band for averaging
        
        elif fusion_method == 'max':
            # Maximum value across layers
            fused_data = np.maximum.reduce(aligned_data)
            
        elif fusion_method == 'min':
            # Minimum value across layers
            fused_data = np.minimum.reduce(aligned_data)
            
        else:
            raise DataFusionError(f"Unsupported fusion method: {fusion_method}")
        
        # Create metadata
        metadata = {
            'crs': target_crs,
            'res': target_resolution,
            'shape': fused_data.shape,
            'fusion_method': fusion_method,
            'source_layers': len(raster_layers),
            'weights': weights
        }
        
        result = {
            'data': fused_data,
            'metadata': metadata,
            'fusion_info': {
                'method': fusion_method,
                'target_crs': target_crs,
                'target_resolution': target_resolution,
                'source_layers': [layer['file_path'] for layer in raster_layers]
            }
        }
        
        logger.info(f"Fused {len(raster_layers)} raster layers using {fusion_method} method")
        return result
        
    except Exception as e:
        logger.error(f"Failed to fuse raster layers: {e}")
        raise DataFusionError(f"Failed to fuse raster layers: {e}")


def fuse_vector_layers(
    vector_layers: List[Dict[str, Any]],
    fusion_method: str = 'union',
    target_crs: Optional[str] = None,
    attribute_mapping: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Fuse multiple vector layers into a unified representation.
    
    Args:
        vector_layers: List of vector data dictionaries
        fusion_method: Fusion method ('union', 'intersection', 'concatenate')
        target_crs: Target coordinate reference system
        attribute_mapping: Mapping of attribute names for concatenation
        
    Returns:
        Dictionary containing fused vector data
        
    Raises:
        DataFusionError: If fusion fails
    """
    try:
        if not GEOPANDAS_AVAILABLE:
            raise DataFusionError("geopandas is not available")
        
        if len(vector_layers) == 0:
            raise DataFusionError("No vector layers provided")
        
        # Determine target CRS
        if target_crs is None:
            target_crs = str(vector_layers[0]['data'].crs)
        
        # Reproject all layers to target CRS
        aligned_layers = []
        for layer in vector_layers:
            gdf = layer['data']
            if str(gdf.crs) != target_crs:
                from .transformation import reproject_vector
                layer = reproject_vector(layer, target_crs)
            aligned_layers.append(layer['data'])
        
        # Perform fusion
        if fusion_method == 'union':
            # Union of all geometries
            from shapely.ops import unary_union
            geometries = []
            for gdf in aligned_layers:
                geometries.extend(gdf.geometry.tolist())
            
            union_geom = unary_union(geometries)
            fused_gdf = gpd.GeoDataFrame(geometry=[union_geom], crs=target_crs)
            
        elif fusion_method == 'intersection':
            # Intersection of all geometries
            from shapely.ops import unary_union
            geometries = []
            for gdf in aligned_layers:
                geometries.extend(gdf.geometry.tolist())
            
            intersection_geom = unary_union(geometries)
            fused_gdf = gpd.GeoDataFrame(geometry=[intersection_geom], crs=target_crs)
            
        elif fusion_method == 'concatenate':
            # Concatenate all layers
            fused_gdf = gpd.GeoDataFrame(pd.concat(aligned_layers, ignore_index=True))
            
            # Apply attribute mapping if provided
            if attribute_mapping:
                fused_gdf = fused_gdf.rename(columns=attribute_mapping)
            
        else:
            raise DataFusionError(f"Unsupported fusion method: {fusion_method}")
        
        # Create metadata
        metadata = {
            'crs': target_crs,
            'bounds': fused_gdf.total_bounds.tolist(),
            'feature_count': len(fused_gdf),
            'geometry_types': fused_gdf.geometry.geom_type.unique().tolist(),
            'columns': fused_gdf.columns.tolist(),
            'fusion_method': fusion_method,
            'source_layers': len(vector_layers)
        }
        
        result = {
            'data': fused_gdf,
            'metadata': metadata,
            'fusion_info': {
                'method': fusion_method,
                'target_crs': target_crs,
                'source_layers': [layer['file_path'] for layer in vector_layers],
                'attribute_mapping': attribute_mapping
            }
        }
        
        logger.info(f"Fused {len(vector_layers)} vector layers using {fusion_method} method")
        return result
        
    except Exception as e:
        logger.error(f"Failed to fuse vector layers: {e}")
        raise DataFusionError(f"Failed to fuse vector layers: {e}")


def combine_datasets(
    datasets: List[Dict[str, Any]],
    combination_method: str = 'merge',
    common_crs: Optional[str] = None,
    common_bounds: Optional[Tuple[float, float, float, float]] = None
) -> Dict[str, Any]:
    """
    Combine multiple datasets of different types into a unified representation.
    
    Args:
        datasets: List of dataset dictionaries
        combination_method: Combination method ('merge', 'overlay', 'intersection')
        common_crs: Common coordinate reference system
        common_bounds: Common bounding box for all datasets
        
    Returns:
        Dictionary containing combined dataset
        
    Raises:
        DataFusionError: If combination fails
    """
    try:
        if len(datasets) == 0:
            raise DataFusionError("No datasets provided")
        
        # Determine common CRS if not specified
        if common_crs is None:
            crs_list = []
            for dataset in datasets:
                if 'metadata' in dataset and 'crs' in dataset['metadata']:
                    crs_list.append(dataset['metadata']['crs'])
            if crs_list:
                common_crs = crs_list[0]  # Use first CRS as common
        
        # Align datasets to common CRS and bounds
        aligned_datasets = []
        for dataset in datasets:
            # Reproject if needed
            if common_crs and 'metadata' in dataset and 'crs' in dataset['metadata']:
                if str(dataset['metadata']['crs']) != common_crs:
                    if 'data' in dataset and hasattr(dataset['data'], 'geometry'):
                        from .transformation import reproject_vector
                        dataset = reproject_vector(dataset, common_crs)
                    else:
                        from .transformation import reproject_raster
                        dataset = reproject_raster(dataset, common_crs)
            
            # Clip to common bounds if specified
            if common_bounds:
                from .transformation import clip_to_bounds
                dataset = clip_to_bounds(dataset, common_bounds)
            
            aligned_datasets.append(dataset)
        
        # Perform combination based on method
        if combination_method == 'merge':
            # Simple merge of datasets
            combined_data = {
                'datasets': aligned_datasets,
                'metadata': {
                    'common_crs': common_crs,
                    'common_bounds': common_bounds,
                    'dataset_count': len(aligned_datasets),
                    'combination_method': combination_method
                }
            }
            
        elif combination_method == 'overlay':
            # Overlay datasets with priority
            combined_data = _overlay_datasets(aligned_datasets)
            
        elif combination_method == 'intersection':
            # Intersection of all datasets
            combined_data = _intersect_datasets(aligned_datasets)
            
        else:
            raise DataFusionError(f"Unsupported combination method: {combination_method}")
        
        result = {
            'data': combined_data,
            'combination_info': {
                'method': combination_method,
                'common_crs': common_crs,
                'common_bounds': common_bounds,
                'source_datasets': [dataset.get('file_path', 'unknown') for dataset in datasets]
            }
        }
        
        logger.info(f"Combined {len(datasets)} datasets using {combination_method} method")
        return result
        
    except Exception as e:
        logger.error(f"Failed to combine datasets: {e}")
        raise DataFusionError(f"Failed to combine datasets: {e}")


def create_unified_dataset(
    raster_layers: Optional[List[Dict[str, Any]]] = None,
    vector_layers: Optional[List[Dict[str, Any]]] = None,
    target_crs: str = "EPSG:4326",
    target_resolution: Tuple[float, float] = (10.0, 10.0),
    fusion_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a unified dataset from multiple raster and vector layers.
    
    Args:
        raster_layers: List of raster data dictionaries
        vector_layers: List of vector data dictionaries
        target_crs: Target coordinate reference system
        target_resolution: Target resolution for raster layers
        fusion_config: Configuration for fusion methods
        
    Returns:
        Dictionary containing unified dataset
        
    Raises:
        DataFusionError: If unification fails
    """
    try:
        if fusion_config is None:
            fusion_config = {
                'raster_fusion': 'stack',
                'vector_fusion': 'union',
                'raster_weights': None,
                'attribute_mapping': None
            }
        
        unified_data = {
            'metadata': {
                'target_crs': target_crs,
                'target_resolution': target_resolution,
                'creation_timestamp': None,  # Will be set below
                'fusion_config': fusion_config
            },
            'raster_data': None,
            'vector_data': None,
            'combined_data': None
        }
        
        # Fuse raster layers if provided
        if raster_layers:
            unified_data['raster_data'] = fuse_raster_layers(
                raster_layers,
                fusion_method=fusion_config['raster_fusion'],
                target_crs=target_crs,
                target_resolution=target_resolution,
                weights=fusion_config.get('raster_weights')
            )
        
        # Fuse vector layers if provided
        if vector_layers:
            unified_data['vector_data'] = fuse_vector_layers(
                vector_layers,
                fusion_method=fusion_config['vector_fusion'],
                target_crs=target_crs,
                attribute_mapping=fusion_config.get('attribute_mapping')
            )
        
        # Combine all data
        all_datasets = []
        if unified_data['raster_data']:
            all_datasets.append(unified_data['raster_data'])
        if unified_data['vector_data']:
            all_datasets.append(unified_data['vector_data'])
        
        if all_datasets:
            unified_data['combined_data'] = combine_datasets(
                all_datasets,
                combination_method='merge',
                common_crs=target_crs
            )
        
        # Add creation timestamp
        from datetime import datetime
        unified_data['metadata']['creation_timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Created unified dataset with {len(raster_layers or [])} raster and {len(vector_layers or [])} vector layers")
        return unified_data
        
    except Exception as e:
        logger.error(f"Failed to create unified dataset: {e}")
        raise DataFusionError(f"Failed to create unified dataset: {e}")


def _overlay_datasets(datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Overlay datasets with priority (internal helper function).
    
    Args:
        datasets: List of aligned datasets
        
    Returns:
        Dictionary containing overlaid data
    """
    # This is a simplified overlay implementation
    # In practice, you might want more sophisticated overlay logic
    return {
        'datasets': datasets,
        'overlay_method': 'priority_based',
        'metadata': {
            'dataset_count': len(datasets),
            'overlay_order': list(range(len(datasets)))
        }
    }


def _intersect_datasets(datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Intersect datasets (internal helper function).
    
    Args:
        datasets: List of aligned datasets
        
    Returns:
        Dictionary containing intersected data
    """
    # This is a simplified intersection implementation
    # In practice, you might want more sophisticated intersection logic
    return {
        'datasets': datasets,
        'intersection_method': 'common_area',
        'metadata': {
            'dataset_count': len(datasets),
            'intersection_type': 'spatial_and_temporal'
        }
    }
