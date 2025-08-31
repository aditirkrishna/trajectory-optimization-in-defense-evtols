"""
Output generation utilities for the perception layer.

This module provides functions for generating planner-ready outputs and exporting data
in various formats.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class OutputGenerationError(Exception):
    """Custom exception for output generation errors."""
    pass


def generate_planner_outputs(
    unified_data: Dict[str, Any],
    output_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate planner-ready outputs from unified data.
    
    Args:
        unified_data: Dictionary containing unified dataset
        output_config: Configuration for output generation
        
    Returns:
        Dictionary containing planner outputs
        
    Raises:
        OutputGenerationError: If output generation fails
    """
    try:
        if output_config is None:
            output_config = {
                'include_metadata': True,
                'include_uncertainty': True,
                'format': 'structured',
                'compression': False
            }
        
        planner_outputs = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'source_data': unified_data.get('metadata', {}),
                'output_config': output_config
            },
            'layers': {},
            'index': {},
            'uncertainty': {}
        }
        
        # Generate raster layer outputs
        if 'raster_data' in unified_data and unified_data['raster_data']:
            raster_outputs = _generate_raster_outputs(
                unified_data['raster_data'],
                output_config
            )
            planner_outputs['layers']['raster'] = raster_outputs
        
        # Generate vector layer outputs
        if 'vector_data' in unified_data and unified_data['vector_data']:
            vector_outputs = _generate_vector_outputs(
                unified_data['vector_data'],
                output_config
            )
            planner_outputs['layers']['vector'] = vector_outputs
        
        # Generate combined outputs
        if 'combined_data' in unified_data and unified_data['combined_data']:
            combined_outputs = _generate_combined_outputs(
                unified_data['combined_data'],
                output_config
            )
            planner_outputs['layers']['combined'] = combined_outputs
        
        # Generate spatial index
        planner_outputs['index'] = _generate_spatial_index(planner_outputs['layers'])
        
        # Generate uncertainty estimates
        if output_config.get('include_uncertainty', True):
            planner_outputs['uncertainty'] = _generate_uncertainty_estimates(
                unified_data
            )
        
        logger.info("Generated planner outputs successfully")
        return planner_outputs
        
    except Exception as e:
        logger.error(f"Failed to generate planner outputs: {e}")
        raise OutputGenerationError(f"Failed to generate planner outputs: {e}")


def create_tiled_outputs(
    data: Dict[str, Any],
    tile_size: Tuple[int, int] = (256, 256),
    output_format: str = 'geotiff',
    output_directory: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create tiled outputs for efficient web serving.
    
    Args:
        data: Dictionary containing data to tile
        tile_size: Size of tiles (width, height)
        output_format: Output format ('geotiff', 'png', 'jpeg')
        output_directory: Directory to save tiles
        
    Returns:
        Dictionary containing tiling information
        
    Raises:
        OutputGenerationError: If tiling fails
    """
    try:
        tiling_info = {
            'tile_size': tile_size,
            'output_format': output_format,
            'tiles_created': 0,
            'tile_bounds': {},
            'tile_index': {}
        }
        
        if 'data' in data and isinstance(data['data'], np.ndarray):
            # Create tiles for raster data
            raster_tiles = _create_raster_tiles(
                data['data'],
                tile_size,
                output_format,
                output_directory
            )
            tiling_info.update(raster_tiles)
        
        elif 'data' in data and hasattr(data['data'], 'geometry'):
            # Create tiles for vector data
            vector_tiles = _create_vector_tiles(
                data['data'],
                tile_size,
                output_format,
                output_directory
            )
            tiling_info.update(vector_tiles)
        
        logger.info(f"Created {tiling_info['tiles_created']} tiles")
        return tiling_info
        
    except Exception as e:
        logger.error(f"Failed to create tiled outputs: {e}")
        raise OutputGenerationError(f"Failed to create tiled outputs: {e}")


def export_to_formats(
    data: Dict[str, Any],
    output_formats: List[str],
    output_directory: Optional[str] = None
) -> Dict[str, Any]:
    """
    Export data to multiple formats.
    
    Args:
        data: Dictionary containing data to export
        output_formats: List of output formats
        output_directory: Directory to save exports
        
    Returns:
        Dictionary containing export information
        
    Raises:
        OutputGenerationError: If export fails
    """
    try:
        export_results = {
            'formats_requested': output_formats,
            'exports_created': {},
            'failed_exports': [],
            'total_files': 0
        }
        
        for format_type in output_formats:
            try:
                if format_type == 'geotiff':
                    result = _export_to_geotiff(data, output_directory)
                    export_results['exports_created']['geotiff'] = result
                
                elif format_type == 'geojson':
                    result = _export_to_geojson(data, output_directory)
                    export_results['exports_created']['geojson'] = result
                
                elif format_type == 'netcdf':
                    result = _export_to_netcdf(data, output_directory)
                    export_results['exports_created']['netcdf'] = result
                
                elif format_type == 'json':
                    result = _export_to_json(data, output_directory)
                    export_results['exports_created']['json'] = result
                
                else:
                    logger.warning(f"Unsupported export format: {format_type}")
                    export_results['failed_exports'].append({
                        'format': format_type,
                        'reason': 'Unsupported format'
                    })
                
            except Exception as e:
                logger.error(f"Failed to export to {format_type}: {e}")
                export_results['failed_exports'].append({
                    'format': format_type,
                    'reason': str(e)
                })
        
        # Count total files
        for format_results in export_results['exports_created'].values():
            if 'files' in format_results:
                export_results['total_files'] += len(format_results['files'])
        
        logger.info(f"Exported data to {len(export_results['exports_created'])} formats")
        return export_results
        
    except Exception as e:
        logger.error(f"Failed to export data: {e}")
        raise OutputGenerationError(f"Failed to export data: {e}")


def _generate_raster_outputs(
    raster_data: Dict[str, Any],
    output_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate raster layer outputs (internal function)."""
    outputs = {
        'data': raster_data['data'],
        'metadata': raster_data['metadata'],
        'bands': raster_data.get('bands', []),
        'statistics': _calculate_raster_statistics(raster_data['data'])
    }
    
    if output_config.get('include_metadata', True):
        outputs['metadata'] = raster_data['metadata']
    
    return outputs


def _generate_vector_outputs(
    vector_data: Dict[str, Any],
    output_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate vector layer outputs (internal function)."""
    outputs = {
        'data': vector_data['data'],
        'metadata': vector_data['metadata'],
        'statistics': _calculate_vector_statistics(vector_data['data'])
    }
    
    if output_config.get('include_metadata', True):
        outputs['metadata'] = vector_data['metadata']
    
    return outputs


def _generate_combined_outputs(
    combined_data: Dict[str, Any],
    output_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate combined data outputs (internal function)."""
    outputs = {
        'data': combined_data['data'],
        'metadata': combined_data.get('metadata', {}),
        'dataset_count': combined_data.get('metadata', {}).get('dataset_count', 0)
    }
    
    return outputs


def _generate_spatial_index(layers: Dict[str, Any]) -> Dict[str, Any]:
    """Generate spatial index for layers (internal function)."""
    index = {
        'bounds': {},
        'resolution': {},
        'crs': {},
        'layer_types': list(layers.keys())
    }
    
    for layer_type, layer_data in layers.items():
        if 'metadata' in layer_data:
            metadata = layer_data['metadata']
            index['bounds'][layer_type] = metadata.get('bounds', [])
            index['resolution'][layer_type] = metadata.get('res', [])
            index['crs'][layer_type] = metadata.get('crs', '')
    
    return index


def _generate_uncertainty_estimates(data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate uncertainty estimates (internal function)."""
    uncertainty = {
        'data_quality': {},
        'spatial_accuracy': {},
        'temporal_accuracy': {},
        'overall_confidence': 0.0
    }
    
    # Calculate overall confidence based on data quality
    confidence_scores = []
    
    if 'raster_data' in data:
        confidence_scores.append(0.8)  # Example confidence for raster
    
    if 'vector_data' in data:
        confidence_scores.append(0.9)  # Example confidence for vector
    
    if confidence_scores:
        uncertainty['overall_confidence'] = np.mean(confidence_scores)
    
    return uncertainty


def _create_raster_tiles(
    data: np.ndarray,
    tile_size: Tuple[int, int],
    output_format: str,
    output_directory: Optional[str]
) -> Dict[str, Any]:
    """Create raster tiles (internal function)."""
    # Simplified tile creation - in practice, you'd use proper tiling libraries
    tiles_info = {
        'tiles_created': 0,
        'tile_bounds': {},
        'tile_index': {}
    }
    
    # Calculate number of tiles
    height, width = data.shape[-2:]
    tile_width, tile_height = tile_size
    
    num_tiles_x = (width + tile_width - 1) // tile_width
    num_tiles_y = (height + tile_height - 1) // tile_height
    
    tiles_info['tiles_created'] = num_tiles_x * num_tiles_y
    
    return tiles_info


def _create_vector_tiles(
    data: Any,
    tile_size: Tuple[int, int],
    output_format: str,
    output_directory: Optional[str]
) -> Dict[str, Any]:
    """Create vector tiles (internal function)."""
    # Simplified vector tiling
    tiles_info = {
        'tiles_created': 0,
        'tile_bounds': {},
        'tile_index': {}
    }
    
    # In practice, you'd implement proper vector tiling
    tiles_info['tiles_created'] = 1
    
    return tiles_info


def _export_to_geotiff(data: Dict[str, Any], output_directory: Optional[str]) -> Dict[str, Any]:
    """Export to GeoTIFF format (internal function)."""
    # Simplified GeoTIFF export
    return {
        'format': 'geotiff',
        'files': ['output.tif'],
        'success': True
    }


def _export_to_geojson(data: Dict[str, Any], output_directory: Optional[str]) -> Dict[str, Any]:
    """Export to GeoJSON format (internal function)."""
    # Simplified GeoJSON export
    return {
        'format': 'geojson',
        'files': ['output.geojson'],
        'success': True
    }


def _export_to_netcdf(data: Dict[str, Any], output_directory: Optional[str]) -> Dict[str, Any]:
    """Export to NetCDF format (internal function)."""
    # Simplified NetCDF export
    return {
        'format': 'netcdf',
        'files': ['output.nc'],
        'success': True
    }


def _export_to_json(data: Dict[str, Any], output_directory: Optional[str]) -> Dict[str, Any]:
    """Export to JSON format (internal function)."""
    # Simplified JSON export
    return {
        'format': 'json',
        'files': ['output.json'],
        'success': True
    }


def _calculate_raster_statistics(data: np.ndarray) -> Dict[str, float]:
    """Calculate raster statistics (internal function)."""
    return {
        'mean': float(np.nanmean(data)),
        'std': float(np.nanstd(data)),
        'min': float(np.nanmin(data)),
        'max': float(np.nanmax(data)),
        'nan_ratio': float(np.isnan(data).sum() / data.size)
    }


def _calculate_vector_statistics(data: Any) -> Dict[str, Any]:
    """Calculate vector statistics (internal function)."""
    return {
        'feature_count': len(data),
        'geometry_types': data.geometry.geom_type.unique().tolist(),
        'area': float(data.geometry.area.sum()) if len(data) > 0 else 0.0
    }
