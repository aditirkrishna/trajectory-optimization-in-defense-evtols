"""
Obstacle Detection Module

This module provides functions for detecting and analyzing obstacles:
- Obstacle detection from DSM/DTM data
- Obstacle classification by height and type
- Obstacle mask generation
- Height-based filtering
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

from utils.validation import validate_raster_data
from utils.coordinates import transform_coordinates

logger = logging.getLogger(__name__)


class ObstacleDetectionError(Exception):
    """Exception raised for obstacle detection errors."""
    pass


def detect_obstacles(
    dsm: np.ndarray,
    dtm: np.ndarray,
    min_height: float = 2.0,
    min_area: int = 10,
    connectivity: int = 8,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Detect obstacles from DSM and DTM data.
    
    Args:
        dsm: Digital Surface Model (includes buildings, trees, etc.)
        dtm: Digital Terrain Model (ground only)
        min_height: Minimum height to consider as obstacle (meters)
        min_area: Minimum area in pixels to consider as obstacle
        connectivity: Connectivity for connected component analysis (4 or 8)
        output_path: Optional path to save obstacle detection results
        
    Returns:
        Dictionary containing obstacle detection results
        
    Raises:
        ObstacleDetectionError: If detection fails
    """
    try:
        validate_raster_data(dsm)
        validate_raster_data(dtm)
        
        if dsm.shape != dtm.shape:
            raise ObstacleDetectionError("DSM and DTM must have the same shape")
            
        # Compute obstacle heights
        obstacle_heights = dsm - dtm
        
        # Create binary obstacle mask
        obstacle_mask = obstacle_heights >= min_height
        
        # Remove small obstacles using connected component analysis
        if SCIPY_AVAILABLE:
            labeled_mask, num_features = ndimage.label(obstacle_mask, structure=np.ones((3, 3)))
            
            # Filter by area
            for i in range(1, num_features + 1):
                area = np.sum(labeled_mask == i)
                if area < min_area:
                    obstacle_mask[labeled_mask == i] = False
                    
            # Relabel after filtering
            labeled_mask, num_features = ndimage.label(obstacle_mask, structure=np.ones((3, 3)))
        else:
            labeled_mask = obstacle_mask.astype(int)
            num_features = 1 if np.any(obstacle_mask) else 0
        
        # Compute obstacle statistics
        obstacle_areas = []
        obstacle_max_heights = []
        obstacle_centroids = []
        
        if num_features > 0:
            for i in range(1, num_features + 1):
                obstacle_pixels = labeled_mask == i
                area = np.sum(obstacle_pixels)
                max_height = np.max(obstacle_heights[obstacle_pixels])
                
                obstacle_areas.append(area)
                obstacle_max_heights.append(max_height)
                
                # Compute centroid
                coords = np.where(obstacle_pixels)
                centroid_y = np.mean(coords[0])
                centroid_x = np.mean(coords[1])
                obstacle_centroids.append((centroid_y, centroid_x))
        
        # Create labeled obstacle array
        labeled_obstacles = labeled_mask.astype(np.uint8)
        
        result = {
            'obstacle_mask': obstacle_mask,
            'labeled_obstacles': labeled_obstacles,
            'obstacle_heights': obstacle_heights,
            'num_obstacles': num_features,
            'obstacle_areas': obstacle_areas,
            'obstacle_max_heights': obstacle_max_heights,
            'obstacle_centroids': obstacle_centroids,
            'min_height_threshold': min_height,
            'min_area_threshold': min_area
        }
        
        if output_path and RASTERIO_AVAILABLE:
            _save_obstacle_detection_results(result, output_path)
            
        logger.info(f"Detected {num_features} obstacles with min_height={min_height}m, min_area={min_area}px")
        return result
        
    except Exception as e:
        raise ObstacleDetectionError(f"Failed to detect obstacles: {str(e)}")


def classify_obstacles(
    obstacle_data: Dict[str, Any],
    height_classes: Optional[Dict[str, Tuple[float, float]]] = None,
    area_classes: Optional[Dict[str, Tuple[int, int]]] = None
) -> Dict[str, Any]:
    """
    Classify obstacles by height and area.
    
    Args:
        obstacle_data: Output from detect_obstacles function
        height_classes: Dictionary of height class names and (min, max) ranges
        area_classes: Dictionary of area class names and (min, max) ranges
        
    Returns:
        Dictionary containing obstacle classifications
        
    Raises:
        ObstacleDetectionError: If classification fails
    """
    try:
        if height_classes is None:
            height_classes = {
                'low': (2.0, 5.0),
                'medium': (5.0, 15.0),
                'high': (15.0, 50.0),
                'very_high': (50.0, float('inf'))
            }
            
        if area_classes is None:
            area_classes = {
                'small': (10, 100),
                'medium': (100, 1000),
                'large': (1000, 10000),
                'very_large': (10000, int(1e6))
            }
        
        labeled_obstacles = obstacle_data['labeled_obstacles']
        obstacle_heights = obstacle_data['obstacle_heights']
        obstacle_areas = obstacle_data['obstacle_areas']
        obstacle_max_heights = obstacle_data['obstacle_max_heights']
        
        # Initialize classification arrays
        height_classification = np.zeros_like(labeled_obstacles, dtype=object)
        area_classification = np.zeros_like(labeled_obstacles, dtype=object)
        
        # Classify each obstacle
        obstacle_classifications = []
        
        for i in range(1, obstacle_data['num_obstacles'] + 1):
            obstacle_pixels = labeled_obstacles == i
            max_height = obstacle_max_heights[i - 1]
            area = obstacle_areas[i - 1]
            
            # Classify by height
            height_class = 'unknown'
            for class_name, (min_h, max_h) in height_classes.items():
                if min_h <= max_height < max_h:
                    height_class = class_name
                    break
            
            # Classify by area
            area_class = 'unknown'
            for class_name, (min_a, max_a) in area_classes.items():
                if min_a <= area < max_a:
                    area_class = class_name
                    break
            
            # Apply classifications to pixels
            height_classification[obstacle_pixels] = height_class
            area_classification[obstacle_pixels] = area_class
            
            obstacle_classifications.append({
                'obstacle_id': i,
                'height_class': height_class,
                'area_class': area_class,
                'max_height': max_height,
                'area': area,
                'centroid': obstacle_data['obstacle_centroids'][i - 1]
            })
        
        # Compute class statistics
        height_class_stats = {}
        for class_name in height_classes.keys():
            class_pixels = height_classification == class_name
            height_class_stats[class_name] = {
                'count': np.sum(class_pixels),
                'mean_height': np.mean(obstacle_heights[class_pixels]) if np.any(class_pixels) else 0
            }
        
        area_class_stats = {}
        for class_name in area_classes.keys():
            class_pixels = area_classification == class_name
            area_class_stats[class_name] = {
                'count': np.sum(class_pixels),
                'mean_area': np.mean(np.array(obstacle_areas)[
                    [i for i, oc in enumerate(obstacle_classifications) if oc['area_class'] == class_name]
                ]) if any(oc['area_class'] == class_name for oc in obstacle_classifications) else 0
            }
        
        result = {
            'height_classification': height_classification,
            'area_classification': area_classification,
            'obstacle_classifications': obstacle_classifications,
            'height_class_stats': height_class_stats,
            'area_class_stats': area_class_stats,
            'height_classes': height_classes,
            'area_classes': area_classes
        }
        
        logger.info(f"Classified {len(obstacle_classifications)} obstacles by height and area")
        return result
        
    except Exception as e:
        raise ObstacleDetectionError(f"Failed to classify obstacles: {str(e)}")


def compute_obstacle_mask(
    obstacle_data: Dict[str, Any],
    height_range: Optional[Tuple[float, float]] = None,
    area_range: Optional[Tuple[int, int]] = None,
    classification: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Generate obstacle mask with optional filtering.
    
    Args:
        obstacle_data: Output from detect_obstacles function
        height_range: Optional (min_height, max_height) filter
        area_range: Optional (min_area, max_area) filter
        classification: Optional classification data for class-based filtering
        output_path: Optional path to save obstacle mask
        
    Returns:
        Binary obstacle mask
        
    Raises:
        ObstacleDetectionError: If mask generation fails
    """
    try:
        obstacle_mask = obstacle_data['obstacle_mask'].copy()
        labeled_obstacles = obstacle_data['labeled_obstacles']
        obstacle_heights = obstacle_data['obstacle_heights']
        obstacle_areas = obstacle_data['obstacle_areas']
        
        # Apply height filter
        if height_range is not None:
            min_h, max_h = height_range
            for i in range(1, obstacle_data['num_obstacles'] + 1):
                obstacle_pixels = labeled_obstacles == i
                max_height = np.max(obstacle_heights[obstacle_pixels])
                
                if not (min_h <= max_height < max_h):
                    obstacle_mask[obstacle_pixels] = False
        
        # Apply area filter
        if area_range is not None:
            min_a, max_a = area_range
            for i in range(1, obstacle_data['num_obstacles'] + 1):
                area = obstacle_areas[i - 1]
                obstacle_pixels = labeled_obstacles == i
                
                if not (min_a <= area < max_a):
                    obstacle_mask[obstacle_pixels] = False
        
        # Apply classification filter
        if classification is not None:
            height_classification = classification['height_classification']
            area_classification = classification['area_classification']
            
            # Example: filter by height class
            # valid_height_classes = ['medium', 'high']
            # height_filter = np.isin(height_classification, valid_height_classes)
            # obstacle_mask = obstacle_mask & height_filter
            
            # Example: filter by area class
            # valid_area_classes = ['medium', 'large']
            # area_filter = np.isin(area_classification, valid_area_classes)
            # obstacle_mask = obstacle_mask & area_filter
        
        if output_path and RASTERIO_AVAILABLE:
            _save_obstacle_mask(obstacle_mask, output_path)
            
        logger.info(f"Generated obstacle mask with {np.sum(obstacle_mask)} obstacle pixels")
        return obstacle_mask
        
    except Exception as e:
        raise ObstacleDetectionError(f"Failed to generate obstacle mask: {str(e)}")


def filter_obstacles_by_height(
    obstacle_data: Dict[str, Any],
    min_height: float = 0.0,
    max_height: float = float('inf'),
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Filter obstacles by height range.
    
    Args:
        obstacle_data: Output from detect_obstacles function
        min_height: Minimum obstacle height (meters)
        max_height: Maximum obstacle height (meters)
        output_path: Optional path to save filtered results
        
    Returns:
        Filtered obstacle data
        
    Raises:
        ObstacleDetectionError: If filtering fails
    """
    try:
        filtered_mask = obstacle_data['obstacle_mask'].copy()
        labeled_obstacles = obstacle_data['labeled_obstacles']
        obstacle_heights = obstacle_data['obstacle_heights']
        
        # Filter obstacles by height
        for i in range(1, obstacle_data['num_obstacles'] + 1):
            obstacle_pixels = labeled_obstacles == i
            max_height = np.max(obstacle_heights[obstacle_pixels])
            
            if not (min_height <= max_height < max_height):
                filtered_mask[obstacle_pixels] = False
        
        # Update obstacle data
        filtered_data = obstacle_data.copy()
        filtered_data['obstacle_mask'] = filtered_mask
        
        # Recompute statistics for filtered obstacles
        if SCIPY_AVAILABLE:
            labeled_filtered, num_filtered = ndimage.label(filtered_mask)
            filtered_data['labeled_obstacles'] = labeled_filtered
            filtered_data['num_obstacles'] = num_filtered
            
            # Update other statistics
            filtered_areas = []
            filtered_max_heights = []
            filtered_centroids = []
            
            for i in range(1, num_filtered + 1):
                obstacle_pixels = labeled_filtered == i
                area = np.sum(obstacle_pixels)
                max_height = np.max(obstacle_heights[obstacle_pixels])
                
                filtered_areas.append(area)
                filtered_max_heights.append(max_height)
                
                coords = np.where(obstacle_pixels)
                centroid_y = np.mean(coords[0])
                centroid_x = np.mean(coords[1])
                filtered_centroids.append((centroid_y, centroid_x))
            
            filtered_data['obstacle_areas'] = filtered_areas
            filtered_data['obstacle_max_heights'] = filtered_max_heights
            filtered_data['obstacle_centroids'] = filtered_centroids
        
        if output_path and RASTERIO_AVAILABLE:
            _save_filtered_obstacles(filtered_data, output_path)
            
        logger.info(f"Filtered obstacles by height {min_height}-{max_height}m: "
                   f"{obstacle_data['num_obstacles']} -> {filtered_data['num_obstacles']}")
        return filtered_data
        
    except Exception as e:
        raise ObstacleDetectionError(f"Failed to filter obstacles by height: {str(e)}")


# Internal helper functions

def _save_obstacle_detection_results(obstacle_data: Dict[str, Any], output_path: str):
    """Save obstacle detection results to file."""
    base_path = Path(output_path)
    
    # Save obstacle mask
    mask_path = base_path.with_suffix('.tif')
    with rasterio.open(
        mask_path, 'w',
        driver='GTiff',
        height=obstacle_data['obstacle_mask'].shape[0],
        width=obstacle_data['obstacle_mask'].shape[1],
        count=1,
        dtype=np.uint8
    ) as dst:
        dst.write(obstacle_data['obstacle_mask'].astype(np.uint8), 1)
    
    # Save labeled obstacles
    labeled_path = base_path.with_name(base_path.stem + '_labeled.tif')
    with rasterio.open(
        labeled_path, 'w',
        driver='GTiff',
        height=obstacle_data['labeled_obstacles'].shape[0],
        width=obstacle_data['labeled_obstacles'].shape[1],
        count=1,
        dtype=np.uint8
    ) as dst:
        dst.write(obstacle_data['labeled_obstacles'], 1)
    
    # Save obstacle heights
    heights_path = base_path.with_name(base_path.stem + '_heights.tif')
    with rasterio.open(
        heights_path, 'w',
        driver='GTiff',
        height=obstacle_data['obstacle_heights'].shape[0],
        width=obstacle_data['obstacle_heights'].shape[1],
        count=1,
        dtype=obstacle_data['obstacle_heights'].dtype
    ) as dst:
        dst.write(obstacle_data['obstacle_heights'], 1)


def _save_obstacle_mask(obstacle_mask: np.ndarray, output_path: str):
    """Save obstacle mask to file."""
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=obstacle_mask.shape[0],
        width=obstacle_mask.shape[1],
        count=1,
        dtype=np.uint8
    ) as dst:
        dst.write(obstacle_mask.astype(np.uint8), 1)


def _save_filtered_obstacles(filtered_data: Dict[str, Any], output_path: str):
    """Save filtered obstacle data to file."""
    base_path = Path(output_path)
    
    # Save filtered mask
    mask_path = base_path.with_suffix('.tif')
    with rasterio.open(
        mask_path, 'w',
        driver='GTiff',
        height=filtered_data['obstacle_mask'].shape[0],
        width=filtered_data['obstacle_mask'].shape[1],
        count=1,
        dtype=np.uint8
    ) as dst:
        dst.write(filtered_data['obstacle_mask'].astype(np.uint8), 1)
    
    # Save labeled filtered obstacles
    labeled_path = base_path.with_name(base_path.stem + '_labeled.tif')
    with rasterio.open(
        labeled_path, 'w',
        driver='GTiff',
        height=filtered_data['labeled_obstacles'].shape[0],
        width=filtered_data['labeled_obstacles'].shape[1],
        count=1,
        dtype=np.uint8
    ) as dst:
        dst.write(filtered_data['labeled_obstacles'], 1)
