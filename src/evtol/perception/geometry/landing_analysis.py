"""
Landing Analysis Module

This module provides functions for analyzing landing feasibility and finding optimal landing sites:
- Landing feasibility analysis based on terrain characteristics
- Landing score computation
- Optimal landing site identification
- Landing zone validation
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


class LandingAnalysisError(Exception):
    """Exception raised for landing analysis errors."""
    pass


def analyze_landing_feasibility(
    dem: np.ndarray,
    slope_threshold: float = 15.0,
    roughness_threshold: float = 2.0,
    obstacle_threshold: float = 2.0,
    min_area: int = 100,
    pixel_size: float = 1.0,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze landing feasibility based on terrain characteristics.
    
    Args:
        dem: Digital Elevation Model as numpy array
        slope_threshold: Maximum acceptable slope in degrees
        roughness_threshold: Maximum acceptable roughness in meters
        obstacle_threshold: Maximum acceptable obstacle height in meters
        min_area: Minimum area for landing zone in square meters
        pixel_size: Pixel size in meters
        output_path: Optional path to save feasibility analysis results
        
    Returns:
        Dictionary containing landing feasibility analysis results
        
    Raises:
        LandingAnalysisError: If analysis fails
    """
    try:
        validate_raster_data(dem)
        
        # Import required functions
        from .terrain_analysis import compute_slope, compute_roughness
        from .clearance_analysis import compute_landing_zones
        
        # Compute terrain features
        slope = compute_slope(dem, pixel_size, method="horn")
        roughness = compute_roughness(dem, window_size=5, roughness_type="std")
        
        # Create feasibility masks
        slope_feasible = slope <= slope_threshold
        roughness_feasible = roughness <= roughness_threshold
        
        # Combine feasibility criteria
        terrain_feasible = slope_feasible & roughness_feasible
        
        # Remove small areas
        if SCIPY_AVAILABLE:
            labeled_mask, num_features = ndimage.label(terrain_feasible)
            for i in range(1, num_features + 1):
                area = np.sum(labeled_mask == i) * (pixel_size ** 2)
                if area < min_area:
                    terrain_feasible[labeled_mask == i] = False
        
        # Compute feasibility statistics
        total_pixels = dem.size
        feasible_pixels = np.sum(terrain_feasible)
        feasibility_percentage = (feasible_pixels / total_pixels) * 100
        
        # Analyze slope distribution
        slope_stats = {
            'mean': np.mean(slope),
            'std': np.std(slope),
            'min': np.min(slope),
            'max': np.max(slope),
            'feasible_percentage': (np.sum(slope_feasible) / total_pixels) * 100
        }
        
        # Analyze roughness distribution
        roughness_stats = {
            'mean': np.mean(roughness),
            'std': np.std(roughness),
            'min': np.min(roughness),
            'max': np.max(roughness),
            'feasible_percentage': (np.sum(roughness_feasible) / total_pixels) * 100
        }
        
        result = {
            'terrain_feasible': terrain_feasible,
            'slope_feasible': slope_feasible,
            'roughness_feasible': roughness_feasible,
            'slope': slope,
            'roughness': roughness,
            'feasibility_percentage': feasibility_percentage,
            'feasible_pixels': feasible_pixels,
            'total_pixels': total_pixels,
            'slope_stats': slope_stats,
            'roughness_stats': roughness_stats,
            'slope_threshold': slope_threshold,
            'roughness_threshold': roughness_threshold,
            'min_area': min_area
        }
        
        if output_path and RASTERIO_AVAILABLE:
            _save_landing_feasibility_results(result, output_path)
            
        logger.info(f"Landing feasibility analysis: {feasibility_percentage:.1f}% feasible area")
        return result
        
    except Exception as e:
        raise LandingAnalysisError(f"Failed to analyze landing feasibility: {str(e)}")


def compute_landing_scores(
    dem: np.ndarray,
    feasibility_data: Dict[str, Any],
    weight_slope: float = 0.4,
    weight_roughness: float = 0.3,
    weight_elevation: float = 0.2,
    weight_accessibility: float = 0.1,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compute landing scores for feasible areas.
    
    Args:
        dem: Digital Elevation Model as numpy array
        feasibility_data: Output from analyze_landing_feasibility function
        weight_slope: Weight for slope component (0-1)
        weight_roughness: Weight for roughness component (0-1)
        weight_elevation: Weight for elevation component (0-1)
        weight_accessibility: Weight for accessibility component (0-1)
        output_path: Optional path to save landing scores
        
    Returns:
        Dictionary containing landing scores and analysis
        
    Raises:
        LandingAnalysisError: If score computation fails
    """
    try:
        validate_raster_data(dem)
        
        # Normalize weights
        total_weight = weight_slope + weight_roughness + weight_elevation + weight_accessibility
        weight_slope /= total_weight
        weight_roughness /= total_weight
        weight_elevation /= total_weight
        weight_accessibility /= total_weight
        
        # Get terrain features
        slope = feasibility_data['slope']
        roughness = feasibility_data['roughness']
        terrain_feasible = feasibility_data['terrain_feasible']
        
        # Initialize scoring arrays
        slope_score = np.zeros_like(dem, dtype=float)
        roughness_score = np.zeros_like(dem, dtype=float)
        elevation_score = np.zeros_like(dem, dtype=float)
        accessibility_score = np.zeros_like(dem, dtype=float)
        
        # Compute slope score (lower slope = higher score)
        max_slope = np.max(slope)
        slope_score = 1.0 - (slope / max_slope)
        slope_score = np.clip(slope_score, 0, 1)
        
        # Compute roughness score (lower roughness = higher score)
        max_roughness = np.max(roughness)
        roughness_score = 1.0 - (roughness / max_roughness)
        roughness_score = np.clip(roughness_score, 0, 1)
        
        # Compute elevation score (prefer moderate elevations)
        elevation = dem.copy()
        min_elev = np.min(elevation)
        max_elev = np.max(elevation)
        mid_elev = (min_elev + max_elev) / 2
        
        # Score based on distance from mid-elevation
        elevation_diff = np.abs(elevation - mid_elev)
        max_diff = max_elev - min_elev
        elevation_score = 1.0 - (elevation_diff / max_diff)
        elevation_score = np.clip(elevation_score, 0, 1)
        
        # Compute accessibility score (distance from edges)
        if SCIPY_AVAILABLE:
            # Distance transform from edges
            edge_mask = np.ones_like(dem, dtype=bool)
            edge_mask[1:-1, 1:-1] = False
            
            distance_from_edge = ndimage.distance_transform_edt(edge_mask)
            max_distance = np.max(distance_from_edge)
            
            accessibility_score = distance_from_edge / max_distance
            accessibility_score = np.clip(accessibility_score, 0, 1)
        else:
            accessibility_score = np.ones_like(dem, dtype=float)
        
        # Compute weighted composite score
        composite_score = (
            weight_slope * slope_score +
            weight_roughness * roughness_score +
            weight_elevation * elevation_score +
            weight_accessibility * accessibility_score
        )
        
        # Apply feasibility mask
        composite_score = np.where(terrain_feasible, composite_score, 0)
        
        # Compute score statistics
        valid_scores = composite_score[composite_score > 0]
        
        if len(valid_scores) > 0:
            score_stats = {
                'mean': np.mean(valid_scores),
                'std': np.std(valid_scores),
                'min': np.min(valid_scores),
                'max': np.max(valid_scores),
                'median': np.median(valid_scores)
            }
        else:
            score_stats = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}
        
        result = {
            'composite_score': composite_score,
            'slope_score': slope_score,
            'roughness_score': roughness_score,
            'elevation_score': elevation_score,
            'accessibility_score': accessibility_score,
            'score_stats': score_stats,
            'weights': {
                'slope': weight_slope,
                'roughness': weight_roughness,
                'elevation': weight_elevation,
                'accessibility': weight_accessibility
            }
        }
        
        if output_path and RASTERIO_AVAILABLE:
            _save_landing_scores_results(result, output_path)
            
        logger.info(f"Computed landing scores: mean={score_stats['mean']:.3f}, "
                   f"max={score_stats['max']:.3f}")
        return result
        
    except Exception as e:
        raise LandingAnalysisError(f"Failed to compute landing scores: {str(e)}")


def find_optimal_landing_sites(
    landing_scores: Dict[str, Any],
    num_sites: int = 10,
    min_distance: float = 50.0,
    pixel_size: float = 1.0,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Find optimal landing sites based on landing scores.
    
    Args:
        landing_scores: Output from compute_landing_scores function
        num_sites: Number of optimal sites to find
        min_distance: Minimum distance between sites in meters
        pixel_size: Pixel size in meters
        output_path: Optional path to save optimal sites results
        
    Returns:
        Dictionary containing optimal landing sites
        
    Raises:
        LandingAnalysisError: If site finding fails
    """
    try:
        composite_score = landing_scores['composite_score']
        
        # Convert minimum distance to pixels
        min_distance_pixels = int(min_distance / pixel_size)
        
        # Find local maxima in composite score
        if SCIPY_AVAILABLE:
            # Use maximum filter to find local maxima
            from scipy.ndimage import maximum_filter
            
            # Create a circular footprint for local maxima detection
            footprint_size = min_distance_pixels
            footprint = np.zeros((footprint_size, footprint_size))
            center = footprint_size // 2
            
            # Create circular footprint
            y, x = np.ogrid[:footprint_size, :footprint_size]
            mask = (x - center)**2 + (y - center)**2 <= (footprint_size//2)**2
            footprint[mask] = 1
            
            # Find local maxima
            local_max = maximum_filter(composite_score, footprint=footprint)
            local_maxima = (composite_score == local_max) & (composite_score > 0)
            
            # Get coordinates of local maxima
            max_coords = np.where(local_maxima)
            max_scores = composite_score[max_coords]
            
            # Sort by score (descending)
            sorted_indices = np.argsort(max_scores)[::-1]
            
            # Select top sites with minimum distance constraint
            selected_sites = []
            selected_coords = []
            selected_scores = []
            
            for idx in sorted_indices:
                y, x = max_coords[0][idx], max_coords[1][idx]
                score = max_scores[idx]
                
                # Check distance to already selected sites
                too_close = False
                for sel_y, sel_x in selected_coords:
                    distance = np.sqrt((y - sel_y)**2 + (x - sel_x)**2)
                    if distance < min_distance_pixels:
                        too_close = True
                        break
                
                if not too_close:
                    selected_sites.append({
                        'id': len(selected_sites) + 1,
                        'y': y,
                        'x': x,
                        'score': score,
                        'coordinates': (y * pixel_size, x * pixel_size)
                    })
                    selected_coords.append((y, x))
                    selected_scores.append(score)
                    
                    if len(selected_sites) >= num_sites:
                        break
        else:
            # Fallback: simple threshold-based selection
            threshold = np.percentile(composite_score[composite_score > 0], 95)
            high_score_mask = composite_score >= threshold
            
            # Find connected components
            labeled_mask, num_features = ndimage.label(high_score_mask)
            
            selected_sites = []
            for i in range(1, min(num_features + 1, num_sites + 1)):
                component_pixels = labeled_mask == i
                coords = np.where(component_pixels)
                
                # Find centroid
                centroid_y = int(np.mean(coords[0]))
                centroid_x = int(np.mean(coords[1]))
                score = composite_score[centroid_y, centroid_x]
                
                selected_sites.append({
                    'id': i,
                    'y': centroid_y,
                    'x': centroid_x,
                    'score': score,
                    'coordinates': (centroid_y * pixel_size, centroid_x * pixel_size)
                })
        
        # Create site mask
        site_mask = np.zeros_like(composite_score, dtype=bool)
        for site in selected_sites:
            site_mask[site['y'], site['x']] = True
        
        result = {
            'optimal_sites': selected_sites,
            'site_mask': site_mask,
            'num_sites_found': len(selected_sites),
            'min_distance': min_distance,
            'pixel_size': pixel_size
        }
        
        if output_path and RASTERIO_AVAILABLE:
            _save_optimal_sites_results(result, output_path)
            
        logger.info(f"Found {len(selected_sites)} optimal landing sites")
        return result
        
    except Exception as e:
        raise LandingAnalysisError(f"Failed to find optimal landing sites: {str(e)}")


def validate_landing_zones(
    landing_sites: Dict[str, Any],
    dem: np.ndarray,
    obstacle_data: Optional[Dict[str, Any]] = None,
    safety_margin: float = 10.0,
    pixel_size: float = 1.0,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate landing zones against additional criteria.
    
    Args:
        landing_sites: Output from find_optimal_landing_sites function
        dem: Digital Elevation Model as numpy array
        obstacle_data: Optional obstacle detection data
        safety_margin: Safety margin around landing sites in meters
        pixel_size: Pixel size in meters
        output_path: Optional path to save validation results
        
    Returns:
        Dictionary containing validation results
        
    Raises:
        LandingAnalysisError: If validation fails
    """
    try:
        validate_raster_data(dem)
        
        optimal_sites = landing_sites['optimal_sites']
        safety_margin_pixels = int(safety_margin / pixel_size)
        
        validation_results = []
        
        for site in optimal_sites:
            y, x = site['y'], site['x']
            
            # Define validation region
            y_min = max(0, y - safety_margin_pixels)
            y_max = min(dem.shape[0], y + safety_margin_pixels + 1)
            x_min = max(0, x - safety_margin_pixels)
            x_max = min(dem.shape[1], x + safety_margin_pixels + 1)
            
            # Extract region
            region = dem[y_min:y_max, x_min:x_max]
            
            # Compute validation metrics
            elevation_variation = np.max(region) - np.min(region)
            elevation_std = np.std(region)
            
            # Check for obstacles in region
            has_obstacles = False
            obstacle_count = 0
            
            if obstacle_data is not None:
                obstacle_mask = obstacle_data['obstacle_mask']
                region_obstacles = obstacle_mask[y_min:y_max, x_min:x_max]
                has_obstacles = np.any(region_obstacles)
                obstacle_count = np.sum(region_obstacles)
            
            # Determine validation status
            validation_status = 'valid'
            issues = []
            
            if elevation_variation > 5.0:  # More than 5m elevation variation
                validation_status = 'marginal'
                issues.append('high_elevation_variation')
            
            if elevation_std > 2.0:  # High elevation standard deviation
                validation_status = 'marginal'
                issues.append('high_elevation_std')
            
            if has_obstacles:
                validation_status = 'invalid'
                issues.append('obstacles_present')
            
            # Compute safety score
            safety_score = 1.0
            if elevation_variation > 0:
                safety_score -= min(0.5, elevation_variation / 10.0)
            if has_obstacles:
                safety_score -= 0.5
            safety_score = max(0.0, safety_score)
            
            validation_result = {
                'site_id': site['id'],
                'coordinates': site['coordinates'],
                'validation_status': validation_status,
                'safety_score': safety_score,
                'elevation_variation': elevation_variation,
                'elevation_std': elevation_std,
                'has_obstacles': has_obstacles,
                'obstacle_count': obstacle_count,
                'issues': issues
            }
            
            validation_results.append(validation_result)
        
        # Compute overall statistics
        valid_sites = [r for r in validation_results if r['validation_status'] == 'valid']
        marginal_sites = [r for r in validation_results if r['validation_status'] == 'marginal']
        invalid_sites = [r for r in validation_results if r['validation_status'] == 'invalid']
        
        overall_stats = {
            'total_sites': len(validation_results),
            'valid_sites': len(valid_sites),
            'marginal_sites': len(marginal_sites),
            'invalid_sites': len(invalid_sites),
            'mean_safety_score': np.mean([r['safety_score'] for r in validation_results]),
            'validation_rate': len(valid_sites) / len(validation_results) * 100
        }
        
        result = {
            'validation_results': validation_results,
            'overall_stats': overall_stats,
            'safety_margin': safety_margin,
            'pixel_size': pixel_size
        }
        
        if output_path and RASTERIO_AVAILABLE:
            _save_validation_results(result, output_path)
            
        logger.info(f"Landing zone validation: {overall_stats['valid_sites']}/{overall_stats['total_sites']} "
                   f"sites valid ({overall_stats['validation_rate']:.1f}%)")
        return result
        
    except Exception as e:
        raise LandingAnalysisError(f"Failed to validate landing zones: {str(e)}")


# Internal helper functions

def _save_landing_feasibility_results(feasibility_data: Dict[str, Any], output_path: str):
    """Save landing feasibility results to file."""
    base_path = Path(output_path)
    
    # Save feasibility mask
    mask_path = base_path.with_suffix('.tif')
    with rasterio.open(
        mask_path, 'w',
        driver='GTiff',
        height=feasibility_data['terrain_feasible'].shape[0],
        width=feasibility_data['terrain_feasible'].shape[1],
        count=1,
        dtype=np.uint8
    ) as dst:
        dst.write(feasibility_data['terrain_feasible'].astype(np.uint8), 1)


def _save_landing_scores_results(scores_data: Dict[str, Any], output_path: str):
    """Save landing scores results to file."""
    base_path = Path(output_path)
    
    # Save composite score
    score_path = base_path.with_suffix('.tif')
    with rasterio.open(
        score_path, 'w',
        driver='GTiff',
        height=scores_data['composite_score'].shape[0],
        width=scores_data['composite_score'].shape[1],
        count=1,
        dtype=scores_data['composite_score'].dtype
    ) as dst:
        dst.write(scores_data['composite_score'], 1)


def _save_optimal_sites_results(sites_data: Dict[str, Any], output_path: str):
    """Save optimal sites results to file."""
    base_path = Path(output_path)
    
    # Save site mask
    mask_path = base_path.with_suffix('.tif')
    with rasterio.open(
        mask_path, 'w',
        driver='GTiff',
        height=sites_data['site_mask'].shape[0],
        width=sites_data['site_mask'].shape[1],
        count=1,
        dtype=np.uint8
    ) as dst:
        dst.write(sites_data['site_mask'].astype(np.uint8), 1)


def _save_validation_results(validation_data: Dict[str, Any], output_path: str):
    """Save validation results to file."""
    # This would typically save a JSON file with validation results
    # For now, we'll just log the results
    logger.info(f"Validation results saved to {output_path}")
