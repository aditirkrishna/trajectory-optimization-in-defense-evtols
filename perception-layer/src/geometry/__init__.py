"""
Geometry Derivatives Package

This package provides functions for computing terrain features and geometry derivatives
that planners need directly - slope, aspect, clearance, roughness, and obstacle detection.
"""

from .terrain_analysis import (
    compute_slope,
    compute_aspect,
    compute_curvature,
    compute_roughness,
    compute_terrain_features,
    TerrainAnalysisError
)

from .clearance_analysis import (
    compute_clearance,
    compute_obstacle_height,
    compute_landing_zones,
    compute_corridor_clearance,
    ClearanceAnalysisError
)

from .obstacle_detection import (
    detect_obstacles,
    classify_obstacles,
    compute_obstacle_mask,
    filter_obstacles_by_height,
    ObstacleDetectionError
)

from .landing_analysis import (
    analyze_landing_feasibility,
    compute_landing_scores,
    find_optimal_landing_sites,
    validate_landing_zones,
    LandingAnalysisError
)

__all__ = [
    # Terrain analysis
    "compute_slope",
    "compute_aspect", 
    "compute_curvature",
    "compute_roughness",
    "compute_terrain_features",
    "TerrainAnalysisError",
    
    # Clearance analysis
    "compute_clearance",
    "compute_obstacle_height",
    "compute_landing_zones",
    "compute_corridor_clearance",
    "ClearanceAnalysisError",
    
    # Obstacle detection
    "detect_obstacles",
    "classify_obstacles",
    "compute_obstacle_mask",
    "filter_obstacles_by_height",
    "ObstacleDetectionError",
    
    # Landing analysis
    "analyze_landing_feasibility",
    "compute_landing_scores",
    "find_optimal_landing_sites",
    "validate_landing_zones",
    "LandingAnalysisError"
]
