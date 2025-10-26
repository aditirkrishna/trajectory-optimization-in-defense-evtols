"""
Perception and Environment Layer for eVTOL Trajectory Optimization

This package provides spatiotemporal maps for eVTOL planning including:
- Terrain and obstacle data
- Atmospheric conditions (wind, turbulence, air density)
- Threat and risk assessment
- Data fusion and uncertainty quantification

The layer produces accurate, versioned, georeferenced, uncertainty-aware,
and fast-to-query maps for the planner.
"""

__version__ = "1.0.0"
__author__ = "eVTOL Perception Team"
__email__ = "perception@evtol-defense.com"

# Import main components for easy access
from .utils.config import Config
from .utils.logging import setup_logging

# Import processing components
from .processing import (
    # Data ingestion
    load_raster_data,
    load_vector_data,
    load_netcdf_data,
    load_point_cloud_data,
    batch_load_data,
    get_data_info,
    # Data transformation
    reproject_raster,
    reproject_vector,
    resample_raster,
    clip_to_bounds,
    normalize_data,
    apply_transformation_pipeline,
    # Data fusion
    fuse_raster_layers,
    fuse_vector_layers,
    combine_datasets,
    create_unified_dataset,
    # Quality control
    perform_quality_checks,
    validate_processing_results,
    generate_quality_report,
    # Output generation
    generate_planner_outputs,
    create_tiled_outputs,
    export_to_formats
)

# Import geometry components
from .geometry import (
    # Terrain analysis
    compute_slope,
    compute_aspect,
    compute_curvature,
    compute_roughness,
    compute_terrain_features,
    TerrainAnalysisError,
    # Clearance analysis
    compute_clearance,
    compute_obstacle_height,
    compute_landing_zones,
    compute_corridor_clearance,
    ClearanceAnalysisError,
    # Obstacle detection
    detect_obstacles,
    classify_obstacles,
    compute_obstacle_mask,
    filter_obstacles_by_height,
    ObstacleDetectionError,
    # Landing analysis
    analyze_landing_feasibility,
    compute_landing_scores,
    find_optimal_landing_sites,
    validate_landing_zones,
    LandingAnalysisError
)

# Main classes for users
__all__ = [
    "Config",
    "setup_logging",
    # Data ingestion
    "load_raster_data",
    "load_vector_data", 
    "load_netcdf_data",
    "load_point_cloud_data",
    "batch_load_data",
    "get_data_info",
    # Data transformation
    "reproject_raster",
    "reproject_vector",
    "resample_raster",
    "clip_to_bounds",
    "normalize_data",
    "apply_transformation_pipeline",
    # Data fusion
    "fuse_raster_layers",
    "fuse_vector_layers",
    "combine_datasets",
    "create_unified_dataset",
    # Quality control
    "perform_quality_checks",
    "validate_processing_results",
    "generate_quality_report",
    # Output generation
    "generate_planner_outputs",
    "create_tiled_outputs",
    "export_to_formats",
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

# Configuration management
def load_config(config_path: str = None) -> Config:
    """Load configuration from YAML file."""
    return Config(config_path)

# Quick setup function
def setup_perception_layer(config_path: str = None, log_level: str = "INFO"):
    """Quick setup of the perception layer with logging."""
    config = load_config(config_path)
    logger = setup_logging(config.logging, log_level)
    return config, logger

