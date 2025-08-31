"""
Urban Obstacles and 3D Objects Package

This package provides comprehensive urban obstacle processing including:
- Building footprint processing and validation
- 3D volume generation and extrusion
- Occupancy grid creation at multiple altitude bands
- Shadow zone and radar occlusion analysis
- Fast collision checking for planners
"""

from .building_processing import (
    # Building footprint processing
    load_building_footprints,
    validate_building_data,
    clean_building_geometries,
    extrude_buildings_to_3d,
    BuildingProcessingError,
    
    # Building height processing
    estimate_building_heights,
    validate_height_data,
    interpolate_missing_heights,
    HeightProcessingError
)

from .volume_generation import (
    # 3D volume creation
    create_3d_volumes,
    voxelize_buildings,
    create_occupancy_grid,
    generate_altitude_bands,
    VolumeGenerationError,
    
    # Volume optimization
    optimize_volume_representation,
    compress_occupancy_data,
    VolumeOptimizationError
)

from .shadow_analysis import (
    # Shadow zone computation
    compute_shadow_zones,
    calculate_radar_occlusion,
    analyze_line_of_sight,
    ShadowAnalysisError,
    
    # Ray casting utilities
    perform_ray_casting,
    compute_occlusion_maps,
    RayCastingError
)

from .collision_detection import (
    # Fast collision checking
    check_collision_at_altitude,
    batch_collision_check,
    create_collision_cache,
    CollisionDetectionError,
    
    # Collision optimization
    optimize_collision_queries,
    CollisionOptimizationError
)

from .urban_analysis import (
    # Urban environment analysis
    analyze_urban_density,
    compute_building_statistics,
    identify_landing_zones,
    UrbanAnalysisError,
    
    # Urban planning utilities
    find_optimal_routes,
    compute_urban_metrics,
    UrbanPlanningError
)

__all__ = [
    # Building processing
    "load_building_footprints",
    "validate_building_data", 
    "clean_building_geometries",
    "extrude_buildings_to_3d",
    "BuildingProcessingError",
    
    # Height processing
    "estimate_building_heights",
    "validate_height_data",
    "interpolate_missing_heights", 
    "HeightProcessingError",
    
    # Volume generation
    "create_3d_volumes",
    "voxelize_buildings",
    "create_occupancy_grid",
    "generate_altitude_bands",
    "VolumeGenerationError",
    
    # Volume optimization
    "optimize_volume_representation",
    "compress_occupancy_data",
    "VolumeOptimizationError",
    
    # Shadow analysis
    "compute_shadow_zones",
    "calculate_radar_occlusion",
    "analyze_line_of_sight",
    "ShadowAnalysisError",
    
    # Ray casting
    "perform_ray_casting",
    "compute_occlusion_maps",
    "RayCastingError",
    
    # Collision detection
    "check_collision_at_altitude",
    "batch_collision_check",
    "create_collision_cache",
    "CollisionDetectionError",
    
    # Collision optimization
    "optimize_collision_queries",
    "CollisionOptimizationError",
    
    # Urban analysis
    "analyze_urban_density",
    "compute_building_statistics",
    "identify_landing_zones",
    "UrbanAnalysisError",
    
    # Urban planning
    "find_optimal_routes",
    "compute_urban_metrics",
    "UrbanPlanningError"
]
