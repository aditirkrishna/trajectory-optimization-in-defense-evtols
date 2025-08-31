"""
Data processing package for the perception layer.

This package provides data processing utilities including:
- Data ingestion from various geospatial formats
- Data transformation and coordinate system handling
- Data fusion and combination
- Quality control and validation
- Output generation for the planner
"""

from .ingestion import (
    load_raster_data,
    load_vector_data,
    load_netcdf_data,
    load_point_cloud_data,
    get_data_info,
    batch_load_data,
    DataIngestionError
)

from .transformation import (
    reproject_raster,
    reproject_vector,
    resample_raster,
    clip_to_bounds,
    normalize_data,
    apply_transformation_pipeline,
    DataTransformationError
)

from .fusion import (
    fuse_raster_layers,
    fuse_vector_layers,
    combine_datasets,
    create_unified_dataset,
    DataFusionError
)

from .quality_control import (
    perform_quality_checks,
    validate_processing_results,
    generate_quality_report,
    QualityControlError
)

from .output import (
    generate_planner_outputs,
    create_tiled_outputs,
    export_to_formats,
    OutputGenerationError
)

__all__ = [
    # Data ingestion
    "load_raster_data",
    "load_vector_data", 
    "load_netcdf_data",
    "load_point_cloud_data",
    "get_data_info",
    "batch_load_data",
    "DataIngestionError",
    
    # Data transformation
    "reproject_raster",
    "reproject_vector",
    "resample_raster",
    "clip_to_bounds",
    "normalize_data",
    "apply_transformation_pipeline",
    "DataTransformationError",
    
    # Data fusion
    "fuse_raster_layers",
    "fuse_vector_layers",
    "combine_datasets",
    "create_unified_dataset",
    "DataFusionError",
    
    # Quality control
    "perform_quality_checks",
    "validate_processing_results",
    "generate_quality_report",
    "QualityControlError",
    
    # Output generation
    "generate_planner_outputs",
    "create_tiled_outputs",
    "export_to_formats",
    "OutputGenerationError",
]
