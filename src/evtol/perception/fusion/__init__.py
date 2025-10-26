"""
Data fusion module for multi-layer perception.

Combines terrain, atmospheric, and threat data into unified
risk and energy maps with uncertainty quantification.
"""

from .data_fusion import (
    FusionMethod,
    LayerData,
    FusedResult,
    PerceptionFusion,
    UncertaintyPropagation,
    compute_calibration_metrics
)

__all__ = [
    "FusionMethod",
    "LayerData",
    "FusedResult",
    "PerceptionFusion",
    "UncertaintyPropagation",
    "compute_calibration_metrics",
]

