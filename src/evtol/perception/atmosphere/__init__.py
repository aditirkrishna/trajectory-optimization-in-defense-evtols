"""
Atmospheric modeling module for eVTOL perception layer.

Provides wind field interpolation, turbulence modeling, and
atmospheric properties calculation.
"""

from .wind_model import (
    WindVector,
    WindMeasurement,
    WindFieldModel,
    load_wind_data_from_csv,
    generate_synthetic_wind_field
)

from .turbulence_model import (
    TurbulenceClass,
    TurbulenceMetrics,
    TurbulenceModel,
    get_atmospheric_stability
)

__all__ = [
    # Wind modeling
    "WindVector",
    "WindMeasurement",
    "WindFieldModel",
    "load_wind_data_from_csv",
    "generate_synthetic_wind_field",
    
    # Turbulence modeling
    "TurbulenceClass",
    "TurbulenceMetrics",
    "TurbulenceModel",
    "get_atmospheric_stability",
]

