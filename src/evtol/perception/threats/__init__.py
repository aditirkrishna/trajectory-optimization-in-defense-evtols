"""
Threat assessment module for eVTOL defense operations.

Provides radar detection modeling, patrol coverage analysis,
and electronic warfare zone assessment.
"""

from .radar_model import (
    RadarType,
    RadarSite,
    DetectionResult,
    RadarDetectionModel,
    RadarNetwork,
    load_radar_sites_from_csv
)

from .patrol_model import (
    PatrolWaypoint,
    PatrolRoute,
    EncounterResult,
    PatrolModel,
    EWZone,
    ElectronicWarfareModel
)

__all__ = [
    # Radar modeling
    "RadarType",
    "RadarSite",
    "DetectionResult",
    "RadarDetectionModel",
    "RadarNetwork",
    "load_radar_sites_from_csv",
    
    # Patrol modeling
    "PatrolWaypoint",
    "PatrolRoute",
    "EncounterResult",
    "PatrolModel",
    
    # Electronic warfare
    "EWZone",
    "ElectronicWarfareModel",
]

