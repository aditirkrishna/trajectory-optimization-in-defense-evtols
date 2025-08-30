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
from .preprocessing.data_loader import DataLoader
from .fusion.map_fusion import MapFusion
from .serving.api import PerceptionAPI

# Main classes for users
__all__ = [
    "Config",
    "DataLoader", 
    "MapFusion",
    "PerceptionAPI",
    "setup_logging"
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

