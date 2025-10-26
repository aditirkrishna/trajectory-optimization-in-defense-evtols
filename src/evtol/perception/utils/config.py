"""
Configuration management for the perception layer.

This module provides a Config class that loads and validates the YAML configuration
file, making all settings easily accessible throughout the perception layer.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class CoordinateSystem:
    """Coordinate reference system settings."""
    working_crs: str
    input_crs: str
    vertical_datum: str
    units: Dict[str, str]


@dataclass
class SpatialSettings:
    """Spatial resolution and tiling settings."""
    base_resolution: float
    tile_size: int
    altitude_bands: List[float]
    mission_buffer: float


@dataclass
class ProcessingSettings:
    """Data processing parameters."""
    interpolation: Dict[str, str]
    gap_filling: Dict[str, Any]
    smoothing: Dict[str, Any]
    quality_control: Dict[str, Any]


@dataclass
class TerrainSettings:
    """Terrain and geometry parameters."""
    slope: Dict[str, Any]
    roughness: Dict[str, Any]
    obstacles: Dict[str, float]


@dataclass
class UrbanSettings:
    """Urban and building parameters."""
    buildings: Dict[str, Any]
    canyon: Dict[str, Any]


@dataclass
class AtmosphereSettings:
    """Atmospheric parameters."""
    wind: Dict[str, Any]
    turbulence: Dict[str, Any]
    air_density: Dict[str, Any]


@dataclass
class ThreatSettings:
    """Threat and risk parameters."""
    radar: Dict[str, Any]
    patrols: Dict[str, Any]
    ew_zones: Dict[str, float]


@dataclass
class FusionSettings:
    """Data fusion parameters."""
    layers: Dict[str, List[str]]
    uncertainty: Dict[str, Any]


@dataclass
class ServingSettings:
    """Serving and API parameters."""
    tiles: Dict[str, Any]
    api: Dict[str, Any]
    cache: Dict[str, Any]


@dataclass
class GovernanceSettings:
    """Data governance and versioning."""
    versioning: Dict[str, Any]
    provenance: Dict[str, bool]
    quality: Dict[str, Any]


@dataclass
class FormatSettings:
    """File formats and storage."""
    raster: Dict[str, str]
    vector: Dict[str, str]
    point_cloud: Dict[str, str]


@dataclass
class LoggingSettings:
    """Logging configuration."""
    level: str
    file: str
    max_file_size: str
    backup_count: int


class Config:
    """
    Configuration management class for the perception layer.
    
    Loads and validates the YAML configuration file, providing easy access
    to all settings throughout the perception layer.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file. If None, uses default.
        """
        if config_path is None:
            # Use default config path
            config_path = Path(__file__).parent.parent.parent / "config" / "perception_config.yaml"
        
        self.config_path = Path(config_path)
        self._raw_config = self._load_yaml()
        self._validate_config()
        
        # Parse into structured dataclasses
        self.coordinate_system = CoordinateSystem(**self._raw_config["coordinate_system"])
        self.spatial = SpatialSettings(**self._raw_config["spatial"])
        self.processing = ProcessingSettings(**self._raw_config["processing"])
        self.terrain = TerrainSettings(**self._raw_config["terrain"])
        self.urban = UrbanSettings(**self._raw_config["urban"])
        self.atmosphere = AtmosphereSettings(**self._raw_config["atmosphere"])
        self.threats = ThreatSettings(**self._raw_config["threats"])
        self.fusion = FusionSettings(**self._raw_config["fusion"])
        self.serving = ServingSettings(**self._raw_config["serving"])
        self.governance = GovernanceSettings(**self._raw_config["governance"])
        self.formats = FormatSettings(**self._raw_config["formats"])
        self.logging = LoggingSettings(**self._raw_config["logging"])
        
        logger.info(f"Configuration loaded from {self.config_path}")
    
    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")
    
    def _validate_config(self):
        """Validate configuration values."""
        required_sections = [
            "coordinate_system", "spatial", "processing", "terrain",
            "urban", "atmosphere", "threats", "fusion", "serving",
            "governance", "formats", "logging"
        ]
        
        for section in required_sections:
            if section not in self._raw_config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate specific values
        if self._raw_config["spatial"]["base_resolution"] <= 0:
            raise ValueError("Base resolution must be positive")
        
        if self._raw_config["spatial"]["tile_size"] <= 0:
            raise ValueError("Tile size must be positive")
        
        # Validate CRS format
        working_crs = self._raw_config["coordinate_system"]["working_crs"]
        if not working_crs.startswith("EPSG:"):
            logger.warning(f"Working CRS may not be in standard format: {working_crs}")
    
    def get_altitude_bands(self) -> List[float]:
        """Get altitude bands for 3D processing."""
        return self.spatial.altitude_bands
    
    def get_working_crs(self) -> str:
        """Get working coordinate reference system."""
        return self.coordinate_system.working_crs
    
    def get_base_resolution(self) -> float:
        """Get base spatial resolution in meters."""
        return self.spatial.base_resolution
    
    def get_tile_size(self) -> int:
        """Get tile size in pixels."""
        return self.spatial.tile_size
    
    def get_processing_version(self) -> str:
        """Get current processing version."""
        return self.governance.versioning["processing_version"]
    
    def get_data_paths(self) -> Dict[str, Path]:
        """Get data directory paths."""
        base_path = Path(__file__).parent.parent.parent / "data"
        return {
            "raw": base_path / "raw",
            "processed": base_path / "processed", 
            "derived": base_path / "derived"
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "coordinate_system": self._raw_config["coordinate_system"],
            "spatial": self._raw_config["spatial"],
            "processing": self._raw_config["processing"],
            "terrain": self._raw_config["terrain"],
            "urban": self._raw_config["urban"],
            "atmosphere": self._raw_config["atmosphere"],
            "threats": self._raw_config["threats"],
            "fusion": self._raw_config["fusion"],
            "serving": self._raw_config["serving"],
            "governance": self._raw_config["governance"],
            "formats": self._raw_config["formats"],
            "logging": self._raw_config["logging"]
        }
    
    def save(self, output_path: str):
        """Save configuration to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {output_path}")
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(working_crs='{self.get_working_crs()}', resolution={self.get_base_resolution()}m)"


# Convenience function for quick config access
def get_config(config_path: Optional[str] = None) -> Config:
    """Get configuration instance."""
    return Config(config_path)
