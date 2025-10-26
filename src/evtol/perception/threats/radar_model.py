"""
Radar Detection Modeling

Models radar detection probability for eVTOL aircraft considering:
- Line-of-sight
- Radar cross-section (RCS)
- Range and altitude
- Terrain masking
- Weather attenuation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RadarType(Enum):
    """Radar system types"""
    SURVEILLANCE = "surveillance"
    TRACKING = "tracking"
    FIRE_CONTROL = "fire_control"
    WEATHER = "weather"


@dataclass
class RadarSite:
    """Radar site configuration"""
    id: str
    lat: float
    lon: float
    elevation_m: float  # Site elevation above sea level
    antenna_height_m: float  # Antenna height above ground
    radar_type: RadarType
    frequency_ghz: float
    max_range_km: float
    peak_power_kw: float
    antenna_gain_db: float
    beamwidth_deg: float
    scan_rate_rpm: Optional[float] = None
    min_elevation_deg: float = 0.0
    max_elevation_deg: float = 90.0
    azimuth_coverage_deg: Tuple[float, float] = (0.0, 360.0)
    operational: bool = True


@dataclass
class DetectionResult:
    """Radar detection analysis result"""
    detected: bool
    detection_prob: float
    range_km: float
    elevation_angle_deg: float
    azimuth_angle_deg: float
    snr_db: float
    los_blocked: bool
    terrain_masking_factor: float


class RadarDetectionModel:
    """
    Radar detection probability model.
    
    Implements radar equation with:
    - Range losses
    - Atmospheric attenuation
    - Terrain masking
    - RCS variations
    - Probabilistic detection
    """
    
    def __init__(self, target_rcs_m2: float = 0.5):
        """
        Initialize radar detection model.
        
        Args:
            target_rcs_m2: Target radar cross-section in square meters
                          (typical small UAV: 0.01-1.0 m²)
        """
        self.target_rcs = target_rcs_m2
        self.earth_radius_km = 6371.0
        
        logger.info(f"Initialized radar model with target RCS = {target_rcs_m2} m²")
    
    def compute_detection_probability(
        self,
        radar: RadarSite,
        target_lat: float,
        target_lon: float,
        target_alt_m: float,
        terrain_elevation_m: Optional[float] = None,
        weather_attenuation_db: float = 0.0
    ) -> DetectionResult:
        """
        Compute radar detection probability for a target.
        
        Args:
            radar: Radar site configuration
            target_lat: Target latitude
            target_lon: Target longitude
            target_alt_m: Target altitude ASL
            terrain_elevation_m: Terrain elevation at target location
            weather_attenuation_db: Additional weather attenuation in dB
            
        Returns:
            DetectionResult with probability and geometry
        """
        # Compute geometry
        range_km, elevation_deg, azimuth_deg = self._compute_geometry(
            radar.lat, radar.lon, radar.elevation_m + radar.antenna_height_m,
            target_lat, target_lon, target_alt_m
        )
        
        # Check if target is within radar coverage
        if range_km > radar.max_range_km:
            return self._no_detection_result(range_km, elevation_deg, azimuth_deg, "out_of_range")
        
        if elevation_deg < radar.min_elevation_deg or elevation_deg > radar.max_elevation_deg:
            return self._no_detection_result(range_km, elevation_deg, azimuth_deg, "outside_elevation")
        
        if not self._in_azimuth_coverage(azimuth_deg, radar.azimuth_coverage_deg):
            return self._no_detection_result(range_km, elevation_deg, azimuth_deg, "outside_azimuth")
        
        # Check line-of-sight and terrain masking
        los_blocked, terrain_factor = self._check_terrain_masking(
            radar, target_lat, target_lon, target_alt_m, terrain_elevation_m
        )
        
        if los_blocked:
            return self._no_detection_result(
                range_km, elevation_deg, azimuth_deg, "terrain_masked",
                terrain_masking_factor=terrain_factor
            )
        
        # Compute radar equation SNR
        snr_db = self._compute_snr(
            radar, range_km, weather_attenuation_db
        )
        
        # Convert SNR to detection probability
        detection_prob = self._snr_to_detection_prob(snr_db)
        
        # Apply terrain masking factor
        detection_prob *= (1.0 - terrain_factor)
        
        return DetectionResult(
            detected=(detection_prob > 0.5),
            detection_prob=detection_prob,
            range_km=range_km,
            elevation_angle_deg=elevation_deg,
            azimuth_angle_deg=azimuth_deg,
            snr_db=snr_db,
            los_blocked=los_blocked,
            terrain_masking_factor=terrain_factor
        )
    
    def _compute_geometry(
        self,
        radar_lat: float, radar_lon: float, radar_alt_m: float,
        target_lat: float, target_lon: float, target_alt_m: float
    ) -> Tuple[float, float, float]:
        """Compute range, elevation, and azimuth from radar to target."""
        # Haversine distance (horizontal)
        dlat = np.radians(target_lat - radar_lat)
        dlon = np.radians(target_lon - radar_lon)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(radar_lat)) * np.cos(np.radians(target_lat)) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        horizontal_range_km = self.earth_radius_km * c
        
        # Altitude difference
        alt_diff_km = (target_alt_m - radar_alt_m) / 1000.0
        
        # Slant range
        range_km = np.sqrt(horizontal_range_km**2 + alt_diff_km**2)
        
        # Elevation angle
        elevation_deg = np.degrees(np.arctan2(alt_diff_km, horizontal_range_km))
        
        # Azimuth angle (bearing from radar to target)
        y = np.sin(dlon) * np.cos(np.radians(target_lat))
        x = np.cos(np.radians(radar_lat)) * np.sin(np.radians(target_lat)) - \
            np.sin(np.radians(radar_lat)) * np.cos(np.radians(target_lat)) * np.cos(dlon)
        azimuth_deg = (np.degrees(np.arctan2(y, x)) + 360) % 360
        
        return float(range_km), float(elevation_deg), float(azimuth_deg)
    
    def _in_azimuth_coverage(self, azimuth: float, coverage: Tuple[float, float]) -> bool:
        """Check if azimuth is within coverage sector."""
        start, end = coverage
        if start <= end:
            return start <= azimuth <= end
        else:  # Sector crosses 0°
            return azimuth >= start or azimuth <= end
    
    def _check_terrain_masking(
        self,
        radar: RadarSite,
        target_lat: float,
        target_lon: float,
        target_alt_m: float,
        terrain_elevation_m: Optional[float]
    ) -> Tuple[bool, float]:
        """
        Check if terrain blocks line-of-sight.
        
        Returns:
            (los_blocked, masking_factor)
        """
        # Simplified terrain masking
        # In production, would sample terrain along ray path
        
        if terrain_elevation_m is None:
            return False, 0.0
        
        # Check if target is below terrain (impossible)
        if target_alt_m < terrain_elevation_m:
            return True, 1.0
        
        # Clearance above terrain
        clearance_m = target_alt_m - terrain_elevation_m
        
        # Masking factor based on clearance
        # More clearance = less masking
        if clearance_m > 100:
            masking_factor = 0.0
        elif clearance_m > 50:
            masking_factor = (100 - clearance_m) / 100 * 0.3
        elif clearance_m > 20:
            masking_factor = (50 - clearance_m) / 50 * 0.5 + 0.3
        else:
            masking_factor = (20 - clearance_m) / 20 * 0.3 + 0.8
        
        masking_factor = np.clip(masking_factor, 0.0, 1.0)
        los_blocked = (masking_factor > 0.9)
        
        return los_blocked, float(masking_factor)
    
    def _compute_snr(
        self,
        radar: RadarSite,
        range_km: float,
        weather_attenuation_db: float
    ) -> float:
        """
        Compute signal-to-noise ratio using radar equation.
        
        Radar equation: SNR = (P_t * G^2 * λ^2 * σ) / ((4π)^3 * R^4 * k * T * B * F * L)
        """
        # Convert to base units
        range_m = range_km * 1000.0
        
        # Wavelength (m)
        c = 3e8  # Speed of light (m/s)
        wavelength_m = c / (radar.frequency_ghz * 1e9)
        
        # Radar parameters
        P_t = radar.peak_power_kw * 1000  # Peak power (W)
        G = 10 ** (radar.antenna_gain_db / 10)  # Linear gain
        sigma = self.target_rcs  # RCS (m²)
        
        # Receiver parameters (typical values)
        k = 1.38e-23  # Boltzmann constant
        T = 290  # System noise temperature (K)
        B = 1e6  # Bandwidth (Hz) - typical for surveillance radar
        F = 3  # Noise figure (linear) ~5dB
        L = 10  # System losses (linear) ~10dB
        
        # Radar equation (simplified)
        numerator = P_t * G * G * wavelength_m**2 * sigma
        denominator = (4 * np.pi)**3 * range_m**4 * k * T * B * F * L
        
        SNR_linear = numerator / denominator
        SNR_db = 10 * np.log10(max(SNR_linear, 1e-20))  # Avoid log(0)
        
        # Apply atmospheric and weather attenuation
        # Atmospheric attenuation (approximately 0.01 dB/km at X-band)
        atm_loss_db = 0.01 * range_km
        
        SNR_db = SNR_db - 2 * atm_loss_db - 2 * weather_attenuation_db  # 2-way path
        
        return float(SNR_db)
    
    def _snr_to_detection_prob(self, snr_db: float, threshold_db: float = 13.0) -> float:
        """
        Convert SNR to detection probability.
        
        Uses a sigmoid function around detection threshold.
        Typical radar detection threshold: 10-15 dB SNR
        """
        # Sigmoid conversion
        # P_d ≈ 0.5 at threshold, approaching 1 for high SNR, 0 for low SNR
        k = 0.5  # Steepness parameter
        prob = 1.0 / (1.0 + np.exp(-k * (snr_db - threshold_db)))
        
        return float(np.clip(prob, 0.0, 1.0))
    
    def _no_detection_result(
        self,
        range_km: float,
        elevation_deg: float,
        azimuth_deg: float,
        reason: str,
        terrain_masking_factor: float = 0.0
    ) -> DetectionResult:
        """Create a no-detection result."""
        return DetectionResult(
            detected=False,
            detection_prob=0.0,
            range_km=range_km,
            elevation_angle_deg=elevation_deg,
            azimuth_angle_deg=azimuth_deg,
            snr_db=-999.0,
            los_blocked=(reason == "terrain_masked"),
            terrain_masking_factor=terrain_masking_factor
        )


class RadarNetwork:
    """Multiple radar sites forming a detection network."""
    
    def __init__(self, radars: List[RadarSite], target_rcs_m2: float = 0.5):
        """
        Initialize radar network.
        
        Args:
            radars: List of radar sites
            target_rcs_m2: Target radar cross-section
        """
        self.radars = radars
        self.model = RadarDetectionModel(target_rcs_m2=target_rcs_m2)
        
        logger.info(f"Initialized radar network with {len(radars)} sites")
    
    def compute_network_detection(
        self,
        target_lat: float,
        target_lon: float,
        target_alt_m: float,
        terrain_elevation_m: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Compute detection probability from entire network.
        
        Uses probability fusion to combine multiple radar detections.
        """
        results = []
        
        for radar in self.radars:
            if not radar.operational:
                continue
            
            result = self.model.compute_detection_probability(
                radar, target_lat, target_lon, target_alt_m, terrain_elevation_m
            )
            results.append((radar.id, result))
        
        # Network detection probability (probability of detection by at least one radar)
        # P(network detection) = 1 - Π(1 - P_i) for all radars
        network_prob = 1.0
        for _, result in results:
            network_prob *= (1.0 - result.detection_prob)
        network_prob = 1.0 - network_prob
        
        # Best detection
        best_result = max(results, key=lambda x: x[1].detection_prob, default=(None, None))
        
        return {
            "network_detection_prob": network_prob,
            "num_radars": len(results),
            "num_detections": sum(1 for _, r in results if r.detected),
            "best_radar_id": best_result[0] if best_result[0] else None,
            "best_snr_db": best_result[1].snr_db if best_result[1] else -999.0,
            "individual_results": dict(results)
        }


def load_radar_sites_from_csv(filepath: str) -> List[RadarSite]:
    """Load radar sites from CSV file."""
    import pandas as pd
    
    df = pd.read_csv(filepath)
    radars = []
    
    for _, row in df.iterrows():
        radar = RadarSite(
            id=row['radar_id'],
            lat=row['lat'],
            lon=row['lon'],
            elevation_m=row['elevation_m'],
            antenna_height_m=row.get('antenna_height_m', 20.0),
            radar_type=RadarType(row.get('radar_type', 'surveillance')),
            frequency_ghz=row.get('frequency_ghz', 10.0),
            max_range_km=row['max_range_km'],
            peak_power_kw=row.get('peak_power_kw', 500.0),
            antenna_gain_db=row.get('antenna_gain_db', 35.0),
            beamwidth_deg=row.get('beamwidth_deg', 2.0),
            operational=row.get('operational', True)
        )
        radars.append(radar)
    
    logger.info(f"Loaded {len(radars)} radar sites from {filepath}")
    return radars


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create a radar site
    radar = RadarSite(
        id="RADAR_001",
        lat=13.0,
        lon=77.6,
        elevation_m=900.0,
        antenna_height_m=30.0,
        radar_type=RadarType.SURVEILLANCE,
        frequency_ghz=10.0,
        max_range_km=100.0,
        peak_power_kw=500.0,
        antenna_gain_db=35.0,
        beamwidth_deg=2.0
    )
    
    # Create detection model
    model = RadarDetectionModel(target_rcs_m2=0.5)
    
    # Compute detection for a target
    result = model.compute_detection_probability(
        radar=radar,
        target_lat=13.05,
        target_lon=77.55,
        target_alt_m=500.0,
        terrain_elevation_m=850.0
    )
    
    print("\nRadar Detection Analysis:")
    print(f"  Detected: {result.detected}")
    print(f"  Detection Probability: {result.detection_prob:.3f}")
    print(f"  Range: {result.range_km:.2f} km")
    print(f"  Elevation Angle: {result.elevation_angle_deg:.1f}°")
    print(f"  Azimuth Angle: {result.azimuth_angle_deg:.1f}°")
    print(f"  SNR: {result.snr_db:.1f} dB")
    print(f"  LOS Blocked: {result.los_blocked}")
    print(f"  Terrain Masking: {result.terrain_masking_factor:.2f}")

