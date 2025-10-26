"""
Wind Field Modeling and Interpolation

Provides wind speed and direction estimation at any point in 3D space
using interpolation from sparse measurements or forecast data.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from scipy.interpolate import LinearNDInterpolator, Rbf, griddata
from scipy.spatial import cKDTree
import logging

logger = logging.getLogger(__name__)


@dataclass
class WindVector:
    """Wind vector at a point"""
    u: float  # East-west component (m/s, positive = eastward)
    v: float  # North-south component (m/s, positive = northward)
    w: float  # Vertical component (m/s, positive = upward)
    speed: float  # Wind speed magnitude (m/s)
    direction: float  # Wind direction (degrees, 0=North, clockwise)
    
    @classmethod
    def from_components(cls, u: float, v: float, w: float = 0.0):
        """Create WindVector from components."""
        speed = np.sqrt(u**2 + v**2 + w**2)
        direction = (np.degrees(np.arctan2(u, v)) + 360) % 360
        return cls(u=u, v=v, w=w, speed=speed, direction=direction)
    
    @classmethod
    def from_speed_direction(cls, speed: float, direction: float, w: float = 0.0):
        """Create WindVector from speed and direction."""
        u = speed * np.sin(np.radians(direction))
        v = speed * np.cos(np.radians(direction))
        return cls(u=u, v=v, w=w, speed=speed, direction=direction)


@dataclass
class WindMeasurement:
    """Wind measurement at a specific location and time"""
    lat: float
    lon: float
    alt_m: float
    time_unix: float
    wind_vector: WindVector
    quality: float = 1.0  # Quality indicator (0-1)


class WindFieldModel:
    """
    Wind field model with 3D interpolation.
    
    Supports multiple interpolation methods:
    - Linear (fast, simple)
    - RBF (smooth, handles scattered data)
    - IDW (inverse distance weighting)
    """
    
    def __init__(self, method: str = "linear", vertical_decay_m: float = 1000.0):
        """
        Initialize wind field model.
        
        Args:
            method: Interpolation method ("linear", "rbf", "idw")
            vertical_decay_m: Vertical decay scale for wind speed (meters)
        """
        self.method = method
        self.vertical_decay_m = vertical_decay_m
        self.measurements: List[WindMeasurement] = []
        self.interpolator_u = None
        self.interpolator_v = None
        self.interpolator_w = None
        self.kdtree = None
        
    def add_measurement(self, measurement: WindMeasurement):
        """Add a wind measurement to the model."""
        self.measurements.append(measurement)
        # Invalidate interpolators
        self.interpolator_u = None
        self.interpolator_v = None
        self.interpolator_w = None
        self.kdtree = None
    
    def add_measurements(self, measurements: List[WindMeasurement]):
        """Add multiple wind measurements."""
        self.measurements.extend(measurements)
        self.interpolator_u = None
        self.interpolator_v = None
        self.interpolator_w = None
        self.kdtree = None
    
    def build_interpolators(self):
        """Build interpolators from measurements."""
        if len(self.measurements) == 0:
            raise ValueError("No measurements available for interpolation")
        
        # Extract coordinates and values
        points = np.array([
            [m.lat, m.lon, m.alt_m] for m in self.measurements
        ])
        u_values = np.array([m.wind_vector.u for m in self.measurements])
        v_values = np.array([m.wind_vector.v for m in self.measurements])
        w_values = np.array([m.wind_vector.w for m in self.measurements])
        
        # Build interpolators based on method
        if self.method == "linear":
            self.interpolator_u = LinearNDInterpolator(points, u_values, fill_value=0.0)
            self.interpolator_v = LinearNDInterpolator(points, v_values, fill_value=0.0)
            self.interpolator_w = LinearNDInterpolator(points, w_values, fill_value=0.0)
        
        elif self.method == "rbf":
            self.interpolator_u = Rbf(
                points[:, 0], points[:, 1], points[:, 2], u_values,
                function='multiquadric', smooth=0.1
            )
            self.interpolator_v = Rbf(
                points[:, 0], points[:, 1], points[:, 2], v_values,
                function='multiquadric', smooth=0.1
            )
            self.interpolator_w = Rbf(
                points[:, 0], points[:, 1], points[:, 2], w_values,
                function='multiquadric', smooth=0.1
            )
        
        elif self.method == "idw":
            # Use KDTree for IDW
            self.kdtree = cKDTree(points)
            self._u_values = u_values
            self._v_values = v_values
            self._w_values = w_values
        
        else:
            raise ValueError(f"Unknown interpolation method: {self.method}")
        
        logger.info(f"Built wind field interpolators using {self.method} method with {len(self.measurements)} measurements")
    
    def query(self, lat: float, lon: float, alt_m: float, time_unix: Optional[float] = None) -> WindVector:
        """
        Query wind at a specific location.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            alt_m: Altitude in meters
            time_unix: Unix timestamp (optional, for temporal interpolation)
            
        Returns:
            WindVector at the query location
        """
        # Build interpolators if not already built
        if self.interpolator_u is None and self.kdtree is None:
            self.build_interpolators()
        
        # Query interpolators
        if self.method == "linear":
            u = float(self.interpolator_u([lat, lon, alt_m])[0])
            v = float(self.interpolator_v([lat, lon, alt_m])[0])
            w = float(self.interpolator_w([lat, lon, alt_m])[0])
        
        elif self.method == "rbf":
            # RBF requires separate arguments, not an array
            u = float(self.interpolator_u(lat, lon, alt_m))
            v = float(self.interpolator_v(lat, lon, alt_m))
            w = float(self.interpolator_w(lat, lon, alt_m))
        
        elif self.method == "idw":
            u, v, w = self._idw_interpolate(lat, lon, alt_m)
        
        return WindVector.from_components(u, v, w)
    
    def _idw_interpolate(self, lat: float, lon: float, alt_m: float, power: float = 2.0, k: int = 4) -> Tuple[float, float, float]:
        """Inverse distance weighting interpolation."""
        query_point = np.array([[lat, lon, alt_m]])
        distances, indices = self.kdtree.query(query_point, k=min(k, len(self.measurements)))
        
        # Avoid division by zero
        distances = np.maximum(distances, 1e-10)
        
        # Compute weights
        weights = 1.0 / (distances ** power)
        weights /= weights.sum()
        
        # Weighted average
        u = np.sum(weights * self._u_values[indices])
        v = np.sum(weights * self._v_values[indices])
        w = np.sum(weights * self._w_values[indices])
        
        return float(u), float(v), float(w)
    
    def apply_vertical_profile(self, wind: WindVector, altitude_m: float, reference_alt_m: float = 10.0) -> WindVector:
        """
        Apply vertical wind profile (logarithmic law).
        
        Args:
            wind: Wind vector at reference altitude
            altitude_m: Target altitude
            reference_alt_m: Reference altitude
            
        Returns:
            Adjusted wind vector
        """
        if altitude_m <= 0:
            altitude_m = 1.0
        
        # Logarithmic wind profile
        z0 = 0.1  # Roughness length (m) - typical for grassland
        scale_factor = np.log(altitude_m / z0) / np.log(reference_alt_m / z0)
        
        return WindVector.from_components(
            wind.u * scale_factor,
            wind.v * scale_factor,
            wind.w
        )
    
    def get_statistics(self) -> Dict[str, float]:
        """Get wind field statistics."""
        if len(self.measurements) == 0:
            return {}
        
        speeds = [m.wind_vector.speed for m in self.measurements]
        directions = [m.wind_vector.direction for m in self.measurements]
        
        return {
            "num_measurements": len(self.measurements),
            "mean_speed_ms": float(np.mean(speeds)),
            "std_speed_ms": float(np.std(speeds)),
            "max_speed_ms": float(np.max(speeds)),
            "mean_direction_deg": float(np.mean(directions)),
            "coverage_lat_deg": float(np.ptp([m.lat for m in self.measurements])),
            "coverage_lon_deg": float(np.ptp([m.lon for m in self.measurements])),
            "coverage_alt_m": float(np.ptp([m.alt_m for m in self.measurements]))
        }


def load_wind_data_from_csv(filepath: str) -> List[WindMeasurement]:
    """
    Load wind measurements from CSV file.
    
    Expected columns: lat, lon, alt_m, time_unix, wind_speed_ms, wind_direction_deg
    """
    import pandas as pd
    
    df = pd.read_csv(filepath)
    measurements = []
    
    for _, row in df.iterrows():
        wind_vector = WindVector.from_speed_direction(
            speed=row['wind_speed_ms'],
            direction=row['wind_direction_deg'],
            w=row.get('vertical_speed_ms', 0.0)
        )
        
        measurement = WindMeasurement(
            lat=row['lat'],
            lon=row['lon'],
            alt_m=row['alt_m'],
            time_unix=row['time_unix'],
            wind_vector=wind_vector,
            quality=row.get('quality', 1.0)
        )
        measurements.append(measurement)
    
    logger.info(f"Loaded {len(measurements)} wind measurements from {filepath}")
    return measurements


def generate_synthetic_wind_field(
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    alt_range: Tuple[float, float],
    num_points: int = 100,
    base_speed: float = 10.0,
    variability: float = 0.3
) -> List[WindMeasurement]:
    """
    Generate synthetic wind field for testing.
    
    Args:
        lat_range: (min_lat, max_lat)
        lon_range: (min_lon, max_lon)
        alt_range: (min_alt, max_alt)
        num_points: Number of measurement points
        base_speed: Base wind speed in m/s
        variability: Variability factor (0-1)
        
    Returns:
        List of synthetic wind measurements
    """
    np.random.seed(42)  # For reproducibility
    
    measurements = []
    
    for i in range(num_points):
        lat = np.random.uniform(*lat_range)
        lon = np.random.uniform(*lon_range)
        alt = np.random.uniform(*alt_range)
        
        # Simulate westerly winds with altitude dependence
        base_direction = 270  # Westerly
        direction = base_direction + np.random.uniform(-30, 30)
        
        # Wind speed increases with altitude
        speed_factor = 1.0 + (alt / 1000)
        speed = base_speed * speed_factor * (1.0 + np.random.uniform(-variability, variability))
        
        wind_vector = WindVector.from_speed_direction(speed, direction)
        
        measurement = WindMeasurement(
            lat=lat,
            lon=lon,
            alt_m=alt,
            time_unix=0.0,
            wind_vector=wind_vector
        )
        measurements.append(measurement)
    
    logger.info(f"Generated {num_points} synthetic wind measurements")
    return measurements


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic wind field
    measurements = generate_synthetic_wind_field(
        lat_range=(12.9, 13.1),
        lon_range=(77.5, 77.7),
        alt_range=(0, 1000),
        num_points=50,
        base_speed=8.0
    )
    
    # Create wind field model
    model = WindFieldModel(method="rbf")
    model.add_measurements(measurements)
    
    # Query wind at a location
    wind = model.query(lat=13.0, lon=77.6, alt_m=500.0)
    print(f"\nWind at (13.0°, 77.6°, 500m):")
    print(f"  Speed: {wind.speed:.2f} m/s")
    print(f"  Direction: {wind.direction:.1f}°")
    print(f"  Components: u={wind.u:.2f}, v={wind.v:.2f}, w={wind.w:.2f}")
    
    # Print statistics
    stats = model.get_statistics()
    print(f"\nWind field statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")

