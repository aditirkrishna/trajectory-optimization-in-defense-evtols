"""
Atmospheric Turbulence Modeling

Implements turbulence intensity estimation and gust modeling for
eVTOL flight safety assessment.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TurbulenceClass(Enum):
    """Turbulence intensity classification"""
    CALM = 0
    LIGHT = 1
    MODERATE = 2
    SEVERE = 3
    EXTREME = 4


@dataclass
class TurbulenceMetrics:
    """Turbulence characterization metrics"""
    intensity: float  # Turbulence intensity (0-1)
    eddy_dissipation_rate: float  # EDR in m^(2/3)/s
    turbulence_class: TurbulenceClass
    rms_velocity_ms: float  # RMS velocity fluctuation (m/s)
    integral_length_scale_m: float  # Turbulence length scale (m)
    gust_factor: float  # Peak gust / mean wind ratio


class TurbulenceModel:
    """
    Atmospheric turbulence model.
    
    Implements:
    - Turbulence intensity estimation
    - Gust modeling
    - Stability effects
    - Terrain-induced turbulence
    """
    
    def __init__(self):
        """Initialize turbulence model."""
        self.base_intensity = 0.1  # Base turbulence intensity
        
    def estimate_turbulence(
        self,
        lat: float,
        lon: float,
        alt_m: float,
        wind_speed_ms: float,
        terrain_roughness: float = 0.1,
        atmospheric_stability: str = "neutral"
    ) -> TurbulenceMetrics:
        """
        Estimate turbulence characteristics at a location.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees  
            alt_m: Altitude in meters AGL
            wind_speed_ms: Mean wind speed in m/s
            terrain_roughness: Terrain roughness parameter (0-1)
            atmospheric_stability: "stable", "neutral", or "unstable"
            
        Returns:
            TurbulenceMetrics object
        """
        # Compute turbulence intensity
        intensity = self._compute_intensity(
            alt_m, wind_speed_ms, terrain_roughness, atmospheric_stability
        )
        
        # Compute eddy dissipation rate (EDR)
        edr = self._compute_edr(intensity, wind_speed_ms, alt_m)
        
        # Classify turbulence
        turb_class = self._classify_turbulence(intensity)
        
        # Compute RMS velocity fluctuation
        rms_velocity = intensity * wind_speed_ms
        
        # Compute integral length scale
        length_scale = self._compute_length_scale(alt_m, atmospheric_stability)
        
        # Compute gust factor
        gust_factor = 1.0 + 2.5 * intensity
        
        return TurbulenceMetrics(
            intensity=intensity,
            eddy_dissipation_rate=edr,
            turbulence_class=turb_class,
            rms_velocity_ms=rms_velocity,
            integral_length_scale_m=length_scale,
            gust_factor=gust_factor
        )
    
    def _compute_intensity(
        self,
        alt_m: float,
        wind_speed_ms: float,
        terrain_roughness: float,
        stability: str
    ) -> float:
        """
        Compute turbulence intensity.
        
        Based on:
        - Altitude (decreases with height)
        - Wind speed (increases with speed)
        - Terrain roughness (increases with roughness)
        - Atmospheric stability
        """
        # Altitude effect (decrease with height)
        if alt_m < 10:
            alt_m = 10  # Avoid division by zero
        alt_factor = (100.0 / alt_m) ** 0.25
        
        # Wind speed effect
        if wind_speed_ms < 1.0:
            wind_factor = 0.5
        else:
            wind_factor = 1.0 + 0.1 * np.log(wind_speed_ms)
        
        # Terrain roughness effect
        terrain_factor = 1.0 + terrain_roughness
        
        # Atmospheric stability effect
        stability_factors = {
            "stable": 0.7,
            "neutral": 1.0,
            "unstable": 1.3
        }
        stability_factor = stability_factors.get(stability, 1.0)
        
        # Combine factors
        intensity = self.base_intensity * alt_factor * wind_factor * terrain_factor * stability_factor
        
        # Clip to reasonable range
        intensity = np.clip(intensity, 0.01, 0.8)
        
        return float(intensity)
    
    def _compute_edr(self, intensity: float, wind_speed_ms: float, alt_m: float) -> float:
        """
        Compute eddy dissipation rate (EDR).
        
        EDR is a cube-root turbulence metric used in aviation.
        Units: m^(2/3)/s
        """
        # Empirical relationship between intensity and EDR
        # Based on aviation turbulence reporting standards
        
        # Velocity variance
        sigma_v_squared = (intensity * wind_speed_ms) ** 2
        
        # Length scale (simplified)
        length_scale = max(alt_m * 0.1, 10.0)
        
        # EDR calculation
        edr = (sigma_v_squared ** 1.5) / length_scale
        
        return float(edr)
    
    def _classify_turbulence(self, intensity: float) -> TurbulenceClass:
        """Classify turbulence intensity."""
        if intensity < 0.05:
            return TurbulenceClass.CALM
        elif intensity < 0.15:
            return TurbulenceClass.LIGHT
        elif intensity < 0.30:
            return TurbulenceClass.MODERATE
        elif intensity < 0.50:
            return TurbulenceClass.SEVERE
        else:
            return TurbulenceClass.EXTREME
    
    def _compute_length_scale(self, alt_m: float, stability: str) -> float:
        """
        Compute turbulence integral length scale.
        
        This represents the characteristic size of turbulent eddies.
        """
        # Base length scale proportional to altitude
        base_scale = alt_m * 0.2
        
        # Stability adjustment
        stability_scales = {
            "stable": 0.5,
            "neutral": 1.0,
            "unstable": 1.5
        }
        scale_factor = stability_scales.get(stability, 1.0)
        
        length_scale = base_scale * scale_factor
        
        # Reasonable bounds
        length_scale = np.clip(length_scale, 10.0, 1000.0)
        
        return float(length_scale)
    
    def generate_gust_time_series(
        self,
        duration_s: float,
        dt: float,
        intensity: float,
        mean_wind_speed_ms: float,
        length_scale_m: float,
        vehicle_speed_ms: float = 20.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate turbulent wind gust time series.
        
        Uses Dryden spectral model for atmospheric turbulence.
        
        Args:
            duration_s: Duration in seconds
            dt: Time step in seconds
            intensity: Turbulence intensity
            mean_wind_speed_ms: Mean wind speed
            length_scale_m: Integral length scale
            vehicle_speed_ms: Vehicle airspeed
            
        Returns:
            (time_array, gust_velocity_array) in m/s
        """
        n_samples = int(duration_s / dt)
        time = np.arange(n_samples) * dt
        
        # RMS turbulence velocity
        sigma = intensity * mean_wind_speed_ms
        
        # Generate white noise
        white_noise = np.random.randn(n_samples)
        
        # Apply Dryden filter (simplified first-order approximation)
        # Actual Dryden requires more complex filtering
        omega_0 = vehicle_speed_ms / length_scale_m
        tau = 1.0 / omega_0
        
        # First-order filter
        gust = np.zeros(n_samples)
        for i in range(1, n_samples):
            gust[i] = gust[i-1] * (1 - dt/tau) + sigma * white_noise[i] * np.sqrt(2*dt/tau)
        
        return time, gust
    
    def assess_flight_impact(self, metrics: TurbulenceMetrics, vehicle_mass_kg: float = 1500.0) -> Dict[str, float]:
        """
        Assess turbulence impact on vehicle flight.
        
        Args:
            metrics: Turbulence metrics
            vehicle_mass_kg: Vehicle mass in kg
            
        Returns:
            Dictionary with impact assessment
        """
        # Load factor due to gusts
        # Simplified - actual calculation requires vehicle parameters
        delta_load_factor = 0.1 * metrics.intensity
        
        # RMS angular rate (roll/pitch)
        # Simplified empirical relationship
        rms_angular_rate_dps = metrics.rms_velocity_ms / metrics.integral_length_scale_m * 57.3  # rad/s to deg/s
        
        # Comfort level (for passengers)
        # Based on ISO 2631 vibration standards
        if metrics.turbulence_class == TurbulenceClass.CALM:
            comfort = "comfortable"
        elif metrics.turbulence_class == TurbulenceClass.LIGHT:
            comfort = "comfortable"
        elif metrics.turbulence_class == TurbulenceClass.MODERATE:
            comfort = "fairly_comfortable"
        elif metrics.turbulence_class == TurbulenceClass.SEVERE:
            comfort = "uncomfortable"
        else:
            comfort = "very_uncomfortable"
        
        # Flight safety risk
        if metrics.turbulence_class in [TurbulenceClass.CALM, TurbulenceClass.LIGHT]:
            risk = "low"
        elif metrics.turbulence_class == TurbulenceClass.MODERATE:
            risk = "medium"
        else:
            risk = "high"
        
        return {
            "delta_load_factor": delta_load_factor,
            "rms_angular_rate_dps": rms_angular_rate_dps,
            "comfort_level": comfort,
            "safety_risk": risk,
            "recommended_speed_reduction": min(0.3 * metrics.intensity, 0.3)  # Up to 30% reduction
        }


def get_atmospheric_stability(
    time_of_day_hour: int,
    cloud_cover_octas: int,
    wind_speed_ms: float
) -> str:
    """
    Estimate atmospheric stability class.
    
    Args:
        time_of_day_hour: Hour of day (0-23)
        cloud_cover_octas: Cloud cover in oktas (0-8)
        wind_speed_ms: Wind speed in m/s
        
    Returns:
        Stability class: "stable", "neutral", or "unstable"
    """
    # Daytime (strong solar heating)
    if 9 <= time_of_day_hour <= 17:
        if wind_speed_ms < 2.0 and cloud_cover_octas < 4:
            return "unstable"  # Strong convection
        elif wind_speed_ms > 5.0:
            return "neutral"  # Wind mixing
        else:
            return "neutral"
    
    # Nighttime (radiative cooling)
    elif time_of_day_hour < 6 or time_of_day_hour > 20:
        if wind_speed_ms < 3.0 and cloud_cover_octas < 4:
            return "stable"  # Radiative inversion
        else:
            return "neutral"
    
    # Transition periods
    else:
        return "neutral"


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    model = TurbulenceModel()
    
    # Estimate turbulence
    metrics = model.estimate_turbulence(
        lat=13.0,
        lon=77.6,
        alt_m=300.0,
        wind_speed_ms=12.0,
        terrain_roughness=0.3,
        atmospheric_stability="neutral"
    )
    
    print("\nTurbulence Assessment:")
    print(f"  Intensity: {metrics.intensity:.3f}")
    print(f"  EDR: {metrics.eddy_dissipation_rate:.4f} m^(2/3)/s")
    print(f"  Classification: {metrics.turbulence_class.name}")
    print(f"  RMS Velocity: {metrics.rms_velocity_ms:.2f} m/s")
    print(f"  Length Scale: {metrics.integral_length_scale_m:.1f} m")
    print(f"  Gust Factor: {metrics.gust_factor:.2f}")
    
    # Assess flight impact
    impact = model.assess_flight_impact(metrics)
    print(f"\nFlight Impact:")
    for key, value in impact.items():
        print(f"  {key}: {value}")
    
    # Generate gust time series
    time, gust = model.generate_gust_time_series(
        duration_s=60.0,
        dt=0.1,
        intensity=metrics.intensity,
        mean_wind_speed_ms=12.0,
        length_scale_m=metrics.integral_length_scale_m,
        vehicle_speed_ms=25.0
    )
    
    print(f"\nGenerated {len(time)} gust samples")
    print(f"  Gust range: {np.min(gust):.2f} to {np.max(gust):.2f} m/s")
    print(f"  Gust std dev: {np.std(gust):.2f} m/s")

