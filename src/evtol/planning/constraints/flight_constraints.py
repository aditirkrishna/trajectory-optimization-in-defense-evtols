"""
Flight Constraints Module

Implements dynamic feasibility constraints for eVTOL flight planning:
- Turn radius constraints
- Climb/descent rate limits
- Speed and acceleration bounds
- Payload capacity checks
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class VehicleConstraints:
    """Physical constraints for the eVTOL vehicle"""
    # Speed limits (m/s)
    min_speed_ms: float = 5.0
    max_speed_ms: float = 60.0
    cruise_speed_ms: float = 35.0
    
    # Acceleration limits (m/s²)
    max_acceleration_ms2: float = 3.0
    max_deceleration_ms2: float = 5.0
    
    # Turn performance
    max_bank_angle_deg: float = 45.0
    min_turn_radius_m: float = 50.0
    max_turn_rate_dps: float = 30.0  # deg/s
    
    # Vertical performance
    max_climb_rate_ms: float = 8.0
    max_descent_rate_ms: float = 6.0
    max_climb_angle_deg: float = 20.0
    max_descent_angle_deg: float = 15.0
    
    # Altitude limits (m)
    min_altitude_agl_m: float = 50.0
    max_altitude_asl_m: float = 5000.0
    service_ceiling_m: float = 4500.0
    
    # Payload constraints (kg)
    max_payload_kg: float = 600.0
    empty_weight_kg: float = 1500.0
    max_takeoff_weight_kg: float = 2100.0
    
    # Endurance limits
    max_flight_time_min: float = 60.0
    min_battery_reserve_percent: float = 20.0


class FlightConstraintChecker:
    """
    Validates flight segments against vehicle constraints.
    
    Checks:
    - Turn radius compliance
    - Climb/descent rate limits
    - Speed envelopes
    - Acceleration limits
    """
    
    def __init__(self, constraints: Optional[VehicleConstraints] = None):
        """
        Initialize constraint checker.
        
        Args:
            constraints: Vehicle constraint parameters
        """
        self.constraints = constraints or VehicleConstraints()
        logger.info("Initialized flight constraint checker")
    
    def check_segment_feasibility(
        self,
        start_lat: float,
        start_lon: float,
        start_alt_m: float,
        end_lat: float,
        end_lon: float,
        end_alt_m: float,
        speed_ms: Optional[float] = None
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Check if a flight segment is feasible.
        
        Args:
            start_lat, start_lon, start_alt_m: Starting position
            end_lat, end_lon, end_alt_m: Ending position
            speed_ms: Flight speed (uses cruise if None)
            
        Returns:
            (is_feasible, violation_details)
        """
        violations = {}
        
        # Use cruise speed if not specified
        if speed_ms is None:
            speed_ms = self.constraints.cruise_speed_ms
        
        # Check speed limits
        if not self.check_speed_limits(speed_ms):
            violations["speed"] = f"Speed {speed_ms:.1f} m/s outside limits"
        
        # Check altitude limits
        if not self.check_altitude_limits(start_alt_m, end_alt_m):
            violations["altitude"] = f"Altitude outside limits"
        
        # Compute segment geometry
        horizontal_dist_m = self._haversine_m(start_lat, start_lon, end_lat, end_lon)
        altitude_change_m = end_alt_m - start_alt_m
        
        # Check climb/descent rates
        if horizontal_dist_m > 0:
            climb_angle_deg = math.degrees(math.atan2(altitude_change_m, horizontal_dist_m))
            
            if altitude_change_m > 0:  # Climbing
                max_allowed_angle = self.constraints.max_climb_angle_deg
                if abs(climb_angle_deg) > max_allowed_angle:
                    violations["climb_angle"] = f"Climb angle {climb_angle_deg:.1f}° exceeds {max_allowed_angle}°"
            elif altitude_change_m < 0:  # Descending
                max_allowed_angle = self.constraints.max_descent_angle_deg
                if abs(climb_angle_deg) > max_allowed_angle:
                    violations["descent_angle"] = f"Descent angle {abs(climb_angle_deg):.1f}° exceeds {max_allowed_angle}°"
        
        is_feasible = len(violations) == 0
        
        return is_feasible, violations
    
    def check_turn_feasibility(
        self,
        prev_lat: float,
        prev_lon: float,
        current_lat: float,
        current_lon: float,
        next_lat: float,
        next_lon: float,
        speed_ms: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Check if a turn at a waypoint is feasible.
        
        Computes turn radius and checks against minimum.
        
        Args:
            prev_lat, prev_lon: Previous waypoint
            current_lat, current_lon: Turn waypoint
            next_lat, next_lon: Next waypoint
            speed_ms: Flight speed
            
        Returns:
            (is_feasible, turn_radius_m)
        """
        if speed_ms is None:
            speed_ms = self.constraints.cruise_speed_ms
        
        # Compute heading change
        heading_in = self._compute_heading(prev_lat, prev_lon, current_lat, current_lon)
        heading_out = self._compute_heading(current_lat, current_lon, next_lat, next_lon)
        
        # Turn angle (smallest angle between headings)
        turn_angle_deg = abs(heading_out - heading_in)
        if turn_angle_deg > 180:
            turn_angle_deg = 360 - turn_angle_deg
        
        # Compute required turn radius
        # R = V² / (g * tan(φ)) where φ is bank angle
        max_bank_rad = math.radians(self.constraints.max_bank_angle_deg)
        g = 9.81  # m/s²
        
        min_possible_radius = (speed_ms ** 2) / (g * math.tan(max_bank_rad))
        
        # Check against constraint
        is_feasible = min_possible_radius >= self.constraints.min_turn_radius_m
        
        return is_feasible, min_possible_radius
    
    def check_speed_limits(self, speed_ms: float) -> bool:
        """Check if speed is within limits."""
        return self.constraints.min_speed_ms <= speed_ms <= self.constraints.max_speed_ms
    
    def check_altitude_limits(self, alt_m: float, end_alt_m: Optional[float] = None) -> bool:
        """Check if altitude is within limits."""
        alts = [alt_m]
        if end_alt_m is not None:
            alts.append(end_alt_m)
        
        for alt in alts:
            if alt < self.constraints.min_altitude_agl_m:
                return False
            if alt > self.constraints.max_altitude_asl_m:
                return False
        
        return True
    
    def compute_minimum_segment_time(
        self,
        start_lat: float,
        start_lon: float,
        start_alt_m: float,
        end_lat: float,
        end_lon: float,
        end_alt_m: float
    ) -> float:
        """
        Compute minimum time to traverse segment given constraints.
        
        Returns:
            Minimum time in seconds
        """
        # Horizontal distance
        horizontal_dist_m = self._haversine_m(start_lat, start_lon, end_lat, end_lon)
        
        # Vertical distance
        altitude_change_m = abs(end_alt_m - start_alt_m)
        
        # Horizontal time at max speed
        horizontal_time_s = horizontal_dist_m / self.constraints.max_speed_ms
        
        # Vertical time at max climb/descent rate
        if altitude_change_m > 0:
            vertical_time_s = altitude_change_m / self.constraints.max_climb_rate_ms
        else:
            vertical_time_s = altitude_change_m / self.constraints.max_descent_rate_ms
        
        # Total time is maximum of the two (they can happen simultaneously)
        return max(horizontal_time_s, vertical_time_s)
    
    def compute_speed_profile(
        self,
        distance_m: float,
        start_speed_ms: float,
        end_speed_ms: float,
        num_points: int = 50
    ) -> np.ndarray:
        """
        Compute speed profile with acceleration constraints.
        
        Args:
            distance_m: Total distance
            start_speed_ms: Initial speed
            end_speed_ms: Final speed
            num_points: Number of profile points
            
        Returns:
            Array of speeds at each point
        """
        # Simple trapezoidal speed profile
        if start_speed_ms < end_speed_ms:
            # Acceleration phase
            accel = self.constraints.max_acceleration_ms2
        else:
            # Deceleration phase
            accel = -self.constraints.max_deceleration_ms2
        
        # Time to reach target speed
        delta_v = end_speed_ms - start_speed_ms
        accel_time = abs(delta_v) / abs(accel)
        accel_dist = start_speed_ms * accel_time + 0.5 * accel * accel_time**2
        
        # Check if we have enough distance
        if accel_dist > distance_m:
            # Can't reach target speed - compute achievable speed
            positions = np.linspace(0, distance_m, num_points)
            speeds = np.zeros(num_points)
            
            for i, pos in enumerate(positions):
                # v² = v0² + 2*a*d
                speeds[i] = np.sqrt(start_speed_ms**2 + 2*accel*pos)
        else:
            # Can reach target speed with cruise in middle
            cruise_dist = distance_m - accel_dist
            
            positions = np.linspace(0, distance_m, num_points)
            speeds = np.zeros(num_points)
            
            for i, pos in enumerate(positions):
                if pos < accel_dist:
                    # Acceleration phase
                    t = np.sqrt(2*pos/accel) if accel > 0 else np.sqrt(2*pos/abs(accel))
                    speeds[i] = start_speed_ms + accel * t
                else:
                    # Cruise phase
                    speeds[i] = end_speed_ms
        
        # Clip to limits
        speeds = np.clip(speeds, self.constraints.min_speed_ms, self.constraints.max_speed_ms)
        
        return speeds
    
    def _haversine_m(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute Haversine distance in meters."""
        R = 6371000.0  # Earth radius in meters
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    
    def _compute_heading(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute heading from point 1 to point 2 in degrees."""
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        y = math.sin(dlon) * math.cos(math.radians(lat2))
        x = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) -
             math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dlon))
        
        heading = math.degrees(math.atan2(y, x))
        return (heading + 360) % 360


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create constraint checker
    checker = FlightConstraintChecker()
    
    # Check segment feasibility
    is_feasible, violations = checker.check_segment_feasibility(
        start_lat=13.0, start_lon=77.5, start_alt_m=100.0,
        end_lat=13.1, end_lon=77.6, end_alt_m=500.0,
        speed_ms=35.0
    )
    
    print(f"Segment feasible: {is_feasible}")
    if violations:
        print(f"Violations: {violations}")
    
    # Check turn feasibility
    is_turn_ok, turn_radius = checker.check_turn_feasibility(
        prev_lat=13.0, prev_lon=77.5,
        current_lat=13.05, current_lon=77.55,
        next_lat=13.1, next_lon=77.5,
        speed_ms=35.0
    )
    
    print(f"Turn feasible: {is_turn_ok}")
    print(f"Turn radius: {turn_radius:.1f} m")
    
    # Compute speed profile
    speeds = checker.compute_speed_profile(
        distance_m=5000,
        start_speed_ms=20.0,
        end_speed_ms=50.0,
        num_points=50
    )
    
    print(f"Speed profile: min={speeds.min():.1f}, max={speeds.max():.1f} m/s")

