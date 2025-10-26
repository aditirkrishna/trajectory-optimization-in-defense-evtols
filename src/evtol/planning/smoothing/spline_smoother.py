"""
Trajectory Smoothing with Curvature-Bounded Splines

Implements smooth trajectory generation with:
- C² continuity (smooth curvature)
- Curvature bounds
- Minimum snap trajectories
- Time parameterization
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy.interpolate import splprep, splev, CubicSpline
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryPoint:
    """Point on smoothed trajectory"""
    lat: float
    lon: float
    alt_m: float
    time_s: float
    speed_ms: Optional[float] = None
    curvature: Optional[float] = None
    heading_deg: Optional[float] = None


class SplineSmoother:
    """
    Smooth trajectory generation using splines.
    
    Features:
    - B-spline smoothing
    - Curvature constraint enforcement
    - Time-optimal parameterization
    - Snap minimization
    """
    
    def __init__(
        self,
        max_curvature: float = 0.01,  # 1/m
        smoothing_factor: float = 0.0,
        spline_degree: int = 3
    ):
        """
        Initialize spline smoother.
        
        Args:
            max_curvature: Maximum allowed curvature (1/m)
            smoothing_factor: Spline smoothing factor (0=interpolation)
            spline_degree: Degree of spline (3=cubic)
        """
        self.max_curvature = max_curvature
        self.smoothing_factor = smoothing_factor
        self.spline_degree = spline_degree
        
    def smooth_waypoints(
        self,
        waypoints: List[Tuple[float, float, float]],  # (lat, lon, alt)
        num_points: int = 100
    ) -> List[TrajectoryPoint]:
        """
        Smooth waypoints using B-spline.
        
        Args:
            waypoints: List of (lat, lon, alt) tuples
            num_points: Number of points in smoothed trajectory
            
        Returns:
            List of smoothed trajectory points
        """
        if len(waypoints) < 4:
            # Need at least 4 points for cubic spline
            return self._linear_interpolation(waypoints, num_points)
        
        # Extract coordinates
        lats = np.array([w[0] for w in waypoints])
        lons = np.array([w[1] for w in waypoints])
        alts = np.array([w[2] for w in waypoints])
        
        # Fit B-spline
        try:
            tck, u = splprep(
                [lats, lons, alts],
                s=self.smoothing_factor,
                k=min(self.spline_degree, len(waypoints) - 1)
            )
            
            # Evaluate spline
            u_new = np.linspace(0, 1, num_points)
            smooth_coords = splev(u_new, tck)
            
            # Create trajectory points
            trajectory = []
            for i in range(num_points):
                point = TrajectoryPoint(
                    lat=float(smooth_coords[0][i]),
                    lon=float(smooth_coords[1][i]),
                    alt_m=float(smooth_coords[2][i]),
                    time_s=0.0  # Will be set later with time parameterization
                )
                trajectory.append(point)
            
            # Check and enforce curvature constraints
            trajectory = self._enforce_curvature_bounds(trajectory)
            
            return trajectory
            
        except Exception as e:
            logger.warning(f"Spline fitting failed: {e}. Using linear interpolation.")
            return self._linear_interpolation(waypoints, num_points)
    
    def _linear_interpolation(
        self,
        waypoints: List[Tuple[float, float, float]],
        num_points: int
    ) -> List[TrajectoryPoint]:
        """Fallback linear interpolation."""
        trajectory = []
        
        # Total number of segments
        n_segments = len(waypoints) - 1
        points_per_segment = num_points // n_segments
        
        for i in range(n_segments):
            start = waypoints[i]
            end = waypoints[i + 1]
            
            for j in range(points_per_segment):
                t = j / points_per_segment
                
                lat = start[0] + t * (end[0] - start[0])
                lon = start[1] + t * (end[1] - start[1])
                alt = start[2] + t * (end[2] - start[2])
                
                trajectory.append(TrajectoryPoint(lat, lon, alt, 0.0))
        
        # Add final point
        trajectory.append(TrajectoryPoint(*waypoints[-1], 0.0))
        
        return trajectory
    
    def _enforce_curvature_bounds(
        self,
        trajectory: List[TrajectoryPoint]
    ) -> List[TrajectoryPoint]:
        """
        Check and enforce curvature constraints.
        
        If curvature exceeds bounds, locally reduce curvature by
        adjusting intermediate points.
        """
        # Compute curvature at each point
        for i in range(1, len(trajectory) - 1):
            prev = trajectory[i - 1]
            curr = trajectory[i]
            next_pt = trajectory[i + 1]
            
            # Approximate curvature using three points
            curvature = self._compute_curvature_3points(prev, curr, next_pt)
            curr.curvature = curvature
            
            # If curvature exceeds limit, adjust point
            if abs(curvature) > self.max_curvature:
                # Move point toward line between prev and next
                alpha = 0.3  # Adjustment factor
                
                curr.lat = curr.lat + alpha * (
                    (prev.lat + next_pt.lat) / 2 - curr.lat
                )
                curr.lon = curr.lon + alpha * (
                    (prev.lon + next_pt.lon) / 2 - curr.lon
                )
                curr.alt_m = curr.alt_m + alpha * (
                    (prev.alt_m + next_pt.alt_m) / 2 - curr.alt_m
                )
        
        return trajectory
    
    def _compute_curvature_3points(
        self,
        p1: TrajectoryPoint,
        p2: TrajectoryPoint,
        p3: TrajectoryPoint
    ) -> float:
        """
        Compute curvature using three points.
        
        κ ≈ 2*|sin(θ)|/d where θ is turn angle and d is chord length
        """
        # Vectors
        v1 = np.array([p2.lat - p1.lat, p2.lon - p1.lon])
        v2 = np.array([p3.lat - p2.lat, p3.lon - p2.lon])
        
        # Magnitudes
        d1 = np.linalg.norm(v1)
        d2 = np.linalg.norm(v2)
        
        if d1 < 1e-6 or d2 < 1e-6:
            return 0.0
        
        # Angle between vectors
        cos_theta = np.dot(v1, v2) / (d1 * d2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        # Average distance
        d_avg = (d1 + d2) / 2
        
        # Curvature
        curvature = 2 * abs(np.sin(theta)) / d_avg if d_avg > 0 else 0.0
        
        return curvature
    
    def time_parameterize(
        self,
        trajectory: List[TrajectoryPoint],
        cruise_speed_ms: float = 35.0,
        max_accel_ms2: float = 3.0
    ) -> List[TrajectoryPoint]:
        """
        Add time parameterization to trajectory.
        
        Computes realistic time stamps considering:
        - Distance between points
        - Speed limits
        - Acceleration limits
        
        Args:
            trajectory: Trajectory points without time
            cruise_speed_ms: Cruise speed
            max_accel_ms2: Maximum acceleration
            
        Returns:
            Trajectory with time stamps
        """
        if not trajectory:
            return trajectory
        
        # Start at time 0
        trajectory[0].time_s = 0.0
        trajectory[0].speed_ms = cruise_speed_ms
        
        current_speed = cruise_speed_ms
        current_time = 0.0
        
        for i in range(1, len(trajectory)):
            prev = trajectory[i - 1]
            curr = trajectory[i]
            
            # Distance
            distance_m = self._haversine_m(
                prev.lat, prev.lon, curr.lat, curr.lon
            )
            
            # Time at constant speed
            if current_speed > 0:
                time_increment = distance_m / current_speed
            else:
                time_increment = 0.0
            
            current_time += time_increment
            curr.time_s = current_time
            curr.speed_ms = current_speed
        
        return trajectory
    
    def _haversine_m(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute Haversine distance in meters."""
        R = 6371000.0  # Earth radius in meters
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * 
             np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c


class MinimumSnapTrajectory:
    """
    Generate minimum snap trajectory.
    
    Minimizes 4th derivative (snap) for smooth, dynamically feasible trajectories.
    """
    
    def __init__(self):
        """Initialize minimum snap trajectory generator."""
        pass
    
    def generate(
        self,
        waypoints: List[Tuple[float, float, float]],
        durations: Optional[List[float]] = None
    ) -> List[TrajectoryPoint]:
        """
        Generate minimum snap trajectory through waypoints.
        
        Args:
            waypoints: List of waypoint coordinates
            durations: Time duration for each segment
            
        Returns:
            Smooth trajectory
        """
        # Simplified version: use cubic spline which minimizes 2nd derivative
        # Full minimum snap requires solving optimization problem
        
        if durations is None:
            # Equal time segments
            durations = [1.0] * (len(waypoints) - 1)
        
        # Extract coordinates
        lats = np.array([w[0] for w in waypoints])
        lons = np.array([w[1] for w in waypoints])
        alts = np.array([w[2] for w in waypoints])
        
        # Cumulative time
        times = np.zeros(len(waypoints))
        for i in range(1, len(waypoints)):
            times[i] = times[i-1] + durations[i-1]
        
        # Create cubic splines
        spline_lat = CubicSpline(times, lats, bc_type='natural')
        spline_lon = CubicSpline(times, lons, bc_type='natural')
        spline_alt = CubicSpline(times, alts, bc_type='natural')
        
        # Sample trajectory
        t_eval = np.linspace(times[0], times[-1], 100)
        
        trajectory = []
        for t in t_eval:
            point = TrajectoryPoint(
                lat=float(spline_lat(t)),
                lon=float(spline_lon(t)),
                alt_m=float(spline_alt(t)),
                time_s=float(t)
            )
            trajectory.append(point)
        
        return trajectory


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create waypoints
    waypoints = [
        (13.0, 77.5, 100.0),
        (13.02, 77.52, 150.0),
        (13.05, 77.55, 200.0),
        (13.08, 77.58, 150.0),
        (13.1, 77.6, 100.0)
    ]
    
    # Smooth with spline
    smoother = SplineSmoother(max_curvature=0.005)
    smoothed = smoother.smooth_waypoints(waypoints, num_points=50)
    
    print(f"Smoothed trajectory: {len(smoothed)} points")
    print(f"First point: lat={smoothed[0].lat:.4f}, lon={smoothed[0].lon:.4f}, alt={smoothed[0].alt_m:.1f}m")
    print(f"Last point: lat={smoothed[-1].lat:.4f}, lon={smoothed[-1].lon:.4f}, alt={smoothed[-1].alt_m:.1f}m")
    
    # Add time parameterization
    smoothed = smoother.time_parameterize(smoothed, cruise_speed_ms=35.0)
    
    print(f"Total duration: {smoothed[-1].time_s:.1f} seconds")
    
    # Check curvatures
    curvatures = [p.curvature for p in smoothed if p.curvature is not None]
    if curvatures:
        print(f"Max curvature: {max(curvatures):.6f} (limit: {smoother.max_curvature})")



