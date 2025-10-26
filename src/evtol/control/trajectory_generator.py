"""
Trajectory Generation Module

Generates flyable trajectories from waypoint sequences with
time parameterization and velocity profiles.
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrajectorySegment:
    """Single segment of trajectory"""
    start_time: float
    end_time: float
    start_position: np.ndarray
    end_position: np.ndarray
    start_velocity: np.ndarray
    end_velocity: np.ndarray
    max_velocity: float = 35.0
    max_acceleration: float = 3.0


class TrajectoryGenerator:
    """
    Generate time-parameterized trajectories.
    
    Produces position, velocity, and acceleration profiles
    that satisfy vehicle constraints.
    """
    
    def __init__(
        self,
        max_velocity: float = 35.0,  # m/s
        max_acceleration: float = 3.0,  # m/s²
        dt: float = 0.01  # Control timestep
    ):
        """
        Initialize trajectory generator.
        
        Args:
            max_velocity: Maximum velocity
            max_acceleration: Maximum acceleration
            dt: Time discretization
        """
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.dt = dt
        
    def generate_trajectory(
        self,
        waypoints: List[np.ndarray],
        initial_velocity: Optional[np.ndarray] = None,
        final_velocity: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Generate trajectory through waypoints.
        
        Args:
            waypoints: List of position waypoints [x, y, z]
            initial_velocity: Starting velocity (None for hover)
            final_velocity: Ending velocity (None for hover)
            
        Returns:
            List of trajectory points with time, position, velocity, acceleration
        """
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")
        
        if initial_velocity is None:
            initial_velocity = np.zeros(3)
        if final_velocity is None:
            final_velocity = np.zeros(3)
        
        # Generate time-optimal segments
        segments = self._compute_segments(
            waypoints, initial_velocity, final_velocity
        )
        
        # Sample trajectory
        trajectory = self._sample_trajectory(segments)
        
        return trajectory
    
    def _compute_segments(
        self,
        waypoints: List[np.ndarray],
        initial_vel: np.ndarray,
        final_vel: np.ndarray
    ) -> List[TrajectorySegment]:
        """Compute trajectory segments between waypoints."""
        segments = []
        current_time = 0.0
        current_vel = initial_vel.copy()
        
        for i in range(len(waypoints) - 1):
            start_pos = waypoints[i]
            end_pos = waypoints[i + 1]
            
            # Determine target velocity
            if i == len(waypoints) - 2:
                target_vel = final_vel
            else:
                # Cruise velocity toward next waypoint
                direction = end_pos - start_pos
                distance = np.linalg.norm(direction)
                if distance > 0:
                    direction = direction / distance
                    target_vel = direction * min(self.max_velocity, distance / 2.0)
                else:
                    target_vel = np.zeros(3)
            
            # Compute segment duration
            distance = np.linalg.norm(end_pos - start_pos)
            avg_velocity = np.linalg.norm(current_vel + target_vel) / 2
            if avg_velocity > 0:
                duration = distance / max(avg_velocity, 1.0)
            else:
                duration = distance / self.max_velocity
            
            segment = TrajectorySegment(
                start_time=current_time,
                end_time=current_time + duration,
                start_position=start_pos,
                end_position=end_pos,
                start_velocity=current_vel,
                end_velocity=target_vel,
                max_velocity=self.max_velocity,
                max_acceleration=self.max_acceleration
            )
            
            segments.append(segment)
            current_time += duration
            current_vel = target_vel
        
        return segments
    
    def _sample_trajectory(
        self,
        segments: List[TrajectorySegment]
    ) -> List[Dict]:
        """Sample trajectory at regular time intervals."""
        trajectory = []
        
        for segment in segments:
            # Time samples for this segment
            t_start = segment.start_time
            t_end = segment.end_time
            times = np.arange(t_start, t_end, self.dt)
            
            for t in times:
                # Normalized time within segment
                tau = (t - t_start) / (t_end - t_start)
                tau = np.clip(tau, 0.0, 1.0)
                
                # Cubic interpolation for smooth velocity
                # p(τ) = (2τ³ - 3τ² + 1)p₀ + (τ³ - 2τ² + τ)v₀ + (-2τ³ + 3τ²)p₁ + (τ³ - τ²)v₁
                
                h1 = 2*tau**3 - 3*tau**2 + 1
                h2 = tau**3 - 2*tau**2 + tau
                h3 = -2*tau**3 + 3*tau**2
                h4 = tau**3 - tau**2
                
                duration = t_end - t_start
                
                position = (
                    h1 * segment.start_position +
                    h2 * duration * segment.start_velocity +
                    h3 * segment.end_position +
                    h4 * duration * segment.end_velocity
                )
                
                # Velocity (derivative of position)
                h1_dot = (6*tau**2 - 6*tau) / duration
                h2_dot = (3*tau**2 - 4*tau + 1)
                h3_dot = (-6*tau**2 + 6*tau) / duration
                h4_dot = (3*tau**2 - 2*tau)
                
                velocity = (
                    h1_dot * segment.start_position +
                    h2_dot * segment.start_velocity +
                    h3_dot * segment.end_position +
                    h4_dot * segment.end_velocity
                )
                
                # Acceleration (derivative of velocity)
                h1_ddot = (12*tau - 6) / duration**2
                h2_ddot = (6*tau - 4) / duration
                h3_ddot = (-12*tau + 6) / duration**2
                h4_ddot = (6*tau - 2) / duration
                
                acceleration = (
                    h1_ddot * segment.start_position +
                    h2_ddot * segment.start_velocity +
                    h3_ddot * segment.end_position +
                    h4_ddot * segment.end_velocity
                )
                
                trajectory.append({
                    'time': float(t),
                    'position': position,
                    'velocity': velocity,
                    'acceleration': acceleration
                })
        
        return trajectory


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    generator = TrajectoryGenerator()
    
    # Define waypoints
    waypoints = [
        np.array([0.0, 0.0, 100.0]),
        np.array([100.0, 50.0, 150.0]),
        np.array([200.0, 100.0, 120.0]),
        np.array([300.0, 50.0, 100.0])
    ]
    
    # Generate trajectory
    trajectory = generator.generate_trajectory(waypoints)
    
    print(f"Generated trajectory: {len(trajectory)} points")
    print(f"Duration: {trajectory[-1]['time']:.2f} seconds")
    print(f"\nSample points:")
    for i in [0, len(trajectory)//2, -1]:
        point = trajectory[i]
        print(f"  t={point['time']:.2f}s: pos={point['position']}, vel_mag={np.linalg.norm(point['velocity']):.2f}m/s")

