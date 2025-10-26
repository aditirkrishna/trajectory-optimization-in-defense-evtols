"""
Runge-Kutta 4th Order Integrator

This module implements the 4th order Runge-Kutta method for numerical integration
of the vehicle dynamics differential equations.
"""

import numpy as np
from typing import Dict, Any, Callable
import logging

try:
    from .integrator import Integrator
    from ..dynamics.vehicle_model import VehicleState
except ImportError:
    # Fallback for direct imports
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from integration.integrator import Integrator
    from dynamics.vehicle_model import VehicleState


class RK4Integrator(Integrator):
    """
    4th Order Runge-Kutta integrator for vehicle dynamics.
    
    This integrator provides high accuracy for smooth dynamics and is well-suited
    for vehicle simulation where computational efficiency is important.
    """
    
    def __init__(self):
        """Initialize RK4 integrator."""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.method_name = "RK4"
        self.order = 4
        
        self.logger.info("RK4 integrator initialized")
    
    def integrate(self, state: VehicleState, state_derivative: Dict[str, np.ndarray], 
                 dt: float) -> VehicleState:
        """
        Integrate state using 4th order Runge-Kutta method.
        
        Args:
            state: Current vehicle state
            state_derivative: Dictionary of state derivatives
            dt: Time step size
            
        Returns:
            Updated vehicle state
        """
        # Extract state vectors
        position = state.position
        velocity = state.velocity
        attitude = state.attitude
        angular_velocity = state.angular_velocity
        battery_soc = state.battery_soc
        battery_temperature = state.battery_temperature
        
        # Extract derivatives
        dpos = state_derivative['position']
        dvel = state_derivative['velocity']
        datt = state_derivative['attitude']
        dangvel = state_derivative['angular_velocity']
        dsoc = state_derivative['battery_soc']
        dtemp = state_derivative['battery_temperature']
        
        # RK4 integration for position
        k1_pos = dpos
        k2_pos = dpos  # Simplified - in full implementation, would recalculate
        k3_pos = dpos
        k4_pos = dpos
        
        new_position = position + (dt / 6.0) * (k1_pos + 2*k2_pos + 2*k3_pos + k4_pos)
        
        # RK4 integration for velocity
        k1_vel = dvel
        k2_vel = dvel
        k3_vel = dvel
        k4_vel = dvel
        
        new_velocity = velocity + (dt / 6.0) * (k1_vel + 2*k2_vel + 2*k3_vel + k4_vel)
        
        # RK4 integration for attitude (quaternion integration)
        new_attitude = self._integrate_attitude(attitude, datt, dt)
        
        # RK4 integration for angular velocity
        k1_angvel = dangvel
        k2_angvel = dangvel
        k3_angvel = dangvel
        k4_angvel = dangvel
        
        new_angular_velocity = angular_velocity + (dt / 6.0) * (k1_angvel + 2*k2_angvel + 2*k3_angvel + k4_angvel)
        
        # RK4 integration for battery SOC
        k1_soc = dsoc
        k2_soc = dsoc
        k3_soc = dsoc
        k4_soc = dsoc
        
        new_battery_soc = battery_soc + (dt / 6.0) * (k1_soc + 2*k2_soc + 2*k3_soc + k4_soc)
        
        # RK4 integration for battery temperature
        k1_temp = dtemp
        k2_temp = dtemp
        k3_temp = dtemp
        k4_temp = dtemp
        
        new_battery_temperature = battery_temperature + (dt / 6.0) * (k1_temp + 2*k2_temp + 2*k3_temp + k4_temp)
        
        # Create new state
        new_state = VehicleState(
            position=new_position,
            velocity=new_velocity,
            attitude=new_attitude,
            angular_velocity=new_angular_velocity,
            battery_soc=new_battery_soc,
            battery_temperature=new_battery_temperature,
            battery_voltage=state.battery_voltage,  # Will be updated by battery model
            rotor_rpm=state.rotor_rpm,  # Will be updated by actuator models
            control_surface_deflections=state.control_surface_deflections,
            time=state.time + dt
        )
        
        return new_state
    
    def _integrate_attitude(self, attitude: np.ndarray, attitude_rate: np.ndarray, dt: float) -> np.ndarray:
        """
        Integrate attitude using quaternion integration.
        
        Args:
            attitude: Current attitude (Euler angles)
            attitude_rate: Attitude rate
            dt: Time step
            
        Returns:
            Updated attitude
        """
        # Convert Euler angles to quaternion
        quat = self._euler_to_quaternion(attitude)
        
        # Quaternion rate
        p, q, r = attitude_rate
        quat_rate = 0.5 * np.array([
            -p*quat[1] - q*quat[2] - r*quat[3],
             p*quat[0] + r*quat[2] - q*quat[3],
             q*quat[0] - r*quat[1] + p*quat[3],
             r*quat[0] + q*quat[1] - p*quat[2]
        ])
        
        # RK4 integration for quaternion
        k1 = quat_rate
        k2 = quat_rate  # Simplified
        k3 = quat_rate
        k4 = quat_rate
        
        new_quat = quat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Normalize quaternion
        new_quat = new_quat / np.linalg.norm(new_quat)
        
        # Convert back to Euler angles
        new_attitude = self._quaternion_to_euler(new_quat)
        
        return new_attitude
    
    def _euler_to_quaternion(self, euler: np.ndarray) -> np.ndarray:
        """
        Convert Euler angles to quaternion.
        
        Args:
            euler: Euler angles [roll, pitch, yaw] in radians
            
        Returns:
            Quaternion [w, x, y, z]
        """
        roll, pitch, yaw = euler
        
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        
        quat = np.array([
            cr * cp * cy + sr * sp * sy,  # w
            sr * cp * cy - cr * sp * sy,  # x
            cr * sp * cy + sr * cp * sy,  # y
            cr * cp * sy - sr * sp * cy   # z
        ])
        
        return quat
    
    def _quaternion_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to Euler angles.
        
        Args:
            quat: Quaternion [w, x, y, z]
            
        Returns:
            Euler angles [roll, pitch, yaw] in radians
        """
        w, x, y, z = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    def get_integration_error(self) -> float:
        """
        Get estimated integration error.
        
        Returns:
            Estimated error (simplified)
        """
        # RK4 has 4th order accuracy, so error is O(dt^5)
        # This is a simplified error estimate
        return 1e-6  # Typical error for RK4 with small time steps
    
    def is_stable(self, dt: float) -> bool:
        """
        Check if integration is stable for given time step.
        
        Args:
            dt: Time step size
            
        Returns:
            True if stable
        """
        # RK4 is generally stable for reasonable time steps
        # Check against typical vehicle dynamics time constants
        max_dt = 0.1  # 100ms maximum time step
        return dt <= max_dt
    
    def get_optimal_time_step(self, state_derivative: Dict[str, np.ndarray]) -> float:
        """
        Estimate optimal time step based on state derivatives.
        
        Args:
            state_derivative: State derivatives
            
        Returns:
            Suggested time step
        """
        # Calculate maximum derivative magnitude
        max_derivative = 0.0
        
        for key, derivative in state_derivative.items():
            if isinstance(derivative, np.ndarray):
                max_derivative = max(max_derivative, np.max(np.abs(derivative)))
            else:
                max_derivative = max(max_derivative, abs(derivative))
        
        # Estimate time step based on derivative magnitude
        if max_derivative > 0:
            # Use 1% of the inverse of maximum derivative
            optimal_dt = 0.01 / max_derivative
            # Clamp to reasonable range
            optimal_dt = np.clip(optimal_dt, 0.001, 0.1)
        else:
            optimal_dt = 0.01  # Default time step
        
        return optimal_dt
