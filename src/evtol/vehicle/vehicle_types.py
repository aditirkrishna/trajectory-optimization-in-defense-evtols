"""
Data Types for Vehicle Layer

This module contains the core data structures used throughout the vehicle layer
to avoid circular import issues.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class VehicleState:
    """Complete vehicle state vector"""
    # Position and attitude
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    attitude: np.ndarray  # [roll, pitch, yaw] in radians
    angular_velocity: np.ndarray  # [p, q, r] in rad/s
    
    # Energy state
    battery_soc: float  # State of charge (0-1)
    battery_temperature: float  # Temperature in Celsius
    battery_voltage: float  # Voltage in Volts
    
    # Actuator states
    rotor_rpm: np.ndarray  # RPM for each rotor
    control_surface_deflections: np.ndarray  # Control surface angles
    
    # Time
    time: float  # Current simulation time


@dataclass
class ControlInputs:
    """Vehicle control inputs"""
    # Rotor commands
    main_rotor_rpm: np.ndarray  # RPM for main rotors
    tail_rotor_rpm: float  # RPM for tail rotor
    lift_fan_rpm: np.ndarray  # RPM for lift fans
    propeller_rpm: np.ndarray  # RPM for propellers
    
    # Control surface deflections
    elevator_deflection: float  # Elevator deflection in radians
    aileron_deflection: float  # Aileron deflection in radians
    rudder_deflection: float  # Rudder deflection in radians
    
    # Throttle and collective
    throttle: float  # Throttle position (0-1)
    collective: float  # Collective pitch (0-1)



