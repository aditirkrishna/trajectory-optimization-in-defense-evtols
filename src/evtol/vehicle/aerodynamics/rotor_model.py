"""
Rotor Model - Comprehensive Rotor Aerodynamics

This module implements detailed rotor aerodynamics including:
- Blade element momentum theory
- Induced velocity calculations
- Thrust and torque generation
- Efficiency modeling
- Ground effect
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from scipy.optimize import fsolve

from ..utils import VehicleConfig


@dataclass
class RotorState:
    """Rotor state variables"""
    rpm: float  # Rotor RPM
    thrust: float  # Thrust in N
    torque: float  # Torque in N·m
    power: float  # Power in W
    efficiency: float  # Efficiency (0-1)
    induced_velocity: float  # Induced velocity in m/s
    advance_ratio: float  # Advance ratio (V/(Ω*R))


class RotorModel:
    """
    Comprehensive rotor model using blade element momentum theory (BEMT)
    with support for various flight conditions and ground effect.
    """
    
    def __init__(self, config: VehicleConfig, rotor_id: str = "main_rotor"):
        """
        Initialize rotor model.
        
        Args:
            config: Vehicle configuration
            rotor_id: Rotor identifier
        """
        self.config = config
        self.rotor_id = rotor_id
        self.logger = logging.getLogger(__name__)
        
        # Load rotor parameters
        self._load_rotor_parameters()
        
        # Initialize state
        self.state = RotorState(
            rpm=0.0,
            thrust=0.0,
            torque=0.0,
            power=0.0,
            efficiency=0.0,
            induced_velocity=0.0,
            advance_ratio=0.0
        )
        
        # Performance lookup tables (from dataset)
        self._load_performance_data()
        
        self.logger.info(f"Rotor model initialized: {rotor_id}")
    
    def _load_rotor_parameters(self) -> None:
        """Load rotor parameters from configuration."""
        if self.rotor_id == "main_rotor":
            rotor_config = self.config.actuators.main_rotors
        elif self.rotor_id == "tail_rotor":
            rotor_config = self.config.actuators.tail_rotor
        elif self.rotor_id == "lift_fan":
            rotor_config = self.config.actuators.lift_fans
        else:
            raise ValueError(f"Unknown rotor type: {self.rotor_id}")
        
        self.diameter = rotor_config.diameter  # m
        self.radius = self.diameter / 2.0  # m
        self.max_thrust = rotor_config.max_thrust  # N
        self.max_torque = rotor_config.max_torque  # N·m
        self.efficiency_peak = rotor_config.efficiency_peak
        
        # Blade parameters (simplified)
        self.num_blades = 4  # Number of blades
        self.blade_chord = 0.15  # m, average chord
        self.blade_twist = -8.0  # degrees, linear twist
        self.blade_taper = 0.8  # Tip chord / root chord ratio
        
        # Airfoil characteristics (simplified)
        self.cl_alpha = 5.73  # Lift curve slope (1/rad)
        self.cd0 = 0.01  # Zero-lift drag coefficient
        self.cd_alpha2 = 0.1  # Drag due to lift coefficient
    
    def _load_performance_data(self) -> None:
        """Load rotor performance data from dataset."""
        # This would load from rotor_thrust_curves_multi.csv
        # For now, we'll create synthetic data based on the dataset structure
        
        # RPM range
        self.rpm_data = np.linspace(800, 6000, 100)
        
        # Altitude and temperature effects
        self.altitude_data = np.array([0, 2000, 4000])  # m
        self.temp_data = np.array([-20, 0, 20])  # °C
        
        # Create performance lookup tables
        self.thrust_table = np.zeros((len(self.altitude_data), len(self.temp_data), len(self.rpm_data)))
        self.torque_table = np.zeros((len(self.altitude_data), len(self.temp_data), len(self.rpm_data)))
        self.efficiency_table = np.zeros((len(self.altitude_data), len(self.temp_data), len(self.rpm_data)))
        
        for i, alt in enumerate(self.altitude_data):
            for j, temp in enumerate(self.temp_data):
                for k, rpm in enumerate(self.rpm_data):
                    # Calculate atmospheric conditions
                    rho = self._calculate_air_density(alt, temp)
                    
                    # Calculate thrust and torque using BEMT
                    thrust, torque, efficiency = self._calculate_bemt_performance(rpm, rho)
                    
                    self.thrust_table[i, j, k] = thrust
                    self.torque_table[i, j, k] = torque
                    self.efficiency_table[i, j, k] = efficiency
    
    def _calculate_air_density(self, altitude: float, temperature: float) -> float:
        """
        Calculate air density at given altitude and temperature.
        
        Args:
            altitude: Altitude in meters
            temperature: Temperature in Celsius
            
        Returns:
            Air density in kg/m³
        """
        # Standard atmosphere model
        T = temperature + 273.15  # Convert to Kelvin
        p0 = 101325  # Pa, sea level pressure
        T0 = 288.15  # K, sea level temperature
        g = 9.81  # m/s², gravitational acceleration
        R = 287.0  # J/kg·K, specific gas constant
        
        # Pressure at altitude
        p = p0 * np.exp(-g * altitude / (R * T0))
        
        # Air density
        rho = p / (R * T)
        
        return rho
    
    def _calculate_bemt_performance(self, rpm: float, rho: float) -> Tuple[float, float, float]:
        """
        Calculate rotor performance using Blade Element Momentum Theory.
        
        Args:
            rpm: Rotor RPM
            rho: Air density in kg/m³
            
        Returns:
            Tuple of (thrust, torque, efficiency)
        """
        if rpm <= 0:
            return 0.0, 0.0, 0.0
        
        # Rotor angular velocity
        omega = rpm * 2 * np.pi / 60  # rad/s
        
        # Blade tip speed
        V_tip = omega * self.radius  # m/s
        
        # Solidity
        sigma = self.num_blades * self.blade_chord / (np.pi * self.radius)
        
        # Thrust coefficient (simplified BEMT)
        # This is a simplified model - full BEMT would require iterative solution
        C_T = 0.1 * (rpm / 3000) ** 2  # Simplified thrust coefficient
        
        # Thrust
        thrust = C_T * rho * (omega * self.radius) ** 2 * np.pi * self.radius ** 2
        
        # Torque coefficient (simplified)
        C_Q = 0.01 * (rpm / 3000) ** 2  # Simplified torque coefficient
        
        # Torque
        torque = C_Q * rho * (omega * self.radius) ** 2 * np.pi * self.radius ** 3
        
        # Power
        power = torque * omega
        
        # Efficiency
        if power > 0:
            efficiency = thrust * self.state.induced_velocity / power
        else:
            efficiency = 0.0
        
        # Apply limits
        thrust = np.clip(thrust, 0, self.max_thrust)
        torque = np.clip(torque, 0, self.max_torque)
        efficiency = np.clip(efficiency, 0, self.efficiency_peak)
        
        return thrust, torque, efficiency
    
    def calculate_performance(self, rpm: float, altitude: float = 0.0, 
                            temperature: float = 20.0, forward_velocity: float = 0.0) -> RotorState:
        """
        Calculate rotor performance at given conditions.
        
        Args:
            rpm: Rotor RPM
            altitude: Altitude in meters
            temperature: Temperature in Celsius
            forward_velocity: Forward velocity in m/s
            
        Returns:
            RotorState with performance parameters
        """
        # Apply RPM limits
        rpm = np.clip(rpm, 800, 6000)
        
        # Calculate air density
        rho = self._calculate_air_density(altitude, temperature)
        
        # Calculate advance ratio
        omega = rpm * 2 * np.pi / 60
        advance_ratio = forward_velocity / (omega * self.radius)
        
        # Interpolate performance from lookup tables
        thrust, torque, efficiency = self._interpolate_performance(rpm, altitude, temperature)
        
        # Apply forward flight effects
        if forward_velocity > 0:
            thrust, torque, efficiency = self._apply_forward_flight_effects(
                thrust, torque, efficiency, advance_ratio
            )
        
        # Calculate induced velocity
        induced_velocity = self._calculate_induced_velocity(thrust, rho)
        
        # Calculate power
        power = torque * omega
        
        # Update state
        self.state = RotorState(
            rpm=rpm,
            thrust=thrust,
            torque=torque,
            power=power,
            efficiency=efficiency,
            induced_velocity=induced_velocity,
            advance_ratio=advance_ratio
        )
        
        return self.state
    
    def _interpolate_performance(self, rpm: float, altitude: float, temperature: float) -> Tuple[float, float, float]:
        """
        Interpolate performance from lookup tables.
        
        Args:
            rpm: Rotor RPM
            altitude: Altitude in meters
            temperature: Temperature in Celsius
            
        Returns:
            Tuple of (thrust, torque, efficiency)
        """
        # Find indices for interpolation
        alt_idx = np.searchsorted(self.altitude_data, altitude)
        temp_idx = np.searchsorted(self.temp_data, temperature)
        rpm_idx = np.searchsorted(self.rpm_data, rpm)
        
        # Clamp indices
        alt_idx = np.clip(alt_idx, 0, len(self.altitude_data) - 1)
        temp_idx = np.clip(temp_idx, 0, len(self.temp_data) - 1)
        rpm_idx = np.clip(rpm_idx, 0, len(self.rpm_data) - 1)
        
        # Simple nearest neighbor interpolation (could be improved with trilinear)
        thrust = self.thrust_table[alt_idx, temp_idx, rpm_idx]
        torque = self.torque_table[alt_idx, temp_idx, rpm_idx]
        efficiency = self.efficiency_table[alt_idx, temp_idx, rpm_idx]
        
        return thrust, torque, efficiency
    
    def _apply_forward_flight_effects(self, thrust: float, torque: float, 
                                    efficiency: float, advance_ratio: float) -> Tuple[float, float, float]:
        """
        Apply forward flight effects to rotor performance.
        
        Args:
            thrust: Hover thrust
            torque: Hover torque
            efficiency: Hover efficiency
            advance_ratio: Advance ratio
            
        Returns:
            Tuple of (thrust, torque, efficiency) with forward flight effects
        """
        # Forward flight reduces thrust and torque requirements
        # This is a simplified model
        
        # Thrust reduction factor
        thrust_factor = 1.0 - 0.3 * advance_ratio
        
        # Torque reduction factor
        torque_factor = 1.0 - 0.4 * advance_ratio
        
        # Efficiency improvement in forward flight
        efficiency_factor = 1.0 + 0.2 * advance_ratio
        
        # Apply factors
        thrust *= thrust_factor
        torque *= torque_factor
        efficiency *= efficiency_factor
        
        # Ensure positive values
        thrust = max(0, thrust)
        torque = max(0, torque)
        efficiency = min(efficiency, self.efficiency_peak)
        
        return thrust, torque, efficiency
    
    def _calculate_induced_velocity(self, thrust: float, rho: float) -> float:
        """
        Calculate induced velocity using momentum theory.
        
        Args:
            thrust: Rotor thrust in N
            rho: Air density in kg/m³
            
        Returns:
            Induced velocity in m/s
        """
        if thrust <= 0:
            return 0.0
        
        # Disk area
        A = np.pi * self.radius ** 2
        
        # Induced velocity from momentum theory
        v_i = np.sqrt(thrust / (2 * rho * A))
        
        return v_i
    
    def calculate_ground_effect(self, altitude: float, thrust: float) -> float:
        """
        Calculate ground effect factor.
        
        Args:
            altitude: Altitude above ground in meters
            thrust: Rotor thrust in N
            
        Returns:
            Ground effect factor (1.0 = no effect, >1.0 = increased thrust)
        """
        if altitude <= 0:
            return 1.0
        
        # Ground effect is significant when altitude < 2 * rotor diameter
        if altitude > 2 * self.diameter:
            return 1.0
        
        # Ground effect factor (simplified model)
        # Based on momentum theory with image vortex
        z_over_R = altitude / self.radius
        
        if z_over_R < 0.5:
            # Strong ground effect
            ground_effect_factor = 1.0 + 0.5 / z_over_R
        else:
            # Moderate ground effect
            ground_effect_factor = 1.0 + 0.1 / z_over_R
        
        return min(ground_effect_factor, 2.0)  # Cap at 2x thrust increase
    
    def get_thrust_from_rpm(self, rpm: float, altitude: float = 0.0, 
                          temperature: float = 20.0) -> float:
        """
        Get thrust for given RPM and conditions.
        
        Args:
            rpm: Rotor RPM
            altitude: Altitude in meters
            temperature: Temperature in Celsius
            
        Returns:
            Thrust in N
        """
        state = self.calculate_performance(rpm, altitude, temperature)
        return state.thrust
    
    def get_rpm_from_thrust(self, thrust: float, altitude: float = 0.0, 
                          temperature: float = 20.0) -> float:
        """
        Get required RPM for given thrust.
        
        Args:
            thrust: Required thrust in N
            altitude: Altitude in meters
            temperature: Temperature in Celsius
            
        Returns:
            Required RPM
        """
        # Use binary search to find RPM for given thrust
        rpm_min, rpm_max = 800, 6000
        
        for _ in range(20):  # Maximum iterations
            rpm_mid = (rpm_min + rpm_max) / 2
            thrust_mid = self.get_thrust_from_rpm(rpm_mid, altitude, temperature)
            
            if abs(thrust_mid - thrust) < 1.0:  # 1N tolerance
                return rpm_mid
            
            if thrust_mid < thrust:
                rpm_min = rpm_mid
            else:
                rpm_max = rpm_mid
        
        return (rpm_min + rpm_max) / 2
    
    def get_current_state(self) -> RotorState:
        """Get current rotor state."""
        return self.state
    
    def get_efficiency(self) -> float:
        """Get current efficiency."""
        return self.state.efficiency
    
    def get_power_consumption(self) -> float:
        """Get current power consumption."""
        return self.state.power
    
    def get_thrust(self) -> float:
        """Get current thrust."""
        return self.state.thrust
    
    def get_torque(self) -> float:
        """Get current torque."""
        return self.state.torque
    
    def get_induced_velocity(self) -> float:
        """Get current induced velocity."""
        return self.state.induced_velocity
    
    def is_operational(self) -> bool:
        """Check if rotor is operational."""
        return (self.state.rpm > 0 and 
                self.state.thrust >= 0 and 
                self.state.torque >= 0 and
                self.state.efficiency > 0)
    
    def get_health_status(self) -> Dict[str, any]:
        """
        Get rotor health status.
        
        Returns:
            Dictionary with health metrics
        """
        return {
            'operational': self.is_operational(),
            'rpm': self.state.rpm,
            'thrust': self.state.thrust,
            'torque': self.state.torque,
            'power': self.state.power,
            'efficiency': self.state.efficiency,
            'induced_velocity': self.state.induced_velocity,
            'advance_ratio': self.state.advance_ratio,
            'efficiency_warning': self.state.efficiency < 0.7,
            'power_warning': self.state.power > self.max_torque * 6000 * 2 * np.pi / 60 * 0.8
        }

