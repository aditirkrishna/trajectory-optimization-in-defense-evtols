"""
Motor Model - Electric Motor Dynamics and Performance

This module implements comprehensive electric motor modeling including:
- Torque-speed characteristics
- Power consumption and efficiency
- Thermal effects
- Fault modeling
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

try:
    from ..utils.config import VehicleConfig
except ImportError:
    # Fallback for direct imports
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.config import VehicleConfig


@dataclass
class MotorState:
    """Motor state variables"""
    rpm: float  # Current RPM
    torque: float  # Current torque in N·m
    current: float  # Current in Amperes
    voltage: float  # Voltage in Volts
    power: float  # Power in Watts
    efficiency: float  # Efficiency (0-1)
    temperature: float  # Temperature in Celsius


class MotorModel:
    """
    Comprehensive electric motor model with performance characteristics,
    thermal effects, and fault modeling.
    """
    
    def __init__(self, config: VehicleConfig, motor_type: str = "main_rotor"):
        """
        Initialize motor model.
        
        Args:
            config: Vehicle configuration
            motor_type: Type of motor (main_rotor, tail_rotor, lift_fan, propeller)
        """
        self.config = config
        self.motor_type = motor_type
        self.logger = logging.getLogger(__name__)
        
        # Load motor parameters from config
        self._load_motor_parameters()
        
        # Initialize state
        self.state = MotorState(
            rpm=0.0,
            torque=0.0,
            current=0.0,
            voltage=0.0,
            power=0.0,
            efficiency=0.0,
            temperature=20.0  # Ambient temperature
        )
        
        # Performance lookup tables (would be loaded from dataset)
        self._initialize_performance_tables()
        
        self.logger.info(f"Motor model initialized: {motor_type}")
    
    def _load_motor_parameters(self) -> None:
        """Load motor parameters from configuration."""
        if self.motor_type == "main_rotor":
            motor_config = self.config.motors.main_rotor
        elif self.motor_type == "tail_rotor":
            motor_config = self.config.motors.tail_rotor
        elif self.motor_type == "lift_fan":
            motor_config = self.config.motors.lift_fan
        elif self.motor_type == "propeller":
            motor_config = self.config.motors.propeller
        else:
            raise ValueError(f"Unknown motor type: {self.motor_type}")
        
        self.kv = motor_config.kv  # RPM/V
        self.resistance = motor_config.resistance  # Ω
        self.inductance = motor_config.inductance  # H
        self.max_current = motor_config.max_current  # A
        self.max_voltage = motor_config.max_voltage  # V
        self.efficiency_peak = motor_config.efficiency_peak
        
        # Thermal parameters
        self.thermal_mass = 2.0  # kg
        self.thermal_resistance = 0.1  # K/W
        self.ambient_temperature = 20.0  # °C
    
    def _initialize_performance_tables(self) -> None:
        """Initialize motor performance lookup tables."""
        # Torque-speed-efficiency table (would be loaded from dataset)
        self.rpm_range = np.linspace(0, 6000, 100)
        self.torque_range = np.linspace(0, 2000, 50)
        
        # Create efficiency map
        self.efficiency_map = np.zeros((len(self.rpm_range), len(self.torque_range)))
        for i, rpm in enumerate(self.rpm_range):
            for j, torque in enumerate(self.torque_range):
                # Simplified efficiency model
                if rpm > 0 and torque > 0:
                    # Peak efficiency at moderate RPM and torque
                    rpm_factor = 1.0 - abs(rpm - 3000) / 3000 * 0.2
                    torque_factor = 1.0 - abs(torque - 1000) / 1000 * 0.3
                    self.efficiency_map[i, j] = self.efficiency_peak * rpm_factor * torque_factor
                else:
                    self.efficiency_map[i, j] = 0.0
    
    def update_state(self, voltage_command: float, load_torque: float, dt: float) -> None:
        """
        Update motor state based on voltage command and load torque.
        
        Args:
            voltage_command: Commanded voltage in Volts
            load_torque: Load torque in N·m
            dt: Time step in seconds
        """
        # Apply voltage limits
        voltage_command = np.clip(voltage_command, 0, self.max_voltage)
        
        # Calculate current (simplified motor model)
        back_emf = self.state.rpm / self.kv  # Back EMF voltage
        voltage_drop = voltage_command - back_emf
        current = voltage_drop / self.resistance
        
        # Apply current limits
        current = np.clip(current, -self.max_current, self.max_current)
        
        # Calculate torque
        torque_constant = 60 / (2 * np.pi * self.kv)  # N·m/A
        motor_torque = torque_constant * current
        
        # Net torque (motor torque - load torque)
        net_torque = motor_torque - load_torque
        
        # Update RPM (simplified dynamics)
        # Assuming constant inertia for now
        inertia = 0.1  # kg·m² (would be from motor specs)
        rpm_acceleration = net_torque / inertia * 60 / (2 * np.pi)  # RPM/s
        
        self.state.rpm += rpm_acceleration * dt
        self.state.rpm = np.clip(self.state.rpm, 0, 6000)  # RPM limits
        
        # Update other state variables
        self.state.torque = motor_torque
        self.state.current = current
        self.state.voltage = voltage_command
        self.state.power = voltage_command * current
        
        # Calculate efficiency
        if self.state.power > 0:
            mechanical_power = self.state.torque * self.state.rpm * 2 * np.pi / 60
            self.state.efficiency = mechanical_power / self.state.power
        else:
            self.state.efficiency = 0.0
        
        # Update temperature
        self._update_temperature(dt)
    
    def _update_temperature(self, dt: float) -> None:
        """
        Update motor temperature based on power losses.
        
        Args:
            dt: Time step in seconds
        """
        # Calculate power losses
        copper_losses = self.state.current**2 * self.resistance
        iron_losses = self.state.rpm**2 * 1e-6  # Simplified iron losses
        total_losses = copper_losses + iron_losses
        
        # Calculate temperature change
        temp_diff = self.state.temperature - self.ambient_temperature
        cooling_rate = temp_diff / self.thermal_resistance
        
        net_heat = total_losses - cooling_rate
        temp_change = net_heat * dt / (self.thermal_mass * 1000)  # J/kg·K
        
        self.state.temperature += temp_change
        self.state.temperature = np.clip(self.state.temperature, -20, 150)  # °C limits
    
    def get_torque_from_rpm(self, rpm: float) -> float:
        """
        Get maximum torque at given RPM.
        
        Args:
            rpm: RPM value
            
        Returns:
            Maximum torque in N·m
        """
        # Simplified torque-speed curve
        if rpm < 1000:
            return 2000.0  # Maximum torque at low RPM
        elif rpm < 3000:
            return 2000.0 - (rpm - 1000) * 0.5  # Linear decrease
        else:
            return 1000.0 - (rpm - 3000) * 0.2  # Further decrease
    
    def get_efficiency(self, rpm: float, torque: float) -> float:
        """
        Get motor efficiency at given RPM and torque.
        
        Args:
            rpm: RPM value
            torque: Torque in N·m
            
        Returns:
            Efficiency (0-1)
        """
        # Interpolate from efficiency map
        rpm_idx = np.searchsorted(self.rpm_range, rpm)
        torque_idx = np.searchsorted(self.torque_range, torque)
        
        if rpm_idx == 0 or rpm_idx >= len(self.rpm_range):
            return 0.0
        if torque_idx == 0 or torque_idx >= len(self.torque_range):
            return 0.0
        
        # Bilinear interpolation
        rpm_ratio = (rpm - self.rpm_range[rpm_idx-1]) / (self.rpm_range[rpm_idx] - self.rpm_range[rpm_idx-1])
        torque_ratio = (torque - self.torque_range[torque_idx-1]) / (self.torque_range[torque_idx] - self.torque_range[torque_idx-1])
        
        efficiency_00 = self.efficiency_map[rpm_idx-1, torque_idx-1]
        efficiency_01 = self.efficiency_map[rpm_idx-1, torque_idx]
        efficiency_10 = self.efficiency_map[rpm_idx, torque_idx-1]
        efficiency_11 = self.efficiency_map[rpm_idx, torque_idx]
        
        efficiency = (efficiency_00 * (1 - rpm_ratio) * (1 - torque_ratio) +
                     efficiency_01 * (1 - rpm_ratio) * torque_ratio +
                     efficiency_10 * rpm_ratio * (1 - torque_ratio) +
                     efficiency_11 * rpm_ratio * torque_ratio)
        
        return efficiency
    
    def get_power_consumption(self, rpm: float, torque: float) -> float:
        """
        Get power consumption at given RPM and torque.
        
        Args:
            rpm: RPM value
            torque: Torque in N·m
            
        Returns:
            Power consumption in Watts
        """
        mechanical_power = torque * rpm * 2 * np.pi / 60
        efficiency = self.get_efficiency(rpm, torque)
        
        if efficiency > 0:
            electrical_power = mechanical_power / efficiency
        else:
            electrical_power = 0.0
        
        return electrical_power
    
    def get_current_state(self) -> MotorState:
        """Get current motor state."""
        return self.state
    
    def get_rpm(self) -> float:
        """Get current RPM."""
        return self.state.rpm
    
    def get_torque(self) -> float:
        """Get current torque."""
        return self.state.torque
    
    def get_current(self) -> float:
        """Get current current."""
        return self.state.current
    
    def get_voltage(self) -> float:
        """Get current voltage."""
        return self.state.voltage
    
    def get_power(self) -> float:
        """Get current power."""
        return self.state.power
    
    def get_efficiency(self) -> float:
        """Get current efficiency."""
        return self.state.efficiency
    
    def get_temperature(self) -> float:
        """Get current temperature."""
        return self.state.temperature
    
    def is_operational(self) -> bool:
        """Check if motor is operational."""
        return (self.state.temperature < 120.0 and  # Temperature limit
                self.state.current < self.max_current and  # Current limit
                self.state.voltage < self.max_voltage)  # Voltage limit
    
    def get_health_status(self) -> Dict[str, any]:
        """
        Get motor health status.
        
        Returns:
            Dictionary with health metrics
        """
        return {
            'operational': self.is_operational(),
            'temperature': self.state.temperature,
            'current': self.state.current,
            'voltage': self.state.voltage,
            'rpm': self.state.rpm,
            'torque': self.state.torque,
            'efficiency': self.state.efficiency,
            'power': self.state.power,
            'temperature_warning': self.state.temperature > 80.0,
            'current_warning': self.state.current > self.max_current * 0.8,
            'efficiency_warning': self.state.efficiency < 0.7
        }
