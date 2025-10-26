"""
Vehicle Configuration Management

This module provides configuration management for the vehicle layer,
including loading and validation of vehicle parameters from YAML files.
"""

import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging
from pathlib import Path


@dataclass
class MassProperties:
    """Vehicle mass and inertia properties"""
    total: float = 1500.0
    empty: float = 900.0
    payload_max: float = 600.0
    cg_offset: list = field(default_factory=lambda: [0.5, 0.0, -0.2])


@dataclass
class InertiaProperties:
    """Vehicle inertia tensor"""
    Ixx: float = 8781.25
    Iyy: float = 12500.0
    Izz: float = 5281.25
    Ixy: float = 0.0
    Ixz: float = 0.0
    Iyz: float = 0.0


@dataclass
class VehicleDimensions:
    """Vehicle physical dimensions"""
    length: float = 8.0
    width: float = 6.0
    height: float = 2.5


@dataclass
class VehicleGeometry:
    """Complete vehicle geometry"""
    mass: MassProperties = field(default_factory=MassProperties)
    inertia: InertiaProperties = field(default_factory=InertiaProperties)
    dimensions: VehicleDimensions = field(default_factory=VehicleDimensions)


@dataclass
class ActuatorLimits:
    """Actuator performance limits"""
    max_deflection: float = 0.35
    max_rate: float = 2.0
    effectiveness: float = 0.8


@dataclass
class ControlSurface:
    """Control surface configuration"""
    elevator: ActuatorLimits = field(default_factory=ActuatorLimits)
    ailerons: ActuatorLimits = field(default_factory=ActuatorLimits)
    rudder: ActuatorLimits = field(default_factory=ActuatorLimits)


@dataclass
class RotorConfig:
    """Rotor configuration"""
    count: int = 4
    diameter: float = 2.5
    rpm_range: list = field(default_factory=lambda: [800, 6000])
    max_thrust: float = 4000.0
    max_torque: float = 2000.0
    efficiency_peak: float = 0.88
    position: list = field(default_factory=list)


@dataclass
class ActuatorConfig:
    """Complete actuator configuration"""
    main_rotors: RotorConfig = field(default_factory=RotorConfig)
    tail_rotor: RotorConfig = field(default_factory=RotorConfig)
    lift_fans: RotorConfig = field(default_factory=RotorConfig)
    propellers: RotorConfig = field(default_factory=RotorConfig)
    control_surfaces: ControlSurface = field(default_factory=ControlSurface)


@dataclass
class BatteryLimits:
    """Battery operational limits"""
    max_discharge_rate: float = 5.0
    max_charge_rate: float = 2.0
    min_soc: float = 0.1
    max_soc: float = 0.95
    min_temperature: float = -20.0
    max_temperature: float = 60.0


@dataclass
class ThermalProperties:
    """Battery thermal properties"""
    mass: float = 50.0
    specific_heat: float = 1000.0
    thermal_conductivity: float = 0.5
    cooling_capacity: float = 5000.0


@dataclass
class BatteryConfig:
    """Battery system configuration"""
    chemistry: str = "Li-ion_NMC"
    capacity_nominal: float = 200.0
    voltage_nominal: float = 400.0
    voltage_range: list = field(default_factory=lambda: [320.0, 450.0])
    thermal: ThermalProperties = field(default_factory=ThermalProperties)
    limits: BatteryLimits = field(default_factory=BatteryLimits)


@dataclass
class FlightEnvelopeConfig:
    """Flight envelope limits"""
    speed: Dict[str, float] = field(default_factory=lambda: {
        'min': 0.0, 'max': 120.0, 'maneuver': 100.0
    })
    altitude: Dict[str, float] = field(default_factory=lambda: {
        'min': 0.0, 'max': 5000.0, 'operational': 4000.0
    })
    load_factors: Dict[str, float] = field(default_factory=lambda: {
        'max_positive': 3.0, 'max_negative': -1.5, 'max_lateral': 2.0
    })
    climb_rate: Dict[str, float] = field(default_factory=lambda: {
        'max': 10.0, 'min': -15.0
    })
    turn_rate: Dict[str, float] = field(default_factory=lambda: {
        'max': 0.5, 'min': -0.5
    })


class VehicleConfig:
    """
    Vehicle configuration manager that loads and validates vehicle parameters.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize vehicle configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize with default values
        self.vehicle = VehicleGeometry()
        self.actuators = ActuatorConfig()
        self.battery = BatteryConfig()
        self.flight_envelope = FlightEnvelopeConfig()
        
        # Load configuration if path provided
        if config_path:
            self.load_config(config_path)
        
        self.logger.info("Vehicle configuration initialized")
    
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            self._parse_config(config_data)
            self.logger.info(f"Configuration loaded from {config_path}")
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")
    
    def _parse_config(self, config_data: Dict[str, Any]) -> None:
        """
        Parse configuration data into dataclass structures.
        
        Args:
            config_data: Raw configuration dictionary
        """
        # Parse vehicle geometry
        if 'vehicle' in config_data:
            vehicle_data = config_data['vehicle']
            
            if 'mass' in vehicle_data:
                mass_data = vehicle_data['mass']
                self.vehicle.mass = MassProperties(
                    total=mass_data.get('total', 1500.0),
                    empty=mass_data.get('empty', 900.0),
                    payload_max=mass_data.get('payload_max', 600.0),
                    cg_offset=mass_data.get('cg_offset', [0.5, 0.0, -0.2])
                )
            
            if 'inertia' in vehicle_data:
                inertia_data = vehicle_data['inertia']
                self.vehicle.inertia = InertiaProperties(
                    Ixx=inertia_data.get('Ixx', 8781.25),
                    Iyy=inertia_data.get('Iyy', 12500.0),
                    Izz=inertia_data.get('Izz', 5281.25),
                    Ixy=inertia_data.get('Ixy', 0.0),
                    Ixz=inertia_data.get('Ixz', 0.0),
                    Iyz=inertia_data.get('Iyz', 0.0)
                )
            
            if 'dimensions' in vehicle_data:
                dim_data = vehicle_data['dimensions']
                self.vehicle.dimensions = VehicleDimensions(
                    length=dim_data.get('length', 8.0),
                    width=dim_data.get('width', 6.0),
                    height=dim_data.get('height', 2.5)
                )
        
        # Parse actuator configuration
        if 'actuators' in config_data:
            actuator_data = config_data['actuators']
            
            if 'main_rotors' in actuator_data:
                rotor_data = actuator_data['main_rotors']
                self.actuators.main_rotors = RotorConfig(
                    count=rotor_data.get('count', 4),
                    diameter=rotor_data.get('diameter', 2.5),
                    rpm_range=rotor_data.get('rpm_range', [800, 6000]),
                    max_thrust=rotor_data.get('max_thrust', 4000.0),
                    max_torque=rotor_data.get('max_torque', 2000.0),
                    efficiency_peak=rotor_data.get('efficiency_peak', 0.88),
                    position=rotor_data.get('position', [])
                )
        
        # Parse battery configuration
        if 'battery' in config_data:
            battery_data = config_data['battery']
            
            self.battery = BatteryConfig(
                chemistry=battery_data.get('chemistry', 'Li-ion_NMC'),
                capacity_nominal=battery_data.get('capacity_nominal', 200.0),
                voltage_nominal=battery_data.get('voltage_nominal', 400.0),
                voltage_range=battery_data.get('voltage_range', [320.0, 450.0])
            )
            
            if 'thermal' in battery_data:
                thermal_data = battery_data['thermal']
                self.battery.thermal = ThermalProperties(
                    mass=thermal_data.get('mass', 50.0),
                    specific_heat=thermal_data.get('specific_heat', 1000.0),
                    thermal_conductivity=thermal_data.get('thermal_conductivity', 0.5),
                    cooling_capacity=thermal_data.get('cooling_capacity', 5000.0)
                )
            
            if 'limits' in battery_data:
                limits_data = battery_data['limits']
                self.battery.limits = BatteryLimits(
                    max_discharge_rate=limits_data.get('max_discharge_rate', 5.0),
                    max_charge_rate=limits_data.get('max_charge_rate', 2.0),
                    min_soc=limits_data.get('min_soc', 0.1),
                    max_soc=limits_data.get('max_soc', 0.95),
                    min_temperature=limits_data.get('min_temperature', -20.0),
                    max_temperature=limits_data.get('max_temperature', 60.0)
                )
        
        # Parse flight envelope
        if 'flight_envelope' in config_data:
            envelope_data = config_data['flight_envelope']
            self.flight_envelope = FlightEnvelopeConfig(
                speed=envelope_data.get('speed', {}),
                altitude=envelope_data.get('altitude', {}),
                load_factors=envelope_data.get('load_factors', {}),
                climb_rate=envelope_data.get('climb_rate', {}),
                turn_rate=envelope_data.get('turn_rate', {})
            )
    
    def validate_config(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if configuration is valid
        """
        errors = []
        
        # Validate mass properties
        if self.vehicle.mass.total <= 0:
            errors.append("Total mass must be positive")
        
        if self.vehicle.mass.payload_max > self.vehicle.mass.total:
            errors.append("Payload mass cannot exceed total mass")
        
        # Validate inertia properties
        if self.vehicle.inertia.Ixx <= 0 or self.vehicle.inertia.Iyy <= 0 or self.vehicle.inertia.Izz <= 0:
            errors.append("Inertia values must be positive")
        
        # Validate battery properties
        if self.battery.capacity_nominal <= 0:
            errors.append("Battery capacity must be positive")
        
        if self.battery.voltage_nominal <= 0:
            errors.append("Battery voltage must be positive")
        
        if self.battery.limits.min_soc >= self.battery.limits.max_soc:
            errors.append("Minimum SOC must be less than maximum SOC")
        
        # Validate flight envelope
        if self.flight_envelope.speed.get('max', 0) <= self.flight_envelope.speed.get('min', 0):
            errors.append("Maximum speed must be greater than minimum speed")
        
        if errors:
            for error in errors:
                self.logger.error(f"Configuration validation error: {error}")
            return False
        
        self.logger.info("Configuration validation passed")
        return True
    
    def get_vehicle_property(self, property_name: str) -> float:
        """
        Get a vehicle property by name.
        
        Args:
            property_name: Name of the property to retrieve
            
        Returns:
            Property value
        """
        # Map property names to actual config values
        property_map = {
            'mass': self.vehicle.mass.total,
            'cg_x': self.vehicle.mass.cg_offset[0],
            'cg_y': self.vehicle.mass.cg_offset[1],
            'cg_z': self.vehicle.mass.cg_offset[2],
            'Ixx': self.vehicle.inertia.Ixx,
            'Iyy': self.vehicle.inertia.Iyy,
            'Izz': self.vehicle.inertia.Izz,
            'Ixy': self.vehicle.inertia.Ixy,
            'Ixz': self.vehicle.inertia.Ixz,
            'Iyz': self.vehicle.inertia.Iyz,
        }
        
        if property_name not in property_map:
            raise ValueError(f"Unknown vehicle property: {property_name}")
        
        return property_map[property_name]
    
    def get_vehicle_mass(self) -> float:
        """Get total vehicle mass."""
        return self.vehicle.mass.total
    
    def get_inertia_tensor(self) -> Dict[str, float]:
        """Get inertia tensor components."""
        return {
            'Ixx': self.vehicle.inertia.Ixx,
            'Iyy': self.vehicle.inertia.Iyy,
            'Izz': self.vehicle.inertia.Izz,
            'Ixy': self.vehicle.inertia.Ixy,
            'Ixz': self.vehicle.inertia.Ixz,
            'Iyz': self.vehicle.inertia.Iyz
        }
    
    def get_battery_capacity(self) -> float:
        """Get battery capacity in Ah."""
        return self.battery.capacity_nominal
    
    def get_battery_voltage(self) -> float:
        """Get battery nominal voltage."""
        return self.battery.voltage_nominal
    
    def get_flight_envelope_limits(self) -> Dict[str, Any]:
        """Get flight envelope limits."""
        return {
            'speed': self.flight_envelope.speed,
            'altitude': self.flight_envelope.altitude,
            'load_factors': self.flight_envelope.load_factors,
            'climb_rate': self.flight_envelope.climb_rate,
            'turn_rate': self.flight_envelope.turn_rate
        }
    
    def save_config(self, config_path: str) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            config_path: Path to save configuration file
        """
        config_data = {
            'vehicle': {
                'mass': {
                    'total': self.vehicle.mass.total,
                    'empty': self.vehicle.mass.empty,
                    'payload_max': self.vehicle.mass.payload_max,
                    'cg_offset': self.vehicle.mass.cg_offset
                },
                'inertia': {
                    'Ixx': self.vehicle.inertia.Ixx,
                    'Iyy': self.vehicle.inertia.Iyy,
                    'Izz': self.vehicle.inertia.Izz,
                    'Ixy': self.vehicle.inertia.Ixy,
                    'Ixz': self.vehicle.inertia.Ixz,
                    'Iyz': self.vehicle.inertia.Iyz
                },
                'dimensions': {
                    'length': self.vehicle.dimensions.length,
                    'width': self.vehicle.dimensions.width,
                    'height': self.vehicle.dimensions.height
                }
            },
            'battery': {
                'chemistry': self.battery.chemistry,
                'capacity_nominal': self.battery.capacity_nominal,
                'voltage_nominal': self.battery.voltage_nominal,
                'voltage_range': self.battery.voltage_range,
                'thermal': {
                    'mass': self.battery.thermal.mass,
                    'specific_heat': self.battery.thermal.specific_heat,
                    'thermal_conductivity': self.battery.thermal.thermal_conductivity,
                    'cooling_capacity': self.battery.thermal.cooling_capacity
                },
                'limits': {
                    'max_discharge_rate': self.battery.limits.max_discharge_rate,
                    'max_charge_rate': self.battery.limits.max_charge_rate,
                    'min_soc': self.battery.limits.min_soc,
                    'max_soc': self.battery.limits.max_soc,
                    'min_temperature': self.battery.limits.min_temperature,
                    'max_temperature': self.battery.limits.max_temperature
                }
            },
            'flight_envelope': {
                'speed': self.flight_envelope.speed,
                'altitude': self.flight_envelope.altitude,
                'load_factors': self.flight_envelope.load_factors,
                'climb_rate': self.flight_envelope.climb_rate,
                'turn_rate': self.flight_envelope.turn_rate
            }
        }
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            raise ValueError(f"Error saving configuration: {e}")

