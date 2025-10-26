"""
Flight Envelope - Comprehensive Constraint Validation

This module provides comprehensive flight envelope constraint checking for eVTOL aircraft,
including speed, altitude, load factor, and operational limits.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

try:
    from ..utils.config import VehicleConfig
    from ..dynamics.vehicle_model import VehicleState, ControlInputs
except ImportError:
    # Fallback for direct imports
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.config import VehicleConfig
    from dynamics.vehicle_model import VehicleState, ControlInputs


@dataclass
class ConstraintViolation:
    """Constraint violation information"""
    constraint_type: str
    constraint_name: str
    current_value: float
    limit_value: float
    severity: float  # 0-1, where 1 is critical
    message: str


class FlightEnvelope:
    """
    Comprehensive flight envelope constraint checker for eVTOL aircraft.
    
    This class validates that the vehicle operates within safe and legal limits
    including speed, altitude, load factors, and operational constraints.
    """
    
    def __init__(self, config: VehicleConfig):
        """
        Initialize flight envelope with configuration.
        
        Args:
            config: Vehicle configuration containing flight envelope limits
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load flight envelope limits
        self.speed_limits = config.flight_envelope.speed
        self.altitude_limits = config.flight_envelope.altitude
        self.load_factor_limits = config.flight_envelope.load_factors
        self.climb_rate_limits = config.flight_envelope.climb_rate
        self.turn_rate_limits = config.flight_envelope.turn_rate
        
        # Additional operational limits
        self.payload_limits = {
            'max_payload': config.vehicle.mass.payload_max,
            'min_payload': 0.0
        }
        
        self.battery_limits = {
            'min_soc': config.battery.limits.min_soc,
            'max_soc': config.battery.limits.max_soc,
            'min_temperature': config.battery.limits.min_temperature,
            'max_temperature': config.battery.limits.max_temperature
        }
        
        # Constraint violation tracking
        self.violation_history: List[ConstraintViolation] = []
        self.active_violations: Dict[str, ConstraintViolation] = {}
        
        self.logger.info("Flight envelope initialized")
    
    def check_constraints(self, state: VehicleState, controls: ControlInputs) -> List[str]:
        """
        Check all flight envelope constraints.
        
        Args:
            state: Current vehicle state
            controls: Current control inputs
            
        Returns:
            List of constraint violation messages
        """
        violations = []
        
        # Check speed constraints
        speed_violations = self._check_speed_constraints(state)
        violations.extend(speed_violations)
        
        # Check altitude constraints
        altitude_violations = self._check_altitude_constraints(state)
        violations.extend(altitude_violations)
        
        # Check load factor constraints
        load_factor_violations = self._check_load_factor_constraints(state)
        violations.extend(load_factor_violations)
        
        # Check climb rate constraints
        climb_rate_violations = self._check_climb_rate_constraints(state)
        violations.extend(climb_rate_violations)
        
        # Check turn rate constraints
        turn_rate_violations = self._check_turn_rate_constraints(state)
        violations.extend(turn_rate_violations)
        
        # Check battery constraints
        battery_violations = self._check_battery_constraints(state)
        violations.extend(battery_violations)
        
        # Check payload constraints
        payload_violations = self._check_payload_constraints(state)
        violations.extend(payload_violations)
        
        # Check actuator constraints
        actuator_violations = self._check_actuator_constraints(controls)
        violations.extend(actuator_violations)
        
        return violations
    
    def _check_speed_constraints(self, state: VehicleState) -> List[str]:
        """Check speed-related constraints."""
        violations = []
        
        # Calculate current speed
        speed = np.linalg.norm(state.velocity)
        
        # Check minimum speed
        if speed < self.speed_limits.get('min', 0.0):
            violation = ConstraintViolation(
                constraint_type='speed',
                constraint_name='minimum_speed',
                current_value=speed,
                limit_value=self.speed_limits['min'],
                severity=0.8,
                message=f"Speed below minimum: {speed:.1f} m/s < {self.speed_limits['min']} m/s"
            )
            violations.append(violation.message)
            self._record_violation(violation)
        
        # Check maximum speed
        if speed > self.speed_limits.get('max', 120.0):
            violation = ConstraintViolation(
                constraint_type='speed',
                constraint_name='maximum_speed',
                current_value=speed,
                limit_value=self.speed_limits['max'],
                severity=1.0,
                message=f"Speed exceeds maximum: {speed:.1f} m/s > {self.speed_limits['max']} m/s"
            )
            violations.append(violation.message)
            self._record_violation(violation)
        
        # Check maneuvering speed
        maneuver_speed = self.speed_limits.get('maneuver', 100.0)
        if speed > maneuver_speed:
            violation = ConstraintViolation(
                constraint_type='speed',
                constraint_name='maneuvering_speed',
                current_value=speed,
                limit_value=maneuver_speed,
                severity=0.6,
                message=f"Speed exceeds maneuvering limit: {speed:.1f} m/s > {maneuver_speed} m/s"
            )
            violations.append(violation.message)
            self._record_violation(violation)
        
        return violations
    
    def _check_altitude_constraints(self, state: VehicleState) -> List[str]:
        """Check altitude-related constraints."""
        violations = []
        
        altitude = state.position[2]  # Z coordinate
        
        # Check minimum altitude
        if altitude < self.altitude_limits.get('min', 0.0):
            violation = ConstraintViolation(
                constraint_type='altitude',
                constraint_name='minimum_altitude',
                current_value=altitude,
                limit_value=self.altitude_limits['min'],
                severity=1.0,
                message=f"Altitude below minimum: {altitude:.1f} m < {self.altitude_limits['min']} m"
            )
            violations.append(violation.message)
            self._record_violation(violation)
        
        # Check maximum altitude
        if altitude > self.altitude_limits.get('max', 5000.0):
            violation = ConstraintViolation(
                constraint_type='altitude',
                constraint_name='maximum_altitude',
                current_value=altitude,
                limit_value=self.altitude_limits['max'],
                severity=1.0,
                message=f"Altitude exceeds maximum: {altitude:.1f} m > {self.altitude_limits['max']} m"
            )
            violations.append(violation.message)
            self._record_violation(violation)
        
        # Check operational altitude
        operational_altitude = self.altitude_limits.get('operational', 4000.0)
        if altitude > operational_altitude:
            violation = ConstraintViolation(
                constraint_type='altitude',
                constraint_name='operational_altitude',
                current_value=altitude,
                limit_value=operational_altitude,
                severity=0.7,
                message=f"Altitude exceeds operational limit: {altitude:.1f} m > {operational_altitude} m"
            )
            violations.append(violation.message)
            self._record_violation(violation)
        
        return violations
    
    def _check_load_factor_constraints(self, state: VehicleState) -> List[str]:
        """Check load factor constraints."""
        violations = []
        
        # Calculate load factors (simplified)
        # In a real implementation, this would use accelerometer data
        acceleration = np.linalg.norm(state.velocity)  # Simplified
        load_factor = 1.0 + acceleration / 9.81  # Approximate load factor
        
        # Check positive load factor
        max_positive = self.load_factor_limits.get('max_positive', 3.0)
        if load_factor > max_positive:
            violation = ConstraintViolation(
                constraint_type='load_factor',
                constraint_name='positive_load_factor',
                current_value=load_factor,
                limit_value=max_positive,
                severity=0.9,
                message=f"Positive load factor exceeded: {load_factor:.2f} g > {max_positive} g"
            )
            violations.append(violation.message)
            self._record_violation(violation)
        
        # Check negative load factor
        max_negative = self.load_factor_limits.get('max_negative', -1.5)
        if load_factor < max_negative:
            violation = ConstraintViolation(
                constraint_type='load_factor',
                constraint_name='negative_load_factor',
                current_value=load_factor,
                limit_value=max_negative,
                severity=0.9,
                message=f"Negative load factor exceeded: {load_factor:.2f} g < {max_negative} g"
            )
            violations.append(violation.message)
            self._record_violation(violation)
        
        return violations
    
    def _check_climb_rate_constraints(self, state: VehicleState) -> List[str]:
        """Check climb rate constraints."""
        violations = []
        
        climb_rate = state.velocity[2]  # Vertical velocity
        
        # Check maximum climb rate
        max_climb = self.climb_rate_limits.get('max', 10.0)
        if climb_rate > max_climb:
            violation = ConstraintViolation(
                constraint_type='climb_rate',
                constraint_name='maximum_climb_rate',
                current_value=climb_rate,
                limit_value=max_climb,
                severity=0.7,
                message=f"Climb rate exceeds maximum: {climb_rate:.1f} m/s > {max_climb} m/s"
            )
            violations.append(violation.message)
            self._record_violation(violation)
        
        # Check maximum descent rate
        max_descent = self.climb_rate_limits.get('min', -15.0)
        if climb_rate < max_descent:
            violation = ConstraintViolation(
                constraint_type='climb_rate',
                constraint_name='maximum_descent_rate',
                current_value=climb_rate,
                limit_value=max_descent,
                severity=0.8,
                message=f"Descent rate exceeds maximum: {climb_rate:.1f} m/s < {max_descent} m/s"
            )
            violations.append(violation.message)
            self._record_violation(violation)
        
        return violations
    
    def _check_turn_rate_constraints(self, state: VehicleState) -> List[str]:
        """Check turn rate constraints."""
        violations = []
        
        # Calculate turn rate magnitude
        turn_rate = np.linalg.norm(state.angular_velocity)
        
        # Check maximum turn rate
        max_turn = self.turn_rate_limits.get('max', 0.5)
        if turn_rate > max_turn:
            violation = ConstraintViolation(
                constraint_type='turn_rate',
                constraint_name='maximum_turn_rate',
                current_value=turn_rate,
                limit_value=max_turn,
                severity=0.8,
                message=f"Turn rate exceeds maximum: {turn_rate:.2f} rad/s > {max_turn} rad/s"
            )
            violations.append(violation.message)
            self._record_violation(violation)
        
        return violations
    
    def _check_battery_constraints(self, state: VehicleState) -> List[str]:
        """Check battery-related constraints."""
        violations = []
        
        # Check SOC limits
        if state.battery_soc < self.battery_limits['min_soc']:
            violation = ConstraintViolation(
                constraint_type='battery',
                constraint_name='minimum_soc',
                current_value=state.battery_soc,
                limit_value=self.battery_limits['min_soc'],
                severity=1.0,
                message=f"Battery SOC below minimum: {state.battery_soc:.3f} < {self.battery_limits['min_soc']}"
            )
            violations.append(violation.message)
            self._record_violation(violation)
        
        if state.battery_soc > self.battery_limits['max_soc']:
            violation = ConstraintViolation(
                constraint_type='battery',
                constraint_name='maximum_soc',
                current_value=state.battery_soc,
                limit_value=self.battery_limits['max_soc'],
                severity=0.8,
                message=f"Battery SOC exceeds maximum: {state.battery_soc:.3f} > {self.battery_limits['max_soc']}"
            )
            violations.append(violation.message)
            self._record_violation(violation)
        
        # Check temperature limits
        if state.battery_temperature < self.battery_limits['min_temperature']:
            violation = ConstraintViolation(
                constraint_type='battery',
                constraint_name='minimum_temperature',
                current_value=state.battery_temperature,
                limit_value=self.battery_limits['min_temperature'],
                severity=0.9,
                message=f"Battery temperature below minimum: {state.battery_temperature:.1f}°C < {self.battery_limits['min_temperature']}°C"
            )
            violations.append(violation.message)
            self._record_violation(violation)
        
        if state.battery_temperature > self.battery_limits['max_temperature']:
            violation = ConstraintViolation(
                constraint_type='battery',
                constraint_name='maximum_temperature',
                current_value=state.battery_temperature,
                limit_value=self.battery_limits['max_temperature'],
                severity=1.0,
                message=f"Battery temperature exceeds maximum: {state.battery_temperature:.1f}°C > {self.battery_limits['max_temperature']}°C"
            )
            violations.append(violation.message)
            self._record_violation(violation)
        
        return violations
    
    def _check_payload_constraints(self, state: VehicleState) -> List[str]:
        """Check payload-related constraints."""
        violations = []
        
        # This would typically check against mission payload
        # For now, we'll use a simplified check
        total_mass = self.config.vehicle.mass.total
        
        if total_mass > self.config.vehicle.mass.total:
            violation = ConstraintViolation(
                constraint_type='payload',
                constraint_name='maximum_payload',
                current_value=total_mass,
                limit_value=self.config.vehicle.mass.total,
                severity=0.8,
                message=f"Total mass exceeds limit: {total_mass:.1f} kg > {self.config.vehicle.mass.total} kg"
            )
            violations.append(violation.message)
            self._record_violation(violation)
        
        return violations
    
    def _check_actuator_constraints(self, controls: ControlInputs) -> List[str]:
        """Check actuator-related constraints."""
        violations = []
        
        # Check rotor RPM limits
        for i, rpm in enumerate(controls.main_rotor_rpm):
            if rpm < 800 or rpm > 6000:  # From config
                violation = ConstraintViolation(
                    constraint_type='actuator',
                    constraint_name=f'main_rotor_{i+1}_rpm',
                    current_value=rpm,
                    limit_value=6000 if rpm > 6000 else 800,
                    severity=0.7,
                    message=f"Main rotor {i+1} RPM out of range: {rpm:.0f} RPM"
                )
                violations.append(violation.message)
                self._record_violation(violation)
        
        # Check control surface deflections
        max_elevator = 0.35  # From config
        if abs(controls.elevator_deflection) > max_elevator:
            violation = ConstraintViolation(
                constraint_type='actuator',
                constraint_name='elevator_deflection',
                current_value=controls.elevator_deflection,
                limit_value=max_elevator,
                severity=0.6,
                message=f"Elevator deflection exceeds limit: {controls.elevator_deflection:.3f} rad"
            )
            violations.append(violation.message)
            self._record_violation(violation)
        
        return violations
    
    def _record_violation(self, violation: ConstraintViolation) -> None:
        """Record a constraint violation."""
        self.violation_history.append(violation)
        self.active_violations[violation.constraint_name] = violation
        
        # Log violation
        if violation.severity >= 0.8:
            self.logger.error(violation.message)
        elif violation.severity >= 0.5:
            self.logger.warning(violation.message)
        else:
            self.logger.info(violation.message)
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """
        Get summary of constraint violations.
        
        Returns:
            Dictionary with violation statistics
        """
        if not self.violation_history:
            return {'total_violations': 0, 'critical_violations': 0}
        
        total_violations = len(self.violation_history)
        critical_violations = sum(1 for v in self.violation_history if v.severity >= 0.8)
        
        # Group by constraint type
        by_type = {}
        for violation in self.violation_history:
            if violation.constraint_type not in by_type:
                by_type[violation.constraint_type] = 0
            by_type[violation.constraint_type] += 1
        
        return {
            'total_violations': total_violations,
            'critical_violations': critical_violations,
            'active_violations': len(self.active_violations),
            'violations_by_type': by_type,
            'recent_violations': [v.message for v in self.violation_history[-5:]]
        }
    
    def clear_violations(self) -> None:
        """Clear violation history."""
        self.violation_history.clear()
        self.active_violations.clear()
        self.logger.info("Constraint violation history cleared")
    
    def is_operational(self) -> bool:
        """
        Check if vehicle is within operational limits.
        
        Returns:
            True if vehicle is operational
        """
        critical_violations = sum(1 for v in self.active_violations.values() if v.severity >= 0.8)
        return critical_violations == 0
    
    def get_operational_margins(self, state: VehicleState) -> Dict[str, float]:
        """
        Get operational margins for current state.
        
        Args:
            state: Current vehicle state
            
        Returns:
            Dictionary of operational margins
        """
        margins = {}
        
        # Speed margin
        speed = np.linalg.norm(state.velocity)
        max_speed = self.speed_limits.get('max', 120.0)
        margins['speed_margin'] = (max_speed - speed) / max_speed
        
        # Altitude margin
        altitude = state.position[2]
        max_altitude = self.altitude_limits.get('max', 5000.0)
        margins['altitude_margin'] = (max_altitude - altitude) / max_altitude
        
        # SOC margin
        min_soc = self.battery_limits['min_soc']
        margins['soc_margin'] = (state.battery_soc - min_soc) / (1.0 - min_soc)
        
        # Temperature margin
        max_temp = self.battery_limits['max_temperature']
        margins['temperature_margin'] = (max_temp - state.battery_temperature) / max_temp
        
        return margins
