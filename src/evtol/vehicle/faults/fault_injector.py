"""
Fault Injector - Advanced Fault Injection and Modeling

This module provides comprehensive fault injection capabilities for reliability
analysis, including actuator faults, sensor faults, and system-level failures.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
import random
from datetime import datetime, timedelta

try:
    from ..utils.config import VehicleConfig
    from ..vehicle_types import ControlInputs
except ImportError:
    # Fallback for direct imports
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.config import VehicleConfig
    from vehicle_types import ControlInputs


class FaultType(Enum):
    """Types of faults that can be injected."""
    STUCK = "stuck"          # Actuator stuck at fixed value
    DRIFT = "drift"          # Gradual parameter drift
    NOISE = "noise"          # Electrical noise injection
    FAILURE = "failure"      # Complete actuator failure
    INTERMITTENT = "intermittent"  # Intermittent operation
    BIAS = "bias"            # Constant bias error
    SCALE = "scale"          # Scaling factor error


class FaultSeverity(Enum):
    """Fault severity levels."""
    LOW = 0.1      # 10% effect
    MEDIUM = 0.5   # 50% effect
    HIGH = 0.8     # 80% effect
    CRITICAL = 1.0 # 100% effect


@dataclass
class FaultDefinition:
    """Definition of a fault to be injected."""
    fault_id: str
    actuator_id: str
    fault_type: FaultType
    severity: float  # 0-1
    start_time: float  # seconds
    end_time: Optional[float]  # seconds, None for permanent
    parameters: Dict[str, Any]  # Fault-specific parameters


@dataclass
class ActiveFault:
    """Currently active fault."""
    definition: FaultDefinition
    start_time: float
    is_active: bool
    current_effect: float  # Current fault effect (0-1)


class FaultInjector:
    """
    Advanced fault injection system for reliability analysis and testing.
    
    This class provides comprehensive fault injection capabilities including:
    - Actuator faults (stuck, drift, noise, failure)
    - Sensor faults (bias, noise, scaling)
    - System-level faults (communication, power)
    - Time-based fault scheduling
    - Fault propagation modeling
    """
    
    def __init__(self, config: VehicleConfig):
        """
        Initialize fault injector.
        
        Args:
            config: Vehicle configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Fault management
        self.fault_definitions: List[FaultDefinition] = []
        self.active_faults: Dict[str, ActiveFault] = {}
        self.fault_history: List[ActiveFault] = []
        
        # Fault injection settings - use defaults if config doesn't have faults
        if hasattr(config, 'faults'):
            self.faults_enabled = config.faults.enabled
            self.fault_types = [FaultType(t) for t in config.faults.types]
            self.injection_times = config.faults.injection_times
            self.severity_range = config.faults.severity_range
        else:
            # Default: faults disabled
            self.faults_enabled = False
            self.fault_types = []
            self.injection_times = []
            self.severity_range = (0.0, 0.0)
        
        # Random fault injection
        self.random_faults_enabled = False
        self.fault_probability = 0.001  # 0.1% per time step
        
        # Fault statistics
        self.fault_statistics = {
            'total_injected': 0,
            'total_active': 0,
            'total_recovered': 0,
            'critical_faults': 0
        }
        
        self.logger.info("Fault injector initialized")
    
    def inject_fault(self, actuator_id: str, fault_type: str, severity: float, 
                    start_time: float, end_time: Optional[float] = None,
                    parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Inject a fault into the specified actuator.
        
        Args:
            actuator_id: ID of the actuator
            fault_type: Type of fault to inject
            severity: Fault severity (0-1)
            start_time: Time to start the fault
            end_time: Time to end the fault (None for permanent)
            parameters: Fault-specific parameters
            
        Returns:
            Fault ID
        """
        if not self.faults_enabled:
            self.logger.warning("Fault injection is disabled")
            return ""
        
        # Create fault definition
        fault_id = f"{actuator_id}_{fault_type}_{int(start_time)}"
        
        fault_def = FaultDefinition(
            fault_id=fault_id,
            actuator_id=actuator_id,
            fault_type=FaultType(fault_type),
            severity=severity,
            start_time=start_time,
            end_time=end_time,
            parameters=parameters or {}
        )
        
        # Add to fault definitions
        self.fault_definitions.append(fault_def)
        
        self.logger.info(f"Fault scheduled: {fault_id}, type={fault_type}, "
                        f"severity={severity:.2f}, start_time={start_time:.1f}s")
        
        return fault_id
    
    def apply_faults(self, controls: ControlInputs, current_time: float) -> ControlInputs:
        """
        Apply active faults to control inputs.
        
        Args:
            controls: Original control inputs
            current_time: Current simulation time
            
        Returns:
            Modified control inputs with faults applied
        """
        if not self.faults_enabled:
            return controls
        
        # Update active faults
        self._update_active_faults(current_time)
        
        # Apply random faults
        if self.random_faults_enabled:
            self._apply_random_faults(current_time)
        
        # Create modified controls
        modified_controls = self._create_modified_controls(controls)
        
        # Apply each active fault
        for fault in self.active_faults.values():
            if fault.is_active:
                modified_controls = self._apply_single_fault(modified_controls, fault)
        
        return modified_controls
    
    def _update_active_faults(self, current_time: float) -> None:
        """Update the list of active faults based on current time."""
        # Check for new faults to activate
        for fault_def in self.fault_definitions:
            if (fault_def.start_time <= current_time and 
                fault_def.fault_id not in self.active_faults):
                
                active_fault = ActiveFault(
                    definition=fault_def,
                    start_time=current_time,
                    is_active=True,
                    current_effect=fault_def.severity
                )
                
                self.active_faults[fault_def.fault_id] = active_fault
                self.fault_statistics['total_injected'] += 1
                
                if fault_def.severity >= 0.8:
                    self.fault_statistics['critical_faults'] += 1
                
                self.logger.info(f"Fault activated: {fault_def.fault_id}")
        
        # Check for faults to deactivate
        faults_to_remove = []
        for fault_id, active_fault in self.active_faults.items():
            if (active_fault.definition.end_time is not None and 
                current_time >= active_fault.definition.end_time):
                
                active_fault.is_active = False
                self.fault_history.append(active_fault)
                faults_to_remove.append(fault_id)
                self.fault_statistics['total_recovered'] += 1
                
                self.logger.info(f"Fault deactivated: {fault_id}")
        
        # Remove deactivated faults
        for fault_id in faults_to_remove:
            del self.active_faults[fault_id]
        
        self.fault_statistics['total_active'] = len(self.active_faults)
    
    def _apply_random_faults(self, current_time: float) -> None:
        """Apply random fault injection."""
        if random.random() < self.fault_probability:
            # Select random actuator and fault type
            actuators = ['main_rotor_1', 'main_rotor_2', 'main_rotor_3', 'main_rotor_4',
                        'tail_rotor', 'elevator', 'aileron', 'rudder']
            
            actuator_id = random.choice(actuators)
            fault_type = random.choice(self.fault_types)
            severity = random.uniform(*self.severity_range)
            
            # Inject random fault
            self.inject_fault(actuator_id, fault_type.value, severity, current_time)
    
    def _create_modified_controls(self, controls: ControlInputs) -> ControlInputs:
        """Create a copy of controls for modification."""
        return ControlInputs(
            main_rotor_rpm=controls.main_rotor_rpm.copy(),
            tail_rotor_rpm=controls.tail_rotor_rpm,
            lift_fan_rpm=controls.lift_fan_rpm.copy(),
            propeller_rpm=controls.propeller_rpm.copy(),
            elevator_deflection=controls.elevator_deflection,
            aileron_deflection=controls.aileron_deflection,
            rudder_deflection=controls.rudder_deflection,
            throttle=controls.throttle,
            collective=controls.collective
        )
    
    def _apply_single_fault(self, controls: ControlInputs, fault: ActiveFault) -> ControlInputs:
        """
        Apply a single fault to the controls.
        
        Args:
            controls: Control inputs to modify
            fault: Active fault to apply
            
        Returns:
            Modified control inputs
        """
        fault_type = fault.definition.fault_type
        severity = fault.current_effect
        actuator_id = fault.definition.actuator_id
        
        if fault_type == FaultType.STUCK:
            controls = self._apply_stuck_fault(controls, actuator_id, severity)
        elif fault_type == FaultType.DRIFT:
            controls = self._apply_drift_fault(controls, actuator_id, severity, fault)
        elif fault_type == FaultType.NOISE:
            controls = self._apply_noise_fault(controls, actuator_id, severity)
        elif fault_type == FaultType.FAILURE:
            controls = self._apply_failure_fault(controls, actuator_id, severity)
        elif fault_type == FaultType.INTERMITTENT:
            controls = self._apply_intermittent_fault(controls, actuator_id, severity)
        elif fault_type == FaultType.BIAS:
            controls = self._apply_bias_fault(controls, actuator_id, severity)
        elif fault_type == FaultType.SCALE:
            controls = self._apply_scale_fault(controls, actuator_id, severity)
        
        return controls
    
    def _apply_stuck_fault(self, controls: ControlInputs, actuator_id: str, severity: float) -> ControlInputs:
        """Apply stuck fault to actuator."""
        if actuator_id.startswith('main_rotor'):
            rotor_idx = int(actuator_id.split('_')[-1]) - 1
            if 0 <= rotor_idx < len(controls.main_rotor_rpm):
                # Stuck at current value with severity-based reduction
                stuck_value = controls.main_rotor_rpm[rotor_idx] * (1.0 - severity * 0.5)
                controls.main_rotor_rpm[rotor_idx] = stuck_value
        
        elif actuator_id == 'tail_rotor':
            controls.tail_rotor_rpm *= (1.0 - severity * 0.5)
        
        elif actuator_id == 'elevator':
            controls.elevator_deflection *= (1.0 - severity * 0.5)
        
        elif actuator_id == 'aileron':
            controls.aileron_deflection *= (1.0 - severity * 0.5)
        
        elif actuator_id == 'rudder':
            controls.rudder_deflection *= (1.0 - severity * 0.5)
        
        return controls
    
    def _apply_drift_fault(self, controls: ControlInputs, actuator_id: str, 
                          severity: float, fault: ActiveFault) -> ControlInputs:
        """Apply drift fault to actuator."""
        # Calculate drift based on time elapsed
        time_elapsed = fault.start_time - fault.definition.start_time
        drift_rate = severity * 0.1  # 10% per second at full severity
        drift_amount = time_elapsed * drift_rate
        
        if actuator_id.startswith('main_rotor'):
            rotor_idx = int(actuator_id.split('_')[-1]) - 1
            if 0 <= rotor_idx < len(controls.main_rotor_rpm):
                controls.main_rotor_rpm[rotor_idx] *= (1.0 - drift_amount)
        
        elif actuator_id == 'tail_rotor':
            controls.tail_rotor_rpm *= (1.0 - drift_amount)
        
        return controls
    
    def _apply_noise_fault(self, controls: ControlInputs, actuator_id: str, severity: float) -> ControlInputs:
        """Apply noise fault to actuator."""
        noise_amplitude = severity * 0.1  # 10% noise at full severity
        
        if actuator_id.startswith('main_rotor'):
            rotor_idx = int(actuator_id.split('_')[-1]) - 1
            if 0 <= rotor_idx < len(controls.main_rotor_rpm):
                noise = np.random.normal(0, noise_amplitude * controls.main_rotor_rpm[rotor_idx])
                controls.main_rotor_rpm[rotor_idx] += noise
        
        elif actuator_id == 'tail_rotor':
            noise = np.random.normal(0, noise_amplitude * controls.tail_rotor_rpm)
            controls.tail_rotor_rpm += noise
        
        return controls
    
    def _apply_failure_fault(self, controls: ControlInputs, actuator_id: str, severity: float) -> ControlInputs:
        """Apply failure fault to actuator."""
        failure_factor = 1.0 - severity  # Complete failure at severity=1.0
        
        if actuator_id.startswith('main_rotor'):
            rotor_idx = int(actuator_id.split('_')[-1]) - 1
            if 0 <= rotor_idx < len(controls.main_rotor_rpm):
                controls.main_rotor_rpm[rotor_idx] *= failure_factor
        
        elif actuator_id == 'tail_rotor':
            controls.tail_rotor_rpm *= failure_factor
        
        elif actuator_id == 'elevator':
            controls.elevator_deflection *= failure_factor
        
        elif actuator_id == 'aileron':
            controls.aileron_deflection *= failure_factor
        
        elif actuator_id == 'rudder':
            controls.rudder_deflection *= failure_factor
        
        return controls
    
    def _apply_intermittent_fault(self, controls: ControlInputs, actuator_id: str, severity: float) -> ControlInputs:
        """Apply intermittent fault to actuator."""
        # Intermittent operation based on time
        time_factor = np.sin(controls.main_rotor_rpm[0] * 0.1)  # Simple time-based function
        intermittent_factor = 1.0 - severity * (1.0 - abs(time_factor))
        
        if actuator_id.startswith('main_rotor'):
            rotor_idx = int(actuator_id.split('_')[-1]) - 1
            if 0 <= rotor_idx < len(controls.main_rotor_rpm):
                controls.main_rotor_rpm[rotor_idx] *= intermittent_factor
        
        return controls
    
    def _apply_bias_fault(self, controls: ControlInputs, actuator_id: str, severity: float) -> ControlInputs:
        """Apply bias fault to actuator."""
        bias = severity * 0.2  # 20% bias at full severity
        
        if actuator_id.startswith('main_rotor'):
            rotor_idx = int(actuator_id.split('_')[-1]) - 1
            if 0 <= rotor_idx < len(controls.main_rotor_rpm):
                controls.main_rotor_rpm[rotor_idx] += bias * 1000  # 1000 RPM bias
        
        elif actuator_id == 'elevator':
            controls.elevator_deflection += bias * 0.1  # 0.1 rad bias
        
        return controls
    
    def _apply_scale_fault(self, controls: ControlInputs, actuator_id: str, severity: float) -> ControlInputs:
        """Apply scale fault to actuator."""
        scale_factor = 1.0 - severity * 0.3  # 30% scaling error at full severity
        
        if actuator_id.startswith('main_rotor'):
            rotor_idx = int(actuator_id.split('_')[-1]) - 1
            if 0 <= rotor_idx < len(controls.main_rotor_rpm):
                controls.main_rotor_rpm[rotor_idx] *= scale_factor
        
        return controls
    
    def get_active_faults(self) -> Dict[str, Any]:
        """
        Get currently active faults.
        
        Returns:
            Dictionary of active faults
        """
        active_faults = {}
        for fault_id, fault in self.active_faults.items():
            if fault.is_active:
                active_faults[fault_id] = {
                    'actuator_id': fault.definition.actuator_id,
                    'fault_type': fault.definition.fault_type.value,
                    'severity': fault.current_effect,
                    'start_time': fault.start_time,
                    'duration': fault.start_time - fault.definition.start_time
                }
        
        return active_faults
    
    def get_fault_statistics(self) -> Dict[str, Any]:
        """
        Get fault injection statistics.
        
        Returns:
            Dictionary of fault statistics
        """
        return {
            'total_injected': self.fault_statistics['total_injected'],
            'total_active': self.fault_statistics['total_active'],
            'total_recovered': self.fault_statistics['total_recovered'],
            'critical_faults': self.fault_statistics['critical_faults'],
            'faults_enabled': self.faults_enabled,
            'random_faults_enabled': self.random_faults_enabled,
            'fault_probability': self.fault_probability
        }
    
    def clear_faults(self) -> None:
        """Clear all fault definitions and active faults."""
        self.fault_definitions.clear()
        self.active_faults.clear()
        self.fault_history.clear()
        self.fault_statistics = {
            'total_injected': 0,
            'total_active': 0,
            'total_recovered': 0,
            'critical_faults': 0
        }
        self.logger.info("All faults cleared")
    
    def enable_random_faults(self, probability: float = 0.001) -> None:
        """
        Enable random fault injection.
        
        Args:
            probability: Probability of fault injection per time step
        """
        self.random_faults_enabled = True
        self.fault_probability = probability
        self.logger.info(f"Random fault injection enabled with probability {probability}")
    
    def disable_random_faults(self) -> None:
        """Disable random fault injection."""
        self.random_faults_enabled = False
        self.logger.info("Random fault injection disabled")
    
    def get_fault_history(self) -> List[Dict[str, Any]]:
        """
        Get fault injection history.
        
        Returns:
            List of fault history entries
        """
        history = []
        for fault in self.fault_history:
            history.append({
                'fault_id': fault.definition.fault_id,
                'actuator_id': fault.definition.actuator_id,
                'fault_type': fault.definition.fault_type.value,
                'severity': fault.definition.severity,
                'start_time': fault.definition.start_time,
                'end_time': fault.definition.end_time,
                'duration': fault.start_time - fault.definition.start_time
            })
        
        return history

