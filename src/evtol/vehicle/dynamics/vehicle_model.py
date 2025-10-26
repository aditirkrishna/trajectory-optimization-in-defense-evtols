"""
Main Vehicle Model - 6-DoF Dynamics Integration

This module provides the main VehicleModel class that integrates all vehicle
dynamics components including 6-DoF motion, energy management, actuator models,
and constraint checking.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

try:
    from ..vehicle_types import VehicleState, ControlInputs
    from ..energy.battery_model import BatteryModel
    from ..actuators.motor_model import MotorModel
    from ..faults.fault_injector import FaultInjector
    from ..integration.rk4_integrator import RK4Integrator
    from ..utils.config import VehicleConfig
except ImportError:
    # Fallback for direct imports
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from vehicle_types import VehicleState, ControlInputs
    from energy.battery_model import BatteryModel
    from actuators.motor_model import MotorModel
    from faults.fault_injector import FaultInjector
    from integration.rk4_integrator import RK4Integrator
    from utils.config import VehicleConfig




class VehicleModel:
    """
    Main vehicle dynamics model integrating 6-DoF motion, energy management,
    actuator models, and constraint checking.
    """
    
    def __init__(self, config: VehicleConfig):
        """
        Initialize the vehicle model with configuration.
        
        Args:
            config: Vehicle configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize component models
        self.battery = BatteryModel(config)
        self.faults = FaultInjector(config)
        self.integrator = RK4Integrator()
        
        # Vehicle parameters
        self.mass = config.get_vehicle_property('mass')
        self.cg = np.array([config.get_vehicle_property('cg_x'),
                           config.get_vehicle_property('cg_y'),
                           config.get_vehicle_property('cg_z')])
        self.inertia_tensor = np.diag([config.get_vehicle_property('Ixx'),
                                      config.get_vehicle_property('Iyy'),
                                      config.get_vehicle_property('Izz')])
        
        # Simulation state
        self.current_state: Optional[VehicleState] = None
        self.simulation_time: float = 0.0
        self.energy_consumed: float = 0.0
        self.fault_status: Dict[str, Any] = {}
        
        # Performance tracking
        self.step_count: int = 0
        self.constraint_violations: List[str] = []
        
        self.logger.info("Vehicle model initialized successfully")
    
    def set_initial_state(self, state: VehicleState) -> None:
        """
        Set the initial vehicle state.
        
        Args:
            state: Initial vehicle state
        """
        self.current_state = state
        self.simulation_time = state.time
        self.energy_consumed = 0.0
        self.step_count = 0
        self.constraint_violations.clear()
        
        # Initialize component states
        self.battery.soc = state.battery_soc
        self.battery.temperature = state.battery_temperature
        
        self.logger.info(f"Initial state set: position={state.position}, "
                        f"velocity={state.velocity}, soc={state.battery_soc:.3f}")
    
    def step(self, controls: ControlInputs, dt: float) -> VehicleState:
        """
        Perform one simulation time step.
        
        Args:
            controls: Control inputs for this time step
            dt: Time step size in seconds
            
        Returns:
            Updated vehicle state
        """
        if self.current_state is None:
            raise ValueError("Initial state not set. Call set_initial_state() first.")
        
        # Apply fault injection
        controls = self.faults.apply_faults(controls, self.simulation_time)
        
        # Calculate forces and moments (simplified)
        forces, moments = self._calculate_forces_and_moments(
            self.current_state, controls, dt
        )
        
        # Calculate power consumption (simplified)
        power_consumed = self._calculate_power_consumption(controls)
        
        # Update battery state
        # Note: BatteryModel.update_state() only needs power_demand and dt
        # It internally manages SOC and temperature
        self.battery.update_state(power_consumed, dt)
        
        # Integrate dynamics (simplified)
        new_state = self._integrate_dynamics(
            self.current_state, forces, moments, controls, dt
        )
        
        # Update simulation state
        self.current_state = new_state
        self.simulation_time += dt
        self.energy_consumed += power_consumed * dt
        self.step_count += 1
        
        return new_state
    
    def simulate(self, initial_state: VehicleState, controls: ControlInputs, 
                 dt: float, duration: float) -> List[VehicleState]:
        """
        Simulate vehicle dynamics over a specified duration.
        
        Args:
            initial_state: Starting vehicle state
            controls: Control inputs (constant or time-varying)
            dt: Time step size in seconds
            duration: Total simulation duration in seconds
            
        Returns:
            List of vehicle states over time
        """
        self.set_initial_state(initial_state)
        
        trajectory = [self.current_state]
        num_steps = int(duration / dt)
        
        self.logger.info(f"Starting simulation: {num_steps} steps, "
                        f"dt={dt:.3f}s, duration={duration:.1f}s")
        
        for step in range(num_steps):
            # Handle time-varying controls if needed
            if callable(controls):
                current_controls = controls(self.simulation_time)
            else:
                current_controls = controls
            
            # Perform one time step
            new_state = self.step(current_controls, dt)
            trajectory.append(new_state)
            
            # Check for simulation termination conditions
            if self._should_terminate_simulation():
                self.logger.warning("Simulation terminated early due to constraints")
                break
        
        self.logger.info(f"Simulation completed: {len(trajectory)} states, "
                        f"final time={self.simulation_time:.1f}s")
        
        return trajectory
    
    def _calculate_state_derivative(self, state: VehicleState, 
                                  forces: np.ndarray, moments: np.ndarray,
                                  controls: ControlInputs) -> Dict[str, np.ndarray]:
        """
        Calculate the time derivative of the vehicle state.
        
        Args:
            state: Current vehicle state
            forces: Total forces in body frame
            moments: Total moments in body frame
            controls: Control inputs
            
        Returns:
            Dictionary of state derivatives
        """
        # Translational dynamics: m * dv/dt = F + m * g
        mass = self.config.vehicle.mass.total
        gravity_body = self.kinematics.transform_gravity_to_body(state.attitude)
        acceleration = (forces + mass * gravity_body) / mass
        
        # Rotational dynamics: I * dω/dt = M - ω × (I * ω)
        inertia = self.config.vehicle.inertia
        I = np.array([
            [inertia.Ixx, inertia.Ixy, inertia.Ixz],
            [inertia.Ixy, inertia.Iyy, inertia.Iyz],
            [inertia.Ixz, inertia.Iyz, inertia.Izz]
        ])
        
        omega = state.angular_velocity
        angular_acceleration = np.linalg.solve(
            I, moments - np.cross(omega, I @ omega)
        )
        
        # Kinematic equations
        velocity_derivative = acceleration
        position_derivative = self.kinematics.transform_velocity_to_inertial(
            state.velocity, state.attitude
        )
        attitude_derivative = self.kinematics.calculate_attitude_rate(
            state.attitude, state.angular_velocity
        )
        
        # Battery dynamics
        soc_derivative = self.battery.get_soc_derivative()
        temp_derivative = self.battery.get_temperature_derivative()
        
        return {
            'position': position_derivative,
            'velocity': velocity_derivative,
            'attitude': attitude_derivative,
            'angular_velocity': angular_acceleration,
            'battery_soc': soc_derivative,
            'battery_temperature': temp_derivative
        }
    
    def _should_terminate_simulation(self) -> bool:
        """
        Check if simulation should be terminated early.
        
        Returns:
            True if simulation should terminate
        """
        # Check battery state
        if self.current_state.battery_soc < 0.05:  # 5% SOC
            self.logger.warning("Battery critically low - terminating simulation")
            return True
        
        # Check altitude limits
        if self.current_state.position[2] < -10.0:  # 10m below ground
            self.logger.warning("Vehicle below ground - terminating simulation")
            return True
        
        # Check excessive constraint violations
        if len(self.constraint_violations) > 100:
            self.logger.warning("Excessive constraint violations - terminating simulation")
            return True
        
        return False
    
    def get_current_state(self) -> VehicleState:
        """Get the current vehicle state."""
        return self.current_state
    
    def get_energy_consumption(self) -> float:
        """Get total energy consumed during simulation."""
        return self.energy_consumed
    
    def get_battery_state_of_charge(self) -> float:
        """Get current battery state of charge."""
        return self.current_state.battery_soc if self.current_state else 0.0
    
    def get_battery_temperature(self) -> float:
        """Get current battery temperature."""
        return self.current_state.battery_temperature if self.current_state else 20.0
    
    def get_power_consumption(self) -> float:
        """Get current power consumption."""
        return self._calculate_power_consumption(ControlInputs(
            main_rotor_rpm=np.array([1000, 1000, 1000, 1000]),
            tail_rotor_rpm=1200,
            lift_fan_rpm=np.array([800, 800]),
            propeller_rpm=np.array([0, 0]),
            elevator_deflection=0.0,
            aileron_deflection=0.0,
            rudder_deflection=0.0,
            throttle=0.7,
            collective=0.5
        ))
    
    def get_fault_status(self) -> Dict[str, Any]:
        """Get current fault status."""
        return self.faults.get_active_faults()
    
    def inject_fault(self, actuator_id: str, fault_type: str, 
                    severity: float, start_time: float) -> None:
        """
        Inject a fault into the specified actuator.
        
        Args:
            actuator_id: ID of the actuator
            fault_type: Type of fault to inject
            severity: Fault severity (0-1)
            start_time: Time to start the fault
        """
        self.faults.inject_fault(actuator_id, fault_type, severity, start_time)
        self.logger.info(f"Fault injected: {actuator_id}, {fault_type}, "
                        f"severity={severity:.2f}, start_time={start_time:.1f}s")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get simulation performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        return {
            'simulation_time': self.simulation_time,
            'step_count': self.step_count,
            'energy_consumed': self.energy_consumed,
            'constraint_violations': len(self.constraint_violations),
            'battery_soc': self.get_battery_state_of_charge(),
            'battery_temperature': self.get_battery_temperature(),
            'faults_active': len(self.get_fault_status())
        }
    
    def _calculate_forces_and_moments(self, state: VehicleState, 
                                    controls: ControlInputs, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate forces and moments acting on the vehicle."""
        # Gravity force
        g = 9.81  # m/s^2
        gravity_force = np.array([0, 0, -self.mass * g])
        
        # Rotor forces (simplified)
        rotor_forces = np.zeros(3)
        for i, rpm in enumerate(controls.main_rotor_rpm):
            # Simplified thrust calculation
            thrust = (rpm / 1000.0) ** 2 * 1000  # N
            rotor_forces[2] += thrust  # Upward thrust
        
        # Total forces
        total_forces = gravity_force + rotor_forces
        
        # Moments (simplified)
        total_moments = np.zeros(3)
        
        return total_forces, total_moments
    
    def _calculate_power_consumption(self, controls: ControlInputs) -> float:
        """Calculate total power consumption."""
        # Simplified power calculation
        main_power = np.sum(controls.main_rotor_rpm) * 10  # 10W per RPM
        tail_power = controls.tail_rotor_rpm * 5
        prop_power = np.sum(controls.propeller_rpm) * 8
        aux_power = 5000  # Auxiliary systems
        
        return main_power + tail_power + prop_power + aux_power
    
    def _integrate_dynamics(self, state: VehicleState, forces: np.ndarray, 
                          moments: np.ndarray, controls: ControlInputs, dt: float) -> VehicleState:
        """Integrate vehicle dynamics."""
        # Translational dynamics
        acceleration = forces / self.mass
        new_velocity = state.velocity + acceleration * dt
        new_position = state.position + new_velocity * dt
        
        # Rotational dynamics (simplified)
        new_angular_velocity = state.angular_velocity  # Simplified
        new_attitude = state.attitude  # Simplified
        
        # Update battery state
        new_battery_soc = self.battery.soc
        new_battery_temperature = self.battery.temperature
        
        # Update rotor RPM (simplified)
        new_rotor_rpm = controls.main_rotor_rpm.copy()
        
        return VehicleState(
            position=new_position,
            velocity=new_velocity,
            attitude=new_attitude,
            angular_velocity=new_angular_velocity,
            battery_soc=new_battery_soc,
            battery_temperature=new_battery_temperature,
            battery_voltage=state.battery_voltage,
            rotor_rpm=new_rotor_rpm,
            control_surface_deflections=state.control_surface_deflections,
            time=state.time + dt
        )
