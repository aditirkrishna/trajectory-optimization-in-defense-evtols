"""
Test Suite for Vehicle Model

This module contains comprehensive tests for the main vehicle model including
dynamics integration, energy management, and constraint checking.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from vehicle_layer import VehicleModel, VehicleConfig, VehicleState, ControlInputs


class TestVehicleModel:
    """Test cases for VehicleModel class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        config = VehicleConfig()
        return config
    
    @pytest.fixture
    def vehicle_model(self, config):
        """Create a vehicle model instance."""
        return VehicleModel(config)
    
    @pytest.fixture
    def initial_state(self):
        """Create initial vehicle state."""
        return VehicleState(
            position=np.array([0.0, 0.0, 100.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            attitude=np.array([0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            battery_soc=0.8,
            battery_temperature=20.0,
            battery_voltage=400.0,
            rotor_rpm=np.array([1000, 1000, 1000, 1000]),
            control_surface_deflections=np.array([0.0, 0.0, 0.0]),
            time=0.0
        )
    
    @pytest.fixture
    def control_inputs(self):
        """Create control inputs."""
        return ControlInputs(
            main_rotor_rpm=np.array([1000, 1000, 1000, 1000]),
            tail_rotor_rpm=1200,
            lift_fan_rpm=np.array([800, 800]),
            propeller_rpm=np.array([0, 0]),
            elevator_deflection=0.0,
            aileron_deflection=0.0,
            rudder_deflection=0.0,
            throttle=0.7,
            collective=0.5
        )
    
    def test_vehicle_model_initialization(self, vehicle_model):
        """Test vehicle model initialization."""
        assert vehicle_model is not None
        assert vehicle_model.config is not None
        assert vehicle_model.current_state is None
        assert vehicle_model.simulation_time == 0.0
        assert vehicle_model.energy_consumed == 0.0
    
    def test_set_initial_state(self, vehicle_model, initial_state):
        """Test setting initial state."""
        vehicle_model.set_initial_state(initial_state)
        
        assert vehicle_model.current_state is not None
        assert vehicle_model.simulation_time == 0.0
        assert vehicle_model.energy_consumed == 0.0
        assert vehicle_model.step_count == 0
    
    def test_step_simulation(self, vehicle_model, initial_state, control_inputs):
        """Test single time step simulation."""
        vehicle_model.set_initial_state(initial_state)
        
        dt = 0.01
        new_state = vehicle_model.step(control_inputs, dt)
        
        assert new_state is not None
        assert new_state.time == dt
        assert vehicle_model.simulation_time == dt
        assert vehicle_model.step_count == 1
    
    def test_simulation_trajectory(self, vehicle_model, initial_state, control_inputs):
        """Test trajectory simulation."""
        vehicle_model.set_initial_state(initial_state)
        
        dt = 0.01
        duration = 1.0
        trajectory = vehicle_model.simulate(initial_state, control_inputs, dt, duration)
        
        assert len(trajectory) > 0
        assert trajectory[0].time == 0.0
        assert trajectory[-1].time == duration
        assert len(trajectory) == int(duration / dt) + 1
    
    def test_energy_consumption_tracking(self, vehicle_model, initial_state, control_inputs):
        """Test energy consumption tracking."""
        vehicle_model.set_initial_state(initial_state)
        
        initial_energy = vehicle_model.get_energy_consumption()
        assert initial_energy == 0.0
        
        # Run simulation
        dt = 0.01
        duration = 10.0
        trajectory = vehicle_model.simulate(initial_state, control_inputs, dt, duration)
        
        final_energy = vehicle_model.get_energy_consumption()
        assert final_energy > 0.0
    
    def test_battery_state_tracking(self, vehicle_model, initial_state, control_inputs):
        """Test battery state tracking."""
        vehicle_model.set_initial_state(initial_state)
        
        initial_soc = vehicle_model.get_battery_state_of_charge()
        assert initial_soc == 0.8
        
        # Run simulation
        dt = 0.01
        duration = 5.0
        trajectory = vehicle_model.simulate(initial_state, control_inputs, dt, duration)
        
        final_soc = vehicle_model.get_battery_state_of_charge()
        assert final_soc < initial_soc  # SOC should decrease
    
    def test_fault_injection(self, vehicle_model, initial_state, control_inputs):
        """Test fault injection capability."""
        vehicle_model.set_initial_state(initial_state)
        
        # Inject fault
        vehicle_model.inject_fault('main_rotor_1_esc', 'stuck', 0.8, 5.0)
        
        fault_status = vehicle_model.get_fault_status()
        assert len(fault_status) > 0
    
    def test_performance_metrics(self, vehicle_model, initial_state, control_inputs):
        """Test performance metrics collection."""
        vehicle_model.set_initial_state(initial_state)
        
        # Run simulation
        dt = 0.01
        duration = 5.0
        trajectory = vehicle_model.simulate(initial_state, control_inputs, dt, duration)
        
        metrics = vehicle_model.get_performance_metrics()
        
        assert 'simulation_time' in metrics
        assert 'step_count' in metrics
        assert 'energy_consumed' in metrics
        assert 'battery_soc' in metrics
        assert 'battery_temperature' in metrics
        
        assert metrics['simulation_time'] == duration
        assert metrics['step_count'] == int(duration / dt)
        assert metrics['energy_consumed'] > 0.0
    
    def test_constraint_violation_handling(self, vehicle_model, initial_state):
        """Test constraint violation handling."""
        vehicle_model.set_initial_state(initial_state)
        
        # Create extreme control inputs that should violate constraints
        extreme_controls = ControlInputs(
            main_rotor_rpm=np.array([7000, 7000, 7000, 7000]),  # Exceeds limits
            tail_rotor_rpm=5000,  # Exceeds limits
            lift_fan_rpm=np.array([4000, 4000]),  # Exceeds limits
            propeller_rpm=np.array([5000, 5000]),  # Exceeds limits
            elevator_deflection=1.0,  # Exceeds limits
            aileron_deflection=1.0,  # Exceeds limits
            rudder_deflection=1.0,  # Exceeds limits
            throttle=1.5,  # Exceeds limits
            collective=1.5  # Exceeds limits
        )
        
        # Run simulation - should handle violations gracefully
        dt = 0.01
        duration = 1.0
        trajectory = vehicle_model.simulate(initial_state, extreme_controls, dt, duration)
        
        # Should complete simulation despite violations
        assert len(trajectory) > 0
        assert len(vehicle_model.constraint_violations) > 0
    
    def test_simulation_termination_conditions(self, vehicle_model, initial_state, control_inputs):
        """Test simulation termination conditions."""
        vehicle_model.set_initial_state(initial_state)
        
        # Set very low SOC to trigger termination
        vehicle_model.current_state.battery_soc = 0.02  # 2% SOC
        
        dt = 0.01
        duration = 10.0
        trajectory = vehicle_model.simulate(initial_state, control_inputs, dt, duration)
        
        # Should terminate early due to low SOC
        assert len(trajectory) < int(duration / dt) + 1
    
    def test_state_derivative_calculation(self, vehicle_model, initial_state, control_inputs):
        """Test state derivative calculation."""
        vehicle_model.set_initial_state(initial_state)
        
        # Mock forces and moments
        forces = np.array([0.0, 0.0, 15000.0])  # 15kN upward force
        moments = np.array([0.0, 0.0, 0.0])  # No moments
        
        state_derivative = vehicle_model._calculate_state_derivative(
            initial_state, forces, moments, control_inputs
        )
        
        assert 'position' in state_derivative
        assert 'velocity' in state_derivative
        assert 'attitude' in state_derivative
        assert 'angular_velocity' in state_derivative
        assert 'battery_soc' in state_derivative
        assert 'battery_temperature' in state_derivative
        
        # Check that derivatives are reasonable
        assert np.all(np.isfinite(state_derivative['position']))
        assert np.all(np.isfinite(state_derivative['velocity']))
        assert np.all(np.isfinite(state_derivative['attitude']))
        assert np.all(np.isfinite(state_derivative['angular_velocity']))
    
    def test_integration_accuracy(self, vehicle_model, initial_state, control_inputs):
        """Test numerical integration accuracy."""
        vehicle_model.set_initial_state(initial_state)
        
        # Run simulation with different time steps
        dt1 = 0.01
        dt2 = 0.005
        
        # First simulation
        trajectory1 = vehicle_model.simulate(initial_state, control_inputs, dt1, 1.0)
        
        # Reset and run second simulation
        vehicle_model.set_initial_state(initial_state)
        trajectory2 = vehicle_model.simulate(initial_state, control_inputs, dt2, 1.0)
        
        # Results should be similar (within reasonable tolerance)
        final_pos1 = trajectory1[-1].position
        final_pos2 = trajectory2[-1].position
        
        position_error = np.linalg.norm(final_pos1 - final_pos2)
        assert position_error < 1.0  # 1 meter tolerance
    
    def test_memory_usage(self, vehicle_model, initial_state, control_inputs):
        """Test memory usage for long simulations."""
        vehicle_model.set_initial_state(initial_state)
        
        # Run long simulation
        dt = 0.01
        duration = 100.0  # 100 seconds
        trajectory = vehicle_model.simulate(initial_state, control_inputs, dt, duration)
        
        # Should complete without memory issues
        assert len(trajectory) > 0
        assert trajectory[-1].time == duration


if __name__ == "__main__":
    pytest.main([__file__])
