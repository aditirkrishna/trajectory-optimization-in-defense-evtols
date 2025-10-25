"""
Advanced Vehicle Simulation Example

This example demonstrates advanced vehicle layer capabilities including:
- Complex mission scenarios
- Fault injection and recovery
- Performance optimization
- Real-time monitoring
- Multi-vehicle operations
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
import logging
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from vehicle_layer import (
    VehicleModel, VehicleConfig, VehicleState, ControlInputs,
    FaultInjector, FlightEnvelope, PerformanceMonitor
)


class MissionPlanner:
    """Mission planning and execution for advanced simulations."""
    
    def __init__(self, vehicle_model: VehicleModel):
        self.vehicle = vehicle_model
        self.mission_waypoints = []
        self.current_waypoint = 0
        self.mission_start_time = 0.0
        
    def load_mission(self, waypoints: List[Dict[str, Any]]):
        """Load mission waypoints."""
        self.mission_waypoints = waypoints
        self.current_waypoint = 0
        self.mission_start_time = 0.0
        
    def get_next_controls(self, current_state: VehicleState, current_time: float) -> ControlInputs:
        """Get control inputs for next waypoint."""
        if self.current_waypoint >= len(self.mission_waypoints):
            return self._create_hover_controls()
        
        target = self.mission_waypoints[self.current_waypoint]
        current_pos = current_state.position
        
        # Calculate desired velocity vector
        target_pos = np.array(target['position'])
        distance = np.linalg.norm(target_pos - current_pos)
        
        if distance < 10.0:  # Within 10m of waypoint
            self.current_waypoint += 1
            return self._create_hover_controls()
        
        # Calculate desired velocity
        desired_velocity = (target_pos - current_pos) / distance * target.get('speed', 20.0)
        
        # Convert to control inputs (simplified)
        return self._velocity_to_controls(desired_velocity, target.get('altitude', current_pos[2]))
    
    def _create_hover_controls(self) -> ControlInputs:
        """Create hover control inputs."""
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
    
    def _velocity_to_controls(self, desired_velocity: np.ndarray, target_altitude: float) -> ControlInputs:
        """Convert desired velocity to control inputs."""
        # Simplified control law
        forward_speed = desired_velocity[0]
        vertical_speed = desired_velocity[2]
        
        # Calculate control inputs based on desired motion
        if forward_speed > 0:
            # Forward flight
            main_rpm = np.array([800, 800, 800, 800])
            prop_rpm = np.array([2000, 2000])
            elevator = 0.1
        else:
            # Hover or backward
            main_rpm = np.array([1000, 1000, 1000, 1000])
            prop_rpm = np.array([0, 0])
            elevator = 0.0
        
        # Altitude control
        if vertical_speed > 0:
            collective = 0.6  # Climb
        elif vertical_speed < 0:
            collective = 0.4  # Descent
        else:
            collective = 0.5  # Maintain altitude
        
        return ControlInputs(
            main_rotor_rpm=main_rpm,
            tail_rotor_rpm=1200,
            lift_fan_rpm=np.array([800, 800]),
            propeller_rpm=prop_rpm,
            elevator_deflection=elevator,
            aileron_deflection=0.0,
            rudder_deflection=0.0,
            throttle=0.8,
            collective=collective
        )


class PerformanceMonitor:
    """Real-time performance monitoring and analysis."""
    
    def __init__(self):
        self.metrics_history = []
        self.start_time = time.time()
        
    def record_metrics(self, state: VehicleState, controls: ControlInputs, 
                      simulation_time: float, step_time: float):
        """Record performance metrics."""
        metrics = {
            'simulation_time': simulation_time,
            'real_time': time.time() - self.start_time,
            'step_time': step_time,
            'position': state.position.copy(),
            'velocity': state.velocity.copy(),
            'battery_soc': state.battery_soc,
            'battery_temperature': state.battery_temperature,
            'power_consumption': self._calculate_power_consumption(controls),
            'efficiency': self._calculate_efficiency(state, controls)
        }
        self.metrics_history.append(metrics)
    
    def _calculate_power_consumption(self, controls: ControlInputs) -> float:
        """Calculate total power consumption."""
        # Simplified power calculation
        main_power = np.sum(controls.main_rotor_rpm) * 10  # 10W per RPM
        tail_power = controls.tail_rotor_rpm * 5
        prop_power = np.sum(controls.propeller_rpm) * 8
        aux_power = 5000  # Auxiliary systems
        
        return main_power + tail_power + prop_power + aux_power
    
    def _calculate_efficiency(self, state: VehicleState, controls: ControlInputs) -> float:
        """Calculate flight efficiency."""
        # Simplified efficiency calculation
        speed = np.linalg.norm(state.velocity)
        power = self._calculate_power_consumption(controls)
        
        if power > 0:
            return speed / power * 1000  # m/s per kW
        return 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics_history:
            return {}
        
        # Calculate statistics
        total_time = self.metrics_history[-1]['simulation_time']
        total_distance = self._calculate_total_distance()
        avg_speed = total_distance / total_time if total_time > 0 else 0
        avg_power = np.mean([m['power_consumption'] for m in self.metrics_history])
        avg_efficiency = np.mean([m['efficiency'] for m in self.metrics_history])
        
        return {
            'total_time': total_time,
            'total_distance': total_distance,
            'average_speed': avg_speed,
            'average_power': avg_power,
            'average_efficiency': avg_efficiency,
            'energy_consumed': avg_power * total_time / 3600,  # kWh
            'real_time_factor': total_time / (time.time() - self.start_time)
        }
    
    def _calculate_total_distance(self) -> float:
        """Calculate total distance traveled."""
        if len(self.metrics_history) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(self.metrics_history)):
            pos1 = self.metrics_history[i-1]['position']
            pos2 = self.metrics_history[i]['position']
            total_distance += np.linalg.norm(pos2 - pos1)
        
        return total_distance


def create_complex_mission() -> List[Dict[str, Any]]:
    """Create a complex mission scenario."""
    return [
        {'position': [0, 0, 100], 'speed': 0, 'altitude': 100, 'description': 'Takeoff'},
        {'position': [100, 0, 150], 'speed': 20, 'altitude': 150, 'description': 'Climb and accelerate'},
        {'position': [500, 0, 200], 'speed': 50, 'altitude': 200, 'description': 'Cruise flight'},
        {'position': [1000, 200, 250], 'speed': 60, 'altitude': 250, 'description': 'Turn and climb'},
        {'position': [1200, 500, 200], 'speed': 40, 'altitude': 200, 'description': 'Approach target'},
        {'position': [1200, 500, 50], 'speed': 10, 'altitude': 50, 'description': 'Descent to target'},
        {'position': [1200, 500, 50], 'speed': 0, 'altitude': 50, 'description': 'Hover at target'},
        {'position': [1200, 500, 100], 'speed': 10, 'altitude': 100, 'description': 'Climb from target'},
        {'position': [800, 300, 150], 'speed': 40, 'altitude': 150, 'description': 'Return flight'},
        {'position': [0, 0, 100], 'speed': 30, 'altitude': 100, 'description': 'Return to base'},
        {'position': [0, 0, 0], 'speed': 5, 'altitude': 0, 'description': 'Landing'}
    ]


def run_advanced_simulation():
    """Run advanced simulation with mission planning and fault injection."""
    print("=== Advanced Vehicle Simulation ===")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    config = VehicleConfig("config/vehicle_config.yaml")
    
    # Create vehicle model
    vehicle = VehicleModel(config)
    
    # Create mission planner
    mission_planner = MissionPlanner(vehicle)
    mission_planner.load_mission(create_complex_mission())
    
    # Create performance monitor
    performance_monitor = PerformanceMonitor()
    
    # Create fault injector
    fault_injector = FaultInjector(config)
    
    # Create flight envelope checker
    flight_envelope = FlightEnvelope(config)
    
    # Set initial state
    initial_state = VehicleState(
        position=np.array([0.0, 0.0, 100.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        attitude=np.array([0.0, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.0]),
        battery_soc=0.9,
        battery_temperature=20.0,
        battery_voltage=400.0,
        rotor_rpm=np.array([1000, 1000, 1000, 1000]),
        control_surface_deflections=np.array([0.0, 0.0, 0.0]),
        time=0.0
    )
    
    vehicle.set_initial_state(initial_state)
    
    # Schedule faults
    fault_injector.inject_fault('main_rotor_1_esc', 'stuck', 0.3, 30.0, 60.0)
    fault_injector.inject_fault('elevator', 'drift', 0.2, 45.0, 75.0)
    fault_injector.inject_fault('battery', 'degradation', 0.1, 60.0)
    
    # Simulation parameters
    dt = 0.01
    max_duration = 120.0  # 2 minutes
    current_time = 0.0
    
    # Simulation loop
    trajectory = [initial_state]
    fault_log = []
    constraint_violations = []
    
    print("Starting advanced simulation...")
    start_time = time.time()
    
    while current_time < max_duration:
        step_start = time.time()
        
        # Get current state
        current_state = trajectory[-1]
        
        # Get mission controls
        controls = mission_planner.get_next_controls(current_state, current_time)
        
        # Apply faults
        controls = fault_injector.apply_faults(controls, current_time)
        
        # Check constraints
        violations = flight_envelope.check_constraints(current_state, controls)
        if violations:
            constraint_violations.extend(violations)
            print(f"Constraint violations at {current_time:.1f}s: {violations}")
        
        # Perform simulation step
        try:
            new_state = vehicle.step(controls, dt)
            trajectory.append(new_state)
            
            # Record performance metrics
            step_time = time.time() - step_start
            performance_monitor.record_metrics(new_state, controls, current_time, step_time)
            
            # Check for mission completion
            if mission_planner.current_waypoint >= len(mission_planner.mission_waypoints):
                print(f"Mission completed at {current_time:.1f}s")
                break
            
            # Check for critical failures
            if new_state.battery_soc < 0.1:
                print(f"Critical battery level reached at {current_time:.1f}s")
                break
            
            current_time += dt
            
        except Exception as e:
            print(f"Simulation error at {current_time:.1f}s: {e}")
            break
    
    total_time = time.time() - start_time
    
    # Generate results
    print("\n=== Simulation Results ===")
    print(f"Total simulation time: {current_time:.1f}s")
    print(f"Real-time execution: {total_time:.1f}s")
    print(f"Real-time factor: {current_time/total_time:.1f}x")
    print(f"Mission waypoints completed: {mission_planner.current_waypoint}/{len(mission_planner.mission_waypoints)}")
    
    # Performance summary
    performance_summary = performance_monitor.get_performance_summary()
    print(f"\nPerformance Summary:")
    print(f"  Total distance: {performance_summary.get('total_distance', 0):.1f} m")
    print(f"  Average speed: {performance_summary.get('average_speed', 0):.1f} m/s")
    print(f"  Average power: {performance_summary.get('average_power', 0):.1f} W")
    print(f"  Energy consumed: {performance_summary.get('energy_consumed', 0):.2f} kWh")
    print(f"  Flight efficiency: {performance_summary.get('average_efficiency', 0):.2f} m/s/kW")
    
    # Fault summary
    fault_stats = fault_injector.get_fault_statistics()
    print(f"\nFault Summary:")
    print(f"  Total faults injected: {fault_stats['total_injected']}")
    print(f"  Active faults: {fault_stats['total_active']}")
    print(f"  Recovered faults: {fault_stats['total_recovered']}")
    print(f"  Critical faults: {fault_stats['critical_faults']}")
    
    # Constraint violations
    print(f"\nConstraint Violations: {len(constraint_violations)}")
    if constraint_violations:
        for violation in constraint_violations[:5]:  # Show first 5
            print(f"  {violation}")
    
    # Final state
    final_state = trajectory[-1]
    print(f"\nFinal State:")
    print(f"  Position: {final_state.position}")
    print(f"  Velocity: {final_state.velocity}")
    print(f"  Battery SOC: {final_state.battery_soc:.3f}")
    print(f"  Battery Temperature: {final_state.battery_temperature:.1f}°C")
    
    return trajectory, performance_summary, fault_stats


def plot_advanced_results(trajectory, performance_summary):
    """Plot advanced simulation results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract data
    times = [state.time for state in trajectory]
    positions = np.array([state.position for state in trajectory])
    velocities = np.array([state.velocity for state in trajectory])
    soc_values = [state.battery_soc for state in trajectory]
    temperatures = [state.battery_temperature for state in trajectory]
    
    # 3D trajectory
    ax = axes[0, 0]
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title('3D Flight Trajectory')
    ax.grid(True)
    
    # Altitude profile
    ax = axes[0, 1]
    ax.plot(times, positions[:, 2], 'g-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Altitude Profile')
    ax.grid(True)
    
    # Speed profile
    ax = axes[0, 2]
    speeds = np.linalg.norm(velocities, axis=1)
    ax.plot(times, speeds, 'r-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Speed Profile')
    ax.grid(True)
    
    # Battery SOC
    ax = axes[1, 0]
    ax.plot(times, soc_values, 'orange', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('State of Charge')
    ax.set_title('Battery SOC')
    ax.grid(True)
    ax.set_ylim(0, 1)
    
    # Battery temperature
    ax = axes[1, 1]
    ax.plot(times, temperatures, 'purple', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Battery Temperature')
    ax.grid(True)
    
    # Performance metrics
    ax = axes[1, 2]
    metrics = ['Distance (km)', 'Avg Speed (m/s)', 'Energy (kWh)', 'Efficiency']
    values = [
        performance_summary.get('total_distance', 0) / 1000,
        performance_summary.get('average_speed', 0),
        performance_summary.get('energy_consumed', 0),
        performance_summary.get('average_efficiency', 0)
    ]
    ax.bar(metrics, values, color=['blue', 'green', 'red', 'orange'])
    ax.set_title('Performance Metrics')
    ax.set_ylabel('Value')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('advanced_simulation_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function for advanced simulation."""
    print("eVTOL Vehicle Layer - Advanced Simulation Example")
    print("=" * 60)
    
    try:
        # Run advanced simulation
        trajectory, performance_summary, fault_stats = run_advanced_simulation()
        
        # Plot results
        plot_advanced_results(trajectory, performance_summary)
        
        print("\nAdvanced simulation completed successfully!")
        print("Results saved to 'advanced_simulation_results.png'")
        
    except Exception as e:
        print(f"Advanced simulation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()