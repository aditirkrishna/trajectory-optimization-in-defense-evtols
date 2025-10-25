"""
Basic Vehicle Simulation Example

This example demonstrates how to use the vehicle layer for basic eVTOL simulation,
including hover, forward flight, and energy consumption tracking.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from vehicle_layer import VehicleModel, VehicleConfig, VehicleState, ControlInputs


def create_initial_state():
    """Create initial vehicle state for simulation."""
    return VehicleState(
        position=np.array([0.0, 0.0, 100.0]),  # 100m altitude
        velocity=np.array([0.0, 0.0, 0.0]),    # Hover condition
        attitude=np.array([0.0, 0.0, 0.0]),    # Level attitude
        angular_velocity=np.array([0.0, 0.0, 0.0]),
        battery_soc=0.8,                       # 80% SOC
        battery_temperature=20.0,              # 20°C
        battery_voltage=400.0,                 # 400V
        rotor_rpm=np.array([1000, 1000, 1000, 1000]),  # 4 main rotors
        control_surface_deflections=np.array([0.0, 0.0, 0.0]),  # Neutral
        time=0.0
    )


def create_hover_controls():
    """Create control inputs for hover condition."""
    return ControlInputs(
        main_rotor_rpm=np.array([1000, 1000, 1000, 1000]),  # Balanced hover
        tail_rotor_rpm=1200,                                 # Yaw control
        lift_fan_rpm=np.array([800, 800]),                  # Lift assistance
        propeller_rpm=np.array([0, 0]),                     # No forward thrust
        elevator_deflection=0.0,
        aileron_deflection=0.0,
        rudder_deflection=0.0,
        throttle=0.7,                                       # 70% throttle
        collective=0.5                                      # 50% collective
    )


def create_forward_flight_controls():
    """Create control inputs for forward flight."""
    return ControlInputs(
        main_rotor_rpm=np.array([800, 800, 800, 800]),      # Reduced for forward flight
        tail_rotor_rpm=1000,                                # Reduced yaw control
        lift_fan_rpm=np.array([600, 600]),                  # Reduced lift
        propeller_rpm=np.array([2000, 2000]),               # Forward thrust
        elevator_deflection=0.1,                            # Slight nose down
        aileron_deflection=0.0,
        rudder_deflection=0.0,
        throttle=0.8,                                       # 80% throttle
        collective=0.3                                      # 30% collective
    )


def simulate_hover():
    """Simulate hover condition for 30 seconds."""
    print("=== Hover Simulation ===")
    
    # Load configuration
    config = VehicleConfig("config/vehicle_config.yaml")
    
    # Create vehicle model
    vehicle = VehicleModel(config)
    
    # Set initial state
    initial_state = create_initial_state()
    vehicle.set_initial_state(initial_state)
    
    # Create hover controls
    controls = create_hover_controls()
    
    # Simulate
    dt = 0.01  # 10ms time step
    duration = 30.0  # 30 seconds
    trajectory = vehicle.simulate(initial_state, controls, dt, duration)
    
    # Extract results
    times = [state.time for state in trajectory]
    positions = np.array([state.position for state in trajectory])
    velocities = np.array([state.velocity for state in trajectory])
    soc_values = [state.battery_soc for state in trajectory]
    temperatures = [state.battery_temperature for state in trajectory]
    
    # Print results
    print(f"Simulation completed: {len(trajectory)} time steps")
    print(f"Final position: {positions[-1]}")
    print(f"Final velocity: {velocities[-1]}")
    print(f"Final SOC: {soc_values[-1]:.3f}")
    print(f"Final temperature: {temperatures[-1]:.1f}°C")
    print(f"Total energy consumed: {vehicle.get_energy_consumption():.1f} Wh")
    
    return trajectory, times, positions, velocities, soc_values, temperatures


def simulate_forward_flight():
    """Simulate forward flight for 60 seconds."""
    print("\n=== Forward Flight Simulation ===")
    
    # Load configuration
    config = VehicleConfig("config/vehicle_config.yaml")
    
    # Create vehicle model
    vehicle = VehicleModel(config)
    
    # Set initial state with forward velocity
    initial_state = create_initial_state()
    initial_state.velocity = np.array([50.0, 0.0, 0.0])  # 50 m/s forward
    vehicle.set_initial_state(initial_state)
    
    # Create forward flight controls
    controls = create_forward_flight_controls()
    
    # Simulate
    dt = 0.01  # 10ms time step
    duration = 60.0  # 60 seconds
    trajectory = vehicle.simulate(initial_state, controls, dt, duration)
    
    # Extract results
    times = [state.time for state in trajectory]
    positions = np.array([state.position for state in trajectory])
    velocities = np.array([state.velocity for state in trajectory])
    soc_values = [state.battery_soc for state in trajectory]
    temperatures = [state.battery_temperature for state in trajectory]
    
    # Print results
    print(f"Simulation completed: {len(trajectory)} time steps")
    print(f"Final position: {positions[-1]}")
    print(f"Final velocity: {velocities[-1]}")
    print(f"Final SOC: {soc_values[-1]:.3f}")
    print(f"Final temperature: {temperatures[-1]:.1f}°C")
    print(f"Total energy consumed: {vehicle.get_energy_consumption():.1f} Wh")
    
    return trajectory, times, positions, velocities, soc_values, temperatures


def simulate_with_fault():
    """Simulate with actuator fault injection."""
    print("\n=== Fault Injection Simulation ===")
    
    # Load configuration
    config = VehicleConfig("config/vehicle_config.yaml")
    
    # Create vehicle model
    vehicle = VehicleModel(config)
    
    # Set initial state
    initial_state = create_initial_state()
    vehicle.set_initial_state(initial_state)
    
    # Create controls
    controls = create_hover_controls()
    
    # Inject fault after 10 seconds
    vehicle.inject_fault('main_rotor_1_esc', 'stuck', severity=0.8, start_time=10.0)
    
    # Simulate
    dt = 0.01
    duration = 30.0
    trajectory = vehicle.simulate(initial_state, controls, dt, duration)
    
    # Extract results
    times = [state.time for state in trajectory]
    positions = np.array([state.position for state in trajectory])
    velocities = np.array([state.velocity for state in trajectory])
    soc_values = [state.battery_soc for state in trajectory]
    
    # Print results
    print(f"Simulation with fault completed: {len(trajectory)} time steps")
    print(f"Final position: {positions[-1]}")
    print(f"Final velocity: {velocities[-1]}")
    print(f"Final SOC: {soc_values[-1]:.3f}")
    print(f"Active faults: {len(vehicle.get_fault_status())}")
    
    return trajectory, times, positions, velocities, soc_values


def plot_results(hover_data, forward_data, fault_data):
    """Plot simulation results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Hover results
    hover_times, hover_positions, hover_velocities, hover_soc, hover_temp = hover_data
    
    # Position
    axes[0, 0].plot(hover_times, hover_positions[:, 2], 'b-', label='Hover')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Altitude (m)')
    axes[0, 0].set_title('Altitude vs Time')
    axes[0, 0].grid(True)
    
    # Velocity
    axes[0, 1].plot(hover_times, np.linalg.norm(hover_velocities, axis=1), 'b-', label='Hover')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Speed (m/s)')
    axes[0, 1].set_title('Speed vs Time')
    axes[0, 1].grid(True)
    
    # SOC
    axes[0, 2].plot(hover_times, hover_soc, 'b-', label='Hover')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('State of Charge')
    axes[0, 2].set_title('Battery SOC vs Time')
    axes[0, 2].grid(True)
    
    # Forward flight results
    forward_times, forward_positions, forward_velocities, forward_soc, forward_temp = forward_data
    
    # Position
    axes[1, 0].plot(forward_times, forward_positions[:, 0], 'r-', label='Forward Flight')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('X Position (m)')
    axes[1, 0].set_title('X Position vs Time')
    axes[1, 0].grid(True)
    
    # Velocity
    axes[1, 1].plot(forward_times, np.linalg.norm(forward_velocities, axis=1), 'r-', label='Forward Flight')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Speed (m/s)')
    axes[1, 1].set_title('Speed vs Time')
    axes[1, 1].grid(True)
    
    # SOC comparison
    axes[1, 2].plot(hover_times, hover_soc, 'b-', label='Hover')
    axes[1, 2].plot(forward_times, forward_soc, 'r-', label='Forward Flight')
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('State of Charge')
    axes[1, 2].set_title('Battery SOC Comparison')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('vehicle_simulation_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main simulation function."""
    print("eVTOL Vehicle Layer - Basic Simulation Example")
    print("=" * 50)
    
    try:
        # Run simulations
        hover_data = simulate_hover()
        forward_data = simulate_forward_flight()
        fault_data = simulate_with_fault()
        
        # Plot results
        plot_results(hover_data, forward_data, fault_data)
        
        print("\nSimulation completed successfully!")
        print("Results saved to 'vehicle_simulation_results.png'")
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
