#!/usr/bin/env python3
"""
eVTOL Trajectory Optimization - Live Demonstration
This script demonstrates the core functionality of the eVTOL trajectory optimization system.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Add src to path
sys.path.append('src')

def demonstrate_planning_layer():
    """Demonstrate the planning layer functionality"""
    print("eVTOL TRAJECTORY OPTIMIZATION DEMONSTRATION")
    print("=" * 60)
    
    try:
        from evtol.planning import RoutePlanner, setup_planning_layer
        
        print("\n1. INITIALIZING PLANNING LAYER...")
        # Explicitly point to built-in demo config to avoid missing-file issues
        config, logger = setup_planning_layer("src/config/planning_config.yaml")
        planner = RoutePlanner(config)
        print("Planning layer initialized successfully")
        
        print("\n2. PLANNING OPTIMAL ROUTE...")
        print("   Start: Bangalore City Center (12.9716°N, 77.5946°E)")
        print("   Goal:  Kempegowda Airport (13.0827°N, 77.5877°E)")
        print("   Altitude: 120m")
        
        start_time = time.time()
        
        # Plan route from Bangalore city center to airport
        route = planner.optimize_route(
            start_lat=12.9716,  # Bangalore City Center
            start_lon=77.5946,
            goal_lat=13.0827,   # Kempegowda Airport
            goal_lon=77.5877,
            start_alt_m=120.0,
            time_iso="2024-01-01T12:00:00"
        )
        
        planning_time = time.time() - start_time
        
        print(f"Route planning completed in {planning_time:.3f} seconds")
        print(f"   Generated {len(route)} waypoints")
        
        # Calculate route metrics
        total_distance = 0.0
        for i in range(len(route) - 1):
            wp1, wp2 = route[i], route[i+1]
            # Simple distance calculation
            lat_diff = wp2.lat - wp1.lat
            lon_diff = wp2.lon - wp1.lon
            distance = np.sqrt(lat_diff**2 + lon_diff**2) * 111.32  # Approximate km
            total_distance += distance
        
        print(f"   Total distance: {total_distance:.2f} km")
        print(f"   Average altitude: {np.mean([wp.alt_m for wp in route]):.1f} m")
        
        return route, planning_time
        
    except Exception as e:
        print(f"Planning layer error: {e}")
        return None, 0

def demonstrate_vehicle_simulation():
    """Demonstrate vehicle simulation capabilities"""
    print("\n3. VEHICLE SIMULATION DEMONSTRATION...")
    
    try:
        from evtol.vehicle import VehicleModel, VehicleConfig, VehicleState, ControlInputs
        
        print("   Initializing vehicle model...")
        config = VehicleConfig("config/default.yaml")
        vehicle = VehicleModel(config)
        
        # Create initial state
        initial_state = VehicleState(
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
        
        # Create hover controls
        controls = ControlInputs(
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
        
        print("   Running 30-second hover simulation...")
        start_time = time.time()
        
        # Simulate hover for 30 seconds
        dt = 0.01  # 10ms time step
        duration = 30.0  # 30 seconds
        trajectory = vehicle.simulate(initial_state, controls, dt, duration)
        
        sim_time = time.time() - start_time
        
        print(f"Simulation completed in {sim_time:.3f} seconds")
        print(f"   Generated {len(trajectory)} simulation steps")
        
        # Extract final state
        final_state = trajectory[-1]
        print(f"   Final position: [{final_state.position[0]:.1f}, {final_state.position[1]:.1f}, {final_state.position[2]:.1f}] m")
        print(f"   Final velocity: [{final_state.velocity[0]:.1f}, {final_state.velocity[1]:.1f}, {final_state.velocity[2]:.1f}] m/s")
        print(f"   Final battery SOC: {final_state.battery_soc:.3f}")
        print(f"   Final temperature: {final_state.battery_temperature:.1f}°C")
        
        return trajectory
        
    except Exception as e:
        print(f"Vehicle simulation error: {e}")
        return None

def demonstrate_perception_layer():
    """Demonstrate perception layer capabilities"""
    print("\n4. PERCEPTION LAYER DEMONSTRATION...")
    
    try:
        from evtol.perception.geometry.terrain_analysis import compute_slope, compute_aspect
        
        print("   Creating sample terrain data...")
        # Create a sample terrain grid
        size = 50
        x = np.linspace(0, 10, size)
        y = np.linspace(0, 10, size)
        X, Y = np.meshgrid(x, y)
        
        # Create a terrain with hills and valleys
        terrain = 100 + 20 * np.sin(X/2) * np.cos(Y/2) + 10 * np.sin(X) * np.sin(Y)
        
        print("   Computing terrain derivatives...")
        slope = compute_slope(terrain, pixel_size=1.0)
        aspect = compute_aspect(terrain, pixel_size=1.0)
        
        print(f"Terrain analysis completed")
        print(f"   Terrain elevation range: {np.min(terrain):.1f}m - {np.max(terrain):.1f}m")
        print(f"   Slope range: {np.min(slope):.1f}° - {np.max(slope):.1f}°")
        print(f"   Aspect range: {np.min(aspect):.1f}° - {np.max(aspect):.1f}°")
        
        return terrain, slope, aspect
        
    except Exception as e:
        print(f"Perception layer error: {e}")
        return None, None, None

def create_visualizations(route, trajectory, terrain_data):
    """Create visualizations of the results"""
    print("\n5. CREATING VISUALIZATIONS...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('eVTOL Trajectory Optimization System - Live Demonstration', fontsize=16, fontweight='bold')
        
        # Route visualization
        if route:
            ax1 = axes[0, 0]
            lats = [wp.lat for wp in route]
            lons = [wp.lon for wp in route]
            alts = [wp.alt_m for wp in route]
            
            ax1.plot(lons, lats, 'b-', linewidth=2, label='Flight Path')
            ax1.scatter(lons[0], lats[0], c='green', s=100, marker='s', label='Start')
            ax1.scatter(lons[-1], lats[-1], c='red', s=100, marker='*', label='Goal')
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')
            ax1.set_title('Optimized Route: Bangalore City -> Airport')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Altitude profile
        if route:
            ax2 = axes[0, 1]
            distances = np.linspace(0, 15, len(route))  # Approximate distances
            ax2.plot(distances, alts, 'g-', linewidth=2)
            ax2.fill_between(distances, 0, alts, alpha=0.3, color='green')
            ax2.set_xlabel('Distance (km)')
            ax2.set_ylabel('Altitude (m)')
            ax2.set_title('Altitude Profile')
            ax2.grid(True, alpha=0.3)
        
        # Vehicle simulation results
        if trajectory:
            ax3 = axes[1, 0]
            times = [state.time for state in trajectory]
            positions = np.array([state.position for state in trajectory])
            soc_values = [state.battery_soc for state in trajectory]
            
            ax3.plot(times, positions[:, 2], 'purple', linewidth=2, label='Altitude')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Altitude (m)')
            ax3.set_title('Vehicle Simulation: Hover Performance')
            ax3.grid(True, alpha=0.3)
            
            # Battery SOC on secondary axis
            ax3_twin = ax3.twinx()
            ax3_twin.plot(times, soc_values, 'orange', linewidth=2, label='Battery SOC')
            ax3_twin.set_ylabel('Battery SOC')
            ax3_twin.set_ylim(0, 1)
        
        # Terrain analysis
        if terrain_data[0] is not None:
            ax4 = axes[1, 1]
            terrain, slope, aspect = terrain_data
            im = ax4.imshow(terrain, cmap='terrain', aspect='auto')
            ax4.set_title('Sample Terrain Analysis')
            ax4.set_xlabel('X (m)')
            ax4.set_ylabel('Y (m)')
            plt.colorbar(im, ax=ax4, label='Elevation (m)')
        
        plt.tight_layout()
        plt.savefig('evtol_demonstration_results.png', dpi=300, bbox_inches='tight')
        print("Visualizations saved as 'evtol_demonstration_results.png'")
        
        return True
        
    except Exception as e:
        print(f"Visualization error: {e}")
        return False

def main():
    """Main demonstration function"""
    print("Starting eVTOL Trajectory Optimization System Demonstration...")
    
    # Demonstrate each layer
    route, planning_time = demonstrate_planning_layer()
    trajectory = demonstrate_vehicle_simulation()
    terrain_data = demonstrate_perception_layer()
    
    # Create visualizations
    viz_success = create_visualizations(route, trajectory, terrain_data)
    
    # Summary
    print("\n" + "=" * 60)
    print("DEMONSTRATION SUMMARY")
    print("=" * 60)
    
    if route:
        print(f"Route Planning: {len(route)} waypoints in {planning_time:.3f}s")
    else:
        print("Route Planning: Failed")
    
    if trajectory:
        print(f"Vehicle Simulation: {len(trajectory)} steps completed")
    else:
        print("Vehicle Simulation: Failed")
    
    if terrain_data[0] is not None:
        print("Terrain Analysis: Terrain derivatives computed")
    else:
        print("Terrain Analysis: Failed")
    
    if viz_success:
        print("Visualizations: Results saved to 'evtol_demonstration_results.png'")
    else:
        print("Visualizations: Failed")
    
    print("\neVTOL Trajectory Optimization System Demonstration Complete!")
    print("   This system demonstrates:")
    print("   • Real-time route optimization with A* pathfinding")
    print("   • 6-DOF vehicle dynamics simulation")
    print("   • Environmental terrain analysis")
    print("   • Multi-objective optimization capabilities")
    print("   • Professional visualization and reporting")

if __name__ == "__main__":
    main()