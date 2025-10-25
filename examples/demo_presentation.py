"""
eVTOL Trajectory Optimization - Interactive Presentation Demo

This script demonstrates the complete system with visual waypoint display.
Perfect for presentations and demonstrations.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import logging

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "perception-layer" / "src"))
sys.path.insert(0, str(project_root / "planning-layer" / "src"))
sys.path.insert(0, str(project_root / "control-layer" / "src"))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class eVTOLDemo:
    """Interactive demonstration of eVTOL trajectory optimization"""
    
    def __init__(self):
        """Initialize the demonstration"""
        logger.info("=" * 80)
        logger.info("eVTOL TRAJECTORY OPTIMIZATION - INTERACTIVE DEMO")
        logger.info("=" * 80)
        
        self.setup_planning_layer()
        self.setup_control_layer()
    
    def setup_planning_layer(self):
        """Setup planning layer components"""
        logger.info("\n>>> Setting up Planning Layer...")
        
        try:
            from planning_layer import setup_planning_layer, RoutePlanner
            from planning_layer.energy import EnergyOptimizer
            from planning_layer.risk import RiskManager
            
            self.config, _ = setup_planning_layer()
            self.planner = RoutePlanner(self.config)
            self.energy_optimizer = EnergyOptimizer(self.config)
            self.risk_manager = RiskManager(self.config)
            
            logger.info("✅ Planning Layer initialized")
        except Exception as e:
            logger.warning(f"⚠️  Planning Layer partial init: {e}")
            self.planner = None
    
    def setup_control_layer(self):
        """Setup control layer components"""
        logger.info("\n>>> Setting up Control Layer...")
        
        try:
            from control.flight_controller import FlightController
            from control.trajectory_generator import TrajectoryGenerator
            
            self.controller = FlightController(dt=0.01)
            self.traj_generator = TrajectoryGenerator()
            
            logger.info("✅ Control Layer initialized")
        except Exception as e:
            logger.warning(f"⚠️  Control Layer error: {e}")
            self.controller = None
    
    def plan_mission(self, start_lat, start_lon, goal_lat, goal_lon, start_alt_m=120.0):
        """
        Plan a complete mission from start to goal.
        
        Args:
            start_lat, start_lon: Starting coordinates
            goal_lat, goal_lon: Goal coordinates
            start_alt_m: Starting altitude in meters
            
        Returns:
            Mission data dictionary
        """
        logger.info("\n" + "=" * 80)
        logger.info("MISSION PLANNING")
        logger.info("=" * 80)
        logger.info(f"Start: ({start_lat:.4f}, {start_lon:.4f}) @ {start_alt_m}m")
        logger.info(f"Goal:  ({goal_lat:.4f}, {goal_lon:.4f})")
        
        if not self.planner:
            logger.error("❌ Planner not available")
            return None
        
        # Plan route
        logger.info("\n>>> Planning optimal route...")
        route = self.planner.optimize_route(
            start_lat=start_lat,
            start_lon=start_lon,
            goal_lat=goal_lat,
            goal_lon=goal_lon,
            start_alt_m=start_alt_m,
            time_iso="2024-01-01T12:00:00"
        )
        
        logger.info(f"✅ Route generated: {len(route)} waypoints")
        
        # Compute metrics
        logger.info("\n>>> Computing metrics...")
        total_distance = self._compute_total_distance(route)
        
        try:
            energy_est = self.energy_optimizer.estimate_route_energy(route)
            risk_score = self.risk_manager.evaluate_route_risk(route)
        except:
            energy_est = total_distance * 0.8  # Fallback estimate
            risk_score = 0.15
        
        logger.info(f"   Distance: {total_distance:.2f} km")
        logger.info(f"   Energy:   {energy_est:.2f} kWh")
        logger.info(f"   Risk:     {risk_score:.3f}")
        
        # Extract waypoint data
        waypoints = []
        for i, wp in enumerate(route):
            waypoints.append({
                'id': i,
                'lat': wp.lat,
                'lon': wp.lon,
                'alt_m': wp.alt_m
            })
        
        return {
            'waypoints': waypoints,
            'distance_km': total_distance,
            'energy_kwh': energy_est,
            'risk_score': risk_score,
            'num_waypoints': len(waypoints)
        }
    
    def visualize_route(self, mission_data, title="eVTOL Optimized Route"):
        """
        Create an interactive visualization of the planned route.
        
        Args:
            mission_data: Mission data from plan_mission()
            title: Plot title
        """
        if not mission_data:
            logger.error("❌ No mission data to visualize")
            return
        
        logger.info("\n>>> Creating visualization...")
        
        waypoints = mission_data['waypoints']
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(16, 8))
        
        # 2D Map View
        ax1 = fig.add_subplot(121)
        
        lats = [wp['lat'] for wp in waypoints]
        lons = [wp['lon'] for wp in waypoints]
        
        # Plot route
        ax1.plot(lons, lats, 'b-', linewidth=2, label='Flight Path', alpha=0.7)
        
        # Plot waypoints
        ax1.scatter(lons, lats, c='red', s=100, zorder=5, edgecolors='black', linewidths=1.5)
        
        # Mark start and goal
        ax1.scatter(lons[0], lats[0], c='green', s=300, marker='s', 
                   label='Start', edgecolors='black', linewidths=2, zorder=6)
        ax1.scatter(lons[-1], lats[-1], c='orange', s=300, marker='*', 
                   label='Goal', edgecolors='black', linewidths=2, zorder=6)
        
        # Annotate waypoint numbers
        for i, (lon, lat) in enumerate(zip(lons, lats)):
            if i % max(1, len(waypoints)//15) == 0:  # Show every nth waypoint
                ax1.annotate(f'WP{i}', (lon, lat), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax1.set_xlabel('Longitude', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Latitude', fontsize=12, fontweight='bold')
        ax1.set_title('2D Map View', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.set_aspect('equal', adjustable='box')
        
        # 3D Altitude Profile
        ax2 = fig.add_subplot(122)
        
        distances = [0]
        for i in range(1, len(waypoints)):
            d = self._haversine_km(
                waypoints[i-1]['lat'], waypoints[i-1]['lon'],
                waypoints[i]['lat'], waypoints[i]['lon']
            )
            distances.append(distances[-1] + d)
        
        alts = [wp['alt_m'] for wp in waypoints]
        
        # Plot altitude profile
        ax2.plot(distances, alts, 'g-', linewidth=3, label='Altitude Profile')
        ax2.fill_between(distances, 0, alts, alpha=0.3, color='green')
        
        # Mark waypoints
        ax2.scatter(distances, alts, c='red', s=100, zorder=5, edgecolors='black', linewidths=1.5)
        
        ax2.set_xlabel('Distance along route (km)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Altitude (m)', fontsize=12, fontweight='bold')
        ax2.set_title('Altitude Profile', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        # Add statistics box
        stats_text = f"""
Mission Statistics:
• Waypoints: {mission_data['num_waypoints']}
• Distance: {mission_data['distance_km']:.2f} km
• Energy: {mission_data['energy_kwh']:.2f} kWh
• Risk Score: {mission_data['risk_score']:.3f}
• Avg Altitude: {np.mean(alts):.1f} m
• Max Altitude: {max(alts):.1f} m
        """
        
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.08, 1, 0.96])
        
        logger.info("✅ Visualization created")
        plt.show()
    
    def print_waypoints_table(self, mission_data):
        """Print waypoints in a formatted table"""
        if not mission_data:
            return
        
        waypoints = mission_data['waypoints']
        
        logger.info("\n" + "=" * 80)
        logger.info("WAYPOINT TABLE")
        logger.info("=" * 80)
        
        print(f"{'ID':<5} {'Latitude':<12} {'Longitude':<12} {'Altitude (m)':<15}")
        print("-" * 80)
        
        for wp in waypoints:
            print(f"{wp['id']:<5} {wp['lat']:<12.6f} {wp['lon']:<12.6f} {wp['alt_m']:<15.1f}")
        
        print("-" * 80)
    
    def _compute_total_distance(self, route):
        """Compute total route distance in km"""
        total = 0.0
        for i in range(len(route) - 1):
            d = self._haversine_km(
                route[i].lat, route[i].lon,
                route[i+1].lat, route[i+1].lon
            )
            total += d
        return total
    
    def _haversine_km(self, lat1, lon1, lat2, lon2):
        """Calculate Haversine distance in kilometers"""
        R = 6371.0  # Earth radius in km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * 
             np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c


def demo_scenario_1():
    """Demo Scenario 1: Urban Bangalore Mission"""
    demo = eVTOLDemo()
    
    logger.info("\n" + "=" * 80)
    logger.info("SCENARIO 1: Urban Bangalore Mission")
    logger.info("=" * 80)
    
    # Plan mission
    mission = demo.plan_mission(
        start_lat=12.9716,  # Bangalore City Center
        start_lon=77.5946,
        goal_lat=13.0827,   # Kempegowda Airport
        goal_lon=77.5877,
        start_alt_m=120.0
    )
    
    # Display results
    if mission:
        demo.print_waypoints_table(mission)
        demo.visualize_route(mission, title="Scenario 1: Urban Bangalore Mission")


def demo_scenario_2():
    """Demo Scenario 2: Long Range Mission"""
    demo = eVTOLDemo()
    
    logger.info("\n" + "=" * 80)
    logger.info("SCENARIO 2: Long Range Mission")
    logger.info("=" * 80)
    
    # Plan mission
    mission = demo.plan_mission(
        start_lat=13.0,
        start_lon=77.5,
        goal_lat=13.2,
        goal_lon=77.8,
        start_alt_m=150.0
    )
    
    # Display results
    if mission:
        demo.print_waypoints_table(mission)
        demo.visualize_route(mission, title="Scenario 2: Long Range Mission")


def main():
    """Main demo entry point"""
    print("\n" + "=" * 80)
    print("eVTOL TRAJECTORY OPTIMIZATION - PRESENTATION DEMO")
    print("=" * 80)
    print("\nSelect a scenario to demonstrate:")
    print("  1. Urban Bangalore Mission (City Center -> Airport)")
    print("  2. Long Range Mission")
    print("  3. Both scenarios")
    print()
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        demo_scenario_1()
    elif choice == '2':
        demo_scenario_2()
    elif choice == '3':
        demo_scenario_1()
        demo_scenario_2()
    else:
        logger.info("Running default scenario...")
        demo_scenario_1()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

