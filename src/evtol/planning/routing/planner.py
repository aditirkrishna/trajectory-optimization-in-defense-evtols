from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
import heapq
import math

import numpy as np

from ..config import PlanningConfig
from ..serving.perception_client import PerceptionClient


@dataclass
class Waypoint:
    lat: float
    lon: float
    alt_m: float
    
    def __hash__(self):
        return hash((round(self.lat, 6), round(self.lon, 6), round(self.alt_m, 1)))
    
    def __eq__(self, other):
        if not isinstance(other, Waypoint):
            return False
        return (round(self.lat, 6) == round(other.lat, 6) and
                round(self.lon, 6) == round(other.lon, 6) and
                round(self.alt_m, 1) == round(other.alt_m, 1))


@dataclass(order=True)
class AStarNode:
    """Node for A* search"""
    f_cost: float  # Total cost (g + h)
    waypoint: Waypoint = field(compare=False)
    g_cost: float = field(compare=False)  # Cost from start
    h_cost: float = field(compare=False)  # Heuristic cost to goal
    parent: Optional[AStarNode] = field(default=None, compare=False)


class RoutePlanner:
    """
    A* based route planner with multi-objective optimization.

    Integrates with perception-layer by querying fused maps for cost/risk.
    Supports dynamic feasibility constraints and multi-objective costs.
    """

    def __init__(self, config: PlanningConfig) -> None:
        self.config = config
        self.perception = PerceptionClient(config)
        
        # Grid resolution for search space
        self.grid_resolution_deg = 0.01  # ~1km at equator
        
        # Cost weights
        self.cost_weights = {
            "distance": 0.3,
            "energy": 0.3,
            "risk": 0.3,
            "time": 0.1
        }

    def optimize_route(
        self,
        start_lat: float,
        start_lon: float,
        goal_lat: float,
        goal_lon: float,
        start_alt_m: float,
        time_iso: str,
        constraints: dict | None = None,
    ) -> List[Waypoint]:
        """
        Find optimal route using A* algorithm.
        
        Args:
            start_lat: Starting latitude
            start_lon: Starting longitude
            goal_lat: Goal latitude
            goal_lon: Goal longitude
            start_alt_m: Starting altitude in meters
            time_iso: ISO timestamp
            constraints: Optional constraints (max_altitude, min_altitude, etc.)
            
        Returns:
            List of waypoints forming the route
        """
        start_wp = Waypoint(start_lat, start_lon, start_alt_m)
        goal_wp = Waypoint(goal_lat, goal_lon, start_alt_m)
        
        # Run A* search
        path = self._astar_search(start_wp, goal_wp, time_iso, constraints)
        
        if not path:
            # Fallback to straight line if A* fails
            path = self._straight_line_fallback(start_wp, goal_wp)
        
        # Smooth the path
        smoothed_path = self._smooth(path)
        
        return smoothed_path

    def _astar_search(
        self,
        start: Waypoint,
        goal: Waypoint,
        time_iso: str,
        constraints: Optional[dict]
    ) -> List[Waypoint]:
        """
        A* pathfinding algorithm.
        
        Returns:
            List of waypoints from start to goal, or empty list if no path found
        """
        # Priority queue: (f_cost, node)
        open_set = []
        heapq.heappush(open_set, AStarNode(
            f_cost=0.0,
            waypoint=start,
            g_cost=0.0,
            h_cost=self._heuristic(start, goal),
            parent=None
        ))
        
        # Tracking visited nodes
        closed_set: Set[Waypoint] = set()
        g_scores: Dict[Waypoint, float] = {start: 0.0}
        
        # Search parameters
        max_iterations = 1000
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            # Get node with lowest f_cost
            current_node = heapq.heappop(open_set)
            current_wp = current_node.waypoint
            
            # Goal reached?
            if self._is_goal(current_wp, goal):
                return self._reconstruct_path(current_node)
            
            # Mark as visited
            if current_wp in closed_set:
                continue
            closed_set.add(current_wp)
            
            # Expand neighbors
            neighbors = self._get_neighbors(current_wp, goal, constraints)
            
            for neighbor_wp in neighbors:
                if neighbor_wp in closed_set:
                    continue
                
                # Compute cost to neighbor
                edge_cost = self._compute_edge_cost(
                    current_wp, neighbor_wp, time_iso
                )
                
                # Skip if infeasible
                if edge_cost < 0:
                    continue
                
                tentative_g = current_node.g_cost + edge_cost
                
                # Better path found?
                if neighbor_wp not in g_scores or tentative_g < g_scores[neighbor_wp]:
                    g_scores[neighbor_wp] = tentative_g
                    h_cost = self._heuristic(neighbor_wp, goal)
                    f_cost = tentative_g + h_cost
                    
                    neighbor_node = AStarNode(
                        f_cost=f_cost,
                        waypoint=neighbor_wp,
                        g_cost=tentative_g,
                        h_cost=h_cost,
                        parent=current_node
                    )
                    
                    heapq.heappush(open_set, neighbor_node)
        
        # No path found
        return []
    
    def _get_neighbors(
        self,
        waypoint: Waypoint,
        goal: Waypoint,
        constraints: Optional[dict]
    ) -> List[Waypoint]:
        """
        Generate neighbor waypoints for expansion.
        
        Uses 8-connected grid in lat/lon plus altitude variations.
        """
        neighbors = []
        
        # 8-connected grid movements
        for dlat in [-1, 0, 1]:
            for dlon in [-1, 0, 1]:
                if dlat == 0 and dlon == 0:
                    continue
                
                new_lat = waypoint.lat + dlat * self.grid_resolution_deg
                new_lon = waypoint.lon + dlon * self.grid_resolution_deg
                
                # Consider altitude variations
                for dalt in [0]:  # Can add [-50, 0, 50] for 3D planning
                    new_alt = waypoint.alt_m + dalt
                    
                    # Apply constraints
                    if constraints:
                        min_alt = constraints.get("min_altitude_m", 50)
                        max_alt = constraints.get("max_altitude_m", 5000)
                        if not (min_alt <= new_alt <= max_alt):
                            continue
                    
                    neighbors.append(Waypoint(new_lat, new_lon, new_alt))
        
        return neighbors
    
    def _compute_edge_cost(
        self,
        from_wp: Waypoint,
        to_wp: Waypoint,
        time_iso: str
    ) -> float:
        """
        Compute cost of edge between two waypoints.
        
        Combines distance, energy, risk, and time costs.
        
        Returns:
            Edge cost, or -1 if infeasible
        """
        try:
            # Query perception layer
            # Query perception (supports local, HTTP, or fake mode)
            result = self.perception.query(
                to_wp.lat, to_wp.lon, to_wp.alt_m, time_iso or "1970-01-01T00:00:00Z"
            )
            
            # Check feasibility
            if not result.feasible:
                return -1.0
            
            # Compute distance
            distance_km = self._haversine_km(
                from_wp.lat, from_wp.lon, to_wp.lat, to_wp.lon
            )
            
            # Multi-objective cost
            distance_cost = distance_km
            energy_cost = result.energy_cost_kwh_per_km * distance_km
            risk_cost = result.risk_score * distance_km
            
            # Assume constant cruise speed for time cost
            cruise_speed_ms = 35.0  # m/s
            time_cost = (distance_km * 1000) / cruise_speed_ms / 3600  # hours
            
            # Weighted combination
            total_cost = (
                self.cost_weights.get("distance", 0.3) * distance_cost +
                self.cost_weights.get("energy", 0.3) * energy_cost +
                self.cost_weights.get("risk", 0.3) * risk_cost +
                self.cost_weights.get("time", 0.1) * time_cost
            )
            
            return total_cost
            
        except Exception as e:
            # If perception query fails, return high cost
            return 10.0
    
    def _heuristic(self, waypoint: Waypoint, goal: Waypoint) -> float:
        """
        Heuristic function for A* (admissible lower bound on cost).
        
        Uses straight-line distance as heuristic.
        """
        distance_km = self._haversine_km(
            waypoint.lat, waypoint.lon, goal.lat, goal.lon
        )
        
        # Scale by minimum cost weight (optimistic estimate)
        min_cost_per_km = min(self.cost_weights.values())
        
        return distance_km * min_cost_per_km
    
    def _is_goal(self, waypoint: Waypoint, goal: Waypoint, tolerance_deg: float = 0.01) -> bool:
        """Check if waypoint is close enough to goal."""
        distance_km = self._haversine_km(
            waypoint.lat, waypoint.lon, goal.lat, goal.lon
        )
        return distance_km < (tolerance_deg * 111)  # ~1km tolerance
    
    def _reconstruct_path(self, node: AStarNode) -> List[Waypoint]:
        """Reconstruct path from goal node back to start."""
        path = []
        current = node
        
        while current is not None:
            path.append(current.waypoint)
            current = current.parent
        
        path.reverse()
        return path
    
    def _straight_line_fallback(self, start: Waypoint, goal: Waypoint) -> List[Waypoint]:
        """Fallback to straight line if A* fails."""
        num_points = 25
        lats = np.linspace(start.lat, goal.lat, num_points)
        lons = np.linspace(start.lon, goal.lon, num_points)
        alts = np.full(num_points, start.alt_m)
        
        return [Waypoint(float(lat), float(lon), float(alt)) 
                for lat, lon, alt in zip(lats, lons, alts)]
    
    def _haversine_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute Haversine distance in km."""
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

    def _smooth(self, waypoints: List[Waypoint]) -> List[Waypoint]:
        window = int(self.config.get("routing.smoothing_window", 5))
        if window < 3 or len(waypoints) < window:
            return waypoints
        arr = np.array([[w.lat, w.lon, w.alt_m] for w in waypoints])
        kernel = np.ones(window) / window
        smoothed = np.vstack([
            np.convolve(arr[:, i], kernel, mode="same") for i in range(3)
        ]).T
        return [Waypoint(float(a), float(b), float(c)) for a, b, c in smoothed]

    def compute_route_cost(self, route: List[Waypoint], time_iso: str) -> float:
        if len(route) < 2:
            return 0.0
        weights = self.config.get("routing.objective_weights", {"time": 0.4, "energy": 0.3, "risk": 0.3})
        cruise_speed_mps = float(self.config.get("energy.cruise_speed_mps", 35.0))

        total_distance_km = 0.0
        total_energy_kwh = 0.0
        total_risk = 0.0

        for a, b in zip(route[:-1], route[1:]):
            result = self.perception.query(a.lat, a.lon, a.alt_m, time_iso)
            # distance
            d_km = self._haversine_km(a.lat, a.lon, b.lat, b.lon)
            total_distance_km += d_km
            # energy
            e_kwh_per_km = result.energy_cost_kwh_per_km
            total_energy_kwh += e_kwh_per_km * d_km
            # risk
            total_risk += result.risk_score * d_km

        # time objective (seconds)
        time_s = (total_distance_km * 1000.0) / max(1e-6, cruise_speed_mps)

        # Normalize with simple scalers to keep magnitudes comparable
        time_norm = time_s / 3600.0  # hours
        energy_norm = total_energy_kwh
        risk_norm = total_risk / max(1e-6, total_distance_km)  # average risk

        return (
            weights.get("time", 0.4) * time_norm
            + weights.get("energy", 0.3) * energy_norm
            + weights.get("risk", 0.3) * risk_norm
        )


