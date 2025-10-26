from __future__ import annotations

from typing import List
import math

from ..config import PlanningConfig
from ..routing.planner import Waypoint
from ..serving.perception_client import PerceptionClient, PerceptionQuery


class RiskManager:
    """Risk evaluation and emergency planning placeholder."""

    def __init__(self, config: PlanningConfig) -> None:
        self.config = config
        self.perception = PerceptionClient(config)

    def evaluate_route_risk(self, route: List[Waypoint], time_iso: str | None = None) -> float:
        # Distance-weighted average risk along the route using perception
        if len(route) < 2:
            return 0.0
        total_risk_distance = 0.0
        total_distance_km = 0.0
        time_iso = time_iso or "1970-01-01T00:00:00Z"
        for a, b in zip(route[:-1], route[1:]):
            result = self.perception.query(a.lat, a.lon, a.alt_m, time_iso)
            risk = result.risk_score
            # approximate distance with haversine
            d_km = self._haversine_km(a.lat, a.lon, b.lat, b.lon)
            total_risk_distance += risk * d_km
            total_distance_km += d_km
        if total_distance_km <= 0.0:
            return 0.0
        avg_risk = total_risk_distance / total_distance_km
        return min(avg_risk, float(self.config.get("risk.max_risk_score", 0.7)))
    
    def _haversine_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate Haversine distance in kilometers"""
        R = 6371.0  # Earth radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

    def propose_contingency(self, route: List[Waypoint]) -> List[Waypoint]:
        # Placeholder: return last half as emergency divert
        return route[len(route) // 2 :]


