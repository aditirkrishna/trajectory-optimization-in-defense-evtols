from __future__ import annotations

from typing import List

from ..config import PlanningConfig
from ..routing.planner import Waypoint
from ..serving.perception_client import PerceptionClient


class EnergyOptimizer:
    """Simple energy estimator and power management placeholder."""

    def __init__(self, config: PlanningConfig) -> None:
        self.config = config
        self.perception = PerceptionClient(config)

    def estimate_route_energy(self, route: List[Waypoint], time_iso: str | None = None) -> float:
        # Distance-weighted energy from perception with reserve enforcement
        if len(route) < 2:
            return 0.0
        capacity_kwh = float(self.config.get("energy.battery_capacity_kwh", 120.0))
        reserve_frac = float(self.config.get("energy.reserve_fraction", 0.15))
        usable_kwh = capacity_kwh * (1.0 - reserve_frac)
        total_energy_kwh = 0.0
        for a, b in zip(route[:-1], route[1:]):
            d_km = _haversine_km(a.lat, a.lon, b.lat, b.lon)
            # Query perception for energy cost (fallback to default if unavailable)
            try:
                from ..serving.perception_client import PerceptionQuery
                query = PerceptionQuery(a.lat, a.lon, a.alt_m, time_iso or "1970-01-01T00:00:00Z")
                result = self.perception.query(query)
                e_per_km = result.energy_cost_kwh_per_km
            except:
                e_per_km = 1.0  # Default energy cost
            total_energy_kwh += e_per_km * d_km
            if total_energy_kwh >= usable_kwh:
                return usable_kwh
        return min(total_energy_kwh, usable_kwh)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import radians, sin, cos, sqrt, atan2

    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


