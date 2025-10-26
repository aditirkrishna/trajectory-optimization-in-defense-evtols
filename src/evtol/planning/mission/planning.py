from __future__ import annotations

from typing import Dict, List

from ..config import PlanningConfig
from ..routing.planner import Waypoint


class MissionPlanner:
    """Mission-level orchestration and sequencing placeholder."""

    def __init__(self, config: PlanningConfig) -> None:
        self.config = config

    def build_single_route_mission(self, route: List[Waypoint]) -> Dict[str, object]:
        return {
            "type": "single_route",
            "num_waypoints": len(route),
            "waypoints": [w.__dict__ for w in route],
        }


