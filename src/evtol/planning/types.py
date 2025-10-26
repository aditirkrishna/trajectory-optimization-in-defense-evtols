from __future__ import annotations

from typing import TypedDict


class RouteSummary(TypedDict):
    distance_km: float
    energy_kwh: float
    risk_score: float


