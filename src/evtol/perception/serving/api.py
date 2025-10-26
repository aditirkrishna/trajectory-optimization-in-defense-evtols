"""
Simple fused query API for perception outputs. Placeholder implementations
that can be swapped with real fused maps (risk, feasibility, energy costs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import math


@dataclass
class QueryPoint:
    lat: float
    lon: float
    alt_m: float
    time_iso: str


def risk_score(point: QueryPoint) -> float:
    """Return a risk score in [0, 1]. Placeholder based on latitude variance."""
    base = 0.1 + 0.1 * (abs(math.sin(math.radians(point.lat))) ** 0.5)
    return float(min(1.0, max(0.0, base)))


def feasible(point: QueryPoint) -> bool:
    """Return whether point is feasible to traverse. Placeholder always True."""
    return True


def energy_cost_kwh_per_km(point: QueryPoint) -> float:
    """Return energy cost per km at this point. Placeholder constant."""
    return 1.0


def summarize_segment(a: QueryPoint, b: QueryPoint) -> Dict[str, float]:
    """Summarize a segment with distance, average risk, and energy per km."""
    d_km = _haversine_km(a.lat, a.lon, b.lat, b.lon)
    avg_risk = 0.5 * (risk_score(a) + risk_score(b))
    e_kwh_per_km = 0.5 * (energy_cost_kwh_per_km(a) + energy_cost_kwh_per_km(b))
    return {"distance_km": d_km, "avg_risk": avg_risk, "energy_kwh_per_km": e_kwh_per_km}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


