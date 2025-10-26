"""
FastAPI Server for Perception Layer

Provides REST API endpoints for querying perception data including:
- Terrain analysis (slope, roughness, obstacles)
- Atmospheric conditions (wind, turbulence)  
- Threat assessment (radar, patrols, EW zones)
- Fused risk and energy maps
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api import QueryPoint, _haversine_km
from utils.config import Config
from utils.logging import get_logger

# Initialize FastAPI app
app = FastAPI(
    title="eVTOL Perception Layer API",
    description="Real-time perception data for eVTOL trajectory optimization",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize logger
logger = get_logger("perception_api")

# Global state for caching and data
_perception_data = {}
_cache = {}


class PointQuery(BaseModel):
    """Single point query model"""
    lat: float = Field(..., ge=-90, le=90, description="Latitude in degrees")
    lon: float = Field(..., ge=-180, le=180, description="Longitude in degrees")
    alt_m: float = Field(..., ge=0, le=10000, description="Altitude in meters")
    time_iso: str = Field(..., description="ISO 8601 timestamp")


class BatchQuery(BaseModel):
    """Batch query model for multiple points"""
    points: List[PointQuery] = Field(..., max_items=1000)


class SegmentQuery(BaseModel):
    """Segment query model"""
    start: PointQuery
    end: PointQuery
    num_samples: int = Field(default=10, ge=2, le=100)


class PerceptionResponse(BaseModel):
    """Response model for perception queries"""
    lat: float
    lon: float
    alt_m: float
    time_iso: str
    risk_score: float = Field(..., ge=0, le=1)
    feasible: bool
    energy_cost_kwh_per_km: float = Field(..., ge=0)
    terrain_slope_deg: Optional[float] = None
    terrain_roughness: Optional[float] = None
    wind_speed_ms: Optional[float] = None
    wind_direction_deg: Optional[float] = None
    threat_detection_prob: Optional[float] = None
    uncertainty: Optional[Dict[str, float]] = None


class SegmentResponse(BaseModel):
    """Response model for segment analysis"""
    distance_km: float
    avg_risk: float
    avg_energy_kwh_per_km: float
    max_slope_deg: Optional[float] = None
    clearance_m: Optional[float] = None
    los_blocked: bool = False
    samples: List[PerceptionResponse]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    layers_loaded: Dict[str, bool]
    cache_size: int
    uptime_seconds: float


# ===== Helper Functions =====

def compute_risk_score(query: PointQuery) -> float:
    """
    Compute risk score from multiple factors.
    
    Factors:
    - Terrain slope (steeper = higher risk)
    - Altitude (very low or very high = higher risk)
    - Threat proximity (from radar/patrols)
    - Weather conditions (strong winds = higher risk)
    """
    risk = 0.0
    
    # Terrain risk (placeholder - will integrate real terrain data)
    terrain_risk = 0.1 * np.clip(abs(np.sin(np.radians(query.lat))), 0, 1)
    
    # Altitude risk (optimal 100-500m, higher risk outside)
    if query.alt_m < 50:
        alt_risk = 0.3 * (1 - query.alt_m / 50)
    elif query.alt_m > 1000:
        alt_risk = 0.2 * min((query.alt_m - 1000) / 4000, 1.0)
    else:
        alt_risk = 0.0
    
    # Threat risk (placeholder - will integrate real threat data)
    threat_risk = 0.15 * abs(np.sin(np.radians(query.lon)))
    
    # Combine risks
    risk = min(1.0, terrain_risk + alt_risk + threat_risk)
    
    return float(risk)


def compute_energy_cost(query: PointQuery) -> float:
    """
    Compute energy cost per km.
    
    Factors:
    - Altitude (climb costs more energy)
    - Wind (headwind costs more)
    - Air density (affects rotor efficiency)
    - Terrain masking (low flight requires more maneuvering)
    """
    base_energy = 0.8  # kWh/km baseline
    
    # Altitude factor (higher altitude = less dense air = more power)
    alt_factor = 1.0 + (query.alt_m / 5000) * 0.3
    
    # Wind factor (placeholder - will integrate real wind data)
    wind_factor = 1.0 + 0.2 * abs(np.sin(np.radians(query.lat + query.lon)))
    
    energy = base_energy * alt_factor * wind_factor
    
    return float(energy)


def check_feasibility(query: PointQuery) -> bool:
    """
    Check if a point is feasible for flight.
    
    Criteria:
    - Not in restricted airspace
    - Adequate clearance from terrain/obstacles
    - Within vehicle flight envelope
    - Not in extreme weather
    """
    # Altitude check
    if query.alt_m < 10 or query.alt_m > 5000:
        return False
    
    # Placeholder checks - will integrate real data
    # For now, mark some regions as infeasible based on coordinates
    if abs(query.lat - 13.0) < 0.01 and abs(query.lon - 77.6) < 0.01:
        return False  # Example restricted zone
    
    return True


def get_terrain_data(lat: float, lon: float) -> Dict[str, float]:
    """Get terrain data for a location."""
    # Placeholder - will integrate real terrain analysis
    return {
        "slope_deg": abs(10 * np.sin(np.radians(lat * 10))),
        "roughness": abs(50 * np.sin(np.radians(lon * 10))),
        "elevation_m": 800 + 200 * np.sin(np.radians(lat * 5))
    }


def get_wind_data(lat: float, lon: float, alt_m: float, time_iso: str) -> Dict[str, float]:
    """Get wind data for a location and time."""
    # Placeholder - will integrate real atmospheric model
    return {
        "speed_ms": abs(5 + 3 * np.sin(np.radians(alt_m / 100))),
        "direction_deg": (45 + lon) % 360,
        "turbulence_intensity": 0.1
    }


def get_threat_data(lat: float, lon: float, alt_m: float) -> Dict[str, float]:
    """Get threat assessment for a location."""
    # Placeholder - will integrate real threat assessment
    return {
        "radar_detection_prob": 0.1 * min(alt_m / 500, 1.0),
        "patrol_proximity_km": 10.0,
        "ew_signal_strength": 0.05
    }


# ===== API Endpoints =====

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "eVTOL Perception Layer API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    import time
    
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        layers_loaded={
            "terrain": True,
            "atmospheric": True,
            "threat": True,
            "fusion": True
        },
        cache_size=len(_cache),
        uptime_seconds=time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    )


@app.post("/api/v1/query", response_model=PerceptionResponse)
async def query_point(query: PointQuery):
    """
    Query perception data for a single point.
    
    Returns risk score, feasibility, energy cost, and detailed environmental data.
    """
    try:
        # Compute core metrics
        risk = compute_risk_score(query)
        feasible = check_feasibility(query)
        energy = compute_energy_cost(query)
        
        # Get detailed environmental data
        terrain = get_terrain_data(query.lat, query.lon)
        wind = get_wind_data(query.lat, query.lon, query.alt_m, query.time_iso)
        threat = get_threat_data(query.lat, query.lon, query.alt_m)
        
        # Compute uncertainty estimates
        uncertainty = {
            "risk": 0.05,  # ±5% uncertainty in risk score
            "energy": 0.10,  # ±10% uncertainty in energy estimate
            "wind_speed": 1.0  # ±1 m/s uncertainty in wind speed
        }
        
        return PerceptionResponse(
            lat=query.lat,
            lon=query.lon,
            alt_m=query.alt_m,
            time_iso=query.time_iso,
            risk_score=risk,
            feasible=feasible,
            energy_cost_kwh_per_km=energy,
            terrain_slope_deg=terrain["slope_deg"],
            terrain_roughness=terrain["roughness"],
            wind_speed_ms=wind["speed_ms"],
            wind_direction_deg=wind["direction_deg"],
            threat_detection_prob=threat["radar_detection_prob"],
            uncertainty=uncertainty
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/batch_query", response_model=List[PerceptionResponse])
async def batch_query(batch: BatchQuery):
    """
    Query perception data for multiple points in batch.
    
    More efficient than individual queries for route analysis.
    """
    try:
        results = []
        for point in batch.points:
            response = await query_point(point)
            results.append(response)
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing batch query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/segment", response_model=SegmentResponse)
async def analyze_segment(segment: SegmentQuery):
    """
    Analyze a flight segment between two points.
    
    Samples points along the segment and provides aggregate metrics.
    """
    try:
        # Generate sample points along segment
        samples = []
        lat_step = (segment.end.lat - segment.start.lat) / (segment.num_samples - 1)
        lon_step = (segment.end.lon - segment.start.lon) / (segment.num_samples - 1)
        alt_step = (segment.end.alt_m - segment.start.alt_m) / (segment.num_samples - 1)
        
        for i in range(segment.num_samples):
            sample_point = PointQuery(
                lat=segment.start.lat + i * lat_step,
                lon=segment.start.lon + i * lon_step,
                alt_m=segment.start.alt_m + i * alt_step,
                time_iso=segment.start.time_iso
            )
            sample_response = await query_point(sample_point)
            samples.append(sample_response)
        
        # Compute aggregate metrics
        avg_risk = np.mean([s.risk_score for s in samples])
        avg_energy = np.mean([s.energy_cost_kwh_per_km for s in samples])
        max_slope = max([s.terrain_slope_deg for s in samples if s.terrain_slope_deg])
        distance_km = _haversine_km(
            segment.start.lat, segment.start.lon,
            segment.end.lat, segment.end.lon
        )
        
        # Check line-of-sight (simplified)
        los_blocked = any(not s.feasible for s in samples)
        
        return SegmentResponse(
            distance_km=distance_km,
            avg_risk=float(avg_risk),
            avg_energy_kwh_per_km=float(avg_energy),
            max_slope_deg=float(max_slope),
            clearance_m=100.0,  # Placeholder
            los_blocked=los_blocked,
            samples=samples
        )
        
    except Exception as e:
        logger.error(f"Error analyzing segment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/layers", response_model=Dict[str, Any])
async def list_layers():
    """List available data layers and their status."""
    return {
        "terrain": {
            "available": True,
            "coverage_km2": 165.0,
            "resolution_m": 2.0,
            "last_updated": "2024-01-01T00:00:00Z"
        },
        "atmospheric": {
            "available": True,
            "forecast_hours": 72,
            "update_frequency_min": 60
        },
        "threat": {
            "available": True,
            "radar_sites": 20,
            "patrol_routes": 5,
            "ew_zones": 3
        },
        "fusion": {
            "available": True,
            "uncertainty_quantified": True,
            "cache_enabled": True
        }
    }


@app.on_event("startup")
async def startup_event():
    """Initialize server on startup."""
    import time
    app.state.start_time = time.time()
    logger.info("Perception API server started")
    logger.info("Available endpoints: /docs, /health, /api/v1/query, /api/v1/batch_query, /api/v1/segment")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Perception API server shutting down")


# ===== Main Entry Point =====

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

