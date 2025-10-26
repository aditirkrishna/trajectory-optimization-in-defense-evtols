"""
Patrol Route and Coverage Modeling

Models patrol aircraft/vehicle coverage and encounter probability.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class PatrolWaypoint:
    """Waypoint along a patrol route"""
    lat: float
    lon: float
    alt_m: float
    time_offset_min: float  # Time from patrol start


@dataclass
class PatrolRoute:
    """Patrol route definition"""
    id: str
    waypoints: List[PatrolWaypoint]
    patrol_speed_ms: float = 50.0  # Patrol speed in m/s
    detection_range_km: float = 10.0  # Visual/sensor detection range
    patrol_frequency_per_day: int = 4  # Number of patrols per day
    patrol_duration_min: float = 120.0  # Duration of each patrol
    active: bool = True


@dataclass
class EncounterResult:
    """Patrol encounter analysis result"""
    encounter_probable: bool
    encounter_probability: float
    closest_approach_km: float
    time_to_encounter_min: Optional[float]
    patrol_position: Optional[Tuple[float, float, float]]  # (lat, lon, alt)


class PatrolModel:
    """
    Patrol coverage and encounter probability model.
    
    Estimates likelihood of encountering patrol assets.
    """
    
    def __init__(self):
        """Initialize patrol model."""
        self.patrols: List[PatrolRoute] = []
        
    def add_patrol(self, patrol: PatrolRoute):
        """Add a patrol route to the model."""
        self.patrols.append(patrol)
        logger.info(f"Added patrol route {patrol.id} with {len(patrol.waypoints)} waypoints")
    
    def compute_encounter_probability(
        self,
        target_lat: float,
        target_lon: float,
        target_alt_m: float,
        time_of_day_hour: float
    ) -> Dict[str, any]:
        """
        Compute encounter probability with all active patrols.
        
        Args:
            target_lat: Target latitude
            target_lon: Target longitude
            target_alt_m: Target altitude
            time_of_day_hour: Time of day (0-24)
            
        Returns:
            Dictionary with encounter analysis
        """
        encounters = []
        
        for patrol in self.patrols:
            if not patrol.active:
                continue
            
            result = self._analyze_patrol_encounter(
                patrol, target_lat, target_lon, target_alt_m, time_of_day_hour
            )
            encounters.append((patrol.id, result))
        
        # Overall encounter probability
        # P(encounter with any patrol) = 1 - Π(1 - P_i)
        overall_prob = 1.0
        for _, result in encounters:
            overall_prob *= (1.0 - result.encounter_probability)
        overall_prob = 1.0 - overall_prob
        
        return {
            "overall_encounter_prob": overall_prob,
            "num_active_patrols": len(encounters),
            "high_risk_patrols": sum(1 for _, r in encounters if r.encounter_probable),
            "individual_encounters": dict(encounters)
        }
    
    def _analyze_patrol_encounter(
        self,
        patrol: PatrolRoute,
        target_lat: float,
        target_lon: float,
        target_alt_m: float,
        time_of_day_hour: float
    ) -> EncounterResult:
        """Analyze encounter probability with a specific patrol."""
        # Find closest point on patrol route
        min_distance_km = float('inf')
        closest_position = None
        
        for waypoint in patrol.waypoints:
            distance_km = self._haversine_km(
                target_lat, target_lon, waypoint.lat, waypoint.lon
            )
            
            # Account for altitude difference
            alt_diff_km = abs(target_alt_m - waypoint.alt_m) / 1000.0
            total_distance_km = np.sqrt(distance_km**2 + alt_diff_km**2)
            
            if total_distance_km < min_distance_km:
                min_distance_km = total_distance_km
                closest_position = (waypoint.lat, waypoint.lon, waypoint.alt_m)
        
        # Compute encounter probability based on distance
        if min_distance_km <= patrol.detection_range_km:
            # Within detection range
            # Probability decreases with distance
            proximity_factor = 1.0 - (min_distance_km / patrol.detection_range_km)
            
            # Temporal factor (patrol frequency)
            temporal_factor = patrol.patrol_frequency_per_day / 24.0  # Fraction of time patrolling
            
            encounter_prob = proximity_factor * temporal_factor
            encounter_probable = (encounter_prob > 0.3)
        else:
            encounter_prob = 0.0
            encounter_probable = False
        
        return EncounterResult(
            encounter_probable=encounter_probable,
            encounter_probability=encounter_prob,
            closest_approach_km=min_distance_km,
            time_to_encounter_min=None,  # Requires temporal prediction
            patrol_position=closest_position
        )
    
    def _haversine_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute Haversine distance in km."""
        R = 6371.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c


@dataclass
class EWZone:
    """Electronic Warfare Zone"""
    id: str
    lat: float
    lon: float
    radius_km: float
    signal_strength_db: float  # Signal strength at center
    frequency_range_ghz: Tuple[float, float]
    zone_type: str  # "jamming", "spoofing", "intercept"
    active: bool = True
    time_varying: bool = False  # Whether zone varies over time


class ElectronicWarfareModel:
    """
    Electronic Warfare zone modeling.
    
    Models GPS jamming, communications jamming, and signal interception zones.
    """
    
    def __init__(self):
        """Initialize EW model."""
        self.zones: List[EWZone] = []
        
    def add_zone(self, zone: EWZone):
        """Add an EW zone."""
        self.zones.append(zone)
        logger.info(f"Added EW zone {zone.id} ({zone.zone_type}) with radius {zone.radius_km} km")
    
    def assess_ew_impact(
        self,
        lat: float,
        lon: float,
        operating_freq_ghz: float = 1.575  # GPS L1 frequency
    ) -> Dict[str, any]:
        """
        Assess electronic warfare impact at a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            operating_freq_ghz: Operating frequency in GHz
            
        Returns:
            Dictionary with EW impact assessment
        """
        impacts = []
        total_jamming_power_db = -999.0  # dBm
        gps_degraded = False
        comms_degraded = False
        
        for zone in self.zones:
            if not zone.active:
                continue
            
            # Check if frequency is affected
            if not (zone.frequency_range_ghz[0] <= operating_freq_ghz <= zone.frequency_range_ghz[1]):
                continue
            
            # Compute distance to zone center
            distance_km = self._haversine_km(lat, lon, zone.lat, zone.lon)
            
            # Check if within zone
            if distance_km <= zone.radius_km:
                # Signal strength decreases with distance (free-space path loss)
                # FSPL(dB) = 20*log10(d) + 20*log10(f) + 32.45 (d in km, f in MHz)
                path_loss_db = 20*np.log10(max(distance_km, 0.001)) + 20*np.log10(operating_freq_ghz*1000) + 32.45
                
                # Effective signal strength at location
                effective_strength_db = zone.signal_strength_db - path_loss_db
                
                # Update max jamming power
                total_jamming_power_db = max(total_jamming_power_db, effective_strength_db)
                
                # Check impact type
                if zone.zone_type == "jamming":
                    if operating_freq_ghz < 2.0:  # GPS frequencies
                        gps_degraded = True
                    else:  # Communications frequencies
                        comms_degraded = True
                
                impacts.append({
                    "zone_id": zone.id,
                    "zone_type": zone.zone_type,
                    "distance_km": distance_km,
                    "effective_strength_db": effective_strength_db,
                    "affected": (effective_strength_db > -100.0)  # Threshold for meaningful impact
                })
        
        return {
            "gps_degraded": gps_degraded,
            "comms_degraded": comms_degraded,
            "jamming_power_db": total_jamming_power_db,
            "num_affecting_zones": len(impacts),
            "zone_impacts": impacts,
            "gps_reliability": 0.0 if gps_degraded else 1.0,
            "comms_reliability": 0.3 if comms_degraded else 1.0
        }
    
    def _haversine_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compute Haversine distance in km."""
        R = 6371.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create patrol route
    waypoints = [
        PatrolWaypoint(lat=13.0, lon=77.5, alt_m=500, time_offset_min=0),
        PatrolWaypoint(lat=13.05, lon=77.55, alt_m=500, time_offset_min=10),
        PatrolWaypoint(lat=13.1, lon=77.6, alt_m=500, time_offset_min=20),
        PatrolWaypoint(lat=13.05, lon=77.65, alt_m=500, time_offset_min=30),
        PatrolWaypoint(lat=13.0, lon=77.6, alt_m=500, time_offset_min=40),
    ]
    
    patrol = PatrolRoute(
        id="PATROL_001",
        waypoints=waypoints,
        patrol_speed_ms=50.0,
        detection_range_km=10.0,
        patrol_frequency_per_day=4
    )
    
    # Create patrol model
    patrol_model = PatrolModel()
    patrol_model.add_patrol(patrol)
    
    # Assess encounter
    result = patrol_model.compute_encounter_probability(
        target_lat=13.02,
        target_lon=77.58,
        target_alt_m=300.0,
        time_of_day_hour=12.0
    )
    
    print("\nPatrol Encounter Analysis:")
    print(f"  Overall Encounter Probability: {result['overall_encounter_prob']:.3f}")
    print(f"  High Risk Patrols: {result['high_risk_patrols']}")
    
    # Create EW zones
    ew_model = ElectronicWarfareModel()
    
    ew_zone = EWZone(
        id="EW_001",
        lat=13.05,
        lon=77.60,
        radius_km=15.0,
        signal_strength_db=60.0,  # dBm at center
        frequency_range_ghz=(1.5, 1.6),  # GPS L1
        zone_type="jamming"
    )
    
    ew_model.add_zone(ew_zone)
    
    # Assess EW impact
    ew_result = ew_model.assess_ew_impact(
        lat=13.02,
        lon=77.58,
        operating_freq_ghz=1.575  # GPS L1
    )
    
    print("\nEW Impact Assessment:")
    print(f"  GPS Degraded: {ew_result['gps_degraded']}")
    print(f"  Comms Degraded: {ew_result['comms_degraded']}")
    print(f"  GPS Reliability: {ew_result['gps_reliability']:.2f}")
    print(f"  Jamming Power: {ew_result['jamming_power_db']:.1f} dBm")

