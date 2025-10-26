"""
Coordinate utilities for the perception layer.

This module provides comprehensive coordinate handling functions including:
- CRS transformations between different coordinate systems
- Distance and bearing calculations
- UTM zone detection and management
- Coordinate validation and range checking
- Integration with the perception layer configuration
"""

import logging
from typing import Tuple, Optional, Union, List
import numpy as np
from pyproj import CRS, Transformer, Geod
from pyproj.exceptions import CRSError
import warnings

logger = logging.getLogger(__name__)


class CoordinateError(Exception):
    """Custom exception for coordinate-related errors."""
    pass


def transform_coordinates(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    z: Optional[Union[float, np.ndarray]] = None,
    src_crs: str = "EPSG:4326",
    dst_crs: str = "EPSG:32632",
    return_z: bool = False
) -> Union[Tuple[float, float], Tuple[float, float, float], 
           Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Transform coordinates between different coordinate reference systems.
    
    Args:
        x: X coordinates (longitude or easting)
        y: Y coordinates (latitude or northing)
        z: Optional Z coordinates (elevation)
        src_crs: Source CRS (default: WGS84)
        dst_crs: Destination CRS (default: UTM zone 32N)
        return_z: Whether to return Z coordinates
        
    Returns:
        Transformed coordinates (x, y) or (x, y, z)
        
    Raises:
        CoordinateError: If transformation fails
    """
    try:
        # Create transformer
        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        
        # Handle single point vs array
        if isinstance(x, (int, float)):
            if z is not None:
                x_out, y_out, z_out = transformer.transform(x, y, z)
                return (x_out, y_out, z_out) if return_z else (x_out, y_out)
            else:
                x_out, y_out = transformer.transform(x, y)
                return (x_out, y_out)
        else:
            # Handle numpy arrays
            if z is not None:
                x_out, y_out, z_out = transformer.transform(x, y, z)
                return (x_out, y_out, z_out) if return_z else (x_out, y_out)
            else:
                x_out, y_out = transformer.transform(x, y)
                return (x_out, y_out)
                
    except (CRSError, Exception) as e:
        logger.error(f"Coordinate transformation failed: {e}")
        raise CoordinateError(f"Failed to transform coordinates: {e}")


def calculate_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    method: str = "geodesic"
) -> float:
    """
    Calculate distance between two points on Earth's surface.
    
    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)
        method: Distance calculation method ("geodesic", "haversine", "euclidean")
        
    Returns:
        Distance in meters
        
    Raises:
        CoordinateError: If calculation fails
    """
    try:
        if method == "geodesic":
            # Most accurate for large distances
            geod = Geod(ellps='WGS84')
            distance = geod.inv(lon1, lat1, lon2, lat2)[2]  # Distance in meters
            return distance
            
        elif method == "haversine":
            # Good approximation for medium distances
            R = 6371000  # Earth's radius in meters
            
            lat1_rad = np.radians(lat1)
            lat2_rad = np.radians(lat2)
            delta_lat = np.radians(lat2 - lat1)
            delta_lon = np.radians(lon2 - lon1)
            
            a = (np.sin(delta_lat / 2) ** 2 + 
                 np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2)
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            distance = R * c
            return distance
            
        elif method == "euclidean":
            # Simple approximation for small distances
            R = 6371000  # Earth's radius in meters
            lat1_rad = np.radians(lat1)
            lat2_rad = np.radians(lat2)
            
            x1 = R * np.cos(lat1_rad) * np.cos(np.radians(lon1))
            y1 = R * np.cos(lat1_rad) * np.sin(np.radians(lon1))
            z1 = R * np.sin(lat1_rad)
            
            x2 = R * np.cos(lat2_rad) * np.cos(np.radians(lon2))
            y2 = R * np.cos(lat2_rad) * np.sin(np.radians(lon2))
            z2 = R * np.sin(lat2_rad)
            
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
            return distance
            
        else:
            raise ValueError(f"Unknown distance method: {method}")
            
    except Exception as e:
        logger.error(f"Distance calculation failed: {e}")
        raise CoordinateError(f"Failed to calculate distance: {e}")


def calculate_bearing(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    Calculate bearing (azimuth) from point 1 to point 2.
    
    Args:
        lat1, lon1: Starting point coordinates (degrees)
        lat2, lon2: Ending point coordinates (degrees)
        
    Returns:
        Bearing in degrees (0-360, where 0 is North)
        
    Raises:
        CoordinateError: If calculation fails
    """
    try:
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lon = np.radians(lon2 - lon1)
        
        y = np.sin(delta_lon) * np.cos(lat2_rad)
        x = (np.cos(lat1_rad) * np.sin(lat2_rad) - 
             np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(delta_lon))
        
        bearing = np.degrees(np.arctan2(y, x))
        bearing = (bearing + 360) % 360  # Normalize to 0-360
        
        return bearing
        
    except Exception as e:
        logger.error(f"Bearing calculation failed: {e}")
        raise CoordinateError(f"Failed to calculate bearing: {e}")


def get_utm_zone(lon: float, lat: float) -> str:
    """
    Determine the appropriate UTM zone for given coordinates.
    
    Args:
        lon: Longitude (degrees)
        lat: Latitude (degrees)
        
    Returns:
        UTM zone string (e.g., "EPSG:32632" for UTM zone 32N)
        
    Raises:
        CoordinateError: If zone determination fails
    """
    try:
        # Calculate UTM zone number
        zone_number = int((lon + 180) / 6) + 1
        
        # Determine hemisphere
        if lat >= 0:
            hemisphere = "N"
            epsg_code = 32600 + zone_number
        else:
            hemisphere = "S"
            epsg_code = 32700 + zone_number
            
        utm_crs = f"EPSG:{epsg_code}"
        
        logger.debug(f"UTM zone determined: {utm_crs} for coordinates ({lon}, {lat})")
        return utm_crs
        
    except Exception as e:
        logger.error(f"UTM zone determination failed: {e}")
        raise CoordinateError(f"Failed to determine UTM zone: {e}")


def validate_coordinates(
    lat: float, lon: float, 
    alt: Optional[float] = None,
    check_range: bool = True
) -> bool:
    """
    Validate coordinate values and ranges.
    
    Args:
        lat: Latitude (degrees)
        lon: Longitude (degrees)
        alt: Optional altitude (meters)
        check_range: Whether to check coordinate ranges
        
    Returns:
        True if coordinates are valid
        
    Raises:
        CoordinateError: If coordinates are invalid
    """
    try:
        # Check for NaN or infinite values
        if not (np.isfinite(lat) and np.isfinite(lon)):
            raise CoordinateError("Coordinates contain NaN or infinite values")
            
        if alt is not None and not np.isfinite(alt):
            raise CoordinateError("Altitude contains NaN or infinite values")
            
        if not check_range:
            return True
            
        # Check latitude range (-90 to 90)
        if not (-90 <= lat <= 90):
            raise CoordinateError(f"Latitude {lat} is outside valid range [-90, 90]")
            
        # Check longitude range (-180 to 180)
        if not (-180 <= lon <= 180):
            raise CoordinateError(f"Longitude {lon} is outside valid range [-180, 180]")
            
        # Check altitude range (reasonable bounds)
        if alt is not None:
            if not (-1000 <= alt <= 50000):  # -1km to 50km
                logger.warning(f"Altitude {alt} is outside typical range [-1000, 50000] meters")
                
        return True
        
    except Exception as e:
        logger.error(f"Coordinate validation failed: {e}")
        raise CoordinateError(f"Invalid coordinates: {e}")


def create_bounding_box(
    center_lat: float, center_lon: float,
    width_m: float, height_m: float
) -> Tuple[float, float, float, float]:
    """
    Create a bounding box around a center point.
    
    Args:
        center_lat, center_lon: Center point coordinates (degrees)
        width_m: Width of bounding box (meters)
        height_m: Height of bounding box (meters)
        
    Returns:
        Bounding box as (min_lat, min_lon, max_lat, max_lon)
        
    Raises:
        CoordinateError: If bounding box creation fails
    """
    try:
        # Convert meters to approximate degrees
        # 1 degree latitude ≈ 111,000 meters
        # 1 degree longitude ≈ 111,000 * cos(latitude) meters
        
        lat_offset = height_m / 111000.0
        lon_offset = width_m / (111000.0 * np.cos(np.radians(center_lat)))
        
        min_lat = center_lat - lat_offset / 2
        max_lat = center_lat + lat_offset / 2
        min_lon = center_lon - lon_offset / 2
        max_lon = center_lon + lon_offset / 2
        
        # Validate the resulting coordinates
        validate_coordinates(min_lat, min_lon)
        validate_coordinates(max_lat, max_lon)
        
        return (min_lat, min_lon, max_lat, max_lon)
        
    except Exception as e:
        logger.error(f"Bounding box creation failed: {e}")
        raise CoordinateError(f"Failed to create bounding box: {e}")


def interpolate_coordinates(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    num_points: int
) -> Tuple[List[float], List[float]]:
    """
    Interpolate coordinates along a line between two points.
    
    Args:
        lat1, lon1: Starting point coordinates (degrees)
        lat2, lon2: Ending point coordinates (degrees)
        num_points: Number of points to interpolate (including endpoints)
        
    Returns:
        Lists of interpolated latitudes and longitudes
        
    Raises:
        CoordinateError: If interpolation fails
    """
    try:
        if num_points < 2:
            raise ValueError("num_points must be at least 2")
            
        # Linear interpolation in lat/lon space
        t = np.linspace(0, 1, num_points)
        
        lats = lat1 + t * (lat2 - lat1)
        lons = lon1 + t * (lon2 - lon1)
        
        # Validate interpolated coordinates
        for lat, lon in zip(lats, lons):
            validate_coordinates(lat, lon, check_range=False)
            
        return lats.tolist(), lons.tolist()
        
    except Exception as e:
        logger.error(f"Coordinate interpolation failed: {e}")
        raise CoordinateError(f"Failed to interpolate coordinates: {e}")


def get_crs_info(crs_string: str) -> dict:
    """
    Get information about a coordinate reference system.
    
    Args:
        crs_string: CRS string (e.g., "EPSG:4326")
        
    Returns:
        Dictionary with CRS information
        
    Raises:
        CoordinateError: If CRS information retrieval fails
    """
    try:
        crs = CRS(crs_string)
        
        info = {
            'name': crs.name,
            'auth_name': getattr(crs, 'auth_name', None),
            'auth_code': getattr(crs, 'auth_code', None),
            'is_geographic': crs.is_geographic,
            'is_projected': crs.is_projected,
            'units': str(crs.axis_info[0].unit_name) if crs.axis_info else None,
            'bounds': getattr(crs, 'bounds', None)
        }
        
        return info
        
    except Exception as e:
        logger.error(f"CRS info retrieval failed: {e}")
        raise CoordinateError(f"Failed to get CRS info: {e}")


def convert_units(
    value: float,
    from_unit: str,
    to_unit: str
) -> float:
    """
    Convert between common coordinate-related units.
    
    Args:
        value: Value to convert
        from_unit: Source unit ("m", "km", "ft", "deg", "rad")
        to_unit: Target unit ("m", "km", "ft", "deg", "rad")
        
    Returns:
        Converted value
        
    Raises:
        CoordinateError: If conversion fails
    """
    try:
        # Define conversion factors
        conversions = {
            'm': 1.0,
            'km': 1000.0,
            'ft': 0.3048,
            'deg': np.pi / 180.0,  # to radians
            'rad': 1.0
        }
        
        if from_unit not in conversions or to_unit not in conversions:
            raise ValueError(f"Unsupported units: {from_unit} or {to_unit}")
            
        # Convert to base unit (meters for distance, radians for angles)
        if from_unit in ['m', 'km', 'ft']:
            # Distance units
            base_value = value * conversions[from_unit]
            return base_value / conversions[to_unit]
        else:
            # Angular units
            base_value = value * conversions[from_unit]
            return base_value / conversions[to_unit]
            
    except Exception as e:
        logger.error(f"Unit conversion failed: {e}")
        raise CoordinateError(f"Failed to convert units: {e}")


# Convenience functions for common operations
def wgs84_to_utm(lat: float, lon: float) -> Tuple[float, float]:
    """Convert WGS84 coordinates to UTM coordinates."""
    return transform_coordinates(lon, lat, src_crs="EPSG:4326", dst_crs=get_utm_zone(lon, lat))


def utm_to_wgs84(easting: float, northing: float, utm_zone: str) -> Tuple[float, float]:
    """Convert UTM coordinates to WGS84 coordinates."""
    return transform_coordinates(easting, northing, src_crs=utm_zone, dst_crs="EPSG:4326")


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance using Haversine formula."""
    return calculate_distance(lat1, lon1, lat2, lon2, method="haversine")


def geodesic_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance using geodesic formula."""
    return calculate_distance(lat1, lon1, lat2, lon2, method="geodesic")
