"""
Serving Module

This module provides API and real-time serving capabilities for the vehicle layer,
including REST API endpoints and real-time simulation interfaces.
"""

from .api import VehicleAPI

__all__ = [
    "VehicleAPI",
]

