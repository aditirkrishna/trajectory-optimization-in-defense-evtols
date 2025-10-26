"""
Core utilities and shared components for the eVTOL trajectory optimization system.

This module contains common data types, configuration management, and utilities
that are shared across all layers of the system.
"""

from .types import *
from .config import *
from .utils import *

__version__ = "0.1.0"
__all__ = [
    # Re-export everything from submodules
]
