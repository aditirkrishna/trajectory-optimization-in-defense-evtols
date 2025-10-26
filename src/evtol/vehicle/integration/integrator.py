"""
Base Integrator Class

This module provides the base class for numerical integrators used in
vehicle dynamics simulation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

try:
    from ..vehicle_types import VehicleState
except ImportError:
    # Fallback for direct imports
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from vehicle_types import VehicleState


class Integrator(ABC):
    """
    Abstract base class for numerical integrators.
    
    This class defines the interface for numerical integration methods
    used to solve the vehicle dynamics differential equations.
    """
    
    def __init__(self):
        """Initialize integrator."""
        self.logger = logging.getLogger(__name__)
        self.method_name = "Base"
        self.order = 1
    
    @abstractmethod
    def integrate(self, state: VehicleState, state_derivative: Dict[str, Any], 
                 dt: float) -> VehicleState:
        """
        Integrate state using the specific integration method.
        
        Args:
            state: Current vehicle state
            state_derivative: Dictionary of state derivatives
            dt: Time step size
            
        Returns:
            Updated vehicle state
        """
        pass
    
    @abstractmethod
    def get_integration_error(self) -> float:
        """
        Get estimated integration error.
        
        Returns:
            Estimated error
        """
        pass
    
    @abstractmethod
    def is_stable(self, dt: float) -> bool:
        """
        Check if integration is stable for given time step.
        
        Args:
            dt: Time step size
            
        Returns:
            True if stable
        """
        pass
    
    def get_method_name(self) -> str:
        """Get integration method name."""
        return self.method_name
    
    def get_order(self) -> int:
        """Get integration method order."""
        return self.order
