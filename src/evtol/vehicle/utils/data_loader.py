"""
Data Loader - CSV Data Loading and Processing

This module provides utilities for loading and processing vehicle data
from CSV files, including interpolation and validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator, interp1d

from .config import VehicleConfig


class DataLoader:
    """
    Utility class for loading and processing vehicle data from CSV files.
    
    This class provides methods for loading various types of vehicle data
    including performance curves, efficiency maps, and fault definitions.
    """
    
    def __init__(self, config: VehicleConfig):
        """
        Initialize data loader.
        
        Args:
            config: Vehicle configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Data cache
        self.data_cache = {}
        
        self.logger.info("Data loader initialized")
    
    def load_rotor_thrust_curves(self, filename: str = "rotor_thrust_curves_multi.csv") -> pd.DataFrame:
        """
        Load rotor thrust curves from CSV file.
        
        Args:
            filename: CSV filename
            
        Returns:
            DataFrame with rotor thrust data
        """
        cache_key = f"rotor_thrust_{filename}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        try:
            file_path = self.config.get_data_path(filename)
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_columns = ['rotor_diameter_m', 'altitude_m', 'temp_offset_C', 
                              'rpm', 'thrust_N', 'torque_Nm', 'efficiency_percent']
            
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in {filename}")
            
            # Cache the data
            self.data_cache[cache_key] = df
            
            self.logger.info(f"Loaded rotor thrust curves: {len(df)} data points")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load rotor thrust curves: {e}")
            raise
    
    def load_battery_specifications(self, filename: str = "battery_specs.csv") -> pd.DataFrame:
        """
        Load battery specifications from CSV file.
        
        Args:
            filename: CSV filename
            
        Returns:
            DataFrame with battery specifications
        """
        cache_key = f"battery_specs_{filename}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        try:
            file_path = self.config.get_data_path(filename)
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_columns = ['chemistry', 'temp_C', 'C_rate', 'capacity_Ah', 
                              'energy_Wh', 'specific_energy_Whkg', 'round_trip_efficiency_percent']
            
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in {filename}")
            
            # Cache the data
            self.data_cache[cache_key] = df
            
            self.logger.info(f"Loaded battery specifications: {len(df)} data points")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load battery specifications: {e}")
            raise
    
    def load_efficiency_maps(self, filename: str = "efficiency_maps.csv") -> pd.DataFrame:
        """
        Load efficiency maps from CSV file.
        
        Args:
            filename: CSV filename
            
        Returns:
            DataFrame with efficiency data
        """
        cache_key = f"efficiency_maps_{filename}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        try:
            file_path = self.config.get_data_path(filename)
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_columns = ['rotor_id', 'rpm', 'thrust_N', 'power_W', 'efficiency_percent']
            
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in {filename}")
            
            # Cache the data
            self.data_cache[cache_key] = df
            
            self.logger.info(f"Loaded efficiency maps: {len(df)} data points")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load efficiency maps: {e}")
            raise
    
    def load_actuator_faults(self, filename: str = "actuator_faults.csv") -> pd.DataFrame:
        """
        Load actuator fault definitions from CSV file.
        
        Args:
            filename: CSV filename
            
        Returns:
            DataFrame with fault definitions
        """
        cache_key = f"actuator_faults_{filename}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        try:
            file_path = self.config.get_data_path(filename)
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_columns = ['actuator_id', 'fault_type', 'start_time_s', 
                              'end_time_s', 'severity', 'notes']
            
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in {filename}")
            
            # Cache the data
            self.data_cache[cache_key] = df
            
            self.logger.info(f"Loaded actuator faults: {len(df)} fault definitions")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load actuator faults: {e}")
            raise
    
    def load_mass_inertia(self, filename: str = "mass_inertia.csv") -> pd.DataFrame:
        """
        Load mass and inertia data from CSV file.
        
        Args:
            filename: CSV filename
            
        Returns:
            DataFrame with mass and inertia data
        """
        cache_key = f"mass_inertia_{filename}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        try:
            file_path = self.config.get_data_path(filename)
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_columns = ['parameter', 'value', 'unit', 'source']
            
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in {filename}")
            
            # Cache the data
            self.data_cache[cache_key] = df
            
            self.logger.info(f"Loaded mass and inertia data: {len(df)} parameters")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load mass and inertia data: {e}")
            raise
    
    def load_limits(self, filename: str = "limits.csv") -> pd.DataFrame:
        """
        Load flight envelope limits from CSV file.
        
        Args:
            filename: CSV filename
            
        Returns:
            DataFrame with flight limits
        """
        cache_key = f"limits_{filename}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        try:
            file_path = self.config.get_data_path(filename)
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_columns = ['payload_kg', 'altitude_max_m', 'speed_max_mps', 
                              'climb_rate_mps', 'endurance_min']
            
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in {filename}")
            
            # Cache the data
            self.data_cache[cache_key] = df
            
            self.logger.info(f"Loaded flight limits: {len(df)} limit sets")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load flight limits: {e}")
            raise
    
    def create_rotor_interpolator(self, rotor_diameter: float, 
                                filename: str = "rotor_thrust_curves_multi.csv") -> RegularGridInterpolator:
        """
        Create interpolator for rotor thrust curves.
        
        Args:
            rotor_diameter: Rotor diameter in meters
            filename: CSV filename
            
        Returns:
            RegularGridInterpolator for thrust/torque lookup
        """
        df = self.load_rotor_thrust_curves(filename)
        
        # Filter for specific rotor diameter
        rotor_data = df[df['rotor_diameter_m'] == rotor_diameter].copy()
        
        if rotor_data.empty:
            raise ValueError(f"No data found for rotor diameter {rotor_diameter}m")
        
        # Get unique values for interpolation grid
        altitudes = sorted(rotor_data['altitude_m'].unique())
        temperatures = sorted(rotor_data['temp_offset_C'].unique())
        rpms = sorted(rotor_data['rpm'].unique())
        
        # Create interpolation grids
        thrust_grid = np.zeros((len(altitudes), len(temperatures), len(rpms)))
        torque_grid = np.zeros((len(altitudes), len(temperatures), len(rpms)))
        efficiency_grid = np.zeros((len(altitudes), len(temperatures), len(rpms)))
        
        # Fill grids
        for i, alt in enumerate(altitudes):
            for j, temp in enumerate(temperatures):
                for k, rpm in enumerate(rpms):
                    mask = ((rotor_data['altitude_m'] == alt) & 
                           (rotor_data['temp_offset_C'] == temp) & 
                           (rotor_data['rpm'] == rpm))
                    
                    if mask.any():
                        thrust_grid[i, j, k] = rotor_data[mask]['thrust_N'].iloc[0]
                        torque_grid[i, j, k] = rotor_data[mask]['torque_Nm'].iloc[0]
                        efficiency_grid[i, j, k] = rotor_data[mask]['efficiency_percent'].iloc[0]
        
        # Create interpolators
        thrust_interp = RegularGridInterpolator(
            (altitudes, temperatures, rpms), thrust_grid, 
            method='linear', fill_value=None
        )
        
        torque_interp = RegularGridInterpolator(
            (altitudes, temperatures, rpms), torque_grid,
            method='linear', fill_value=None
        )
        
        efficiency_interp = RegularGridInterpolator(
            (altitudes, temperatures, rpms), efficiency_grid,
            method='linear', fill_value=None
        )
        
        return {
            'thrust': thrust_interp,
            'torque': torque_interp,
            'efficiency': efficiency_interp
        }
    
    def create_battery_interpolator(self, chemistry: str, 
                                  filename: str = "battery_specs.csv") -> RegularGridInterpolator:
        """
        Create interpolator for battery specifications.
        
        Args:
            chemistry: Battery chemistry type
            filename: CSV filename
            
        Returns:
            RegularGridInterpolator for battery properties
        """
        df = self.load_battery_specifications(filename)
        
        # Filter for specific chemistry
        battery_data = df[df['chemistry'] == chemistry].copy()
        
        if battery_data.empty:
            raise ValueError(f"No data found for battery chemistry {chemistry}")
        
        # Get unique values for interpolation grid
        temperatures = sorted(battery_data['temp_C'].unique())
        c_rates = sorted(battery_data['C_rate'].unique())
        
        # Create interpolation grids
        capacity_grid = np.zeros((len(temperatures), len(c_rates)))
        energy_grid = np.zeros((len(temperatures), len(c_rates)))
        efficiency_grid = np.zeros((len(temperatures), len(c_rates)))
        
        # Fill grids
        for i, temp in enumerate(temperatures):
            for j, c_rate in enumerate(c_rates):
                mask = ((battery_data['temp_C'] == temp) & 
                       (battery_data['C_rate'] == c_rate))
                
                if mask.any():
                    capacity_grid[i, j] = battery_data[mask]['capacity_Ah'].iloc[0]
                    energy_grid[i, j] = battery_data[mask]['energy_Wh'].iloc[0]
                    efficiency_grid[i, j] = battery_data[mask]['round_trip_efficiency_percent'].iloc[0]
        
        # Create interpolators
        capacity_interp = RegularGridInterpolator(
            (temperatures, c_rates), capacity_grid,
            method='linear', fill_value=None
        )
        
        energy_interp = RegularGridInterpolator(
            (temperatures, c_rates), energy_grid,
            method='linear', fill_value=None
        )
        
        efficiency_interp = RegularGridInterpolator(
            (temperatures, c_rates), efficiency_grid,
            method='linear', fill_value=None
        )
        
        return {
            'capacity': capacity_interp,
            'energy': energy_interp,
            'efficiency': efficiency_interp
        }
    
    def get_parameter_value(self, parameter_name: str, 
                          filename: str = "mass_inertia.csv") -> float:
        """
        Get a specific parameter value from mass/inertia data.
        
        Args:
            parameter_name: Name of the parameter
            filename: CSV filename
            
        Returns:
            Parameter value
        """
        df = self.load_mass_inertia(filename)
        
        param_data = df[df['parameter'] == parameter_name]
        
        if param_data.empty:
            raise ValueError(f"Parameter '{parameter_name}' not found in {filename}")
        
        return float(param_data['value'].iloc[0])
    
    def get_flight_limits(self, payload_kg: float, 
                         filename: str = "limits.csv") -> Dict[str, float]:
        """
        Get flight limits for specific payload.
        
        Args:
            payload_kg: Payload weight in kg
            filename: CSV filename
            
        Returns:
            Dictionary of flight limits
        """
        df = self.load_limits(filename)
        
        # Find closest payload value
        payload_diff = np.abs(df['payload_kg'] - payload_kg)
        closest_idx = payload_diff.idxmin()
        
        limits = {
            'max_altitude': float(df.loc[closest_idx, 'altitude_max_m']),
            'max_speed': float(df.loc[closest_idx, 'speed_max_mps']),
            'max_climb_rate': float(df.loc[closest_idx, 'climb_rate_mps']),
            'endurance': float(df.loc[closest_idx, 'endurance_min'])
        }
        
        return limits
    
    def clear_cache(self) -> None:
        """Clear data cache."""
        self.data_cache.clear()
        self.logger.info("Data cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached data.
        
        Returns:
            Dictionary with cache information
        """
        return {
            'cached_files': list(self.data_cache.keys()),
            'cache_size': len(self.data_cache),
            'memory_usage': sum(df.memory_usage(deep=True).sum() 
                              for df in self.data_cache.values() if isinstance(df, pd.DataFrame))
        }
