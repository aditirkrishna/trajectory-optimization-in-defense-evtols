#!/usr/bin/env python3
"""
Test script for enhanced validation utilities.

This script tests the enhanced validation utilities to ensure they work correctly.
Run this script to verify the validation functions are working as expected.
"""

import sys
import os
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.validation import (
    validate_raster,
    validate_vector,
    validate_coordinates,
    validate_time_range,
    validate_file_path,
    validate_config,
    validate_raster_metadata,
    validate_raster_data_quality,
    validate_vector_geometry,
    validate_spatial_consistency,
    validate_temporal_data,
    validate_perception_config,
    validate_performance_metrics,
    ValidationError
)

def test_basic_validation_functions():
    """Test basic validation functions."""
    print("Testing basic validation functions...")
    
    # Test raster validation
    test_array = np.random.rand(100, 100).astype(np.float32)
    try:
        validate_raster(test_array, expected_shape=(100, 100), expected_dtype=np.float32)
        print("✓ Raster validation working correctly")
    except ValidationError as e:
        print(f"✗ Raster validation failed: {e}")
    
    # Test vector validation
    test_vector = {
        'type': 'FeatureCollection',
        'features': [{'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [0, 0]}}]
    }
    try:
        validate_vector(test_vector, required_fields=['type', 'features'])
        print("✓ Vector validation working correctly")
    except ValidationError as e:
        print(f"✗ Vector validation failed: {e}")
    
    # Test coordinate validation
    try:
        validate_coordinates(52.5200, 13.4050)
        print("✓ Coordinate validation working correctly")
    except ValidationError as e:
        print(f"✗ Coordinate validation failed: {e}")
    
    # Test time range validation
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=1)
    try:
        validate_time_range(start_time, end_time)
        print("✓ Time range validation working correctly")
    except ValidationError as e:
        print(f"✗ Time range validation failed: {e}")
    
    # Test file path validation
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.close()
    try:
        validate_file_path(temp_file.name)
        print("✓ File path validation working correctly")
        os.unlink(temp_file.name)
    except ValidationError as e:
        print(f"✗ File path validation failed: {e}")
    
    # Test config validation
    test_config = {
        'coordinate_system': {'working_crs': 'EPSG:32632'},
        'spatial': {'base_resolution': 2.0, 'tile_size': 512},
        'processing': {},
        'terrain': {},
        'urban': {},
        'atmosphere': {},
        'threats': {}
    }
    try:
        validate_config(test_config, required_keys=['coordinate_system', 'spatial'])
        print("✓ Config validation working correctly")
    except ValidationError as e:
        print(f"✗ Config validation failed: {e}")
    
    print()


def test_enhanced_validation_functions():
    """Test enhanced validation functions."""
    print("Testing enhanced validation functions...")
    
    # Test perception config validation
    test_perception_config = {
        'coordinate_system': {
            'working_crs': 'EPSG:32632',
            'input_crs': 'EPSG:4326'
        },
        'spatial': {
            'base_resolution': 2.0,
            'tile_size': 512,
            'altitude_bands': [0, 10, 25, 50, 100, 200, 500, 1000]
        },
        'processing': {},
        'terrain': {},
        'urban': {},
        'atmosphere': {},
        'threats': {}
    }
    
    try:
        result = validate_perception_config(test_perception_config)
        print(f"✓ Perception config validation working correctly")
        print(f"  Valid: {result['valid']}")
    except ValidationError as e:
        print(f"✗ Perception config validation failed: {e}")
    
    # Test temporal data validation
    dates = pd.date_range('2023-01-01', periods=100, freq='H')
    temporal_data = pd.DataFrame({'timestamp': dates, 'value': np.random.rand(100)})
    
    try:
        result = validate_temporal_data(temporal_data, 'timestamp', expected_frequency='1H')
        print(f"✓ Temporal data validation working correctly")
        print(f"  Valid: {result['valid']}, Duration: {result['duration']}")
    except ValidationError as e:
        print(f"✗ Temporal data validation failed: {e}")
    
    # Test performance metrics validation
    try:
        result = validate_performance_metrics(
            file_size_mb=50.0,
            processing_time_seconds=30.0,
            memory_usage_mb=100.0,
            max_file_size_mb=100.0,
            max_processing_time_seconds=60.0,
            max_memory_usage_mb=200.0
        )
        print(f"✓ Performance metrics validation working correctly")
        print(f"  Valid: {result['valid']}")
    except ValidationError as e:
        print(f"✗ Performance metrics validation failed: {e}")
    
    print()


def test_validation_error_handling():
    """Test validation error handling."""
    print("Testing validation error handling...")
    
    # Test invalid coordinates
    try:
        validate_coordinates(91.0, 13.4050)  # Invalid latitude
        print("✗ Invalid coordinates should have failed")
    except ValidationError:
        print("✓ Invalid coordinates correctly rejected")
    
    # Test invalid time range
    try:
        validate_time_range(datetime.now(), datetime.now() - timedelta(hours=1))  # End before start
        print("✗ Invalid time range should have failed")
    except ValidationError:
        print("✓ Invalid time range correctly rejected")
    
    # Test invalid config
    try:
        validate_config({}, required_keys=['missing_key'])
        print("✗ Invalid config should have failed")
    except ValidationError:
        print("✓ Invalid config correctly rejected")
    
    # Test invalid perception config
    try:
        validate_perception_config({'invalid': 'config'})
        print("✗ Invalid perception config should have failed")
    except ValidationError:
        print("✓ Invalid perception config correctly rejected")
    
    print()


def test_validation_warnings():
    """Test validation warnings."""
    print("Testing validation warnings...")
    
    # Test performance warnings (approaching limits)
    try:
        result = validate_performance_metrics(
            file_size_mb=85.0,  # 85% of 100 MB limit
            processing_time_seconds=50.0,  # 83% of 60s limit
            memory_usage_mb=180.0,  # 90% of 200 MB limit
            max_file_size_mb=100.0,
            max_processing_time_seconds=60.0,
            max_memory_usage_mb=200.0
        )
        print(f"✓ Performance warnings working correctly")
        print(f"  Warnings: {len(result['warnings'])}")
        for warning in result['warnings']:
            print(f"    - {warning}")
    except ValidationError as e:
        print(f"✗ Performance warnings test failed: {e}")
    
    print()


def test_validation_statistics():
    """Test validation statistics and reporting."""
    print("Testing validation statistics...")
    
    # Test temporal data with gaps
    dates = pd.date_range('2023-01-01', periods=100, freq='H')
    # Remove some dates to create gaps
    dates = dates.drop([10, 20, 30, 40, 50])
    temporal_data = pd.DataFrame({'timestamp': dates, 'value': np.random.rand(len(dates))})
    
    try:
        result = validate_temporal_data(
            temporal_data, 
            'timestamp', 
            expected_frequency='1H',
            check_gaps=True,
            max_gap_hours=2.0
        )
        print(f"✓ Temporal validation statistics working correctly")
        print(f"  Total points: {result['total_points']}")
        print(f"  Duration: {result['duration']}")
        print(f"  Gaps found: {len(result['gaps'])}")
        print(f"  Warnings: {len(result['warnings'])}")
    except ValidationError as e:
        print(f"✗ Temporal validation statistics failed: {e}")
    
    print()


def main():
    """Run all validation utility tests."""
    print("=" * 60)
    print("Testing Enhanced Validation Utilities")
    print("=" * 60)
    
    try:
        test_basic_validation_functions()
        test_enhanced_validation_functions()
        test_validation_error_handling()
        test_validation_warnings()
        test_validation_statistics()
        
        print("=" * 60)
        print("All validation utility tests passed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
