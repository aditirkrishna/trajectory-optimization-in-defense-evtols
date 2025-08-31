"""
Quality control utilities for the perception layer.

This module provides functions for validating data quality and generating quality reports.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class QualityControlError(Exception):
    """Custom exception for quality control errors."""
    pass


def perform_quality_checks(
    data: Dict[str, Any],
    checks: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Perform quality checks on processed data.
    
    Args:
        data: Dictionary containing processed data
        checks: List of quality checks to perform
        
    Returns:
        Dictionary containing quality check results
        
    Raises:
        QualityControlError: If quality checks fail
    """
    try:
        if checks is None:
            checks = ['basic', 'spatial', 'temporal', 'statistical']
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'checks_performed': checks,
            'overall_quality': 'unknown',
            'issues': [],
            'recommendations': []
        }
        
        for check in checks:
            if check == 'basic':
                basic_results = _perform_basic_checks(data)
                results.update(basic_results)
            
            elif check == 'spatial':
                spatial_results = _perform_spatial_checks(data)
                results.update(spatial_results)
            
            elif check == 'temporal':
                temporal_results = _perform_temporal_checks(data)
                results.update(temporal_results)
            
            elif check == 'statistical':
                statistical_results = _perform_statistical_checks(data)
                results.update(statistical_results)
            
            else:
                logger.warning(f"Unknown quality check: {check}")
        
        # Determine overall quality
        results['overall_quality'] = _determine_overall_quality(results)
        
        logger.info(f"Performed {len(checks)} quality checks")
        return results
        
    except Exception as e:
        logger.error(f"Failed to perform quality checks: {e}")
        raise QualityControlError(f"Failed to perform quality checks: {e}")


def validate_processing_results(
    results: Dict[str, Any],
    validation_criteria: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate processing results against criteria.
    
    Args:
        results: Dictionary containing processing results
        validation_criteria: Criteria for validation
        
    Returns:
        Dictionary containing validation results
        
    Raises:
        QualityControlError: If validation fails
    """
    try:
        if validation_criteria is None:
            validation_criteria = {
                'min_data_quality': 0.7,
                'max_missing_data': 0.1,
                'spatial_consistency': True,
                'temporal_consistency': True
            }
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'criteria': validation_criteria,
            'passed': True,
            'issues': [],
            'details': {}
        }
        
        # Check data quality
        if 'quality_metrics' in results:
            quality_score = results['quality_metrics'].get('overall_score', 0.0)
            if quality_score < validation_criteria['min_data_quality']:
                validation_results['passed'] = False
                validation_results['issues'].append(f"Data quality too low: {quality_score}")
        
        # Check missing data
        if 'missing_data_ratio' in results:
            missing_ratio = results['missing_data_ratio']
            if missing_ratio > validation_criteria['max_missing_data']:
                validation_results['passed'] = False
                validation_results['issues'].append(f"Too much missing data: {missing_ratio}")
        
        # Check spatial consistency
        if validation_criteria['spatial_consistency']:
            if 'spatial_issues' in results and results['spatial_issues']:
                validation_results['passed'] = False
                validation_results['issues'].append("Spatial consistency issues detected")
        
        # Check temporal consistency
        if validation_criteria['temporal_consistency']:
            if 'temporal_issues' in results and results['temporal_issues']:
                validation_results['passed'] = False
                validation_results['issues'].append("Temporal consistency issues detected")
        
        logger.info(f"Validation {'passed' if validation_results['passed'] else 'failed'}")
        return validation_results
        
    except Exception as e:
        logger.error(f"Failed to validate processing results: {e}")
        raise QualityControlError(f"Failed to validate processing results: {e}")


def generate_quality_report(
    quality_results: Dict[str, Any],
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive quality report.
    
    Args:
        quality_results: Dictionary containing quality check results
        output_path: Path to save the report (optional)
        
    Returns:
        Dictionary containing the quality report
        
    Raises:
        QualityControlError: If report generation fails
    """
    try:
        report = {
            'report_id': f"QC_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'overall_quality': quality_results.get('overall_quality', 'unknown'),
                'checks_performed': len(quality_results.get('checks_performed', [])),
                'issues_found': len(quality_results.get('issues', [])),
                'recommendations': len(quality_results.get('recommendations', []))
            },
            'detailed_results': quality_results,
            'metadata': {
                'version': '1.0',
                'generator': 'perception_layer_quality_control'
            }
        }
        
        # Save report if output path provided
        if output_path:
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from utils.file_utils import save_json
            save_json(report, output_path)
            logger.info(f"Quality report saved to {output_path}")
        
        return report
        
    except Exception as e:
        logger.error(f"Failed to generate quality report: {e}")
        raise QualityControlError(f"Failed to generate quality report: {e}")


def _perform_basic_checks(data: Dict[str, Any]) -> Dict[str, Any]:
    """Perform basic quality checks (internal function)."""
    results = {
        'basic_checks': {
            'data_exists': 'data' in data,
            'metadata_exists': 'metadata' in data,
            'data_type_valid': True,
            'file_path_valid': 'file_path' in data
        },
        'issues': [],
        'recommendations': []
    }
    
    # Check data type
    if 'data' in data:
        if isinstance(data['data'], np.ndarray):
            results['basic_checks']['data_shape'] = data['data'].shape
            results['basic_checks']['data_dtype'] = str(data['data'].dtype)
        elif hasattr(data['data'], 'geometry'):  # GeoDataFrame
            results['basic_checks']['feature_count'] = len(data['data'])
            results['basic_checks']['geometry_types'] = data['data'].geometry.geom_type.unique().tolist()
    
    return results


def _perform_spatial_checks(data: Dict[str, Any]) -> Dict[str, Any]:
    """Perform spatial quality checks (internal function)."""
    results = {
        'spatial_checks': {},
        'spatial_issues': [],
        'recommendations': []
    }
    
    if 'metadata' in data and 'bounds' in data['metadata']:
        bounds = data['metadata']['bounds']
        if len(bounds) == 4:
            minx, miny, maxx, maxy = bounds
            results['spatial_checks']['bounds_valid'] = minx < maxx and miny < maxy
            results['spatial_checks']['area'] = (maxx - minx) * (maxy - miny)
            
            if not results['spatial_checks']['bounds_valid']:
                results['spatial_issues'].append("Invalid spatial bounds")
    
    return results


def _perform_temporal_checks(data: Dict[str, Any]) -> Dict[str, Any]:
    """Perform temporal quality checks (internal function)."""
    results = {
        'temporal_checks': {},
        'temporal_issues': [],
        'recommendations': []
    }
    
    # Add temporal checks if temporal data is present
    if 'metadata' in data and 'temporal_info' in data['metadata']:
        temporal_info = data['metadata']['temporal_info']
        results['temporal_checks']['temporal_data_present'] = True
        results['temporal_checks']['temporal_info'] = temporal_info
    else:
        results['temporal_checks']['temporal_data_present'] = False
    
    return results


def _perform_statistical_checks(data: Dict[str, Any]) -> Dict[str, Any]:
    """Perform statistical quality checks (internal function)."""
    results = {
        'statistical_checks': {},
        'statistical_issues': [],
        'recommendations': []
    }
    
    if 'data' in data and isinstance(data['data'], np.ndarray):
        data_array = data['data']
        
        # Basic statistics
        results['statistical_checks']['mean'] = float(np.nanmean(data_array))
        results['statistical_checks']['std'] = float(np.nanstd(data_array))
        results['statistical_checks']['min'] = float(np.nanmin(data_array))
        results['statistical_checks']['max'] = float(np.nanmax(data_array))
        results['statistical_checks']['nan_count'] = int(np.isnan(data_array).sum())
        results['statistical_checks']['total_elements'] = int(data_array.size)
        
        # Calculate missing data ratio
        missing_ratio = results['statistical_checks']['nan_count'] / results['statistical_checks']['total_elements']
        results['statistical_checks']['missing_data_ratio'] = float(missing_ratio)
        
        if missing_ratio > 0.1:
            results['statistical_issues'].append(f"High missing data ratio: {missing_ratio:.2%}")
            results['recommendations'].append("Consider data interpolation or gap filling")
    
    return results


def _determine_overall_quality(results: Dict[str, Any]) -> str:
    """Determine overall quality score (internal function)."""
    issues = results.get('issues', [])
    
    if not issues:
        return 'excellent'
    elif len(issues) <= 2:
        return 'good'
    elif len(issues) <= 5:
        return 'fair'
    else:
        return 'poor'
