"""
File utilities for the perception layer.

This module provides file and directory management functions including:
- Directory creation and management
- File information extraction
- Temporary file handling
- Cleanup operations
"""

import os
import shutil
import tempfile
import logging
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import hashlib
import json
import gzip
import zipfile
from datetime import datetime
import fnmatch
import glob

# Optional imports for geospatial file handling
try:
    import rasterio
    from rasterio.crs import CRS
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

try:
    import netCDF4 as nc
    NETCDF4_AVAILABLE = True
except ImportError:
    NETCDF4_AVAILABLE = False

logger = logging.getLogger(__name__)


class FileUtilsError(Exception):
    """Custom exception for file utility errors."""
    pass


def ensure_directory(directory_path: str, create_parents: bool = True) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        create_parents: Whether to create parent directories
        
    Returns:
        Absolute path to the directory
        
    Raises:
        FileUtilsError: If directory creation fails
    """
    try:
        directory_path = os.path.abspath(directory_path)
        
        if not os.path.exists(directory_path):
            if create_parents:
                os.makedirs(directory_path, exist_ok=True)
                logger.info(f"Created directory: {directory_path}")
            else:
                os.mkdir(directory_path)
                logger.info(f"Created directory: {directory_path}")
        else:
            logger.debug(f"Directory already exists: {directory_path}")
        
        return directory_path
        
    except Exception as e:
        logger.error(f"Failed to create directory {directory_path}: {e}")
        raise FileUtilsError(f"Failed to create directory: {e}")


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get comprehensive information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary containing file information
        
    Raises:
        FileUtilsError: If file information retrieval fails
    """
    try:
        if not os.path.exists(file_path):
            raise FileUtilsError(f"File does not exist: {file_path}")
        
        stat_info = os.stat(file_path)
        
        info = {
            'path': os.path.abspath(file_path),
            'name': os.path.basename(file_path),
            'directory': os.path.dirname(os.path.abspath(file_path)),
            'size_bytes': stat_info.st_size,
            'size_mb': stat_info.st_size / (1024 * 1024),
            'created_time': datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
            'modified_time': datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            'accessed_time': datetime.fromtimestamp(stat_info.st_atime).isoformat(),
            'is_file': os.path.isfile(file_path),
            'is_directory': os.path.isdir(file_path),
            'is_readable': os.access(file_path, os.R_OK),
            'is_writable': os.access(file_path, os.W_OK),
            'is_executable': os.access(file_path, os.X_OK),
            'extension': os.path.splitext(file_path)[1],
            'exists': True
        }
        
        # Calculate file hash for integrity checking
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
                info['md5_hash'] = file_hash
        except Exception as e:
            logger.warning(f"Could not calculate MD5 hash for {file_path}: {e}")
            info['md5_hash'] = None
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get file info for {file_path}: {e}")
        raise FileUtilsError(f"Failed to get file info: {e}")


def create_temp_file(
    suffix: str = "",
    prefix: str = "perception_",
    directory: Optional[str] = None,
    delete_on_exit: bool = True
) -> str:
    """
    Create a temporary file.
    
    Args:
        suffix: File suffix
        prefix: File prefix
        directory: Directory to create file in (None for system temp dir)
        delete_on_exit: Whether to delete file when program exits
        
    Returns:
        Path to the temporary file
        
    Raises:
        FileUtilsError: If temporary file creation fails
    """
    try:
        if directory:
            ensure_directory(directory)
        
        temp_file = tempfile.NamedTemporaryFile(
            suffix=suffix,
            prefix=prefix,
            dir=directory,
            delete=False
        )
        temp_file.close()
        
        logger.debug(f"Created temporary file: {temp_file.name}")
        return temp_file.name
        
    except Exception as e:
        logger.error(f"Failed to create temporary file: {e}")
        raise FileUtilsError(f"Failed to create temporary file: {e}")


def create_temp_directory(
    suffix: str = "",
    prefix: str = "perception_",
    directory: Optional[str] = None
) -> str:
    """
    Create a temporary directory.
    
    Args:
        suffix: Directory suffix
        prefix: Directory prefix
        directory: Parent directory (None for system temp dir)
        
    Returns:
        Path to the temporary directory
        
    Raises:
        FileUtilsError: If temporary directory creation fails
    """
    try:
        if directory:
            ensure_directory(directory)
        
        temp_dir = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=directory)
        
        logger.debug(f"Created temporary directory: {temp_dir}")
        return temp_dir
        
    except Exception as e:
        logger.error(f"Failed to create temporary directory: {e}")
        raise FileUtilsError(f"Failed to create temporary directory: {e}")


def cleanup_temp_files(file_paths: List[str]) -> None:
    """
    Clean up temporary files.
    
    Args:
        file_paths: List of file paths to delete
        
    Raises:
        FileUtilsError: If cleanup fails
    """
    try:
        for file_path in file_paths:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"Deleted temporary file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {file_path}: {e}")
            else:
                logger.debug(f"Temporary file does not exist: {file_path}")
                
    except Exception as e:
        logger.error(f"Failed to cleanup temporary files: {e}")
        raise FileUtilsError(f"Failed to cleanup temporary files: {e}")


def cleanup_temp_directories(directory_paths: List[str]) -> None:
    """
    Clean up temporary directories.
    
    Args:
        directory_paths: List of directory paths to delete
        
    Raises:
        FileUtilsError: If cleanup fails
    """
    try:
        for directory_path in directory_paths:
            if os.path.exists(directory_path):
                try:
                    shutil.rmtree(directory_path)
                    logger.debug(f"Deleted temporary directory: {directory_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary directory {directory_path}: {e}")
            else:
                logger.debug(f"Temporary directory does not exist: {directory_path}")
                
    except Exception as e:
        logger.error(f"Failed to cleanup temporary directories: {e}")
        raise FileUtilsError(f"Failed to cleanup temporary directories: {e}")


def copy_file(
    source_path: str,
    destination_path: str,
    overwrite: bool = False
) -> str:
    """
    Copy a file from source to destination.
    
    Args:
        source_path: Source file path
        destination_path: Destination file path
        overwrite: Whether to overwrite existing file
        
    Returns:
        Path to the copied file
        
    Raises:
        FileUtilsError: If copy operation fails
    """
    try:
        if not os.path.exists(source_path):
            raise FileUtilsError(f"Source file does not exist: {source_path}")
        
        # Ensure destination directory exists
        dest_dir = os.path.dirname(destination_path)
        if dest_dir:
            ensure_directory(dest_dir)
        
        # Check if destination exists
        if os.path.exists(destination_path) and not overwrite:
            raise FileUtilsError(f"Destination file exists and overwrite=False: {destination_path}")
        
        shutil.copy2(source_path, destination_path)
        logger.info(f"Copied file from {source_path} to {destination_path}")
        
        return destination_path
        
    except Exception as e:
        logger.error(f"Failed to copy file from {source_path} to {destination_path}: {e}")
        raise FileUtilsError(f"Failed to copy file: {e}")


def move_file(
    source_path: str,
    destination_path: str,
    overwrite: bool = False
) -> str:
    """
    Move a file from source to destination.
    
    Args:
        source_path: Source file path
        destination_path: Destination file path
        overwrite: Whether to overwrite existing file
        
    Returns:
        Path to the moved file
        
    Raises:
        FileUtilsError: If move operation fails
    """
    try:
        if not os.path.exists(source_path):
            raise FileUtilsError(f"Source file does not exist: {source_path}")
        
        # Ensure destination directory exists
        dest_dir = os.path.dirname(destination_path)
        if dest_dir:
            ensure_directory(dest_dir)
        
        # Check if destination exists
        if os.path.exists(destination_path) and not overwrite:
            raise FileUtilsError(f"Destination file exists and overwrite=False: {destination_path}")
        
        shutil.move(source_path, destination_path)
        logger.info(f"Moved file from {source_path} to {destination_path}")
        
        return destination_path
        
    except Exception as e:
        logger.error(f"Failed to move file from {source_path} to {destination_path}: {e}")
        raise FileUtilsError(f"Failed to move file: {e}")


def list_files(
    directory_path: str,
    pattern: Optional[str] = None,
    recursive: bool = False
) -> List[str]:
    """
    List files in a directory.
    
    Args:
        directory_path: Directory to list files from
        pattern: File pattern to match (e.g., "*.tif")
        recursive: Whether to search recursively
        
    Returns:
        List of file paths
        
    Raises:
        FileUtilsError: If listing fails
    """
    try:
        if not os.path.exists(directory_path):
            raise FileUtilsError(f"Directory does not exist: {directory_path}")
        
        if not os.path.isdir(directory_path):
            raise FileUtilsError(f"Path is not a directory: {directory_path}")
        
        files = []
        
        if recursive:
            for root, dirs, filenames in os.walk(directory_path):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    if pattern is None or filename.endswith(pattern.replace("*", "")):
                        files.append(file_path)
        else:
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                if os.path.isfile(file_path):
                    if pattern is None or filename.endswith(pattern.replace("*", "")):
                        files.append(file_path)
        
        logger.debug(f"Found {len(files)} files in {directory_path}")
        return files
        
    except Exception as e:
        logger.error(f"Failed to list files in {directory_path}: {e}")
        raise FileUtilsError(f"Failed to list files: {e}")


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in megabytes
        
    Raises:
        FileUtilsError: If size retrieval fails
    """
    try:
        if not os.path.exists(file_path):
            raise FileUtilsError(f"File does not exist: {file_path}")
        
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        
        return size_mb
        
    except Exception as e:
        logger.error(f"Failed to get file size for {file_path}: {e}")
        raise FileUtilsError(f"Failed to get file size: {e}")


def save_json(data: Dict[str, Any], file_path: str, indent: int = 2) -> str:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the JSON file
        indent: JSON indentation
        
    Returns:
        Path to the saved file
        
    Raises:
        FileUtilsError: If save operation fails
    """
    try:
        # Ensure directory exists
        dest_dir = os.path.dirname(file_path)
        if dest_dir:
            ensure_directory(dest_dir)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        
        logger.info(f"Saved JSON data to {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}")
        raise FileUtilsError(f"Failed to save JSON: {e}")


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data
        
    Raises:
        FileUtilsError: If load operation fails
    """
    try:
        if not os.path.exists(file_path):
            raise FileUtilsError(f"JSON file does not exist: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.debug(f"Loaded JSON data from {file_path}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load JSON from {file_path}: {e}")
        raise FileUtilsError(f"Failed to load JSON: {e}")


# Enhanced Geospatial File Utilities

def get_geospatial_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get comprehensive information about geospatial files.
    
    Args:
        file_path: Path to the geospatial file
        
    Returns:
        Dictionary containing geospatial file information
        
    Raises:
        FileUtilsError: If file information retrieval fails
    """
    try:
        # Get basic file info
        basic_info = get_file_info(file_path)
        file_ext = basic_info['extension'].lower()
        
        geospatial_info = {
            **basic_info,
            'file_type': 'unknown',
            'geospatial_metadata': {},
            'supported_format': False
        }
        
        # Raster files
        if file_ext in ['.tif', '.tiff', '.geotiff']:
            if RASTERIO_AVAILABLE:
                try:
                    with rasterio.open(file_path) as src:
                        geospatial_info.update({
                            'file_type': 'raster',
                            'supported_format': True,
                            'geospatial_metadata': {
                                'crs': str(src.crs),
                                'bounds': src.bounds,
                                'shape': src.shape,
                                'count': src.count,
                                'dtype': src.dtypes[0],
                                'resolution': src.res,
                                'nodata': src.nodata,
                                'transform': src.transform
                            }
                        })
                except Exception as e:
                    logger.warning(f"Could not read raster metadata for {file_path}: {e}")
            else:
                geospatial_info['file_type'] = 'raster (rasterio not available)'
        
        # Vector files
        elif file_ext in ['.shp', '.geojson', '.gpkg', '.kml', '.kmz']:
            if GEOPANDAS_AVAILABLE:
                try:
                    gdf = gpd.read_file(file_path)
                    geospatial_info.update({
                        'file_type': 'vector',
                        'supported_format': True,
                        'geospatial_metadata': {
                            'crs': str(gdf.crs),
                            'bounds': gdf.total_bounds.tolist(),
                            'feature_count': len(gdf),
                            'geometry_types': gdf.geometry.geom_type.unique().tolist(),
                            'columns': gdf.columns.tolist()
                        }
                    })
                except Exception as e:
                    logger.warning(f"Could not read vector metadata for {file_path}: {e}")
            else:
                geospatial_info['file_type'] = 'vector (geopandas not available)'
        
        # NetCDF files
        elif file_ext in ['.nc', '.netcdf']:
            if NETCDF4_AVAILABLE:
                try:
                    with nc.Dataset(file_path, 'r') as dataset:
                        geospatial_info.update({
                            'file_type': 'netcdf',
                            'supported_format': True,
                            'geospatial_metadata': {
                                'dimensions': dict(dataset.dimensions),
                                'variables': list(dataset.variables.keys()),
                                'attributes': dict(dataset.__dict__)
                            }
                        })
                except Exception as e:
                    logger.warning(f"Could not read NetCDF metadata for {file_path}: {e}")
            else:
                geospatial_info['file_type'] = 'netcdf (netCDF4 not available)'
        
        # Compressed files
        elif file_ext in ['.gz', '.zip', '.tar.gz']:
            geospatial_info['file_type'] = 'compressed'
            geospatial_info['supported_format'] = True
        
        return geospatial_info
        
    except Exception as e:
        logger.error(f"Failed to get geospatial file info for {file_path}: {e}")
        raise FileUtilsError(f"Failed to get geospatial file info: {e}")


def compress_file(
    source_path: str,
    destination_path: Optional[str] = None,
    compression_type: str = 'gzip',
    compression_level: int = 6
) -> str:
    """
    Compress a file using various compression methods.
    
    Args:
        source_path: Path to the source file
        destination_path: Path for compressed file (None for auto-generated)
        compression_type: Type of compression ('gzip', 'zip')
        compression_level: Compression level (1-9 for gzip, 1-9 for zip)
        
    Returns:
        Path to the compressed file
        
    Raises:
        FileUtilsError: If compression fails
    """
    try:
        if not os.path.exists(source_path):
            raise FileUtilsError(f"Source file does not exist: {source_path}")
        
        if destination_path is None:
            if compression_type == 'gzip':
                destination_path = f"{source_path}.gz"
            elif compression_type == 'zip':
                destination_path = f"{source_path}.zip"
            else:
                raise FileUtilsError(f"Unsupported compression type: {compression_type}")
        
        # Ensure destination directory exists
        dest_dir = os.path.dirname(destination_path)
        if dest_dir:
            ensure_directory(dest_dir)
        
        if compression_type == 'gzip':
            with open(source_path, 'rb') as f_in:
                with gzip.open(destination_path, 'wb', compresslevel=compression_level) as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        elif compression_type == 'zip':
            with zipfile.ZipFile(destination_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=compression_level) as zipf:
                zipf.write(source_path, os.path.basename(source_path))
        
        else:
            raise FileUtilsError(f"Unsupported compression type: {compression_type}")
        
        logger.info(f"Compressed {source_path} to {destination_path}")
        return destination_path
        
    except Exception as e:
        logger.error(f"Failed to compress {source_path}: {e}")
        raise FileUtilsError(f"Failed to compress file: {e}")


def decompress_file(
    source_path: str,
    destination_path: Optional[str] = None
) -> str:
    """
    Decompress a file.
    
    Args:
        source_path: Path to the compressed file
        destination_path: Path for decompressed file (None for auto-generated)
        
    Returns:
        Path to the decompressed file
        
    Raises:
        FileUtilsError: If decompression fails
    """
    try:
        if not os.path.exists(source_path):
            raise FileUtilsError(f"Source file does not exist: {source_path}")
        
        file_ext = os.path.splitext(source_path)[1].lower()
        
        if destination_path is None:
            if file_ext == '.gz':
                destination_path = source_path[:-3]  # Remove .gz
            elif file_ext == '.zip':
                base_name = os.path.splitext(source_path)[0]
                destination_path = base_name
            else:
                raise FileUtilsError(f"Unsupported compression format: {file_ext}")
        
        # Ensure destination directory exists
        dest_dir = os.path.dirname(destination_path)
        if dest_dir:
            ensure_directory(dest_dir)
        
        if file_ext == '.gz':
            with gzip.open(source_path, 'rb') as f_in:
                with open(destination_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        elif file_ext == '.zip':
            with zipfile.ZipFile(source_path, 'r') as zipf:
                zipf.extractall(os.path.dirname(destination_path))
                # For single file archives, return the extracted file path
                if len(zipf.namelist()) == 1:
                    destination_path = os.path.join(os.path.dirname(destination_path), zipf.namelist()[0])
        
        else:
            raise FileUtilsError(f"Unsupported compression format: {file_ext}")
        
        logger.info(f"Decompressed {source_path} to {destination_path}")
        return destination_path
        
    except Exception as e:
        logger.error(f"Failed to decompress {source_path}: {e}")
        raise FileUtilsError(f"Failed to decompress file: {e}")


def batch_process_files(
    input_directory: str,
    output_directory: str,
    file_pattern: str = "*",
    process_function: callable = None,
    recursive: bool = False,
    max_files: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Batch process files in a directory.
    
    Args:
        input_directory: Directory containing input files
        output_directory: Directory for output files
        file_pattern: File pattern to match (e.g., "*.tif")
        process_function: Function to apply to each file
        recursive: Whether to search recursively
        max_files: Maximum number of files to process (None for all)
        
    Returns:
        List of processing results
        
    Raises:
        FileUtilsError: If batch processing fails
    """
    try:
        # Ensure output directory exists
        ensure_directory(output_directory)
        
        # Find matching files
        matching_files = []
        
        if recursive:
            for root, dirs, files in os.walk(input_directory):
                for file in files:
                    if fnmatch.fnmatch(file, file_pattern):
                        matching_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(input_directory):
                if fnmatch.fnmatch(file, file_pattern):
                    matching_files.append(os.path.join(input_directory, file))
        
        # Limit number of files if specified
        if max_files:
            matching_files = matching_files[:max_files]
        
        logger.info(f"Found {len(matching_files)} files to process")
        
        results = []
        for i, file_path in enumerate(matching_files, 1):
            try:
                logger.info(f"Processing file {i}/{len(matching_files)}: {file_path}")
                
                if process_function:
                    result = process_function(file_path, output_directory)
                else:
                    # Default: copy file
                    filename = os.path.basename(file_path)
                    output_path = os.path.join(output_directory, filename)
                    result = copy_file(file_path, output_path)
                
                results.append({
                    'input_file': file_path,
                    'output_file': result if isinstance(result, str) else result.get('output_file', ''),
                    'success': True,
                    'error': None
                })
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results.append({
                    'input_file': file_path,
                    'output_file': '',
                    'success': False,
                    'error': str(e)
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to batch process files: {e}")
        raise FileUtilsError(f"Failed to batch process files: {e}")


def validate_file_integrity(file_path: str, expected_hash: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate file integrity using checksums.
    
    Args:
        file_path: Path to the file
        expected_hash: Expected MD5 hash (optional)
        
    Returns:
        Dictionary with integrity validation results
        
    Raises:
        FileUtilsError: If validation fails
    """
    try:
        if not os.path.exists(file_path):
            raise FileUtilsError(f"File does not exist: {file_path}")
        
        # Calculate file hash
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        validation_result = {
            'file_path': file_path,
            'file_hash': file_hash,
            'integrity_valid': True,
            'file_size': os.path.getsize(file_path),
            'error': None
        }
        
        # Compare with expected hash if provided
        if expected_hash:
            if file_hash.lower() != expected_hash.lower():
                validation_result['integrity_valid'] = False
                validation_result['error'] = f"Hash mismatch: expected {expected_hash}, got {file_hash}"
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Failed to validate file integrity for {file_path}: {e}")
        raise FileUtilsError(f"Failed to validate file integrity: {e}")


def create_file_manifest(
    directory_path: str,
    output_path: Optional[str] = None,
    include_hashes: bool = True,
    file_pattern: str = "*"
) -> Dict[str, Any]:
    """
    Create a manifest of files in a directory.
    
    Args:
        directory_path: Directory to create manifest for
        output_path: Path to save manifest (None for auto-generated)
        include_hashes: Whether to include file hashes
        file_pattern: File pattern to include
        
    Returns:
        Dictionary containing file manifest
        
    Raises:
        FileUtilsError: If manifest creation fails
    """
    try:
        if not os.path.exists(directory_path):
            raise FileUtilsError(f"Directory does not exist: {directory_path}")
        
        if output_path is None:
            output_path = os.path.join(directory_path, "file_manifest.json")
        
        manifest = {
            'directory': directory_path,
            'created_at': datetime.now().isoformat(),
            'total_files': 0,
            'total_size_bytes': 0,
            'files': []
        }
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if fnmatch.fnmatch(file, file_pattern):
                    file_path = os.path.join(root, file)
                    file_info = get_file_info(file_path)
                    
                    file_entry = {
                        'path': file_path,
                        'name': file_info['name'],
                        'size_bytes': file_info['size_bytes'],
                        'modified_time': file_info['modified_time']
                    }
                    
                    if include_hashes:
                        file_entry['md5_hash'] = file_info['md5_hash']
                    
                    manifest['files'].append(file_entry)
                    manifest['total_files'] += 1
                    manifest['total_size_bytes'] += file_info['size_bytes']
        
        # Save manifest
        save_json(manifest, output_path)
        
        logger.info(f"Created file manifest with {manifest['total_files']} files")
        return manifest
        
    except Exception as e:
        logger.error(f"Failed to create file manifest: {e}")
        raise FileUtilsError(f"Failed to create file manifest: {e}")


def optimize_geospatial_file(
    input_path: str,
    output_path: Optional[str] = None,
    optimization_type: str = 'compression',
    **kwargs
) -> str:
    """
    Optimize geospatial files for better performance.
    
    Args:
        input_path: Path to input file
        output_path: Path for optimized file (None for auto-generated)
        optimization_type: Type of optimization ('compression', 'tiling', 'overviews')
        **kwargs: Additional optimization parameters
        
    Returns:
        Path to the optimized file
        
    Raises:
        FileUtilsError: If optimization fails
    """
    try:
        if not os.path.exists(input_path):
            raise FileUtilsError(f"Input file does not exist: {input_path}")
        
        if output_path is None:
            base_name = os.path.splitext(input_path)[0]
            output_path = f"{base_name}_optimized{os.path.splitext(input_path)[1]}"
        
        # Ensure output directory exists
        dest_dir = os.path.dirname(output_path)
        if dest_dir:
            ensure_directory(dest_dir)
        
        if optimization_type == 'compression':
            # Simple compression optimization
            return compress_file(input_path, output_path, **kwargs)
        
        elif optimization_type == 'copy':
            # Simple copy (no optimization)
            return copy_file(input_path, output_path)
        
        else:
            raise FileUtilsError(f"Unsupported optimization type: {optimization_type}")
        
    except Exception as e:
        logger.error(f"Failed to optimize file {input_path}: {e}")
        raise FileUtilsError(f"Failed to optimize file: {e}")


def get_supported_formats() -> Dict[str, List[str]]:
    """
    Get list of supported geospatial file formats.
    
    Returns:
        Dictionary of supported formats by category
    """
    formats = {
        'raster': ['.tif', '.tiff', '.geotiff', '.img', '.hdf', '.hdf5'],
        'vector': ['.shp', '.geojson', '.gpkg', '.kml', '.kmz', '.gml'],
        'point_cloud': ['.las', '.laz', '.xyz', '.pts'],
        'compressed': ['.gz', '.zip', '.tar.gz', '.7z'],
        'metadata': ['.xml', '.json', '.txt', '.md']
    }
    
    # Add NetCDF if available
    if NETCDF4_AVAILABLE:
        formats['netcdf'] = ['.nc', '.netcdf', '.cdf']
    
    return formats
