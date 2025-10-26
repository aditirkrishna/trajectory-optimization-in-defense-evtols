#!/usr/bin/env python3
"""
Simple Terrain Analysis Demo - Working Components Only
Using the perception layer with real project datasets
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add perception layer to path
sys.path.append('perception-layer/src')

# Import perception layer functions
from utils.config import Config
from geometry.terrain_analysis import compute_slope, compute_aspect, compute_roughness, compute_terrain_features
from geometry.obstacle_detection import detect_obstacles
from utils.validation import validate_raster

def load_terrain_data():
    """Load terrain elevation data from CSV"""
    print("Loading terrain elevation data...")
    terrain_df = pd.read_csv('project-datasets/0_raw/environment/terrain/terrain_elevation.csv')
    print(f"Loaded {len(terrain_df)} terrain points")
    print(f"Elevation range: {terrain_df['elevation_m'].min():.1f}m - {terrain_df['elevation_m'].max():.1f}m")
    return terrain_df

def load_building_data():
    """Load building footprint data from CSV"""
    print("Loading building footprint data...")
    buildings_df = pd.read_csv('project-datasets/0_raw/environment/terrain/building_footprints.csv')
    print(f"Loaded {len(buildings_df)} buildings")
    print(f"Building height range: {buildings_df['height_m'].min():.1f}m - {buildings_df['height_m'].max():.1f}m")
    return buildings_df

def create_dem_grid(terrain_df, resolution=0.01):
    """Create a DEM grid from point data"""
    print("Creating DEM grid from point data...")
    
    # Get bounds
    min_lat, max_lat = terrain_df['latitude'].min(), terrain_df['latitude'].max()
    min_lon, max_lon = terrain_df['longitude'].min(), terrain_df['longitude'].max()
    
    # Create grid
    lats = np.arange(min_lat, max_lat + resolution, resolution)
    lons = np.arange(min_lon, max_lon + resolution, resolution)
    
    # Create elevation grid by interpolating from points
    elevation_grid = np.zeros((len(lats), len(lons)))
    
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            # Find closest terrain point
            distances = np.sqrt((terrain_df['latitude'] - lat)**2 + (terrain_df['longitude'] - lon)**2)
            closest_idx = distances.idxmin()
            elevation_grid[i, j] = terrain_df.loc[closest_idx, 'elevation_m']
    
    print(f"Created DEM grid: {elevation_grid.shape[0]}x{elevation_grid.shape[1]} points")
    print(f"Grid resolution: {resolution:.4f} degrees")
    
    return elevation_grid, lats, lons

def create_dsm_grid(terrain_df, buildings_df, resolution=0.01):
    """Create a DSM grid including buildings"""
    print("Creating DSM grid with buildings...")
    
    # Start with DEM
    dem_grid, lats, lons = create_dem_grid(terrain_df, resolution)
    dsm_grid = dem_grid.copy()
    
    # Add building heights
    for _, building in buildings_df.iterrows():
        # Find grid cell for building
        lat_idx = np.argmin(np.abs(lats - building['latitude']))
        lon_idx = np.argmin(np.abs(lons - building['longitude']))
        
        # Add building height to DSM
        dsm_grid[lat_idx, lon_idx] += building['height_m']
    
    print(f"Created DSM grid with {len(buildings_df)} buildings")
    return dsm_grid, lats, lons

def analyze_terrain(elevation_grid):
    """Perform terrain analysis using perception layer"""
    print("\n=== TERRAIN ANALYSIS ===")
    
    # Convert to float32 for processing
    dem = elevation_grid.astype(np.float32)
    
    # Validate input
    try:
        validate_raster(dem)
        print("✓ Input DEM validation passed")
    except Exception as e:
        print(f"✗ Input validation failed: {e}")
        return None
    
    # Compute terrain derivatives
    print("Computing terrain derivatives...")
    
    # Slope
    slope = compute_slope(dem, pixel_size=1.0, method="horn")
    print(f"✓ Slope computed: range [{np.min(slope):.2f}°, {np.max(slope):.2f}°]")
    
    # Aspect
    aspect = compute_aspect(dem, pixel_size=1.0, method="horn")
    print(f"✓ Aspect computed: range [{np.min(aspect):.2f}°, {np.max(aspect):.2f}°]")
    
    # Roughness
    roughness = compute_roughness(dem, window_size=5, roughness_type="std")
    print(f"✓ Roughness computed: range [{np.min(roughness):.2f}, {np.max(roughness):.2f}]")
    
    # Terrain features
    features = compute_terrain_features(dem, pixel_size=1.0, window_size=5)
    print(f"✓ Terrain features computed: {list(features.keys())}")
    
    return {
        'dem': dem,
        'slope': slope,
        'aspect': aspect,
        'roughness': roughness,
        'features': features
    }

def detect_obstacles_simple(dem, dsm):
    """Simple obstacle detection using perception layer"""
    print("\n=== OBSTACLE DETECTION ===")
    
    # Convert to float32
    dem = dem.astype(np.float32)
    dsm = dsm.astype(np.float32)
    
    # Detect obstacles (using the working function)
    obstacles = detect_obstacles(dem, dsm, min_height=2.0)
    print(f"✓ Obstacles detected successfully")
    
    # Extract obstacle mask from the result
    if isinstance(obstacles, dict):
        obstacle_mask = obstacles.get('obstacle_mask', np.zeros_like(dem, dtype=bool))
        num_obstacles = obstacles.get('num_obstacles', 0)
    else:
        obstacle_mask = obstacles
        num_obstacles = np.sum(obstacle_mask)
    
    print(f"✓ Found {num_obstacles} obstacle pixels")
    
    return {
        'obstacles': obstacle_mask,
        'num_obstacles': num_obstacles
    }

def visualize_results(terrain_results, obstacle_results, lats, lons):
    """Create visualizations of the results"""
    print("\n=== CREATING VISUALIZATIONS ===")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Terrain Analysis and Obstacle Detection Results', fontsize=16)
    
    # DEM
    im1 = axes[0, 0].imshow(terrain_results['dem'], cmap='terrain', aspect='auto')
    axes[0, 0].set_title('Digital Elevation Model (DEM)')
    axes[0, 0].set_xlabel('Longitude Index')
    axes[0, 0].set_ylabel('Latitude Index')
    plt.colorbar(im1, ax=axes[0, 0], label='Elevation (m)')
    
    # Slope
    im2 = axes[0, 1].imshow(terrain_results['slope'], cmap='hot', aspect='auto')
    axes[0, 1].set_title('Slope (degrees)')
    axes[0, 1].set_xlabel('Longitude Index')
    axes[0, 1].set_ylabel('Latitude Index')
    plt.colorbar(im2, ax=axes[0, 1], label='Slope (°)')
    
    # Roughness
    im3 = axes[0, 2].imshow(terrain_results['roughness'], cmap='viridis', aspect='auto')
    axes[0, 2].set_title('Surface Roughness')
    axes[0, 2].set_xlabel('Longitude Index')
    axes[0, 2].set_ylabel('Latitude Index')
    plt.colorbar(im3, ax=axes[0, 2], label='Roughness')
    
    # Obstacles
    im4 = axes[1, 0].imshow(obstacle_results['obstacles'], cmap='Reds', aspect='auto')
    axes[1, 0].set_title('Detected Obstacles')
    axes[1, 0].set_xlabel('Longitude Index')
    axes[1, 0].set_ylabel('Latitude Index')
    plt.colorbar(im4, ax=axes[1, 0], label='Obstacle (0/1)')
    
    # Height difference (DSM - DEM)
    height_diff = terrain_results['dsm'] - terrain_results['dem']
    im5 = axes[1, 1].imshow(height_diff, cmap='plasma', aspect='auto')
    axes[1, 1].set_title('Building Heights (DSM - DEM)')
    axes[1, 1].set_xlabel('Longitude Index')
    axes[1, 1].set_ylabel('Latitude Index')
    plt.colorbar(im5, ax=axes[1, 1], label='Height (m)')
    
    # Combined view
    combined = np.zeros_like(terrain_results['dem'])
    combined[obstacle_results['obstacles']] = 1
    combined[~obstacle_results['obstacles']] = 0.3
    im6 = axes[1, 2].imshow(combined, cmap='RdYlGn', aspect='auto')
    axes[1, 2].set_title('Flight Safety Map')
    axes[1, 2].set_xlabel('Longitude Index')
    axes[1, 2].set_ylabel('Latitude Index')
    plt.colorbar(im6, ax=axes[1, 2], label='Safety (0=Obstacle, 1=Safe)')
    
    plt.tight_layout()
    plt.savefig('terrain_analysis_results.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved as 'terrain_analysis_results.png'")
    
    return fig

def print_summary_stats(terrain_results, obstacle_results):
    """Print summary statistics"""
    print("\n=== SUMMARY STATISTICS ===")
    
    # Terrain statistics
    dem = terrain_results['dem']
    slope = terrain_results['slope']
    roughness = terrain_results['roughness']
    
    print(f"Terrain Analysis:")
    print(f"  - Elevation range: {np.min(dem):.1f}m - {np.max(dem):.1f}m")
    print(f"  - Mean elevation: {np.mean(dem):.1f}m")
    print(f"  - Slope range: {np.min(slope):.1f}° - {np.max(slope):.1f}°")
    print(f"  - Mean slope: {np.mean(slope):.1f}°")
    print(f"  - Roughness range: {np.min(roughness):.3f} - {np.max(roughness):.3f}")
    print(f"  - Mean roughness: {np.mean(roughness):.3f}")
    
    # Obstacle statistics
    obstacles = obstacle_results['obstacles']
    total_pixels = obstacles.size
    obstacle_pixels = np.sum(obstacles)
    
    print(f"\nObstacle Detection:")
    print(f"  - Total pixels: {total_pixels:,}")
    print(f"  - Obstacle pixels: {obstacle_pixels:,} ({obstacle_pixels/total_pixels*100:.1f}%)")
    print(f"  - Safe flight pixels: {total_pixels - obstacle_pixels:,} ({(total_pixels - obstacle_pixels)/total_pixels*100:.1f}%)")
    
    # Building statistics
    if 'dsm' in terrain_results:
        height_diff = terrain_results['dsm'] - terrain_results['dem']
        building_pixels = np.sum(height_diff > 2.0)  # Buildings > 2m
        print(f"\nBuilding Analysis:")
        print(f"  - Building pixels (>2m): {building_pixels:,} ({building_pixels/total_pixels*100:.1f}%)")
        print(f"  - Max building height: {np.max(height_diff):.1f}m")
        print(f"  - Mean building height: {np.mean(height_diff[height_diff > 0]):.1f}m")

def main():
    """Main function to run terrain analysis and obstacle detection"""
    print("=" * 60)
    print("TERRAIN ANALYSIS AND OBSTACLE DETECTION DEMO")
    print("Using Perception Layer with Real Project Data")
    print("=" * 60)
    
    try:
        # Load configuration
        print("\n1. Loading configuration...")
        config = Config('perception-layer/config/perception_config.yaml')
        print(f"✓ Configuration loaded: Working CRS = {config.coordinate_system.working_crs}")
        
        # Load data
        print("\n2. Loading datasets...")
        terrain_df = load_terrain_data()
        buildings_df = load_building_data()
        
        # Create grids
        print("\n3. Creating elevation grids...")
        dem_grid, lats, lons = create_dem_grid(terrain_df, resolution=0.005)
        dsm_grid, _, _ = create_dsm_grid(terrain_df, buildings_df, resolution=0.005)
        
        # Terrain analysis
        print("\n4. Performing terrain analysis...")
        terrain_results = analyze_terrain(dem_grid)
        
        if terrain_results is None:
            print("✗ Terrain analysis failed")
            return
        
        # Add DSM to results
        terrain_results['dsm'] = dsm_grid
        
        # Obstacle detection
        print("\n5. Performing obstacle detection...")
        obstacle_results = detect_obstacles_simple(dem_grid, dsm_grid)
        
        # Visualizations
        print("\n6. Creating visualizations...")
        fig = visualize_results(terrain_results, obstacle_results, lats, lons)
        
        # Summary statistics
        print_summary_stats(terrain_results, obstacle_results)
        
        print("\n" + "=" * 60)
        print("✓ ANALYSIS COMPLETE!")
        print("✓ Results saved to 'terrain_analysis_results.png'")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


