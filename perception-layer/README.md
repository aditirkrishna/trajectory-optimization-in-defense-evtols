# Perception and Environment Layer for eVTOL Trajectory Optimization

## Overview

The Perception and Environment Layer is the first of four layers in trajectory optimization in the eVTOL defense system. It produces spatiotemporal maps that the planner can query at runtime, including:

- **Terrain and obstacles**: Elevation, slope, roughness, building footprints
- **Atmospheric conditions**: Wind fields, turbulence, air density, temperature
- **Threat assessment**: Radar detection, patrol coverage, electronic warfare zones
- **Risk evaluation**: Detection probabilities, communication degradation, GPS reliability

The layer provides **accurate, versioned, georeferenced, uncertainty-aware, and fast-to-query** maps for the eVTOL planner.

## Key Features

- **Multi-source data fusion**: Combines DEM, building data, weather, threats
- **3D spatiotemporal processing**: Handles altitude and time dimensions
- **Uncertainty quantification**: Provides confidence intervals for all outputs
- **Fast query interface**: Optimized for real-time planner access
- **Research-grade quality**: Comprehensive validation and provenance tracking

## Project Structure

```
perception-layer/
├── config/                     # Configuration files
│   └── perception_config.yaml  # Main configuration
├── src/                        # Source code
│   ├── preprocessing/          # Data ingestion and preprocessing
│   ├── geometry/              # Terrain derivatives (slope, roughness)
│   ├── urban/                 # Building and urban obstacle processing
│   ├── atmosphere/            # Wind and atmospheric modeling
│   ├── fusion/                # Data fusion and layer stacking
│   ├── threats/               # Threat and risk assessment
│   ├── uncertainty/           # Uncertainty quantification
│   ├── derived/               # Precomputed derived maps
│   ├── serving/               # API and tile serving
│   └── utils/                 # Utilities and helpers
├── data/                      # Data directories
│   ├── raw/                   # Raw input data
│   ├── processed/             # Preprocessed data
│   └── derived/               # Derived products
├── tests/                     # Unit and integration tests
├── docs/                      # Documentation
├── scripts/                   # Processing scripts
├── examples/                  # Usage examples
└── requirements.txt           # Python dependencies
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd perception-layer

# Install dependencies
pip install -r requirements.txt

# Install GDAL (system dependency)
# On Ubuntu: sudo apt-get install gdal-bin libgdal-dev
# On Windows: Download from OSGeo4W
# On macOS: brew install gdal
```

### 2. Basic Usage

```python
from perception_layer import setup_perception_layer, DataLoader

# Setup with default configuration
config, logger = setup_perception_layer()

# Load and process data
loader = DataLoader(config)
elevation_data = loader.load_elevation("path/to/dem.tif")
wind_data = loader.load_wind("path/to/wind.nc")

# Query the fused map
from perception_layer import MapFusion
fusion = MapFusion(config)
result = fusion.query(lat=45.0, lon=-122.0, alt=100, time="2024-01-01T12:00:00")
```

### 3. Configuration

Edit `config/perception_config.yaml` to customize:

- **Coordinate systems**: Working CRS and input formats
- **Spatial resolution**: Base resolution and tile sizes
- **Processing parameters**: Interpolation methods, quality thresholds
- **Threat models**: Radar detection, patrol parameters
- **Serving options**: API settings, caching configuration

## Data Requirements

### Input Data Sources

1. **Terrain Data**
   - DEM/DSM: SRTM, Copernicus, LiDAR point clouds
   - Format: GeoTIFF, COG, LAS/LAZ

2. **Building Data**
   - Building footprints and heights: OSM, municipal data
   - Format: GeoJSON, Shapefile

3. **Atmospheric Data**
   - Wind fields: ERA5, WRF, local sensors
   - Format: NetCDF, Zarr

4. **Threat Data**
   - Radar sites: Location, parameters, schedules
   - Patrol routes: Trajectories, detection ranges
   - Format: GeoJSON, CSV

### Data Organization

```
data/
├── raw/
│   ├── terrain/
│   │   ├── dem/
│   │   └── buildings/
│   ├── atmosphere/
│   │   ├── wind/
│   │   └── temperature/
│   └── threats/
│       ├── radar/
│       └── patrols/
├── processed/
│   ├── elevation/
│   ├── slope/
│   └── roughness/
└── derived/
    ├── fusion_maps/
    ├── risk_assessment/
    └── energy_cost/
```

## Processing Pipeline

### Phase 1: Preprocessing
1. **Coordinate transformation**: Convert to working CRS
2. **Resampling**: Standardize spatial resolution
3. **Gap filling**: Interpolate missing data
4. **Quality control**: Validate data integrity

### Phase 2: Geometry Processing
1. **Slope calculation**: Terrain steepness
2. **Roughness computation**: Surface variability
3. **Obstacle detection**: Identify barriers
4. **Building processing**: 3D urban volumes

### Phase 3: Atmospheric Modeling
1. **Wind interpolation**: 3D wind fields
2. **Turbulence modeling**: Gust effects
3. **Air density**: Standard atmosphere calculations

### Phase 4: Threat Assessment
1. **Radar detection**: Line-of-sight analysis
2. **Patrol coverage**: Time-expanded detection
3. **EW zones**: Communication degradation

### Phase 5: Data Fusion
1. **Layer stacking**: Combine all data sources
2. **Uncertainty propagation**: Confidence intervals
3. **Derived products**: Energy costs, feasibility masks

## API Reference

### Core Classes

#### `Config`
Configuration management for all processing parameters.

```python
config = Config("path/to/config.yaml")
resolution = config.get_base_resolution()
crs = config.get_working_crs()
```

#### `DataLoader`
Handles data ingestion and preprocessing.

```python
loader = DataLoader(config)
elevation = loader.load_elevation("dem.tif")
buildings = loader.load_buildings("buildings.geojson")
```

#### `MapFusion`
Combines all layers into queryable maps.

```python
fusion = MapFusion(config)
result = fusion.query(lat, lon, alt, time)
```

#### `PerceptionAPI`
FastAPI server for remote access.

```python
api = PerceptionAPI(config)
api.start()  # Starts server on localhost:8000
```

### Query Interface

```python
# Single point query
result = fusion.query(
    lat=45.0, 
    lon=-122.0, 
    alt=100, 
    time="2024-01-01T12:00:00"
)

# Batch query
results = fusion.batch_query([
    {"lat": 45.0, "lon": -122.0, "alt": 100, "time": "2024-01-01T12:00:00"},
    {"lat": 45.1, "lon": -122.1, "alt": 150, "time": "2024-01-01T12:00:00"}
])

# Line-of-sight analysis
los = fusion.line_of_sight(
    start=(lat1, lon1, alt1),
    end=(lat2, lon2, alt2),
    time="2024-01-01T12:00:00"
)
```

## Performance

### Benchmarks
- **Query time**: < 10ms for single point queries
- **Batch queries**: < 100ms for 1000 points
- **Memory usage**: < 2GB for typical mission area
- **Storage**: ~1GB per 100km² at 2m resolution

### Optimization
- **Tiling**: 512x512 pixel tiles for efficient access
- **Caching**: Redis-based caching for hot data
- **Compression**: COG format for fast windowed reads
- **Indexing**: Quadkey/H3 spatial indexing

## Quality Assurance

### Validation
- **Cross-validation**: Compare against withheld data
- **Synthetic tests**: Known obstacle detection
- **Sensitivity analysis**: Parameter uncertainty
- **Performance monitoring**: Query latency tracking

### Reports
- **Weekly QC reports**: Data quality metrics
- **Processing logs**: Full provenance tracking
- **Error alerts**: Automated quality monitoring

## Development

### Testing
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_preprocessing.py::test_elevation_loading
```

### Code Quality
```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

## Acknowledgments

- GDAL/OGR for geospatial data handling
- xarray for multi-dimensional data
- FastAPI for web serving
- Scientific Python community for core libraries
