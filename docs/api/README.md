# API Documentation

This directory contains API documentation for the eVTOL Trajectory Optimization System.

## Structure

- `perception/` - Perception layer API documentation
- `planning/` - Planning layer API documentation  
- `vehicle/` - Vehicle layer API documentation
- `control/` - Control layer API documentation

## Generating Documentation

To generate API documentation:

```bash
# Generate documentation for all layers
python -m sphinx docs/ docs/_build/html

# Generate documentation for specific layer
python -m sphinx docs/api/perception/ docs/_build/html/perception
```

## API Reference

### Perception Layer
- Environment analysis APIs
- Threat detection APIs
- Data fusion APIs

### Planning Layer
- Route optimization APIs
- Mission planning APIs
- Risk assessment APIs

### Vehicle Layer
- Dynamics simulation APIs
- Energy management APIs
- Actuator control APIs

### Control Layer
- Flight controller APIs
- Trajectory generation APIs