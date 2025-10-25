# Docker Deployment

This directory contains Docker configuration files for deploying the eVTOL system.

## Files
- `Dockerfile` - Main application Docker image
- `docker-compose.yml` - Multi-service deployment
- `docker-compose.prod.yml` - Production deployment
- `docker-compose.dev.yml` - Development deployment

## Usage

### Development
```bash
docker-compose -f docker-compose.dev.yml up
```

### Production
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Build Custom Image
```bash
docker build -t evtol-system .
```

## Services
- **evtol-api**: Main API service
- **evtol-perception**: Perception layer service
- **evtol-planning**: Planning layer service
- **evtol-vehicle**: Vehicle layer service
- **evtol-control**: Control layer service
- **redis**: Caching service
- **postgres**: Database service


