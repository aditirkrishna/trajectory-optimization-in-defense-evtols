# Kubernetes Deployment

This directory contains Kubernetes configuration files for deploying the eVTOL system.

## Files
- `namespace.yaml` - Kubernetes namespace
- `configmap.yaml` - Configuration management
- `secrets.yaml` - Secret management
- `deployments/` - Application deployments
- `services/` - Service definitions
- `ingress/` - Ingress configuration

## Usage

### Deploy to Kubernetes
```bash
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml
kubectl apply -f deployments/
kubectl apply -f services/
kubectl apply -f ingress/
```

### Check Deployment Status
```bash
kubectl get pods -n evtol-system
kubectl get services -n evtol-system
```

## Components
- **evtol-api**: Main API deployment
- **evtol-perception**: Perception service deployment
- **evtol-planning**: Planning service deployment
- **evtol-vehicle**: Vehicle service deployment
- **evtol-control**: Control service deployment


