# eVTOL Trajectory Optimization for Defense Applications

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive, research-grade trajectory optimization system for electric Vertical Take-Off and Landing (eVTOL) aircraft in defense applications.

## 🎯 Overview

This project implements a complete 4-layer architecture for eVTOL trajectory optimization:

1. **Perception Layer** - Environmental intelligence and situational awareness
2. **Planning Layer** - Route optimization and mission planning  
3. **Vehicle Layer** - High-fidelity dynamics modeling
4. **Control Layer** - Flight control and trajectory tracking

## ✨ Key Features

### Perception & Environment Analysis
- ✅ Wind field modeling with 3D interpolation (Linear, RBF, IDW)
- ✅ Turbulence intensity estimation and gust modeling
- ✅ Radar detection probability (physics-based radar equation)
- ✅ Patrol coverage and encounter probability analysis
- ✅ Electronic warfare zone assessment
- ✅ Multi-source data fusion with uncertainty quantification
- ✅ FastAPI server for real-time queries

### Advanced Planning
- ✅ A* pathfinding with multi-objective costs
- ✅ Dynamic feasibility constraints (turn radius, climb/descent rates)
- ✅ Pareto frontier multi-objective optimization
- ✅ Robust planning with chance constraints
- ✅ Trajectory smoothing with curvature bounds
- ✅ K-best diverse route generation

### Vehicle Dynamics
- ✅ 6-DOF rigid body dynamics simulation
- ✅ Battery modeling (Li-ion NMC, Li-S, Solid State)
- ✅ Motor and actuator models with fault injection
- ✅ Flight envelope constraints

### Flight Control
- ✅ Hierarchical PID control (position → velocity → attitude)
- ✅ Trajectory tracking with feedforward
- ✅ Time-optimal trajectory generation
- ✅ Anti-windup protection

### Research Infrastructure
- ✅ MLflow experiment tracking
- ✅ Comprehensive benchmark suite (8 scenarios)
- ✅ Automated benchmarking pipeline
- ✅ GitHub Actions CI/CD
- ✅ Pre-commit hooks (Black, Ruff, mypy, bandit)
- ✅ Sphinx documentation

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd trajectory-optimization-in-defense-evtols

# Create conda environment
conda env create -f environment.yml
conda activate evtol

# Install all layers
pip install -e perception-layer/
pip install -e planning-layer/
pip install -e vehicle-layer/

# Setup pre-commit hooks
pre-commit install
```

### Basic Usage

```python
from planning_layer import setup_planning_layer, RoutePlanner, EnergyOptimizer

# Setup planning layer
config, logger = setup_planning_layer()
planner = RoutePlanner(config)

# Plan a route
route = planner.optimize_route(
    start_lat=13.0, start_lon=77.5,
    goal_lat=13.1, goal_lon=77.6,
    start_alt_m=120.0,
    time_iso="2024-01-01T12:00:00"
)

# Assess energy and risk
energy = EnergyOptimizer(config).estimate_route_energy(route)
print(f"Route: {len(route)} waypoints, Energy: {energy:.2f} kWh")
```

### Running Benchmarks

```bash
# Run all benchmarks
python benchmarks/run_benchmarks.py

# View results
cat benchmarks/results/benchmark_report_summary.txt

# View in MLflow
mlflow ui --backend-store-uri file:./mlruns
```

### Integration Testing

```bash
# Run comprehensive integration tests
python integration_test.py
```

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    eVTOL System                         │
├─────────────────────────────────────────────────────────┤
│  Layer 4: Control      │  Flight Control & Tracking     │
│  Layer 3: Vehicle      │  Dynamics & Energy Management  │
│  Layer 2: Planning     │  Route Planning & Optimization │
│  Layer 1: Perception   │  Environment Intelligence      │
├─────────────────────────────────────────────────────────┤
│            Data, Infrastructure & Benchmarks            │
└─────────────────────────────────────────────────────────┘
```

## 📈 Performance

- **Query Time**: < 10ms for single point queries
- **Planning Time**: < 500ms for typical routes (A* with 1000 nodes)
- **Trajectory Generation**: < 100ms for 60s flight
- **Memory Usage**: < 2GB per mission area

## 🔬 Research Quality

This project follows research-grade best practices:

- ✅ Reproducible experiments with MLflow tracking
- ✅ Comprehensive benchmark suite with 8 diverse scenarios
- ✅ Uncertainty quantification throughout pipeline
- ✅ Automated testing (unit + integration)
- ✅ Code quality enforcement (>85% coverage target)
- ✅ Complete documentation with Sphinx

## 📚 Documentation

- **Full Documentation**: See `docs/` directory
- **Architecture**: `architecture/architecture.md`
- **API Reference**: Auto-generated from docstrings
- **Examples**: Check `*/examples/` directories
- **Contributing**: See `CONTRIBUTING.md`

## 🗂️ Project Structure

```
trajectory-optimization-in-defense-evtols/
├── perception-layer/        # Layer 1: Environment perception
│   ├── src/
│   │   ├── atmosphere/     # Wind & turbulence models
│   │   ├── threats/        # Radar, patrols, EW zones
│   │   ├── fusion/         # Multi-source data fusion
│   │   ├── geometry/       # Terrain analysis
│   │   └── serving/        # FastAPI server
│   └── tests/
├── planning-layer/          # Layer 2: Route planning
│   ├── src/planning_layer/
│   │   ├── routing/        # A* and graph planners
│   │   ├── optimization/   # Pareto frontier
│   │   ├── robust/         # Uncertainty-aware planning
│   │   ├── constraints/    # Flight constraints
│   │   └── smoothing/      # Trajectory smoothing
│   └── tests/
├── vehicle-layer/           # Layer 3: Vehicle dynamics
│   ├── src/
│   │   ├── dynamics/       # 6-DOF dynamics
│   │   ├── energy/         # Battery models
│   │   ├── actuators/      # Motor & ESC models
│   │   └── constraints/    # Flight envelope
│   └── tests/
├── control-layer/           # Layer 4: Flight control
│   └── src/control/
│       ├── flight_controller.py
│       └── trajectory_generator.py
├── benchmarks/              # Benchmark suite
│   ├── scenarios.yaml
│   └── run_benchmarks.py
├── docs/                    # Sphinx documentation
├── scripts/                 # Utility scripts
│   ├── setup_mlflow.py
│   └── run_experiment.py
└── project-datasets/        # Mission data
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=perception-layer/src --cov=planning-layer/src --cov=vehicle-layer/src

# Run specific layer
pytest perception-layer/tests -v

# Run integration tests
python integration_test.py
```

## 🤝 Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines.

### Development Workflow

1. Create a feature branch
2. Make changes with tests
3. Run pre-commit hooks: `pre-commit run --all-files`
4. Submit pull request

## 📊 Benchmark Scenarios

The project includes 8 comprehensive benchmark scenarios:

1. **Urban Bangalore - Short Distance** (Easy)
2. **Urban Bangalore - Complex Route** (Medium)
3. **Mountainous Terrain Navigation** (Hard)
4. **GPS-Denied Zone Operation** (Hard)
5. **High-Threat Penetration Mission** (Extreme)
6. **Long-Range Endurance Mission** (Medium)
7. **Multiple Delivery Points** (Medium)
8. **Emergency Landing Scenario** (Extreme)

## 📜 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- GDAL/OGR for geospatial data handling
- FastAPI for web serving
- MLflow for experiment tracking
- Scientific Python community

## 📧 Contact

For questions or collaboration: IISc Research Team

---

**Project Status**: Active Development | Version 0.1.0  
**Last Updated**: October 2024

