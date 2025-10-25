eVTOL Trajectory Optimization for Defense Applications
=====================================================

Welcome to the documentation for the eVTOL Trajectory Optimization System - a comprehensive, research-grade framework for safe, efficient, and stealthy trajectory planning in defense applications.

.. image:: https://img.shields.io/badge/Python-3.11-blue
   :alt: Python 3.11

.. image:: https://img.shields.io/badge/License-MIT-green
   :alt: MIT License

Overview
--------

This system provides a complete 4-layer architecture for eVTOL trajectory optimization:

1. **Perception Layer**: Environmental intelligence and situational awareness
2. **Planning Layer**: Route optimization and mission planning
3. **Vehicle Layer**: High-fidelity dynamics modeling
4. **Control Layer**: Flight control and trajectory tracking

Key Features
-----------

✅ **Multi-Layer Architecture**
   - Modular design with clear separation of concerns
   - Each layer independently testable and deployable

✅ **Perception & Fusion**
   - Wind field modeling with 3D interpolation
   - Turbulence estimation and flight impact assessment
   - Radar detection modeling with physics-based equations
   - Patrol coverage and encounter probability
   - Electronic warfare zone assessment
   - Multi-source data fusion with uncertainty quantification

✅ **Advanced Planning**
   - A* pathfinding with multi-objective costs
   - Dynamic feasibility constraints (turn radius, climb/descent)
   - Pareto frontier multi-objective optimization
   - Robust planning with uncertainty propagation
   - Trajectory smoothing with curvature bounds

✅ **Vehicle Dynamics**
   - 6-DOF rigid body dynamics
   - Battery modeling (Li-ion NMC, Li-S, Solid State)
   - Actuator models with fault injection
   - Flight envelope constraints

✅ **Flight Control**
   - Hierarchical PID control (position, velocity, attitude)
   - Trajectory tracking with feedforward
   - Time-optimal trajectory generation

✅ **Research Quality**
   - MLflow experiment tracking
   - Comprehensive benchmarking suite
   - CI/CD with GitHub Actions
   - Pre-commit hooks for code quality

Quick Start
-----------

Installation
^^^^^^^^^^^

.. code-block:: bash

   # Create conda environment
   conda env create -f environment.yml
   conda activate evtol

   # Install all layers
   pip install -e perception-layer/
   pip install -e planning-layer/
   pip install -e vehicle-layer/

   # Install pre-commit hooks
   pre-commit install

Basic Usage
^^^^^^^^^^

.. code-block:: python

   from planning_layer import setup_planning_layer, RoutePlanner

   # Setup
   config, logger = setup_planning_layer()
   planner = RoutePlanner(config)

   # Plan a route
   route = planner.optimize_route(
       start_lat=13.0, start_lon=77.5,
       goal_lat=13.1, goal_lon=77.6,
       start_alt_m=120.0,
       time_iso="2024-01-01T12:00:00"
   )

   print(f"Route generated with {len(route)} waypoints")

System Architecture
------------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────┐
   │                    eVTOL System                         │
   ├─────────────────────────────────────────────────────────┤
   │  Layer 4: Control      │  Flight Control & Tracking     │
   │  Layer 3: Vehicle      │  Dynamics & Energy Management  │
   │  Layer 2: Planning     │  Route Planning & Optimization │
   │  Layer 1: Perception   │  Environment Intelligence      │
   ├─────────────────────────────────────────────────────────┤
   │                Data & Infrastructure                     │
   └─────────────────────────────────────────────────────────┘

Table of Contents
----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorials
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/perception
   api/planning
   api/vehicle
   api/control

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   architecture
   benchmarks
   testing

.. toctree::
   :maxdepth: 1
   :caption: Additional

   changelog
   license
   references

Performance
-----------

Benchmarked Performance:

- **Query Time**: < 10ms for single point queries
- **Planning Time**: < 500ms for typical routes
- **Trajectory Generation**: < 100ms for 60s flight
- **Memory Usage**: < 2GB per mission

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



