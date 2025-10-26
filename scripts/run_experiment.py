"""
Experiment Runner for Reproducible Research

This script runs experiments with MLflow tracking for reproducibility.
"""

import argparse
import time
import yaml
import numpy as np
from pathlib import Path
import sys

# Add layers to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "perception-layer" / "src"))
sys.path.insert(0, str(project_root / "planning-layer" / "src"))
sys.path.insert(0, str(project_root / "vehicle-layer" / "src"))

import mlflow
from setup_mlflow import setup_mlflow, log_planning_metrics


def run_planning_experiment(
    start_lat: float,
    start_lon: float,
    goal_lat: float,
    goal_lon: float,
    alt_m: float = 120.0,
    algorithm: str = "a_star",
    seed: int = 42
):
    """
    Run a planning experiment with full MLflow tracking.
    
    Args:
        start_lat: Start latitude
        start_lon: Start longitude
        goal_lat: Goal latitude
        goal_lon: Goal longitude
        alt_m: Altitude in meters
        algorithm: Planning algorithm to use
        seed: Random seed for reproducibility
    """
    # Set random seeds for reproducibility
    np.random.seed(seed)
    
    # Import planning layer
    from planning_layer import setup_planning_layer, RoutePlanner, EnergyOptimizer, RiskManager
    
    # Setup
    config, logger = setup_planning_layer()
    planner = RoutePlanner(config)
    energy_optimizer = EnergyOptimizer(config)
    risk_manager = RiskManager(config)
    
    # Plan route with timing
    start_time = time.time()
    route = planner.optimize_route(
        start_lat=start_lat,
        start_lon=start_lon,
        goal_lat=goal_lat,
        goal_lon=goal_lon,
        start_alt_m=alt_m,
        time_iso="2024-01-01T12:00:00"
    )
    planning_time_ms = (time.time() - start_time) * 1000
    
    # Compute metrics
    energy_kwh = energy_optimizer.estimate_route_energy(route)
    risk_score = risk_manager.evaluate_route_risk(route)
    route_cost = planner.compute_route_cost(route, time_iso="2024-01-01T12:00:00")
    
    # Log to MLflow
    run_id = log_planning_metrics(
        run_name=f"{algorithm}_{start_lat}_{start_lon}_to_{goal_lat}_{goal_lon}",
        route_cost=route_cost,
        energy_kwh=energy_kwh,
        risk_score=risk_score,
        num_waypoints=len(route),
        planning_time_ms=planning_time_ms,
        params={
            "algorithm": algorithm,
            "start_lat": start_lat,
            "start_lon": start_lon,
            "goal_lat": goal_lat,
            "goal_lon": goal_lon,
            "alt_m": alt_m,
            "seed": seed
        }
    )
    
    print(f"\n=== Experiment Results ===")
    print(f"Run ID: {run_id}")
    print(f"Algorithm: {algorithm}")
    print(f"Route cost: {route_cost:.3f}")
    print(f"Energy: {energy_kwh:.2f} kWh")
    print(f"Risk score: {risk_score:.3f}")
    print(f"Waypoints: {len(route)}")
    print(f"Planning time: {planning_time_ms:.1f} ms")
    
    return run_id, route


def main():
    parser = argparse.ArgumentParser(description="Run reproducible planning experiments")
    parser.add_argument("--start-lat", type=float, default=45.0, help="Start latitude")
    parser.add_argument("--start-lon", type=float, default=-122.0, help="Start longitude")
    parser.add_argument("--goal-lat", type=float, default=45.2, help="Goal latitude")
    parser.add_argument("--goal-lon", type=float, default=-122.3, help="Goal longitude")
    parser.add_argument("--alt", type=float, default=120.0, help="Altitude in meters")
    parser.add_argument("--algorithm", type=str, default="a_star", help="Planning algorithm")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Setup MLflow
    setup_mlflow(create_experiments=False)
    
    # Run experiment
    run_planning_experiment(
        start_lat=args.start_lat,
        start_lon=args.start_lon,
        goal_lat=args.goal_lat,
        goal_lon=args.goal_lon,
        alt_m=args.alt,
        algorithm=args.algorithm,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

