"""
MLflow Setup and Configuration

This script initializes MLflow for experiment tracking, including:
- Creating tracking URI and artifacts directory
- Setting up experiments for each layer
- Providing helper functions for logging
"""

import os
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient


def setup_mlflow(tracking_uri: str = "file:./mlruns", create_experiments: bool = True):
    """
    Setup MLflow tracking server and experiments.
    
    Args:
        tracking_uri: URI for MLflow tracking server
        create_experiments: Whether to create default experiments
    """
    # Set tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
    if create_experiments:
        client = MlflowClient()
        
        # Define experiments for each layer
        experiments = [
            {
                "name": "perception-layer",
                "tags": {
                    "layer": "perception",
                    "description": "Environment perception and data fusion experiments"
                }
            },
            {
                "name": "planning-layer",
                "tags": {
                    "layer": "planning",
                    "description": "Route planning and optimization experiments"
                }
            },
            {
                "name": "vehicle-layer",
                "tags": {
                    "layer": "vehicle",
                    "description": "Vehicle dynamics simulation experiments"
                }
            },
            {
                "name": "integration",
                "tags": {
                    "layer": "integration",
                    "description": "Full system integration experiments"
                }
            },
            {
                "name": "benchmarks",
                "tags": {
                    "layer": "benchmarks",
                    "description": "Benchmark and baseline comparison experiments"
                }
            }
        ]
        
        for exp_config in experiments:
            try:
                exp_id = client.create_experiment(
                    name=exp_config["name"],
                    tags=exp_config["tags"]
                )
                print(f"Created experiment: {exp_config['name']} (ID: {exp_id})")
            except Exception as e:
                if "already exists" in str(e):
                    print(f"Experiment already exists: {exp_config['name']}")
                else:
                    print(f"Error creating experiment {exp_config['name']}: {e}")
    
    print("\nMLflow setup complete!")
    print(f"View experiments: mlflow ui --backend-store-uri {tracking_uri}")


def log_perception_metrics(
    experiment_name: str = "perception-layer",
    run_name: str = None,
    metrics: dict = None,
    params: dict = None,
    artifacts: dict = None
):
    """
    Log perception layer metrics to MLflow.
    
    Args:
        experiment_name: Name of experiment
        run_name: Name for this run
        metrics: Dictionary of metrics to log
        params: Dictionary of parameters to log
        artifacts: Dictionary of artifacts to log (path -> artifact_name)
    """
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        if params:
            for key, value in params.items():
                mlflow.log_param(key, value)
        
        # Log metrics
        if metrics:
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
        
        # Log artifacts
        if artifacts:
            for path, artifact_name in artifacts.items():
                mlflow.log_artifact(path, artifact_name)
        
        print(f"Logged run: {run.info.run_id}")
        return run.info.run_id


def log_planning_metrics(
    experiment_name: str = "planning-layer",
    run_name: str = None,
    route_cost: float = None,
    energy_kwh: float = None,
    risk_score: float = None,
    num_waypoints: int = None,
    planning_time_ms: float = None,
    params: dict = None
):
    """
    Log planning layer metrics to MLflow.
    
    Args:
        experiment_name: Name of experiment
        run_name: Name for this run
        route_cost: Total route cost
        energy_kwh: Estimated energy consumption in kWh
        risk_score: Risk assessment score
        num_waypoints: Number of waypoints in route
        planning_time_ms: Planning computation time in ms
        params: Additional parameters
    """
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=run_name) as run:
        # Log metrics
        if route_cost is not None:
            mlflow.log_metric("route_cost", route_cost)
        if energy_kwh is not None:
            mlflow.log_metric("energy_kwh", energy_kwh)
        if risk_score is not None:
            mlflow.log_metric("risk_score", risk_score)
        if num_waypoints is not None:
            mlflow.log_metric("num_waypoints", num_waypoints)
        if planning_time_ms is not None:
            mlflow.log_metric("planning_time_ms", planning_time_ms)
        
        # Log parameters
        if params:
            for key, value in params.items():
                mlflow.log_param(key, value)
        
        print(f"Logged planning run: {run.info.run_id}")
        return run.info.run_id


if __name__ == "__main__":
    # Setup MLflow
    project_root = Path(__file__).parent.parent
    tracking_uri = f"file://{project_root}/mlruns"
    
    setup_mlflow(tracking_uri=tracking_uri, create_experiments=True)
    
    # Example usage
    print("\nExample: Log a sample perception run")
    log_perception_metrics(
        run_name="terrain_analysis_demo",
        metrics={
            "mean_slope_deg": 33.6,
            "mean_roughness": 54.8,
            "obstacle_pixels": 0,
            "safe_flight_percentage": 100.0
        },
        params={
            "base_resolution_m": 2.0,
            "dem_method": "horn",
            "area_km2": 165.0
        }
    )
    
    print("\nExample: Log a sample planning run")
    log_planning_metrics(
        run_name="bangalore_mission_1",
        route_cost=42.5,
        energy_kwh=15.3,
        risk_score=0.23,
        num_waypoints=25,
        planning_time_ms=87.3,
        params={
            "algorithm": "a_star",
            "objective": "multi_objective",
            "start_lat": 12.9716,
            "start_lon": 77.5946
        }
    )

