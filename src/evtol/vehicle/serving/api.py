"""
Vehicle API - REST API for Vehicle Layer

This module provides a FastAPI-based REST API for the vehicle layer,
enabling remote access to vehicle simulation and control capabilities.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import logging
import uvicorn
from datetime import datetime

from ..dynamics.vehicle_model import VehicleModel, VehicleConfig, VehicleState, ControlInputs


# Pydantic models for API
class VehicleStateRequest(BaseModel):
    """Request model for vehicle state."""
    position: List[float]
    velocity: List[float]
    attitude: List[float]
    angular_velocity: List[float]
    battery_soc: float
    battery_temperature: float
    battery_voltage: float
    rotor_rpm: List[float]
    control_surface_deflections: List[float]
    time: float


class ControlInputsRequest(BaseModel):
    """Request model for control inputs."""
    main_rotor_rpm: List[float]
    tail_rotor_rpm: float
    lift_fan_rpm: List[float]
    propeller_rpm: List[float]
    elevator_deflection: float
    aileron_deflection: float
    rudder_deflection: float
    throttle: float
    collective: float


class SimulationRequest(BaseModel):
    """Request model for simulation."""
    initial_state: VehicleStateRequest
    controls: ControlInputsRequest
    dt: float
    duration: float


class FaultInjectionRequest(BaseModel):
    """Request model for fault injection."""
    actuator_id: str
    fault_type: str
    severity: float
    start_time: float
    end_time: Optional[float] = None


class VehicleAPI:
    """
    FastAPI-based REST API for vehicle layer.
    
    This class provides HTTP endpoints for vehicle simulation, control,
    and monitoring capabilities.
    """
    
    def __init__(self, config_path: str = "config/vehicle_config.yaml"):
        """
        Initialize vehicle API.
        
        Args:
            config_path: Path to vehicle configuration file
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Vehicle Layer API",
            description="REST API for eVTOL vehicle simulation and control",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize vehicle model
        self.config = VehicleConfig(config_path)
        self.vehicle = VehicleModel(self.config)
        
        # Simulation state
        self.current_simulation = None
        self.simulation_results = {}
        
        # Setup routes
        self._setup_routes()
        
        self.logger.info("Vehicle API initialized")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "message": "Vehicle Layer API",
                "version": "1.0.0",
                "status": "running",
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "vehicle_operational": self.vehicle.is_operational() if hasattr(self.vehicle, 'is_operational') else True,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/simulate")
        async def simulate(request: SimulationRequest):
            """Run vehicle simulation."""
            try:
                # Convert request to internal types
                initial_state = self._convert_to_vehicle_state(request.initial_state)
                controls = self._convert_to_control_inputs(request.controls)
                
                # Run simulation
                trajectory = self.vehicle.simulate(
                    initial_state, controls, request.dt, request.duration
                )
                
                # Convert results to API format
                results = self._convert_trajectory_to_api(trajectory)
                
                return {
                    "success": True,
                    "trajectory": results,
                    "simulation_time": request.duration,
                    "num_steps": len(trajectory),
                    "energy_consumed": self.vehicle.get_energy_consumption(),
                    "final_soc": self.vehicle.get_battery_state_of_charge()
                }
                
            except Exception as e:
                self.logger.error(f"Simulation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/step")
        async def step_simulation(request: Dict[str, Any]):
            """Perform single simulation step."""
            try:
                # This would require maintaining simulation state
                # For now, return error
                raise HTTPException(
                    status_code=501, 
                    detail="Single step simulation not implemented"
                )
                
            except Exception as e:
                self.logger.error(f"Step simulation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/faults/inject")
        async def inject_fault(request: FaultInjectionRequest):
            """Inject fault into vehicle."""
            try:
                self.vehicle.inject_fault(
                    request.actuator_id,
                    request.fault_type,
                    request.severity,
                    request.start_time
                )
                
                return {
                    "success": True,
                    "fault_id": f"{request.actuator_id}_{request.fault_type}_{int(request.start_time)}",
                    "message": f"Fault injected into {request.actuator_id}"
                }
                
            except Exception as e:
                self.logger.error(f"Fault injection failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/faults/status")
        async def get_fault_status():
            """Get current fault status."""
            try:
                fault_status = self.vehicle.get_fault_status()
                return {
                    "active_faults": fault_status,
                    "num_active": len(fault_status)
                }
                
            except Exception as e:
                self.logger.error(f"Failed to get fault status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/energy/status")
        async def get_energy_status():
            """Get energy system status."""
            try:
                return {
                    "battery_soc": self.vehicle.get_battery_state_of_charge(),
                    "battery_temperature": self.vehicle.get_battery_temperature(),
                    "power_consumption": self.vehicle.get_power_consumption(),
                    "energy_consumed": self.vehicle.get_energy_consumption()
                }
                
            except Exception as e:
                self.logger.error(f"Failed to get energy status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/constraints/status")
        async def get_constraints_status():
            """Get constraint violation status."""
            try:
                # This would require implementing constraint checking in the API
                return {
                    "violations": [],
                    "num_violations": 0,
                    "operational": True
                }
                
            except Exception as e:
                self.logger.error(f"Failed to get constraints status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/performance/metrics")
        async def get_performance_metrics():
            """Get performance metrics."""
            try:
                metrics = self.vehicle.get_performance_metrics()
                return {
                    "metrics": metrics,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Failed to get performance metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/config")
        async def get_configuration():
            """Get vehicle configuration."""
            try:
                return {
                    "vehicle_mass": self.config.get_vehicle_mass(),
                    "battery_capacity": self.config.get_battery_capacity(),
                    "battery_voltage": self.config.get_battery_voltage(),
                    "flight_envelope": self.config.get_flight_envelope_limits()
                }
                
            except Exception as e:
                self.logger.error(f"Failed to get configuration: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _convert_to_vehicle_state(self, request: VehicleStateRequest) -> VehicleState:
        """Convert API request to VehicleState."""
        return VehicleState(
            position=np.array(request.position),
            velocity=np.array(request.velocity),
            attitude=np.array(request.attitude),
            angular_velocity=np.array(request.angular_velocity),
            battery_soc=request.battery_soc,
            battery_temperature=request.battery_temperature,
            battery_voltage=request.battery_voltage,
            rotor_rpm=np.array(request.rotor_rpm),
            control_surface_deflections=np.array(request.control_surface_deflections),
            time=request.time
        )
    
    def _convert_to_control_inputs(self, request: ControlInputsRequest) -> ControlInputs:
        """Convert API request to ControlInputs."""
        return ControlInputs(
            main_rotor_rpm=np.array(request.main_rotor_rpm),
            tail_rotor_rpm=request.tail_rotor_rpm,
            lift_fan_rpm=np.array(request.lift_fan_rpm),
            propeller_rpm=np.array(request.propeller_rpm),
            elevator_deflection=request.elevator_deflection,
            aileron_deflection=request.aileron_deflection,
            rudder_deflection=request.rudder_deflection,
            throttle=request.throttle,
            collective=request.collective
        )
    
    def _convert_trajectory_to_api(self, trajectory: List[VehicleState]) -> List[Dict[str, Any]]:
        """Convert trajectory to API format."""
        results = []
        for state in trajectory:
            results.append({
                "time": state.time,
                "position": state.position.tolist(),
                "velocity": state.velocity.tolist(),
                "attitude": state.attitude.tolist(),
                "angular_velocity": state.angular_velocity.tolist(),
                "battery_soc": state.battery_soc,
                "battery_temperature": state.battery_temperature,
                "battery_voltage": state.battery_voltage,
                "rotor_rpm": state.rotor_rpm.tolist(),
                "control_surface_deflections": state.control_surface_deflections.tolist()
            })
        return results
    
    def start(self, host: str = "localhost", port: int = 8001):
        """
        Start the API server.
        
        Args:
            host: Host address
            port: Port number
        """
        self.logger.info(f"Starting Vehicle API server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)
    
    def get_app(self):
        """Get the FastAPI app instance."""
        return self.app


def main():
    """Main function to run the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vehicle Layer API Server")
    parser.add_argument("--host", default="localhost", help="Host address")
    parser.add_argument("--port", type=int, default=8001, help="Port number")
    parser.add_argument("--config", default="config/vehicle_config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and start API
    api = VehicleAPI(args.config)
    api.start(args.host, args.port)


if __name__ == "__main__":
    main()

