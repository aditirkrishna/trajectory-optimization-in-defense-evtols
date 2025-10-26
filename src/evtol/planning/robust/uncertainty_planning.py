"""
Robust Planning with Uncertainty

Implements chance-constrained and robust optimization for route planning
that accounts for uncertainties in perception data.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class UncertainParameter:
    """Parameter with uncertainty"""
    mean: float
    std: float
    distribution: str = "normal"  # "normal", "uniform", "lognormal"
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Generate random samples."""
        if self.distribution == "normal":
            return np.random.normal(self.mean, self.std, n_samples)
        elif self.distribution == "uniform":
            half_width = self.std * np.sqrt(3)  # Convert std to uniform half-width
            return np.random.uniform(self.mean - half_width, self.mean + half_width, n_samples)
        elif self.distribution == "lognormal":
            return np.random.lognormal(np.log(self.mean), self.std, n_samples)
        else:
            return np.full(n_samples, self.mean)


@dataclass
class ChanceConstraint:
    """Chance constraint with confidence level"""
    constraint_type: str  # "risk", "energy", "time"
    threshold: float
    confidence_level: float  # Probability of satisfying constraint (0-1)


class RobustPlanner:
    """
    Robust route planner with uncertainty propagation.
    
    Methods:
    - Chance-constrained optimization
    - Worst-case optimization
    - Robust counterpart formulation
    - Monte Carlo validation
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize robust planner.
        
        Args:
            confidence_level: Default confidence level for constraints
        """
        self.confidence_level = confidence_level
        self.monte_carlo_samples = 1000
        
    def evaluate_route_robustness(
        self,
        route_cost_params: Dict[str, UncertainParameter],
        constraints: List[ChanceConstraint]
    ) -> Dict[str, any]:
        """
        Evaluate robustness of a route using Monte Carlo simulation.
        
        Args:
            route_cost_params: Uncertain parameters (risk, energy, etc.)
            constraints: List of chance constraints
            
        Returns:
            Robustness metrics
        """
        # Monte Carlo sampling
        samples = {}
        for param_name, param in route_cost_params.items():
            samples[param_name] = param.sample(self.monte_carlo_samples)
        
        # Check constraint violations
        violations = {c.constraint_type: 0 for c in constraints}
        
        for i in range(self.monte_carlo_samples):
            for constraint in constraints:
                param_value = samples[constraint.constraint_type][i]
                if param_value > constraint.threshold:
                    violations[constraint.constraint_type] += 1
        
        # Compute violation rates
        violation_rates = {
            constraint_type: count / self.monte_carlo_samples
            for constraint_type, count in violations.items()
        }
        
        # Overall robustness score (probability all constraints satisfied)
        overall_robustness = 1.0
        for constraint in constraints:
            satisfaction_rate = 1.0 - violation_rates[constraint.constraint_type]
            overall_robustness *= satisfaction_rate
        
        return {
            "overall_robustness": overall_robustness,
            "violation_rates": violation_rates,
            "satisfied_constraints": sum(
                1 for rate in violation_rates.values() 
                if (1.0 - rate) >= self.confidence_level
            ),
            "total_constraints": len(constraints)
        }
    
    def chance_constrained_cost(
        self,
        cost_param: UncertainParameter,
        constraint: ChanceConstraint
    ) -> float:
        """
        Compute deterministic equivalent of chance constraint.
        
        For normal distribution:
        P(X ≤ threshold) ≥ α  →  mean + z_α * std ≤ threshold
        
        Args:
            cost_param: Uncertain cost parameter
            constraint: Chance constraint
            
        Returns:
            Penalty if constraint violated, 0 otherwise
        """
        # Get z-score for confidence level
        z_score = stats.norm.ppf(constraint.confidence_level)
        
        # Deterministic equivalent (upper bound with confidence)
        upper_bound = cost_param.mean + z_score * cost_param.std
        
        # Penalty if violated
        if upper_bound > constraint.threshold:
            penalty = (upper_bound - constraint.threshold) * 10.0  # Large penalty
            return penalty
        
        return 0.0
    
    def worst_case_cost(
        self,
        cost_param: UncertainParameter,
        uncertainty_budget: float = 2.0
    ) -> float:
        """
        Compute worst-case cost within uncertainty budget.
        
        Robust optimization approach: considers worst case within
        specified number of standard deviations.
        
        Args:
            cost_param: Uncertain cost parameter
            uncertainty_budget: Number of std devs for worst case
            
        Returns:
            Worst-case cost
        """
        return cost_param.mean + uncertainty_budget * cost_param.std
    
    def robust_route_comparison(
        self,
        route_costs: List[Dict[str, UncertainParameter]],
        risk_aversion: float = 1.0
    ) -> List[Tuple[int, float]]:
        """
        Compare routes under uncertainty using robust metric.
        
        Uses mean-variance criterion:
        robust_cost = mean + λ * variance
        
        Args:
            route_costs: List of uncertain cost parameters for each route
            risk_aversion: Risk aversion parameter (0=risk-neutral, higher=more conservative)
            
        Returns:
            List of (route_id, robust_score) sorted by score
        """
        scores = []
        
        for route_id, costs in enumerate(route_costs):
            # Aggregate costs
            total_mean = sum(param.mean for param in costs.values())
            total_variance = sum(param.std**2 for param in costs.values())
            
            # Mean-variance score
            robust_score = total_mean + risk_aversion * np.sqrt(total_variance)
            
            scores.append((route_id, robust_score))
        
        # Sort by robust score (ascending = better)
        scores.sort(key=lambda x: x[1])
        
        return scores
    
    def compute_value_at_risk(
        self,
        cost_param: UncertainParameter,
        percentile: float = 0.95
    ) -> float:
        """
        Compute Value at Risk (VaR) for cost.
        
        VaR_α = value at α-percentile of cost distribution
        
        Args:
            cost_param: Uncertain cost parameter
            percentile: Percentile for VaR (0-1)
            
        Returns:
            VaR value
        """
        # For normal distribution
        z_score = stats.norm.ppf(percentile)
        var = cost_param.mean + z_score * cost_param.std
        
        return var
    
    def compute_conditional_value_at_risk(
        self,
        cost_param: UncertainParameter,
        percentile: float = 0.95
    ) -> float:
        """
        Compute Conditional Value at Risk (CVaR).
        
        CVaR_α = expected cost in worst α% cases
        
        Args:
            cost_param: Uncertain cost parameter
            percentile: Percentile for CVaR (0-1)
            
        Returns:
            CVaR value
        """
        # For normal distribution
        z_score = stats.norm.ppf(percentile)
        phi_z = stats.norm.pdf(z_score)
        
        # CVaR formula for normal distribution
        cvar = cost_param.mean + (phi_z / (1 - percentile)) * cost_param.std
        
        return cvar
    
    def adaptive_uncertainty_budget(
        self,
        historical_violations: List[bool],
        initial_budget: float = 2.0,
        learning_rate: float = 0.1
    ) -> float:
        """
        Adaptively adjust uncertainty budget based on historical performance.
        
        Args:
            historical_violations: List of constraint violations (True/False)
            initial_budget: Starting uncertainty budget
            learning_rate: Adaptation rate
            
        Returns:
            Adjusted uncertainty budget
        """
        if not historical_violations:
            return initial_budget
        
        # Compute violation rate
        violation_rate = sum(historical_violations) / len(historical_violations)
        
        # Target violation rate (complement of confidence level)
        target_rate = 1.0 - self.confidence_level
        
        # Adjust budget
        error = violation_rate - target_rate
        adjusted_budget = initial_budget + learning_rate * error * len(historical_violations)
        
        # Clip to reasonable range
        adjusted_budget = np.clip(adjusted_budget, 0.5, 5.0)
        
        return adjusted_budget


class ScenarioBasedPlanner:
    """
    Scenario-based robust planning.
    
    Generates multiple scenarios and ensures feasibility across scenarios.
    """
    
    def __init__(self, n_scenarios: int = 10):
        """
        Initialize scenario planner.
        
        Args:
            n_scenarios: Number of scenarios to generate
        """
        self.n_scenarios = n_scenarios
    
    def generate_scenarios(
        self,
        uncertain_params: Dict[str, UncertainParameter]
    ) -> List[Dict[str, float]]:
        """
        Generate scenarios by sampling uncertain parameters.
        
        Args:
            uncertain_params: Dictionary of uncertain parameters
            
        Returns:
            List of scenarios (each is dict of parameter values)
        """
        scenarios = []
        
        for _ in range(self.n_scenarios):
            scenario = {}
            for param_name, param in uncertain_params.items():
                scenario[param_name] = param.sample(1)[0]
            scenarios.append(scenario)
        
        return scenarios
    
    def evaluate_route_across_scenarios(
        self,
        route_cost_func,
        scenarios: List[Dict[str, float]]
    ) -> Dict[str, any]:
        """
        Evaluate route performance across scenarios.
        
        Args:
            route_cost_func: Function that computes cost given scenario
            scenarios: List of scenarios
            
        Returns:
            Performance metrics
        """
        costs = []
        
        for scenario in scenarios:
            cost = route_cost_func(scenario)
            costs.append(cost)
        
        costs = np.array(costs)
        
        return {
            "mean_cost": float(np.mean(costs)),
            "std_cost": float(np.std(costs)),
            "min_cost": float(np.min(costs)),
            "max_cost": float(np.max(costs)),
            "median_cost": float(np.median(costs)),
            "percentile_95": float(np.percentile(costs, 95))
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create uncertain parameters
    risk_param = UncertainParameter(mean=0.3, std=0.05)
    energy_param = UncertainParameter(mean=15.0, std=2.0)
    
    # Create chance constraints
    constraints = [
        ChanceConstraint("risk", threshold=0.5, confidence_level=0.95),
        ChanceConstraint("energy", threshold=20.0, confidence_level=0.90)
    ]
    
    # Robust planner
    planner = RobustPlanner(confidence_level=0.95)
    
    # Evaluate robustness
    robustness = planner.evaluate_route_robustness(
        route_cost_params={"risk": risk_param, "energy": energy_param},
        constraints=constraints
    )
    
    print("Robustness Analysis:")
    print(f"  Overall robustness: {robustness['overall_robustness']:.3f}")
    print(f"  Violation rates: {robustness['violation_rates']}")
    print(f"  Satisfied constraints: {robustness['satisfied_constraints']}/{robustness['total_constraints']}")
    
    # Compute VaR and CVaR
    var = planner.compute_value_at_risk(energy_param, percentile=0.95)
    cvar = planner.compute_conditional_value_at_risk(energy_param, percentile=0.95)
    
    print(f"\nRisk Metrics:")
    print(f"  VaR (95%): {var:.2f}")
    print(f"  CVaR (95%): {cvar:.2f}")
    
    # Scenario-based analysis
    scenario_planner = ScenarioBasedPlanner(n_scenarios=100)
    scenarios = scenario_planner.generate_scenarios({
        "risk": risk_param,
        "energy": energy_param
    })
    
    print(f"\nGenerated {len(scenarios)} scenarios")

