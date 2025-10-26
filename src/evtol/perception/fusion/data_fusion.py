"""
Data Fusion Module

Combines multiple perception layers (terrain, atmospheric, threats) into
fused risk and energy cost maps with uncertainty quantification.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FusionMethod(Enum):
    """Data fusion methods"""
    WEIGHTED_SUM = "weighted_sum"
    BAYESIAN = "bayesian"
    DEMPSTER_SHAFER = "dempster_shafer"
    FUZZY_LOGIC = "fuzzy_logic"


@dataclass
class LayerData:
    """Data from a single perception layer"""
    layer_name: str
    value: float  # Normalized value (0-1)
    confidence: float  # Confidence in this measurement (0-1)
    uncertainty: float  # Uncertainty estimate (0-1)
    weight: float = 1.0  # Layer weight in fusion


@dataclass
class FusedResult:
    """Fused perception result"""
    risk_score: float  # Overall risk (0-1)
    energy_cost_factor: float  # Energy cost multiplier (0.5-3.0)
    feasible: bool  # Whether location is feasible
    overall_confidence: float  # Confidence in fusion result (0-1)
    uncertainty: Dict[str, float]  # Uncertainty breakdown by source
    contributing_layers: List[str]  # Layers that contributed


class PerceptionFusion:
    """
    Multi-layer perception data fusion.
    
    Combines:
    - Terrain risk (slope, obstacles, roughness)
    - Atmospheric risk (wind, turbulence)
    - Threat risk (radar, patrols, EW)
    
    Into unified risk and energy cost estimates.
    """
    
    def __init__(
        self,
        method: FusionMethod = FusionMethod.WEIGHTED_SUM,
        risk_weights: Optional[Dict[str, float]] = None,
        energy_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize fusion engine.
        
        Args:
            method: Fusion method to use
            risk_weights: Custom weights for risk fusion
            energy_weights: Custom weights for energy fusion
        """
        self.method = method
        
        # Default risk weights
        self.risk_weights = risk_weights or {
            "terrain": 0.30,
            "atmospheric": 0.25,
            "radar": 0.25,
            "patrol": 0.10,
            "ew": 0.10
        }
        
        # Default energy weights
        self.energy_weights = energy_weights or {
            "terrain": 0.20,
            "altitude": 0.30,
            "wind": 0.40,
            "turbulence": 0.10
        }
        
        logger.info(f"Initialized fusion engine with method: {method.value}")
    
    def fuse_risk_data(self, layers: List[LayerData]) -> FusedResult:
        """
        Fuse risk data from multiple layers.
        
        Args:
            layers: List of LayerData from different sources
            
        Returns:
            FusedResult with combined risk assessment
        """
        if not layers:
            return self._default_result()
        
        # Apply fusion method
        if self.method == FusionMethod.WEIGHTED_SUM:
            risk, confidence, uncertainty = self._weighted_sum_fusion(layers)
        elif self.method == FusionMethod.BAYESIAN:
            risk, confidence, uncertainty = self._bayesian_fusion(layers)
        else:
            risk, confidence, uncertainty = self._weighted_sum_fusion(layers)
        
        # Determine feasibility
        feasible = (risk < 0.7) and all(layer.value < 0.9 for layer in layers)
        
        # Compute energy cost factor
        energy_factor = self._compute_energy_factor(layers)
        
        return FusedResult(
            risk_score=risk,
            energy_cost_factor=energy_factor,
            feasible=feasible,
            overall_confidence=confidence,
            uncertainty={layer.layer_name: layer.uncertainty for layer in layers},
            contributing_layers=[layer.layer_name for layer in layers]
        )
    
    def _weighted_sum_fusion(
        self,
        layers: List[LayerData]
    ) -> Tuple[float, float, float]:
        """
        Weighted sum fusion method.
        
        Returns:
            (fused_value, confidence, uncertainty)
        """
        total_weight = 0.0
        weighted_sum = 0.0
        weighted_confidence = 0.0
        weighted_uncertainty = 0.0
        
        for layer in layers:
            # Get weight for this layer
            weight = self.risk_weights.get(layer.layer_name, layer.weight)
            
            # Weight by confidence
            effective_weight = weight * layer.confidence
            
            weighted_sum += layer.value * effective_weight
            weighted_confidence += layer.confidence * weight
            weighted_uncertainty += layer.uncertainty * weight
            total_weight += effective_weight
        
        if total_weight == 0:
            return 0.0, 0.0, 1.0
        
        fused_value = weighted_sum / total_weight
        avg_confidence = weighted_confidence / sum(self.risk_weights.get(l.layer_name, l.weight) for l in layers)
        avg_uncertainty = weighted_uncertainty / sum(self.risk_weights.get(l.layer_name, l.weight) for l in layers)
        
        return (
            float(np.clip(fused_value, 0.0, 1.0)),
            float(np.clip(avg_confidence, 0.0, 1.0)),
            float(np.clip(avg_uncertainty, 0.0, 1.0))
        )
    
    def _bayesian_fusion(
        self,
        layers: List[LayerData]
    ) -> Tuple[float, float, float]:
        """
        Bayesian fusion method.
        
        Uses Bayesian updating to combine evidence from multiple sources.
        
        Returns:
            (fused_value, confidence, uncertainty)
        """
        # Prior (assume moderate risk initially)
        prior_risk = 0.3
        prior_confidence = 0.5
        
        # Likelihood ratio method
        log_odds = np.log(prior_risk / (1 - prior_risk))
        total_confidence = prior_confidence
        total_uncertainty = 0.3
        
        for layer in layers:
            # Convert layer value to likelihood
            # High value = high evidence of risk
            if layer.value > 0.5:
                likelihood_ratio = (layer.value / (1 - layer.value)) * layer.confidence
            else:
                likelihood_ratio = ((1 - layer.value) / layer.value) * layer.confidence
                likelihood_ratio = 1.0 / likelihood_ratio
            
            # Update log odds
            log_odds += np.log(likelihood_ratio)
            
            # Update confidence
            total_confidence = min(1.0, total_confidence + layer.confidence * 0.1)
            
            # Propagate uncertainty
            total_uncertainty = np.sqrt(total_uncertainty**2 + layer.uncertainty**2)
        
        # Convert back to probability
        posterior_risk = 1.0 / (1.0 + np.exp(-log_odds))
        
        return (
            float(np.clip(posterior_risk, 0.0, 1.0)),
            float(np.clip(total_confidence, 0.0, 1.0)),
            float(np.clip(total_uncertainty, 0.0, 1.0))
        )
    
    def _compute_energy_factor(self, layers: List[LayerData]) -> float:
        """
        Compute energy cost factor from layers.
        
        Returns:
            Energy cost multiplier (0.5-3.0)
        """
        base_factor = 1.0
        
        for layer in layers:
            weight = self.energy_weights.get(layer.layer_name, 0.0)
            if weight > 0:
                # Convert layer value to energy impact
                # Higher values = more energy cost
                impact = 1.0 + (layer.value * 2.0 * weight)
                base_factor *= impact
        
        return float(np.clip(base_factor, 0.5, 3.0))
    
    def _default_result(self) -> FusedResult:
        """Return default result when no layers available."""
        return FusedResult(
            risk_score=0.5,
            energy_cost_factor=1.0,
            feasible=True,
            overall_confidence=0.0,
            uncertainty={},
            contributing_layers=[]
        )


class UncertaintyPropagation:
    """
    Uncertainty quantification and propagation.
    
    Tracks and propagates uncertainties from source data through
    processing pipeline to final outputs.
    """
    
    def __init__(self):
        """Initialize uncertainty propagator."""
        self.source_uncertainties: Dict[str, float] = {}
    
    def add_source_uncertainty(self, source: str, uncertainty: float):
        """Add uncertainty from a data source."""
        self.source_uncertainties[source] = uncertainty
    
    def propagate_linear(
        self,
        value: float,
        source_uncertainties: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """
        Propagate uncertainty through linear combination.
        
        For y = Σ(w_i * x_i):
        σ_y² = Σ(w_i² * σ_i²)
        
        Args:
            value: Computed value
            source_uncertainties: Uncertainties of input variables
            weights: Weights used in linear combination
            
        Returns:
            Output uncertainty
        """
        variance = 0.0
        
        for source, uncertainty in source_uncertainties.items():
            weight = weights.get(source, 1.0)
            variance += (weight * uncertainty) ** 2
        
        return float(np.sqrt(variance))
    
    def propagate_monte_carlo(
        self,
        func,
        params: Dict[str, Tuple[float, float]],  # param: (mean, std)
        n_samples: int = 1000
    ) -> Tuple[float, float]:
        """
        Propagate uncertainty using Monte Carlo sampling.
        
        Args:
            func: Function to evaluate
            params: Dictionary of parameter: (mean, std_dev)
            n_samples: Number of Monte Carlo samples
            
        Returns:
            (mean_output, std_output)
        """
        # Generate samples
        samples = {}
        for param, (mean, std) in params.items():
            samples[param] = np.random.normal(mean, std, n_samples)
        
        # Evaluate function for each sample
        outputs = []
        for i in range(n_samples):
            sample_params = {param: samples[param][i] for param in params}
            outputs.append(func(**sample_params))
        
        outputs = np.array(outputs)
        
        return float(np.mean(outputs)), float(np.std(outputs))
    
    def compute_confidence_interval(
        self,
        mean: float,
        std: float,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute confidence interval.
        
        Args:
            mean: Mean value
            std: Standard deviation
            confidence_level: Confidence level (0-1)
            
        Returns:
            (lower_bound, upper_bound)
        """
        from scipy import stats
        
        # Z-score for confidence level
        z = stats.norm.ppf((1 + confidence_level) / 2)
        
        margin = z * std
        
        return (mean - margin, mean + margin)


def compute_calibration_metrics(
    predicted_probs: np.ndarray,
    actual_outcomes: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute calibration metrics for probabilistic predictions.
    
    Args:
        predicted_probs: Predicted probabilities (0-1)
        actual_outcomes: Actual binary outcomes (0 or 1)
        n_bins: Number of bins for calibration curve
        
    Returns:
        Dictionary with calibration metrics (ECE, MCE, Brier score)
    """
    # Expected Calibration Error (ECE)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0  # Maximum Calibration Error
    
    for i in range(n_bins):
        mask = (predicted_probs >= bin_edges[i]) & (predicted_probs < bin_edges[i+1])
        
        if np.sum(mask) > 0:
            bin_confidence = np.mean(predicted_probs[mask])
            bin_accuracy = np.mean(actual_outcomes[mask])
            bin_size = np.sum(mask) / len(predicted_probs)
            
            calibration_error = abs(bin_confidence - bin_accuracy)
            ece += bin_size * calibration_error
            mce = max(mce, calibration_error)
    
    # Brier score
    brier = np.mean((predicted_probs - actual_outcomes) ** 2)
    
    return {
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(mce),
        "brier_score": float(brier)
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create fusion engine
    fusion = PerceptionFusion(method=FusionMethod.WEIGHTED_SUM)
    
    # Create sample layer data
    layers = [
        LayerData(layer_name="terrain", value=0.3, confidence=0.9, uncertainty=0.05),
        LayerData(layer_name="atmospheric", value=0.4, confidence=0.7, uncertainty=0.10),
        LayerData(layer_name="radar", value=0.6, confidence=0.8, uncertainty=0.08),
        LayerData(layer_name="patrol", value=0.2, confidence=0.6, uncertainty=0.15),
    ]
    
    # Fuse data
    result = fusion.fuse_risk_data(layers)
    
    print("\nFusion Result:")
    print(f"  Risk Score: {result.risk_score:.3f}")
    print(f"  Energy Factor: {result.energy_cost_factor:.2f}")
    print(f"  Feasible: {result.feasible}")
    print(f"  Confidence: {result.overall_confidence:.3f}")
    print(f"  Contributing Layers: {', '.join(result.contributing_layers)}")
    print(f"  Uncertainties: {result.uncertainty}")
    
    # Test Bayesian fusion
    fusion_bayesian = PerceptionFusion(method=FusionMethod.BAYESIAN)
    result_bayes = fusion_bayesian.fuse_risk_data(layers)
    
    print(f"\nBayesian Fusion Risk: {result_bayes.risk_score:.3f}")
    
    # Test uncertainty propagation
    propagator = UncertaintyPropagation()
    
    uncertainty = propagator.propagate_linear(
        value=0.5,
        source_uncertainties={"terrain": 0.05, "wind": 0.10},
        weights={"terrain": 0.3, "wind": 0.7}
    )
    
    print(f"\nPropagated Uncertainty: {uncertainty:.4f}")

