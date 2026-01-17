"""
Model comparison analysis.

Compares outputs of multiple pricing models under identical inputs.
Uses Black-Scholes as the baseline reference model.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time

from core.option import Option
from core.market import MarketEnvironment
from core.inputs import PricingInputs
from models.base import PricingModel, PricingResult
from models.black_scholes import BlackScholesModel


@dataclass
class ComparisonMetrics:
    """
    Metrics for comparing a model against the baseline.
    """
    model_name: str
    price: float
    absolute_error: float
    relative_error: float
    runtime_ms: float
    baseline_price: float
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "price": self.price,
            "absolute_error": self.absolute_error,
            "relative_error": self.relative_error,
            "runtime_ms": self.runtime_ms,
            "baseline_price": self.baseline_price,
            "warnings": self.warnings,
        }


@dataclass 
class ComparisonResult:
    """
    Complete comparison result across all models.
    """
    baseline_model: str
    baseline_price: float
    model_metrics: List[ComparisonMetrics]
    input_signature: str
    summary: str = ""
    disagreement_explanation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_model": self.baseline_model,
            "baseline_price": self.baseline_price,
            "model_metrics": [m.to_dict() for m in self.model_metrics],
            "input_signature": self.input_signature,
            "summary": self.summary,
            "disagreement_explanation": self.disagreement_explanation,
        }


class ModelComparison:
    """
    Compare multiple pricing models against a baseline.
    
    Black-Scholes is always used as the reference baseline,
    even when not included in the selected models.
    """
    
    def __init__(self, baseline_model: Optional[PricingModel] = None):
        """
        Initialize comparison engine.
        
        Args:
            baseline_model: Reference model (defaults to Black-Scholes)
        """
        self.baseline = baseline_model or BlackScholesModel()
    
    def compare(self, models: List[PricingModel], 
                option: Option, 
                market: MarketEnvironment,
                inputs: Optional[PricingInputs] = None) -> ComparisonResult:
        """
        Compare all models against the baseline.
        
        Args:
            models: List of models to compare
            option: Option contract to price
            market: Market environment
            inputs: Numerical parameters (optional)
            
        Returns:
            ComparisonResult with metrics for each model
        """
        # Get baseline price
        start = time.perf_counter()
        baseline_price = self.baseline.price(option, market, inputs)
        baseline_runtime = (time.perf_counter() - start) * 1000
        
        metrics = []
        
        # Add baseline to metrics
        metrics.append(ComparisonMetrics(
            model_name=self.baseline.name,
            price=baseline_price,
            absolute_error=0.0,
            relative_error=0.0,
            runtime_ms=baseline_runtime,
            baseline_price=baseline_price,
            warnings=[],
        ))
        
        # Compare each model
        for model in models:
            if model.name == self.baseline.name:
                continue  # Already added
                
            start = time.perf_counter()
            price = model.price(option, market, inputs)
            runtime = (time.perf_counter() - start) * 1000
            
            abs_error = abs(price - baseline_price)
            rel_error = abs_error / baseline_price if baseline_price != 0 else 0
            
            # Get diagnostics for warnings
            result = model.compute(option, market, inputs)
            
            metrics.append(ComparisonMetrics(
                model_name=model.name,
                price=price,
                absolute_error=abs_error,
                relative_error=rel_error,
                runtime_ms=runtime,
                baseline_price=baseline_price,
                warnings=result.warnings,
            ))
        
        # Generate summary and disagreement explanation
        summary = self._generate_summary(metrics)
        disagreement = self._explain_disagreement(metrics, option, market)
        
        return ComparisonResult(
            baseline_model=self.baseline.name,
            baseline_price=baseline_price,
            model_metrics=metrics,
            input_signature=inputs.input_signature() if inputs else "",
            summary=summary,
            disagreement_explanation=disagreement,
        )
    
    def _generate_summary(self, metrics: List[ComparisonMetrics]) -> str:
        """Generate a summary of the comparison."""
        if len(metrics) <= 1:
            return "Single model - no comparison available"
        
        max_error = max(m.relative_error for m in metrics)
        avg_error = sum(m.relative_error for m in metrics[1:]) / (len(metrics) - 1)
        
        if max_error < 0.001:
            return "Excellent agreement: all models within 0.1% of baseline"
        elif max_error < 0.01:
            return f"Good agreement: max relative error {max_error:.2%}"
        elif max_error < 0.05:
            return f"Moderate agreement: max relative error {max_error:.2%}"
        else:
            return f"Significant divergence: max relative error {max_error:.2%}"
    
    def _explain_disagreement(self, metrics: List[ComparisonMetrics],
                               option: Option, market: MarketEnvironment) -> str:
        """Explain why models might disagree."""
        explanations = []
        
        # Check for numerical models
        numerical_models = [m for m in metrics if m.model_name != "Black-Scholes"]
        
        if not numerical_models:
            return ""
        
        for m in numerical_models:
            if m.relative_error > 0.01:  # More than 1% error
                if "Binomial" in m.model_name:
                    explanations.append(
                        f"{m.model_name}: Discrete approximation. "
                        "Error decreases with more steps."
                    )
                elif "Monte Carlo" in m.model_name:
                    explanations.append(
                        f"{m.model_name}: Stochastic simulation. "
                        "Error decreases with more paths (O(1/√N))."
                    )
        
        # Check for extreme moneyness
        moneyness = market.spot / option.strike
        if moneyness < 0.8 or moneyness > 1.2:
            explanations.append(
                f"Deep {'ITM' if moneyness > 1.2 else 'OTM'} options "
                "may show larger numerical differences."
            )
        
        # Check for short maturity
        if option.maturity < 0.1:
            explanations.append(
                "Short maturity reduces time for numerical approximation, "
                "potentially increasing discretization error."
            )
        
        return " ".join(explanations)


def compare_all_models(option: Option, market: MarketEnvironment,
                       inputs: Optional[PricingInputs] = None) -> ComparisonResult:
    """
    Convenience function to compare all available models.
    """
    from models.black_scholes import BlackScholesModel
    from models.binomial import BinomialModel
    from models.monte_carlo import MonteCarloModel
    
    models = [
        BlackScholesModel(),
        BinomialModel(),
        MonteCarloModel(),
    ]
    
    comparison = ModelComparison()
    return comparison.compare(models, option, market, inputs)
