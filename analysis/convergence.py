"""
Convergence analysis tools.

Analyzes how numerical models converge to analytical solutions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from core.option import Option
from core.market import MarketEnvironment
from core.inputs import PricingInputs, NumericalParams, OptionParams, MarketParams
from core.enums import OptionType
from models.black_scholes import BlackScholesModel
from models.binomial import BinomialModel
from models.monte_carlo import MonteCarloModel


@dataclass
class ConvergencePoint:
    """Single point in a convergence study."""
    parameter_value: int  # e.g., number of steps or paths
    price: float
    error: float
    relative_error: float
    runtime_ms: float


@dataclass
class ConvergenceResult:
    """
    Results of a convergence analysis.
    """
    model_name: str
    parameter_name: str
    baseline_price: float
    points: List[ConvergencePoint]
    convergence_rate: Optional[float] = None
    final_error: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "parameter_name": self.parameter_name,
            "baseline_price": self.baseline_price,
            "points": [
                {
                    "parameter_value": p.parameter_value,
                    "price": p.price,
                    "error": p.error,
                    "relative_error": p.relative_error,
                    "runtime_ms": p.runtime_ms,
                }
                for p in self.points
            ],
            "convergence_rate": self.convergence_rate,
            "final_error": self.final_error,
        }


class ConvergenceAnalyzer:
    """
    Analyze convergence behavior of numerical models.
    """
    
    def __init__(self):
        self.baseline = BlackScholesModel()
    
    def analyze_binomial_convergence(
        self,
        option: Option,
        market: MarketEnvironment,
        step_values: Optional[List[int]] = None,
        seed: int = 42,
    ) -> ConvergenceResult:
        """
        Analyze binomial model convergence as steps increase.
        
        Args:
            option: Option to price
            market: Market environment
            step_values: List of step counts to test (default: 10 to 500)
            seed: Random seed (not used for binomial, for consistency)
            
        Returns:
            ConvergenceResult with prices at each step count
        """
        if step_values is None:
            step_values = [10, 25, 50, 75, 100, 150, 200, 300, 400, 500]
        
        # Get baseline price
        baseline_price = self.baseline.price(option, market)
        
        points = []
        model = BinomialModel()
        
        for steps in step_values:
            option_params = OptionParams(option.strike, option.maturity, option.option_type)
            market_params = MarketParams(market.spot, market.volatility, market.risk_free_rate)
            numerical = NumericalParams(num_steps=steps, num_paths=1, seed=seed)
            inputs = PricingInputs(option_params, market_params, numerical)
            
            import time
            start = time.perf_counter()
            price = model.price(option, market, inputs)
            runtime = (time.perf_counter() - start) * 1000
            
            error = abs(price - baseline_price)
            rel_error = error / baseline_price if baseline_price != 0 else 0
            
            points.append(ConvergencePoint(
                parameter_value=steps,
                price=price,
                error=error,
                relative_error=rel_error,
                runtime_ms=runtime,
            ))
        
        # Estimate convergence rate (error should decrease as O(1/N))
        convergence_rate = self._estimate_convergence_rate(
            [p.parameter_value for p in points],
            [p.error for p in points]
        )
        
        return ConvergenceResult(
            model_name="Binomial (CRR)",
            parameter_name="steps",
            baseline_price=baseline_price,
            points=points,
            convergence_rate=convergence_rate,
            final_error=points[-1].error if points else None,
        )
    
    def analyze_monte_carlo_convergence(
        self,
        option: Option,
        market: MarketEnvironment,
        path_values: Optional[List[int]] = None,
        seed: int = 42,
    ) -> ConvergenceResult:
        """
        Analyze Monte Carlo convergence as paths increase.
        
        Args:
            option: Option to price
            market: Market environment
            path_values: List of path counts to test
            seed: Random seed for reproducibility
            
        Returns:
            ConvergenceResult with prices at each path count
        """
        if path_values is None:
            path_values = [100, 500, 1000, 5000, 10000, 25000, 50000, 100000]
        
        # Get baseline price
        baseline_price = self.baseline.price(option, market)
        
        points = []
        model = MonteCarloModel()
        
        for paths in path_values:
            option_params = OptionParams(option.strike, option.maturity, option.option_type)
            market_params = MarketParams(market.spot, market.volatility, market.risk_free_rate)
            numerical = NumericalParams(num_steps=1, num_paths=paths, seed=seed)
            inputs = PricingInputs(option_params, market_params, numerical)
            
            import time
            start = time.perf_counter()
            price = model.price(option, market, inputs)
            runtime = (time.perf_counter() - start) * 1000
            
            error = abs(price - baseline_price)
            rel_error = error / baseline_price if baseline_price != 0 else 0
            
            points.append(ConvergencePoint(
                parameter_value=paths,
                price=price,
                error=error,
                relative_error=rel_error,
                runtime_ms=runtime,
            ))
        
        # Estimate convergence rate (error should decrease as O(1/√N))
        convergence_rate = self._estimate_convergence_rate(
            [p.parameter_value for p in points],
            [p.error for p in points],
            expected_rate=-0.5  # O(1/√N)
        )
        
        return ConvergenceResult(
            model_name="Monte Carlo",
            parameter_name="paths",
            baseline_price=baseline_price,
            points=points,
            convergence_rate=convergence_rate,
            final_error=points[-1].error if points else None,
        )
    
    def _estimate_convergence_rate(
        self,
        x_values: List[int],
        errors: List[float],
        expected_rate: float = -1.0,
    ) -> Optional[float]:
        """
        Estimate convergence rate from error data.
        
        Fits error ~ x^rate and returns the rate.
        """
        if len(x_values) < 3 or any(e <= 0 for e in errors):
            return None
        
        try:
            # Log-log regression: log(error) = rate * log(x) + const
            log_x = np.log(x_values)
            log_err = np.log(errors)
            
            # Simple linear regression
            n = len(log_x)
            sum_x = np.sum(log_x)
            sum_y = np.sum(log_err)
            sum_xy = np.sum(log_x * log_err)
            sum_xx = np.sum(log_x * log_x)
            
            rate = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
            
            return float(rate)
        except Exception:
            return None


def analyze_all_convergence(
    option: Option,
    market: MarketEnvironment,
) -> Dict[str, ConvergenceResult]:
    """
    Convenience function to analyze convergence for all numerical models.
    """
    analyzer = ConvergenceAnalyzer()
    
    return {
        "binomial": analyzer.analyze_binomial_convergence(option, market),
        "monte_carlo": analyzer.analyze_monte_carlo_convergence(option, market),
    }
