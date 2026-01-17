"""
Experiment runner for parameter sweeps.

Enables batch experiments over parameter grids for research.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.option import Option
from core.market import MarketEnvironment
from core.inputs import PricingInputs, NumericalParams, OptionParams, MarketParams
from core.enums import OptionType
from models.base import PricingModel, PricingResult
from models.black_scholes import BlackScholesModel


@dataclass
class ExperimentPoint:
    """Single point in a parameter sweep."""
    parameter_name: str
    parameter_value: float
    prices: Dict[str, float]
    errors: Dict[str, float]  # Error vs Black-Scholes
    greeks: Dict[str, Dict[str, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter_name": self.parameter_name,
            "parameter_value": self.parameter_value,
            "prices": self.prices,
            "errors": self.errors,
            "greeks": self.greeks,
        }


@dataclass
class ExperimentResult:
    """
    Results of a parameter sweep experiment.
    """
    experiment_name: str
    parameter_name: str
    parameter_values: List[float]
    points: List[ExperimentPoint]
    base_option: Dict[str, Any]
    base_market: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "parameter_name": self.parameter_name,
            "parameter_values": self.parameter_values,
            "points": [p.to_dict() for p in self.points],
            "base_option": self.base_option,
            "base_market": self.base_market,
        }
    
    def get_prices_array(self, model_name: str) -> List[float]:
        """Get array of prices for a specific model."""
        return [p.prices.get(model_name, 0) for p in self.points]
    
    def get_errors_array(self, model_name: str) -> List[float]:
        """Get array of errors for a specific model."""
        return [p.errors.get(model_name, 0) for p in self.points]


class ExperimentRunner:
    """
    Run parameter sweep experiments across multiple models.
    """
    
    def __init__(self, models: Optional[List[PricingModel]] = None):
        """
        Initialize experiment runner.
        
        Args:
            models: List of models to include in experiments
        """
        if models is None:
            from models.binomial import BinomialModel
            from models.monte_carlo import MonteCarloModel
            models = [
                BlackScholesModel(),
                BinomialModel(),
                MonteCarloModel(),
            ]
        
        self.models = models
        self.baseline = BlackScholesModel()
    
    def volatility_sweep(
        self,
        base_option: Option,
        base_market: MarketEnvironment,
        vol_range: Optional[List[float]] = None,
        inputs: Optional[PricingInputs] = None,
    ) -> ExperimentResult:
        """
        Sweep volatility and observe price changes.
        
        Args:
            base_option: Base option contract
            base_market: Base market (volatility will be varied)
            vol_range: List of volatilities to test
            inputs: Numerical parameters
            
        Returns:
            ExperimentResult with prices at each volatility
        """
        if vol_range is None:
            vol_range = np.linspace(0.05, 0.80, 16).tolist()
        
        points = []
        
        for vol in vol_range:
            market = MarketEnvironment(base_market.spot, vol, base_market.risk_free_rate)
            
            prices = {}
            errors = {}
            greeks = {}
            
            baseline_price = self.baseline.price(base_option, market, inputs)
            
            for model in self.models:
                price = model.price(base_option, market, inputs)
                greek_values = model.greeks(base_option, market, inputs)
                
                prices[model.name] = price
                errors[model.name] = abs(price - baseline_price)
                greeks[model.name] = greek_values.to_dict()
            
            points.append(ExperimentPoint(
                parameter_name="volatility",
                parameter_value=vol,
                prices=prices,
                errors=errors,
                greeks=greeks,
            ))
        
        return ExperimentResult(
            experiment_name="Volatility Sweep",
            parameter_name="volatility",
            parameter_values=vol_range,
            points=points,
            base_option={"strike": base_option.strike, "maturity": base_option.maturity, 
                         "type": base_option.option_type.name},
            base_market={"spot": base_market.spot, "rate": base_market.risk_free_rate},
        )
    
    def maturity_sweep(
        self,
        base_option: Option,
        base_market: MarketEnvironment,
        maturity_range: Optional[List[float]] = None,
        inputs: Optional[PricingInputs] = None,
    ) -> ExperimentResult:
        """
        Sweep maturity and observe price changes.
        """
        if maturity_range is None:
            maturity_range = np.linspace(0.1, 2.0, 20).tolist()
        
        points = []
        
        for T in maturity_range:
            option = Option(base_option.strike, T, base_option.option_type)
            
            prices = {}
            errors = {}
            greeks = {}
            
            baseline_price = self.baseline.price(option, base_market, inputs)
            
            for model in self.models:
                price = model.price(option, base_market, inputs)
                greek_values = model.greeks(option, base_market, inputs)
                
                prices[model.name] = price
                errors[model.name] = abs(price - baseline_price)
                greeks[model.name] = greek_values.to_dict()
            
            points.append(ExperimentPoint(
                parameter_name="maturity",
                parameter_value=T,
                prices=prices,
                errors=errors,
                greeks=greeks,
            ))
        
        return ExperimentResult(
            experiment_name="Maturity Sweep",
            parameter_name="maturity",
            parameter_values=maturity_range,
            points=points,
            base_option={"strike": base_option.strike, "type": base_option.option_type.name},
            base_market={"spot": base_market.spot, "volatility": base_market.volatility,
                         "rate": base_market.risk_free_rate},
        )
    
    def moneyness_sweep(
        self,
        base_option: Option,
        base_market: MarketEnvironment,
        moneyness_range: Optional[List[float]] = None,
        inputs: Optional[PricingInputs] = None,
    ) -> ExperimentResult:
        """
        Sweep moneyness (S/K ratio) and observe price changes.
        
        Varies the spot price while keeping strike constant.
        """
        if moneyness_range is None:
            moneyness_range = np.linspace(0.7, 1.3, 13).tolist()
        
        points = []
        K = base_option.strike
        
        for m in moneyness_range:
            spot = K * m  # S/K = m, so S = K*m
            market = MarketEnvironment(spot, base_market.volatility, base_market.risk_free_rate)
            
            prices = {}
            errors = {}
            greeks = {}
            
            baseline_price = self.baseline.price(base_option, market, inputs)
            
            for model in self.models:
                price = model.price(base_option, market, inputs)
                greek_values = model.greeks(base_option, market, inputs)
                
                prices[model.name] = price
                errors[model.name] = abs(price - baseline_price)
                greeks[model.name] = greek_values.to_dict()
            
            points.append(ExperimentPoint(
                parameter_name="moneyness",
                parameter_value=m,
                prices=prices,
                errors=errors,
                greeks=greeks,
            ))
        
        return ExperimentResult(
            experiment_name="Moneyness Sweep",
            parameter_name="moneyness",
            parameter_values=moneyness_range,
            points=points,
            base_option={"strike": K, "maturity": base_option.maturity,
                         "type": base_option.option_type.name},
            base_market={"volatility": base_market.volatility, 
                         "rate": base_market.risk_free_rate},
        )
    
    def run_all_sweeps(
        self,
        option: Option,
        market: MarketEnvironment,
        inputs: Optional[PricingInputs] = None,
    ) -> Dict[str, ExperimentResult]:
        """
        Run all standard parameter sweeps.
        """
        return {
            "volatility": self.volatility_sweep(option, market, inputs=inputs),
            "maturity": self.maturity_sweep(option, market, inputs=inputs),
            "moneyness": self.moneyness_sweep(option, market, inputs=inputs),
        }
