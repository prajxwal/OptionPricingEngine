"""
Abstract base class for pricing models.

Defines the standard interface all pricing models must implement,
along with the structured PricingResult output contract.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time

from core.inputs import PricingInputs
from core.option import Option
from core.market import MarketEnvironment


@dataclass
class Greeks:
    """
    Container for option Greeks (sensitivities).
    
    All values are per-unit of the option.
    """
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    rho: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Optional[float]]:
        """Convert to dictionary representation."""
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
        }


@dataclass
class Diagnostics:
    """
    Model-specific diagnostic information.
    
    Used for convergence analysis and debugging.
    """
    convergence_metric: Optional[float] = None
    standard_error: Optional[float] = None
    confidence_interval: Optional[tuple] = None
    iterations: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "convergence_metric": self.convergence_metric,
            "standard_error": self.standard_error,
            "confidence_interval": self.confidence_interval,
            "iterations": self.iterations,
        }
        result.update(self.extra)
        return result


@dataclass
class PricingResult:
    """
    Standardized output contract for all pricing models.
    
    Every model returns this structure to ensure consistent
    comparison and reporting.
    """
    price: float
    greeks: Greeks
    diagnostics: Diagnostics
    runtime_ms: float
    warnings: List[str] = field(default_factory=list)
    model_name: str = ""
    input_signature: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "price": self.price,
            "greeks": self.greeks.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
            "runtime_ms": self.runtime_ms,
            "warnings": self.warnings,
            "model_name": self.model_name,
            "input_signature": self.input_signature,
        }


class PricingModel(ABC):
    """
    Abstract base class for option pricing models.
    
    All pricing models must inherit from this class and implement
    the required methods to ensure consistent interface.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the model name."""
        pass
    
    @abstractmethod
    def price(self, option: Option, market: MarketEnvironment, 
              inputs: Optional[PricingInputs] = None) -> float:
        """
        Calculate the option price.
        
        Args:
            option: The option contract to price
            market: Market environment parameters
            inputs: Full pricing inputs (optional, for numerical params)
            
        Returns:
            The calculated option price
        """
        pass
    
    @abstractmethod
    def greeks(self, option: Option, market: MarketEnvironment,
               inputs: Optional[PricingInputs] = None) -> Greeks:
        """
        Calculate option Greeks (sensitivities).
        
        Args:
            option: The option contract
            market: Market environment parameters
            inputs: Full pricing inputs (optional)
            
        Returns:
            Greeks object with calculated sensitivities
        """
        pass
    
    @abstractmethod
    def assumptions(self) -> List[str]:
        """
        Return list of model assumptions.
        
        This is critical for transparency - every model must
        explicitly state its assumptions.
        """
        pass
    
    def diagnostics(self, option: Option, market: MarketEnvironment,
                    inputs: Optional[PricingInputs] = None) -> Diagnostics:
        """
        Return model-specific diagnostic information.
        
        Override in subclasses for model-specific diagnostics.
        """
        return Diagnostics()
    
    def compute(self, option: Option, market: MarketEnvironment,
                inputs: Optional[PricingInputs] = None) -> PricingResult:
        """
        Full computation returning standardized PricingResult.
        
        This is the primary method for model comparison.
        """
        warnings: List[str] = []
        
        # Check for extreme conditions
        if option.maturity < 0.01:
            warnings.append("Near-zero maturity may cause numerical instability")
        
        moneyness = market.spot / option.strike
        if moneyness < 0.5 or moneyness > 2.0:
            warnings.append(f"Extreme moneyness ({moneyness:.2f}) may affect accuracy")
        
        # Time the computation
        start = time.perf_counter()
        price = self.price(option, market, inputs)
        greeks = self.greeks(option, market, inputs)
        diagnostics = self.diagnostics(option, market, inputs)
        runtime_ms = (time.perf_counter() - start) * 1000
        
        return PricingResult(
            price=price,
            greeks=greeks,
            diagnostics=diagnostics,
            runtime_ms=runtime_ms,
            warnings=warnings,
            model_name=self.name,
            input_signature=inputs.input_signature() if inputs else "",
        )
