"""
Immutable input objects for the Option Pricing Engine.

All inputs are frozen dataclasses that are hashable for reproducibility.
Each provides an input_signature() method for deterministic hash generation.
"""

from dataclasses import dataclass
from typing import Optional
import hashlib
import json

from .enums import OptionType


@dataclass(frozen=True)
class OptionParams:
    """
    Immutable option contract parameters.
    
    Attributes:
        strike: Strike price (must be positive)
        maturity: Time to maturity in years (must be positive)
        option_type: CALL or PUT
    """
    strike: float
    maturity: float
    option_type: OptionType
    
    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError(f"Strike must be positive, got {self.strike}")
        if self.maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {self.maturity}")
        if not isinstance(self.option_type, OptionType):
            raise TypeError(f"option_type must be OptionType, got {type(self.option_type)}")
    
    def input_signature(self) -> str:
        """Generate deterministic hash for reproducibility metadata."""
        data = {
            "strike": self.strike,
            "maturity": self.maturity,
            "option_type": self.option_type.name
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class MarketParams:
    """
    Immutable market environment parameters.
    
    Attributes:
        spot: Current underlying price (must be positive)
        volatility: Annualized volatility as decimal (must be positive)
        risk_free_rate: Annualized risk-free rate as decimal
    """
    spot: float
    volatility: float
    risk_free_rate: float
    
    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError(f"Spot price must be positive, got {self.spot}")
        if self.volatility <= 0:
            raise ValueError(f"Volatility must be positive, got {self.volatility}")
        # Risk-free rate can be negative (e.g., some European bonds)
    
    def input_signature(self) -> str:
        """Generate deterministic hash for reproducibility metadata."""
        data = {
            "spot": self.spot,
            "volatility": self.volatility,
            "risk_free_rate": self.risk_free_rate
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class NumericalParams:
    """
    Immutable numerical control parameters.
    
    Attributes:
        num_steps: Number of steps for binomial tree (default: 100)
        num_paths: Number of paths for Monte Carlo (default: 100000)
        seed: Random seed for reproducibility (default: 42)
    """
    num_steps: int = 100
    num_paths: int = 100000
    seed: int = 42
    
    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")
        if self.num_paths <= 0:
            raise ValueError(f"num_paths must be positive, got {self.num_paths}")
    
    def input_signature(self) -> str:
        """Generate deterministic hash for reproducibility metadata."""
        data = {
            "num_steps": self.num_steps,
            "num_paths": self.num_paths,
            "seed": self.seed
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class PricingInputs:
    """
    Complete pricing inputs combining all parameters.
    
    This is the primary input object passed to pricing models.
    """
    option: OptionParams
    market: MarketParams
    numerical: NumericalParams = NumericalParams()
    
    def input_signature(self) -> str:
        """Generate deterministic input hash for run identification and reproducibility checks."""
        combined = f"{self.option.input_signature()}-{self.market.input_signature()}-{self.numerical.input_signature()}"
        return hashlib.sha256(combined.encode()).hexdigest()[:24]
