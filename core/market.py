"""
Market environment for the Option Pricing Engine.

Encapsulates market assumptions separately from the option contract.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketEnvironment:
    """
    Immutable market environment.
    
    Attributes:
        spot: Current underlying price
        volatility: Annualized volatility (as decimal, e.g., 0.20 for 20%)
        risk_free_rate: Annualized risk-free interest rate (as decimal)
    
    Notes:
        - Designed for future extension to stochastic volatility models
        - Currently assumes constant parameters
    """
    spot: float
    volatility: float
    risk_free_rate: float
    
    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError(f"Spot price must be positive, got {self.spot}")
        if self.volatility <= 0:
            raise ValueError(f"Volatility must be positive, got {self.volatility}")
        # Note: risk_free_rate can be negative
    
    @property
    def sigma(self) -> float:
        """Alias for volatility."""
        return self.volatility
    
    @property
    def r(self) -> float:
        """Alias for risk_free_rate."""
        return self.risk_free_rate
    
    @property
    def S(self) -> float:
        """Alias for spot price."""
        return self.spot
    
    def __str__(self) -> str:
        return f"Market | S={self.spot:.2f} | σ={self.volatility:.2%} | r={self.risk_free_rate:.2%}"
