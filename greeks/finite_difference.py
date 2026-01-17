"""
Finite difference Greeks calculations.

Provides numerical Greeks for any pricing model using bump-and-reprice.
Uses relative bump strategy for numerical stability.
"""

from dataclasses import dataclass
from typing import Dict, Callable, Optional

from core.option import Option
from core.market import MarketEnvironment
from core.inputs import PricingInputs
from models.base import PricingModel


@dataclass
class FiniteDifferenceConfig:
    """
    Configuration for finite difference calculations.
    
    Uses relative bump strategy: epsilon = max(relative_bump * param, min_bump)
    This prevents numerical instability for small parameter values.
    """
    relative_bump_spot: float = 0.01      # 1% of spot
    relative_bump_vol: float = 0.01       # 1% of vol
    relative_bump_rate: float = 0.0001    # 1 bp
    bump_time: float = 1/365              # 1 day
    
    min_bump_spot: float = 0.01
    min_bump_vol: float = 0.001
    min_bump_rate: float = 0.0001
    
    def get_spot_bump(self, spot: float) -> float:
        """Calculate spot bump size."""
        return max(self.relative_bump_spot * spot, self.min_bump_spot)
    
    def get_vol_bump(self, vol: float) -> float:
        """Calculate volatility bump size."""
        return max(self.relative_bump_vol * vol, self.min_bump_vol)
    
    def get_rate_bump(self, rate: float) -> float:
        """Calculate rate bump size (use absolute for rates near zero)."""
        return max(self.relative_bump_rate * abs(rate), self.min_bump_rate)


@dataclass
class FiniteDifferenceGreeks:
    """
    Greeks calculated via finite differences.
    
    Includes estimation quality indicators.
    """
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    
    # Bump sizes used (for diagnostics)
    spot_bump: float = 0.0
    vol_bump: float = 0.0
    rate_bump: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
            "spot_bump": self.spot_bump,
            "vol_bump": self.vol_bump,
            "rate_bump": self.rate_bump,
        }


class FiniteDifferenceCalculator:
    """
    Calculate Greeks using finite difference methods.
    
    Works with any pricing model that implements the PricingModel interface.
    Uses central differences for better accuracy.
    """
    
    def __init__(self, model: PricingModel, config: Optional[FiniteDifferenceConfig] = None):
        """
        Initialize calculator.
        
        Args:
            model: Any pricing model implementing PricingModel interface
            config: Finite difference configuration (uses defaults if None)
        """
        self.model = model
        self.config = config or FiniteDifferenceConfig()
    
    def _price(self, option: Option, market: MarketEnvironment,
               inputs: Optional[PricingInputs] = None) -> float:
        """Helper to get price from model."""
        return self.model.price(option, market, inputs)
    
    def calculate_delta(self, option: Option, market: MarketEnvironment,
                        inputs: Optional[PricingInputs] = None) -> float:
        """
        Calculate Delta using central difference.
        
        Delta ≈ (V(S+h) - V(S-h)) / (2h)
        """
        h = self.config.get_spot_bump(market.spot)
        
        market_up = MarketEnvironment(market.spot + h, market.volatility, market.risk_free_rate)
        market_down = MarketEnvironment(market.spot - h, market.volatility, market.risk_free_rate)
        
        price_up = self._price(option, market_up, inputs)
        price_down = self._price(option, market_down, inputs)
        
        return (price_up - price_down) / (2 * h)
    
    def calculate_gamma(self, option: Option, market: MarketEnvironment,
                        inputs: Optional[PricingInputs] = None) -> float:
        """
        Calculate Gamma using central second difference.
        
        Gamma ≈ (V(S+h) - 2V(S) + V(S-h)) / h²
        """
        h = self.config.get_spot_bump(market.spot)
        
        market_up = MarketEnvironment(market.spot + h, market.volatility, market.risk_free_rate)
        market_down = MarketEnvironment(market.spot - h, market.volatility, market.risk_free_rate)
        
        price_up = self._price(option, market_up, inputs)
        price_mid = self._price(option, market, inputs)
        price_down = self._price(option, market_down, inputs)
        
        return (price_up - 2 * price_mid + price_down) / (h ** 2)
    
    def calculate_vega(self, option: Option, market: MarketEnvironment,
                       inputs: Optional[PricingInputs] = None) -> float:
        """
        Calculate Vega using central difference.
        
        Returns per 1% volatility change.
        """
        h = self.config.get_vol_bump(market.volatility)
        
        market_up = MarketEnvironment(market.spot, market.volatility + h, market.risk_free_rate)
        market_down = MarketEnvironment(market.spot, market.volatility - h, market.risk_free_rate)
        
        price_up = self._price(option, market_up, inputs)
        price_down = self._price(option, market_down, inputs)
        
        # Per 1% vol change
        return (price_up - price_down) / (2 * h) / 100
    
    def calculate_theta(self, option: Option, market: MarketEnvironment,
                        inputs: Optional[PricingInputs] = None) -> float:
        """
        Calculate Theta using forward difference on time.
        
        Returns per day.
        """
        h = self.config.bump_time
        
        if option.maturity <= h:
            # Option too close to expiry
            return 0.0
        
        # Create shorter-dated option
        option_shorter = Option(option.strike, option.maturity - h, option.option_type)
        
        price_current = self._price(option, market, inputs)
        price_shorter = self._price(option_shorter, market, inputs)
        
        # Theta = (V(T-h) - V(T)) / h, per day
        return (price_shorter - price_current) / h / 365
    
    def calculate_rho(self, option: Option, market: MarketEnvironment,
                      inputs: Optional[PricingInputs] = None) -> float:
        """
        Calculate Rho using central difference.
        
        Returns per 1% rate change.
        """
        h = self.config.get_rate_bump(market.risk_free_rate)
        
        market_up = MarketEnvironment(market.spot, market.volatility, market.risk_free_rate + h)
        market_down = MarketEnvironment(market.spot, market.volatility, market.risk_free_rate - h)
        
        price_up = self._price(option, market_up, inputs)
        price_down = self._price(option, market_down, inputs)
        
        # Per 1% rate change
        return (price_up - price_down) / (2 * h) / 100
    
    def calculate_all(self, option: Option, market: MarketEnvironment,
                      inputs: Optional[PricingInputs] = None) -> FiniteDifferenceGreeks:
        """
        Calculate all Greeks using finite differences.
        
        More efficient than calling individual methods as it
        reuses some price calculations.
        """
        S = market.spot
        sigma = market.volatility
        r = market.risk_free_rate
        
        h_s = self.config.get_spot_bump(S)
        h_v = self.config.get_vol_bump(sigma)
        h_r = self.config.get_rate_bump(r)
        h_t = self.config.bump_time
        
        # Spot bumps for delta and gamma
        market_s_up = MarketEnvironment(S + h_s, sigma, r)
        market_s_down = MarketEnvironment(S - h_s, sigma, r)
        
        price_mid = self._price(option, market, inputs)
        price_s_up = self._price(option, market_s_up, inputs)
        price_s_down = self._price(option, market_s_down, inputs)
        
        delta = (price_s_up - price_s_down) / (2 * h_s)
        gamma = (price_s_up - 2 * price_mid + price_s_down) / (h_s ** 2)
        
        # Vol bumps for vega
        market_v_up = MarketEnvironment(S, sigma + h_v, r)
        market_v_down = MarketEnvironment(S, sigma - h_v, r)
        
        price_v_up = self._price(option, market_v_up, inputs)
        price_v_down = self._price(option, market_v_down, inputs)
        
        vega = (price_v_up - price_v_down) / (2 * h_v) / 100
        
        # Rate bumps for rho
        market_r_up = MarketEnvironment(S, sigma, r + h_r)
        market_r_down = MarketEnvironment(S, sigma, r - h_r)
        
        price_r_up = self._price(option, market_r_up, inputs)
        price_r_down = self._price(option, market_r_down, inputs)
        
        rho = (price_r_up - price_r_down) / (2 * h_r) / 100
        
        # Time bump for theta
        if option.maturity > h_t:
            option_shorter = Option(option.strike, option.maturity - h_t, option.option_type)
            price_shorter = self._price(option_shorter, market, inputs)
            theta = (price_shorter - price_mid) / h_t / 365
        else:
            theta = 0.0
        
        return FiniteDifferenceGreeks(
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho,
            spot_bump=h_s,
            vol_bump=h_v,
            rate_bump=h_r,
        )


def validate_against_analytical(fd_greeks: FiniteDifferenceGreeks,
                                 analytical_greeks: Dict[str, float],
                                 tolerance: float = 0.01) -> Dict[str, bool]:
    """
    Validate finite difference Greeks against analytical values.
    
    Args:
        fd_greeks: Finite difference calculated Greeks
        analytical_greeks: Dictionary of analytical Greek values
        tolerance: Relative tolerance for comparison
        
    Returns:
        Dictionary indicating which Greeks pass validation
    """
    results = {}
    
    for greek_name in ['delta', 'gamma', 'vega', 'theta', 'rho']:
        fd_value = getattr(fd_greeks, greek_name)
        analytical_value = analytical_greeks.get(greek_name, 0.0)
        
        if analytical_value == 0:
            # Use absolute comparison for zero values
            results[greek_name] = abs(fd_value) < tolerance
        else:
            relative_error = abs(fd_value - analytical_value) / abs(analytical_value)
            results[greek_name] = relative_error < tolerance
    
    return results
