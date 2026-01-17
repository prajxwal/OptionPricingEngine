"""
Black-Scholes analytical pricing model.

Provides closed-form pricing and Greeks for European options.
This is the benchmark model for numerical method validation.
"""

import numpy as np
from scipy.stats import norm
from typing import List, Optional

from core.option import Option
from core.market import MarketEnvironment
from core.enums import OptionType
from core.inputs import PricingInputs
from models.base import PricingModel, Greeks, Diagnostics


class BlackScholesModel(PricingModel):
    """
    Black-Scholes-Merton analytical pricing model.
    
    Assumptions:
        - Constant volatility
        - Continuous trading
        - No dividends
        - Log-normal asset price distribution
        - Risk-free rate is constant
        - No transaction costs
        - European exercise only
    """
    
    @property
    def name(self) -> str:
        return "Black-Scholes"
    
    def _d1(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter."""
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    def _d2(self, d1: float, sigma: float, T: float) -> float:
        """Calculate d2 parameter."""
        return d1 - sigma * np.sqrt(T)
    
    def price(self, option: Option, market: MarketEnvironment,
              inputs: Optional[PricingInputs] = None) -> float:
        """
        Calculate Black-Scholes option price.
        
        Call: S*N(d1) - K*e^(-rT)*N(d2)
        Put:  K*e^(-rT)*N(-d2) - S*N(-d1)
        """
        S = market.spot
        K = option.strike
        T = option.maturity
        r = market.risk_free_rate
        sigma = market.volatility
        
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(d1, sigma, T)
        
        discount = np.exp(-r * T)
        
        if option.is_call:
            price = S * norm.cdf(d1) - K * discount * norm.cdf(d2)
        else:
            price = K * discount * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return float(price)
    
    def greeks(self, option: Option, market: MarketEnvironment,
               inputs: Optional[PricingInputs] = None) -> Greeks:
        """Calculate analytical Greeks."""
        S = market.spot
        K = option.strike
        T = option.maturity
        r = market.risk_free_rate
        sigma = market.volatility
        
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(d1, sigma, T)
        
        sqrt_T = np.sqrt(T)
        discount = np.exp(-r * T)
        pdf_d1 = norm.pdf(d1)
        
        # Gamma is the same for calls and puts
        gamma = pdf_d1 / (S * sigma * sqrt_T)
        
        # Vega is the same for calls and puts (per 1% vol move = /100)
        vega = S * pdf_d1 * sqrt_T / 100
        
        if option.is_call:
            delta = norm.cdf(d1)
            theta = (-(S * pdf_d1 * sigma) / (2 * sqrt_T) 
                     - r * K * discount * norm.cdf(d2)) / 365  # Per day
            rho = K * T * discount * norm.cdf(d2) / 100  # Per 1% rate move
        else:
            delta = norm.cdf(d1) - 1
            theta = (-(S * pdf_d1 * sigma) / (2 * sqrt_T) 
                     + r * K * discount * norm.cdf(-d2)) / 365  # Per day
            rho = -K * T * discount * norm.cdf(-d2) / 100  # Per 1% rate move
        
        return Greeks(
            delta=float(delta),
            gamma=float(gamma),
            vega=float(vega),
            theta=float(theta),
            rho=float(rho),
        )
    
    def assumptions(self) -> List[str]:
        """Return Black-Scholes model assumptions."""
        return [
            "Constant volatility throughout option life",
            "Continuous trading with no gaps",
            "No dividends paid during option life",
            "Log-normal distribution of asset prices",
            "Constant risk-free interest rate",
            "No transaction costs or taxes",
            "European exercise only (no early exercise)",
            "Markets are frictionless and liquid",
        ]
    
    def diagnostics(self, option: Option, market: MarketEnvironment,
                    inputs: Optional[PricingInputs] = None) -> Diagnostics:
        """Return diagnostic information."""
        # Put-call parity check
        S = market.spot
        K = option.strike
        T = option.maturity
        r = market.risk_free_rate
        
        call_option = Option(K, T, OptionType.CALL)
        put_option = Option(K, T, OptionType.PUT)
        
        call_price = self.price(call_option, market)
        put_price = self.price(put_option, market)
        
        # C - P = S - K*e^(-rT)
        parity_lhs = call_price - put_price
        parity_rhs = S - K * np.exp(-r * T)
        parity_error = abs(parity_lhs - parity_rhs)
        
        return Diagnostics(
            extra={
                "put_call_parity_error": parity_error,
                "d1": self._d1(S, K, T, r, market.volatility),
                "d2": self._d2(self._d1(S, K, T, r, market.volatility), 
                              market.volatility, T),
            }
        )
    
    def put_call_parity_check(self, option: Option, market: MarketEnvironment,
                               tolerance: float = 1e-10) -> bool:
        """
        Verify put-call parity holds.
        
        C - P = S - K*e^(-rT)
        """
        S = market.spot
        K = option.strike
        T = option.maturity
        r = market.risk_free_rate
        
        call_option = Option(K, T, OptionType.CALL)
        put_option = Option(K, T, OptionType.PUT)
        
        call_price = self.price(call_option, market)
        put_price = self.price(put_option, market)
        
        lhs = call_price - put_price
        rhs = S - K * np.exp(-r * T)
        
        return abs(lhs - rhs) < tolerance
