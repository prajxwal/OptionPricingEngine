"""
Binomial tree pricing model.

Cox-Ross-Rubinstein (CRR) implementation for European options.
Converges to Black-Scholes as the number of steps increases.
"""

import numpy as np
from typing import List, Optional

from core.option import Option
from core.market import MarketEnvironment
from core.inputs import PricingInputs
from models.base import PricingModel, Greeks, Diagnostics


class BinomialModel(PricingModel):
    """
    Cox-Ross-Rubinstein binomial tree pricing model.
    
    Assumptions:
        - CRR parameterization: u = e^(σ√Δt), d = 1/u
        - Risk-neutral probability: p = (e^(rΔt) - d) / (u - d)
        - European exercise only
        - No dividends
    """
    
    def __init__(self, default_steps: int = 100):
        """
        Initialize binomial model.
        
        Args:
            default_steps: Default number of time steps if not specified
        """
        self.default_steps = default_steps
    
    @property
    def name(self) -> str:
        return "Binomial (CRR)"
    
    def _get_steps(self, inputs: Optional[PricingInputs]) -> int:
        """Get number of steps from inputs or default."""
        if inputs is not None:
            return inputs.numerical.num_steps
        return self.default_steps
    
    def price(self, option: Option, market: MarketEnvironment,
              inputs: Optional[PricingInputs] = None) -> float:
        """
        Calculate option price using CRR binomial tree.
        
        Uses backward induction from terminal payoffs.
        """
        N = self._get_steps(inputs)
        
        S = market.spot
        K = option.strike
        T = option.maturity
        r = market.risk_free_rate
        sigma = market.volatility
        
        # Time step
        dt = T / N
        
        # CRR parameters
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        
        # Risk-neutral probability
        p = (np.exp(r * dt) - d) / (u - d)
        
        # Discount factor per step
        discount = np.exp(-r * dt)
        
        # Build terminal asset prices
        # S_T[j] = S * u^j * d^(N-j) for j = 0, 1, ..., N
        j = np.arange(N + 1)
        S_T = S * (u ** j) * (d ** (N - j))
        
        # Terminal payoffs
        payoffs = option.payoff(S_T)
        
        # Backward induction
        for i in range(N - 1, -1, -1):
            payoffs = discount * (p * payoffs[1:] + (1 - p) * payoffs[:-1])
        
        return float(payoffs[0])
    
    def greeks(self, option: Option, market: MarketEnvironment,
               inputs: Optional[PricingInputs] = None) -> Greeks:
        """
        Calculate Greeks using finite differences on the tree.
        
        This provides tree-based Greeks which may differ slightly
        from Black-Scholes analytical Greeks.
        """
        N = self._get_steps(inputs)
        
        S = market.spot
        K = option.strike
        T = option.maturity
        r = market.risk_free_rate
        sigma = market.volatility
        
        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        
        # For delta and gamma, we need option values at t=dt
        # Build the tree for first two time steps
        
        # Current price
        V_0 = self.price(option, market, inputs)
        
        # Prices at first step
        S_u = S * u
        S_d = S * d
        
        market_u = MarketEnvironment(S_u, sigma, r)
        market_d = MarketEnvironment(S_d, sigma, r)
        
        # Create shorter-dated options for tree nodes
        if T - dt > 0:
            option_dt = Option(K, T - dt, option.option_type)
            V_u = self.price(option_dt, market_u, inputs)
            V_d = self.price(option_dt, market_d, inputs)
        else:
            V_u = option.payoff(S_u)
            V_d = option.payoff(S_d)
        
        # Delta: (V_u - V_d) / (S_u - S_d)
        delta = (V_u - V_d) / (S_u - S_d)
        
        # Gamma: need V_uu, V_ud, V_dd
        if T - 2*dt > 0:
            option_2dt = Option(K, T - 2*dt, option.option_type)
            
            S_uu = S * u * u
            S_ud = S  # u * d = 1
            S_dd = S * d * d
            
            market_uu = MarketEnvironment(S_uu, sigma, r)
            market_ud = MarketEnvironment(S_ud, sigma, r)
            market_dd = MarketEnvironment(S_dd, sigma, r)
            
            V_uu = self.price(option_2dt, market_uu, inputs)
            V_ud = self.price(option_2dt, market_ud, inputs)
            V_dd = self.price(option_2dt, market_dd, inputs)
            
            delta_u = (V_uu - V_ud) / (S_uu - S_ud)
            delta_d = (V_ud - V_dd) / (S_ud - S_dd)
            
            gamma = (delta_u - delta_d) / (0.5 * (S_uu - S_dd))
        else:
            gamma = 0.0
        
        # Theta: (V_ud - V_0) / (2*dt) converted to per day
        theta = (V_ud - V_0) / (2 * dt) / 365 if T - 2*dt > 0 else 0.0
        
        # Vega and Rho via bump-and-reprice
        bump = 0.01
        
        market_vol_up = MarketEnvironment(S, sigma + bump, r)
        market_vol_down = MarketEnvironment(S, sigma - bump, r)
        V_vol_up = self.price(option, market_vol_up, inputs)
        V_vol_down = self.price(option, market_vol_down, inputs)
        vega = (V_vol_up - V_vol_down) / (2 * bump) / 100  # Per 1% vol
        
        market_r_up = MarketEnvironment(S, sigma, r + bump)
        market_r_down = MarketEnvironment(S, sigma, r - bump)
        V_r_up = self.price(option, market_r_up, inputs)
        V_r_down = self.price(option, market_r_down, inputs)
        rho = (V_r_up - V_r_down) / (2 * bump) / 100  # Per 1% rate
        
        return Greeks(
            delta=float(delta),
            gamma=float(gamma),
            vega=float(vega),
            theta=float(theta),
            rho=float(rho),
        )
    
    def assumptions(self) -> List[str]:
        """Return binomial model assumptions."""
        return [
            "CRR parameterization: u = e^(σ√Δt), d = 1/u",
            "Risk-neutral probability derived from no-arbitrage",
            "Discrete time steps (approximation of continuous)",
            "European exercise only",
            "No dividends",
            "Converges to Black-Scholes as steps → ∞",
        ]
    
    def diagnostics(self, option: Option, market: MarketEnvironment,
                    inputs: Optional[PricingInputs] = None) -> Diagnostics:
        """Return convergence diagnostics."""
        N = self._get_steps(inputs)
        
        warnings_list = []
        if N < 50:
            warnings_list.append(f"Low step count ({N}) may affect accuracy")
        
        # Calculate convergence indicators
        dt = option.maturity / N
        u = np.exp(market.volatility * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(market.risk_free_rate * dt) - d) / (u - d)
        
        return Diagnostics(
            iterations=N,
            extra={
                "up_factor": u,
                "down_factor": d,
                "risk_neutral_prob": p,
                "time_step": dt,
            }
        )
