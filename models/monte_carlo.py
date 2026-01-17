"""
Monte Carlo pricing model.

GBM-based path simulation for European option pricing.
Provides confidence intervals and convergence diagnostics.
"""

import numpy as np
from typing import List, Optional, Tuple

from core.option import Option
from core.market import MarketEnvironment
from core.inputs import PricingInputs
from models.base import PricingModel, Greeks, Diagnostics


class MonteCarloModel(PricingModel):
    """
    Monte Carlo simulation pricing model.
    
    Assumptions:
        - Geometric Brownian Motion (GBM) for asset dynamics
        - Euler-Maruyama discretization
        - Risk-neutral measure for pricing
        - European exercise only
        - No dividends
    """
    
    def __init__(self, default_paths: int = 100000, default_seed: int = 42):
        """
        Initialize Monte Carlo model.
        
        Args:
            default_paths: Default number of simulation paths
            default_seed: Default random seed for reproducibility
        """
        self.default_paths = default_paths
        self.default_seed = default_seed
        self._last_std_error: Optional[float] = None
        self._last_conf_interval: Optional[Tuple[float, float]] = None
    
    @property
    def name(self) -> str:
        return "Monte Carlo"
    
    def _get_params(self, inputs: Optional[PricingInputs]) -> Tuple[int, int]:
        """Get simulation parameters from inputs or defaults."""
        if inputs is not None:
            return inputs.numerical.num_paths, inputs.numerical.seed
        return self.default_paths, self.default_seed
    
    def price(self, option: Option, market: MarketEnvironment,
              inputs: Optional[PricingInputs] = None) -> float:
        """
        Calculate option price using Monte Carlo simulation.
        
        Simulates GBM paths and averages discounted payoffs.
        """
        num_paths, seed = self._get_params(inputs)
        
        S = market.spot
        T = option.maturity
        r = market.risk_free_rate
        sigma = market.volatility
        
        # Set random seed for reproducibility
        rng = np.random.default_rng(seed)
        
        # Simulate terminal prices using GBM closed-form solution
        # S_T = S * exp((r - 0.5*σ²)*T + σ*√T*Z)
        Z = rng.standard_normal(num_paths)
        S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        
        # Calculate payoffs
        payoffs = option.payoff(S_T)
        
        # Discount and average
        discount = np.exp(-r * T)
        discounted_payoffs = discount * payoffs
        
        price = np.mean(discounted_payoffs)
        
        # Store statistics for diagnostics
        self._last_std_error = np.std(discounted_payoffs) / np.sqrt(num_paths)
        self._last_conf_interval = (
            price - 1.96 * self._last_std_error,
            price + 1.96 * self._last_std_error
        )
        
        return float(price)
    
    def greeks(self, option: Option, market: MarketEnvironment,
               inputs: Optional[PricingInputs] = None) -> Greeks:
        """
        Calculate Greeks using pathwise and finite difference methods.
        
        Delta uses pathwise estimator, others use finite differences.
        """
        num_paths, seed = self._get_params(inputs)
        
        S = market.spot
        K = option.strike
        T = option.maturity
        r = market.risk_free_rate
        sigma = market.volatility
        
        rng = np.random.default_rng(seed)
        
        # Generate paths once for consistency
        Z = rng.standard_normal(num_paths)
        S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        
        discount = np.exp(-r * T)
        
        # Delta: Pathwise estimator
        # d/dS E[e^(-rT) * payoff(S_T)] = e^(-rT) * E[payoff'(S_T) * dS_T/dS]
        # dS_T/dS = S_T / S
        if option.is_call:
            indicator = (S_T > K).astype(float)
            delta = discount * np.mean(indicator * S_T / S)
        else:
            indicator = (S_T < K).astype(float)
            delta = discount * np.mean(-indicator * S_T / S)
        
        # Gamma via finite difference
        bump_s = max(0.01 * S, 0.01)
        market_up = MarketEnvironment(S + bump_s, sigma, r)
        market_down = MarketEnvironment(S - bump_s, sigma, r)
        
        # Use same seed for consistency
        rng_up = np.random.default_rng(seed)
        rng_down = np.random.default_rng(seed)
        
        Z_up = rng_up.standard_normal(num_paths)
        Z_down = rng_down.standard_normal(num_paths)
        
        S_T_up = (S + bump_s) * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_up)
        S_T_down = (S - bump_s) * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_down)
        
        price_up = discount * np.mean(option.payoff(S_T_up))
        price_down = discount * np.mean(option.payoff(S_T_down))
        price_mid = discount * np.mean(option.payoff(S_T))
        
        gamma = (price_up - 2 * price_mid + price_down) / (bump_s ** 2)
        
        # Vega via finite difference
        bump_vol = max(0.01 * sigma, 0.001)
        rng_vol = np.random.default_rng(seed)
        Z_vol = rng_vol.standard_normal(num_paths)
        
        S_T_vol_up = S * np.exp((r - 0.5 * (sigma + bump_vol)**2) * T + (sigma + bump_vol) * np.sqrt(T) * Z_vol)
        S_T_vol_down = S * np.exp((r - 0.5 * (sigma - bump_vol)**2) * T + (sigma - bump_vol) * np.sqrt(T) * Z_vol)
        
        price_vol_up = discount * np.mean(option.payoff(S_T_vol_up))
        price_vol_down = discount * np.mean(option.payoff(S_T_vol_down))
        
        vega = (price_vol_up - price_vol_down) / (2 * bump_vol) / 100  # Per 1%
        
        # Theta via finite difference on maturity
        bump_t = 1/365  # 1 day
        if T > bump_t:
            rng_t = np.random.default_rng(seed)
            Z_t = rng_t.standard_normal(num_paths)
            
            T_down = T - bump_t
            S_T_t = S * np.exp((r - 0.5 * sigma**2) * T_down + sigma * np.sqrt(T_down) * Z_t)
            price_t = np.exp(-r * T_down) * np.mean(option.payoff(S_T_t))
            
            theta = (price_t - price_mid) / bump_t / 365  # Per day
        else:
            theta = 0.0
        
        # Rho via finite difference
        bump_r = 0.0001  # 1 bp
        rng_r = np.random.default_rng(seed)
        Z_r = rng_r.standard_normal(num_paths)
        
        S_T_r_up = S * np.exp(((r + bump_r) - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_r)
        S_T_r_down = S * np.exp(((r - bump_r) - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_r)
        
        price_r_up = np.exp(-(r + bump_r) * T) * np.mean(option.payoff(S_T_r_up))
        price_r_down = np.exp(-(r - bump_r) * T) * np.mean(option.payoff(S_T_r_down))
        
        rho = (price_r_up - price_r_down) / (2 * bump_r) / 100  # Per 1%
        
        return Greeks(
            delta=float(delta),
            gamma=float(gamma),
            vega=float(vega),
            theta=float(theta),
            rho=float(rho),
        )
    
    def assumptions(self) -> List[str]:
        """Return Monte Carlo model assumptions."""
        return [
            "Geometric Brownian Motion asset dynamics",
            "Euler-Maruyama discretization (single step for European)",
            "Risk-neutral probability measure",
            "Variance reduction not applied (plain MC)",
            "European exercise only",
            "No dividends",
            "Convergence rate: O(1/√N)",
        ]
    
    def diagnostics(self, option: Option, market: MarketEnvironment,
                    inputs: Optional[PricingInputs] = None) -> Diagnostics:
        """Return Monte Carlo specific diagnostics."""
        num_paths, seed = self._get_params(inputs)
        
        # Run pricing to get statistics if not already done
        if self._last_std_error is None:
            self.price(option, market, inputs)
        
        warnings_list = []
        if num_paths < 10000:
            warnings_list.append(f"Low path count ({num_paths}) may have high variance")
        if self._last_std_error and self._last_std_error > 0.1:
            warnings_list.append(f"High standard error ({self._last_std_error:.4f})")
        
        return Diagnostics(
            standard_error=self._last_std_error,
            confidence_interval=self._last_conf_interval,
            iterations=num_paths,
            extra={
                "seed": seed,
                "relative_error": self._last_std_error / self.price(option, market, inputs) 
                                  if self._last_std_error else None,
            }
        )
