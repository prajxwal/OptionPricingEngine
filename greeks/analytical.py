"""
Analytical Greeks calculations.

Provides exact closed-form Greeks for Black-Scholes model.
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Dict

from core.option import Option
from core.market import MarketEnvironment
from core.enums import OptionType


@dataclass
class AnalyticalGreeks:
    """
    Complete set of analytical Greeks.
    
    All values are computed using Black-Scholes closed-form formulas.
    """
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    vanna: float = 0.0  # d(delta)/d(sigma)
    charm: float = 0.0  # d(delta)/d(t)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
            "vanna": self.vanna,
            "charm": self.charm,
        }


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d1 parameter."""
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(d1: float, sigma: float, T: float) -> float:
    """Calculate d2 parameter."""
    return d1 - sigma * np.sqrt(T)


def calculate_delta(option: Option, market: MarketEnvironment) -> float:
    """
    Calculate Delta: sensitivity of option price to underlying price.
    
    Call Delta: N(d1)
    Put Delta: N(d1) - 1
    
    Delta ranges from 0 to 1 for calls, -1 to 0 for puts.
    """
    S, K, T = market.spot, option.strike, option.maturity
    r, sigma = market.risk_free_rate, market.volatility
    
    d1 = _d1(S, K, T, r, sigma)
    
    if option.is_call:
        return float(norm.cdf(d1))
    else:
        return float(norm.cdf(d1) - 1)


def calculate_gamma(option: Option, market: MarketEnvironment) -> float:
    """
    Calculate Gamma: rate of change of delta with respect to underlying price.
    
    Gamma = N'(d1) / (S * σ * √T)
    
    Gamma is the same for calls and puts.
    """
    S, K, T = market.spot, option.strike, option.maturity
    r, sigma = market.risk_free_rate, market.volatility
    
    d1 = _d1(S, K, T, r, sigma)
    
    return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def calculate_vega(option: Option, market: MarketEnvironment) -> float:
    """
    Calculate Vega: sensitivity of option price to volatility.
    
    Vega = S * N'(d1) * √T
    
    Vega is the same for calls and puts.
    Returns value per 1% change in volatility (divided by 100).
    """
    S, K, T = market.spot, option.strike, option.maturity
    r, sigma = market.risk_free_rate, market.volatility
    
    d1 = _d1(S, K, T, r, sigma)
    
    # Per 1% vol change
    return float(S * norm.pdf(d1) * np.sqrt(T) / 100)


def calculate_theta(option: Option, market: MarketEnvironment) -> float:
    """
    Calculate Theta: sensitivity of option price to time decay.
    
    Returns value per day (divided by 365).
    Theta is typically negative (options lose value over time).
    """
    S, K, T = market.spot, option.strike, option.maturity
    r, sigma = market.risk_free_rate, market.volatility
    
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(d1, sigma, T)
    
    sqrt_T = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    discount = np.exp(-r * T)
    
    if option.is_call:
        theta = (-(S * pdf_d1 * sigma) / (2 * sqrt_T) 
                 - r * K * discount * norm.cdf(d2))
    else:
        theta = (-(S * pdf_d1 * sigma) / (2 * sqrt_T) 
                 + r * K * discount * norm.cdf(-d2))
    
    # Per day
    return float(theta / 365)


def calculate_rho(option: Option, market: MarketEnvironment) -> float:
    """
    Calculate Rho: sensitivity of option price to interest rate.
    
    Call Rho: K * T * e^(-rT) * N(d2)
    Put Rho: -K * T * e^(-rT) * N(-d2)
    
    Returns value per 1% change in interest rate (divided by 100).
    """
    S, K, T = market.spot, option.strike, option.maturity
    r, sigma = market.risk_free_rate, market.volatility
    
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(d1, sigma, T)
    
    discount = np.exp(-r * T)
    
    if option.is_call:
        rho = K * T * discount * norm.cdf(d2)
    else:
        rho = -K * T * discount * norm.cdf(-d2)
    
    # Per 1% rate change
    return float(rho / 100)


def calculate_vanna(option: Option, market: MarketEnvironment) -> float:
    """
    Calculate Vanna: sensitivity of delta to volatility (cross-gamma).
    
    Vanna = d(delta)/d(sigma) = -N'(d1) * d2 / sigma
    """
    S, K, T = market.spot, option.strike, option.maturity
    r, sigma = market.risk_free_rate, market.volatility
    
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(d1, sigma, T)
    
    return float(-norm.pdf(d1) * d2 / sigma)


def calculate_charm(option: Option, market: MarketEnvironment) -> float:
    """
    Calculate Charm: rate of change of delta over time (delta decay).
    
    Charm = d(delta)/dt
    """
    S, K, T = market.spot, option.strike, option.maturity
    r, sigma = market.risk_free_rate, market.volatility
    
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(d1, sigma, T)
    
    pdf_d1 = norm.pdf(d1)
    sqrt_T = np.sqrt(T)
    
    charm = pdf_d1 * (2 * r * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)
    
    if option.is_put:
        charm = charm
    
    # Per day
    return float(charm / 365)


def calculate_all_greeks(option: Option, market: MarketEnvironment) -> AnalyticalGreeks:
    """
    Calculate all analytical Greeks at once.
    
    This is more efficient than calling individual functions
    as it reuses intermediate calculations.
    """
    S, K, T = market.spot, option.strike, option.maturity
    r, sigma = market.risk_free_rate, market.volatility
    
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(d1, sigma, T)
    
    sqrt_T = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    discount = np.exp(-r * T)
    
    # Delta
    if option.is_call:
        delta = cdf_d1
    else:
        delta = cdf_d1 - 1
    
    # Gamma (same for call/put)
    gamma = pdf_d1 / (S * sigma * sqrt_T)
    
    # Vega (same for call/put, per 1%)
    vega = S * pdf_d1 * sqrt_T / 100
    
    # Theta
    if option.is_call:
        theta = (-(S * pdf_d1 * sigma) / (2 * sqrt_T) 
                 - r * K * discount * cdf_d2) / 365
    else:
        theta = (-(S * pdf_d1 * sigma) / (2 * sqrt_T) 
                 + r * K * discount * norm.cdf(-d2)) / 365
    
    # Rho (per 1%)
    if option.is_call:
        rho = K * T * discount * cdf_d2 / 100
    else:
        rho = -K * T * discount * norm.cdf(-d2) / 100
    
    # Vanna
    vanna = -pdf_d1 * d2 / sigma
    
    # Charm (per day)
    charm = pdf_d1 * (2 * r * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T) / 365
    
    return AnalyticalGreeks(
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
        vanna=float(vanna),
        charm=float(charm),
    )
