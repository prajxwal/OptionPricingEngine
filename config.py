"""
Configuration for the Option Pricing Engine.

Separates numerical defaults from Flask configuration.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NumericalDefaults:
    """Default values for numerical computations."""
    # Binomial model
    default_steps: int = 100
    max_steps: int = 1000
    
    # Monte Carlo
    default_paths: int = 100000
    max_paths: int = 500000
    default_seed: int = 42
    
    # Finite difference
    relative_bump_spot: float = 0.01
    relative_bump_vol: float = 0.01
    relative_bump_rate: float = 0.0001
    min_bump: float = 0.0001


@dataclass(frozen=True)
class FlaskConfig:
    """Flask application configuration."""
    SECRET_KEY: str = "option-pricing-engine-dev-key-change-in-production"
    DEBUG: bool = True
    HOST: str = "127.0.0.1"
    PORT: int = 5000
    
    # Request limits
    MAX_CONTENT_LENGTH: int = 1024 * 1024  # 1MB
    
    # Session
    SESSION_COOKIE_SECURE: bool = False
    SESSION_COOKIE_HTTPONLY: bool = True


@dataclass(frozen=True)
class ValidationLimits:
    """Input validation limits."""
    # Spot price
    min_spot: float = 0.01
    max_spot: float = 100000.0
    
    # Strike
    min_strike: float = 0.01
    max_strike: float = 100000.0
    
    # Volatility
    min_volatility: float = 0.01
    max_volatility: float = 5.0  # 500%
    
    # Interest rate
    min_rate: float = -0.10  # -10%
    max_rate: float = 1.0    # 100%
    
    # Maturity
    min_maturity: float = 0.001  # ~8 hours
    max_maturity: float = 30.0   # 30 years
    
    # Numerical params
    min_steps: int = 2
    max_steps: int = 1000
    min_paths: int = 100
    max_paths: int = 500000


# Global configuration instances
NUMERICAL_DEFAULTS = NumericalDefaults()
FLASK_CONFIG = FlaskConfig()
VALIDATION_LIMITS = ValidationLimits()


# Reproducibility disclaimer
REPRODUCIBILITY_NOTE = """
Reproducibility is guaranteed on the same platform and Python version.
Floating-point results may vary slightly across different environments
due to differences in numerical precision and library implementations.
"""
