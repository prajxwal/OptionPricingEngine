"""
Enums for the Option Pricing Engine.

Defines standardized types for option contracts and pricing models.
"""

from enum import Enum, auto


class OptionType(Enum):
    """European option type."""
    CALL = auto()
    PUT = auto()
    
    def __str__(self) -> str:
        return self.name


class ModelType(Enum):
    """Available pricing models."""
    BLACK_SCHOLES = auto()
    BINOMIAL = auto()
    MONTE_CARLO = auto()
    
    def __str__(self) -> str:
        return self.name.replace("_", " ").title()
