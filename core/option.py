"""
Option contract abstraction for the Option Pricing Engine.

Represents European options as first-class domain objects.
"""

from dataclasses import dataclass
from typing import Union
import numpy as np
import numpy.typing as npt

from .enums import OptionType


@dataclass(frozen=True)
class Option:
    """
    European option contract.
    
    Attributes:
        strike: Strike price
        maturity: Time to maturity in years
        option_type: CALL or PUT
    
    Constraints:
        - European exercise only
        - No dividends
    """
    strike: float
    maturity: float
    option_type: OptionType
    
    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError(f"Strike must be positive, got {self.strike}")
        if self.maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {self.maturity}")
    
    def payoff(self, spot_at_maturity: Union[float, npt.NDArray[np.float64]]) -> Union[float, npt.NDArray[np.float64]]:
        """
        Calculate option payoff at maturity.
        
        Args:
            spot_at_maturity: Underlying price(s) at expiration. 
                              Can be scalar or numpy array for vectorized computation.
        
        Returns:
            Payoff value(s): max(S - K, 0) for calls, max(K - S, 0) for puts.
        """
        spot = np.asarray(spot_at_maturity)
        
        if self.option_type == OptionType.CALL:
            return np.maximum(spot - self.strike, 0.0)
        else:  # PUT
            return np.maximum(self.strike - spot, 0.0)
    
    @property
    def is_call(self) -> bool:
        """Return True if this is a call option."""
        return self.option_type == OptionType.CALL
    
    @property
    def is_put(self) -> bool:
        """Return True if this is a put option."""
        return self.option_type == OptionType.PUT
    
    def __str__(self) -> str:
        return f"European {self.option_type.name} | K={self.strike:.2f} | T={self.maturity:.4f}y"
