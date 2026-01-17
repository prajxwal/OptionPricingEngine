"""
Summary report generation.

Creates structured summaries of pricing results and analysis.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

from core.option import Option
from core.market import MarketEnvironment
from core.inputs import PricingInputs
from models.base import PricingResult


@dataclass
class ModelSummary:
    """Summary of a single model's results."""
    name: str
    price: float
    assumptions: List[str]
    limitations: List[str]
    warnings: List[str]
    diagnostics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "price": self.price,
            "assumptions": self.assumptions,
            "limitations": self.limitations,
            "warnings": self.warnings,
            "diagnostics": self.diagnostics,
        }


@dataclass
class PricingSummary:
    """
    Complete summary of a pricing session.
    
    Includes reproducibility metadata and model disagreement analysis.
    """
    timestamp: str
    input_signature: str
    option_description: str
    market_description: str
    model_summaries: List[ModelSummary]
    comparison_summary: str
    disagreement_explanation: str
    limitations_disclaimer: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "input_signature": self.input_signature,
            "option_description": self.option_description,
            "market_description": self.market_description,
            "model_summaries": [m.to_dict() for m in self.model_summaries],
            "comparison_summary": self.comparison_summary,
            "disagreement_explanation": self.disagreement_explanation,
            "limitations_disclaimer": self.limitations_disclaimer,
        }


# Standard limitations disclaimer
LIMITATIONS_DISCLAIMER = """
IMPORTANT LIMITATIONS:
• Black-Scholes assumptions are unrealistic in practice
• Constant volatility is a simplification - real markets exhibit volatility smiles/skews
• Monte Carlo results are stochastic and may vary slightly between runs
• No dividends, transaction costs, or market frictions are modeled
• These prices are for educational/research purposes only - not for trading decisions
• Numerical methods (Binomial, MC) are approximations with finite precision
• Results assume European exercise - not valid for American options
"""


# Model-specific limitations
MODEL_LIMITATIONS = {
    "Black-Scholes": [
        "Assumes constant volatility",
        "Assumes continuous trading",
        "No dividends modeled",
        "Log-normal distribution may not match reality",
        "Not suitable for options with dividend-paying underlyings",
    ],
    "Binomial (CRR)": [
        "Discrete time approximation",
        "Accuracy depends on number of steps",
        "CRR parameterization assumes specific up/down factors",
        "May exhibit oscillation around true value for low step counts",
    ],
    "Monte Carlo": [
        "Stochastic - results vary by random seed",
        "Computationally expensive for high accuracy",
        "Standard error decreases as O(1/√N)",
        "No variance reduction techniques applied",
        "Greeks via finite difference may be noisy",
    ],
}


class SummaryGenerator:
    """Generate structured summaries of pricing results."""
    
    def generate_pricing_summary(
        self,
        results: List[PricingResult],
        option: Option,
        market: MarketEnvironment,
        inputs: Optional[PricingInputs] = None,
    ) -> PricingSummary:
        """
        Generate a complete pricing summary.
        
        Args:
            results: List of PricingResult from different models
            option: The option that was priced
            market: Market environment used
            inputs: Input parameters (for signature)
            
        Returns:
            PricingSummary with all metadata
        """
        model_summaries = []
        
        for result in results:
            assumptions = self._get_model_assumptions(result.model_name)
            limitations = MODEL_LIMITATIONS.get(result.model_name, [])
            
            model_summaries.append(ModelSummary(
                name=result.model_name,
                price=result.price,
                assumptions=assumptions,
                limitations=limitations,
                warnings=result.warnings,
                diagnostics=result.diagnostics.to_dict(),
            ))
        
        # Generate comparison summary
        comparison = self._generate_comparison_summary(results)
        disagreement = self._generate_disagreement_explanation(results, option, market)
        
        return PricingSummary(
            timestamp=datetime.now().isoformat(),
            input_signature=inputs.input_signature() if inputs else "",
            option_description=str(option),
            market_description=str(market),
            model_summaries=model_summaries,
            comparison_summary=comparison,
            disagreement_explanation=disagreement,
            limitations_disclaimer=LIMITATIONS_DISCLAIMER,
        )
    
    def _get_model_assumptions(self, model_name: str) -> List[str]:
        """Get assumptions for a model."""
        assumptions_map = {
            "Black-Scholes": [
                "Constant volatility throughout option life",
                "Continuous trading with no gaps",
                "No dividends paid during option life",
                "Log-normal distribution of asset prices",
                "Constant risk-free interest rate",
                "No transaction costs or taxes",
            ],
            "Binomial (CRR)": [
                "CRR parameterization: u = e^(σ√Δt)",
                "Risk-neutral probability from no-arbitrage",
                "Discrete time approximation",
            ],
            "Monte Carlo": [
                "Geometric Brownian Motion dynamics",
                "Euler-Maruyama discretization",
                "Risk-neutral measure for pricing",
            ],
        }
        return assumptions_map.get(model_name, [])
    
    def _generate_comparison_summary(self, results: List[PricingResult]) -> str:
        """Generate a summary of model comparison."""
        if len(results) <= 1:
            return "Single model - no comparison available"
        
        prices = [r.price for r in results]
        names = [r.model_name for r in results]
        
        min_price = min(prices)
        max_price = max(prices)
        spread = max_price - min_price
        avg_price = sum(prices) / len(prices)
        
        if spread < 0.01 * avg_price:
            return f"Excellent agreement: All models within 1% ({spread:.4f} spread)"
        elif spread < 0.05 * avg_price:
            return f"Good agreement: Spread of {spread:.4f} ({spread/avg_price:.1%} of average)"
        else:
            return f"Significant divergence: Spread of {spread:.4f} ({spread/avg_price:.1%})"
    
    def _generate_disagreement_explanation(
        self,
        results: List[PricingResult],
        option: Option,
        market: MarketEnvironment,
    ) -> str:
        """Explain why models might disagree."""
        if len(results) <= 1:
            return ""
        
        explanations = []
        
        # Find the largest deviations
        bs_price = None
        for r in results:
            if r.model_name == "Black-Scholes":
                bs_price = r.price
                break
        
        if bs_price is None:
            return "No Black-Scholes baseline available for comparison"
        
        for r in results:
            if r.model_name == "Black-Scholes":
                continue
            
            error = abs(r.price - bs_price)
            rel_error = error / bs_price if bs_price != 0 else 0
            
            if rel_error > 0.01:  # More than 1%
                if "Binomial" in r.model_name:
                    explanations.append(
                        f"{r.model_name} shows {rel_error:.2%} deviation. "
                        "This is expected for discrete approximations. "
                        "Increasing steps will reduce this error."
                    )
                elif "Monte Carlo" in r.model_name:
                    explanations.append(
                        f"{r.model_name} shows {rel_error:.2%} deviation. "
                        "Monte Carlo has statistical uncertainty. "
                        "Increasing paths will reduce standard error."
                    )
        
        # Check for problematic option characteristics
        moneyness = market.spot / option.strike
        if moneyness < 0.8 or moneyness > 1.25:
            explanations.append(
                f"Deep {'ITM' if moneyness > 1 else 'OTM'} options (moneyness={moneyness:.2f}) "
                "may show larger numerical differences between models."
            )
        
        if option.maturity < 0.1:
            explanations.append(
                "Very short maturity options have less time for "
                "numerical methods to approximate the continuous process."
            )
        
        return " ".join(explanations) if explanations else "Models are in close agreement."


def generate_summary(
    results: List[PricingResult],
    option: Option,
    market: MarketEnvironment,
    inputs: Optional[PricingInputs] = None,
) -> PricingSummary:
    """Convenience function to generate a pricing summary."""
    generator = SummaryGenerator()
    return generator.generate_pricing_summary(results, option, market, inputs)
