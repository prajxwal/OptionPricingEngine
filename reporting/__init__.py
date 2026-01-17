"""
Reporting module for the Option Pricing Engine.

Provides plotting and summary generation.
"""

from .plots import (
    fig_to_base64,
    plot_price_vs_volatility,
    plot_price_vs_maturity,
    plot_convergence,
    plot_model_comparison,
    plot_greeks_comparison,
    plot_moneyness_analysis,
    plot_error_heatmap,
)
from .summary import (
    ModelSummary,
    PricingSummary,
    SummaryGenerator,
    generate_summary,
    LIMITATIONS_DISCLAIMER,
    MODEL_LIMITATIONS,
)

__all__ = [
    "fig_to_base64",
    "plot_price_vs_volatility",
    "plot_price_vs_maturity",
    "plot_convergence",
    "plot_model_comparison",
    "plot_greeks_comparison",
    "plot_moneyness_analysis",
    "plot_error_heatmap",
    "ModelSummary",
    "PricingSummary",
    "SummaryGenerator",
    "generate_summary",
    "LIMITATIONS_DISCLAIMER",
    "MODEL_LIMITATIONS",
]
