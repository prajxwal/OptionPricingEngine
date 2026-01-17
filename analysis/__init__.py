"""
Analysis module for the Option Pricing Engine.

Provides model comparison, convergence analysis, and experiments.
"""

from .comparison import (
    ComparisonMetrics,
    ComparisonResult,
    ModelComparison,
    compare_all_models,
)
from .convergence import (
    ConvergencePoint,
    ConvergenceResult,
    ConvergenceAnalyzer,
    analyze_all_convergence,
)
from .experiments import (
    ExperimentPoint,
    ExperimentResult,
    ExperimentRunner,
)

__all__ = [
    "ComparisonMetrics",
    "ComparisonResult",
    "ModelComparison",
    "compare_all_models",
    "ConvergencePoint",
    "ConvergenceResult",
    "ConvergenceAnalyzer",
    "analyze_all_convergence",
    "ExperimentPoint",
    "ExperimentResult",
    "ExperimentRunner",
]
