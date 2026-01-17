"""
Service layer bridging web interface and pricing engine.

This is critical for maintaining architecture cleanliness.
All business logic coordination happens here - NOT in routes.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enums import OptionType, ModelType
from core.inputs import OptionParams, MarketParams, NumericalParams, PricingInputs
from core.option import Option
from core.market import MarketEnvironment
from models.base import PricingModel, PricingResult
from models.black_scholes import BlackScholesModel
from models.binomial import BinomialModel
from models.monte_carlo import MonteCarloModel
from analysis.comparison import ModelComparison, ComparisonResult
from analysis.convergence import ConvergenceAnalyzer, ConvergenceResult
from reporting.plots import (
    plot_model_comparison,
    plot_convergence,
    plot_greeks_comparison,
)
from reporting.summary import generate_summary, PricingSummary


@dataclass
class PricingRequest:
    """Structured request from web form."""
    # Option
    option_type: str  # 'call' or 'put'
    strike: float
    maturity: float
    
    # Market
    spot: float
    volatility: float
    risk_free_rate: float
    
    # Models to run
    use_black_scholes: bool
    use_binomial: bool
    use_monte_carlo: bool
    
    # Numerical params
    num_steps: int
    num_paths: int
    seed: int
    
    # Analysis options
    run_convergence: bool


@dataclass
class PricingResponse:
    """Complete response for web display."""
    # Core results
    results: List[PricingResult]
    comparison: ComparisonResult
    summary: PricingSummary
    
    # Plots (base64 encoded)
    comparison_plot: str
    greeks_plot: str
    convergence_plots: Dict[str, str]  # model_name -> plot
    
    # Metadata
    input_signature: str
    option_description: str
    market_description: str


class PricingService:
    """
    Bridge between web forms and pricing engine.
    
    Responsibilities:
    - Convert form inputs to domain objects
    - Run selected models
    - Coordinate comparison and analysis
    - Generate plots and summaries
    """
    
    def __init__(self):
        self.models: Dict[str, PricingModel] = {
            "Black-Scholes": BlackScholesModel(),
            "Binomial (CRR)": BinomialModel(),
            "Monte Carlo": MonteCarloModel(),
        }
        self.comparison = ModelComparison()
        self.convergence = ConvergenceAnalyzer()
    
    def process_request(self, request: PricingRequest) -> PricingResponse:
        """
        Process a pricing request from the web form.
        
        This is the main entry point for the service layer.
        """
        # Convert to domain objects
        option, market, inputs = self._create_domain_objects(request)
        
        # Select models
        selected_models = self._get_selected_models(request)
        
        # Run pricing
        results = self._run_models(selected_models, option, market, inputs)
        
        # Run comparison
        comparison_result = self.comparison.compare(
            selected_models, option, market, inputs
        )
        
        # Generate summary
        summary = generate_summary(results, option, market, inputs)
        
        # Generate plots
        comparison_plot = self._create_comparison_plot(results)
        greeks_plot = self._create_greeks_plot(results)
        
        # Convergence analysis if requested
        convergence_plots = {}
        if request.run_convergence:
            convergence_plots = self._run_convergence_analysis(option, market)
        
        return PricingResponse(
            results=results,
            comparison=comparison_result,
            summary=summary,
            comparison_plot=comparison_plot,
            greeks_plot=greeks_plot,
            convergence_plots=convergence_plots,
            input_signature=inputs.input_signature(),
            option_description=str(option),
            market_description=str(market),
        )
    
    def _create_domain_objects(
        self, request: PricingRequest
    ) -> Tuple[Option, MarketEnvironment, PricingInputs]:
        """Convert request to domain objects."""
        option_type = OptionType.CALL if request.option_type == 'call' else OptionType.PUT
        
        option = Option(
            strike=request.strike,
            maturity=request.maturity,
            option_type=option_type,
        )
        
        market = MarketEnvironment(
            spot=request.spot,
            volatility=request.volatility,
            risk_free_rate=request.risk_free_rate,
        )
        
        option_params = OptionParams(request.strike, request.maturity, option_type)
        market_params = MarketParams(request.spot, request.volatility, request.risk_free_rate)
        numerical_params = NumericalParams(
            num_steps=request.num_steps,
            num_paths=request.num_paths,
            seed=request.seed,
        )
        
        inputs = PricingInputs(option_params, market_params, numerical_params)
        
        return option, market, inputs
    
    def _get_selected_models(self, request: PricingRequest) -> List[PricingModel]:
        """Get list of selected pricing models."""
        models = []
        
        if request.use_black_scholes:
            models.append(self.models["Black-Scholes"])
        if request.use_binomial:
            models.append(self.models["Binomial (CRR)"])
        if request.use_monte_carlo:
            models.append(self.models["Monte Carlo"])
        
        return models
    
    def _run_models(
        self,
        models: List[PricingModel],
        option: Option,
        market: MarketEnvironment,
        inputs: PricingInputs,
    ) -> List[PricingResult]:
        """Run all selected models and collect results."""
        results = []
        
        for model in models:
            result = model.compute(option, market, inputs)
            results.append(result)
        
        return results
    
    def _create_comparison_plot(self, results: List[PricingResult]) -> str:
        """Create model comparison bar chart."""
        if not results:
            return ""
        
        model_names = [r.model_name for r in results]
        prices = [r.price for r in results]
        
        # Calculate errors vs first model (Black-Scholes if present)
        baseline_price = results[0].price
        errors = [abs(r.price - baseline_price) for r in results]
        
        return plot_model_comparison(
            model_names=model_names,
            prices=prices,
            errors=errors,
            baseline_name=results[0].model_name,
        )
    
    def _create_greeks_plot(self, results: List[PricingResult]) -> str:
        """Create Greeks comparison chart."""
        if not results:
            return ""
        
        greeks_data = {r.model_name: r.greeks.to_dict() for r in results}
        return plot_greeks_comparison(greeks_data)
    
    def _run_convergence_analysis(
        self,
        option: Option,
        market: MarketEnvironment,
    ) -> Dict[str, str]:
        """Run convergence analysis and generate plots."""
        plots = {}
        
        # Binomial convergence
        binomial_result = self.convergence.analyze_binomial_convergence(option, market)
        plots["Binomial (CRR)"] = plot_convergence(
            parameter_values=[p.parameter_value for p in binomial_result.points],
            errors=[p.error for p in binomial_result.points],
            model_name="Binomial (CRR)",
            parameter_name="steps",
            baseline_price=binomial_result.baseline_price,
        )
        
        # Monte Carlo convergence
        mc_result = self.convergence.analyze_monte_carlo_convergence(option, market)
        plots["Monte Carlo"] = plot_convergence(
            parameter_values=[p.parameter_value for p in mc_result.points],
            errors=[p.error for p in mc_result.points],
            model_name="Monte Carlo",
            parameter_name="paths",
            baseline_price=mc_result.baseline_price,
        )
        
        return plots


# Convenience function for route handlers
def process_pricing_form(form_data: Dict[str, Any]) -> PricingResponse:
    """
    Process pricing form data and return response.
    
    This is the main interface used by routes.
    """
    request = PricingRequest(
        option_type=form_data.get('option_type', 'call'),
        strike=float(form_data.get('strike', 100)),
        maturity=float(form_data.get('maturity', 1.0)),
        spot=float(form_data.get('spot', 100)),
        volatility=float(form_data.get('volatility', 0.20)),
        risk_free_rate=float(form_data.get('risk_free_rate', 0.05)),
        use_black_scholes=form_data.get('use_black_scholes', True),
        use_binomial=form_data.get('use_binomial', True),
        use_monte_carlo=form_data.get('use_monte_carlo', True),
        num_steps=int(form_data.get('num_steps', 100)),
        num_paths=int(form_data.get('num_paths', 100000)),
        seed=int(form_data.get('seed', 42)),
        run_convergence=form_data.get('run_convergence', False),
    )
    
    service = PricingService()
    return service.process_request(request)
