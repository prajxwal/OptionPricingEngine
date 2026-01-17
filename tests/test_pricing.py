"""
Unit tests for the Option Pricing Engine.

Tests cover:
1. Put-call parity for Black-Scholes
2. Binomial convergence to BS
3. Monte Carlo convergence to BS
4. Finite-difference Greeks converge to analytical Greeks
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enums import OptionType
from core.option import Option
from core.market import MarketEnvironment
from core.inputs import OptionParams, MarketParams, NumericalParams, PricingInputs
from models.black_scholes import BlackScholesModel
from models.binomial import BinomialModel
from models.monte_carlo import MonteCarloModel
from greeks.analytical import calculate_all_greeks
from greeks.finite_difference import FiniteDifferenceCalculator, FiniteDifferenceConfig


class TestBlackScholes:
    """Tests for Black-Scholes model."""
    
    def setup_method(self):
        self.model = BlackScholesModel()
        self.option_call = Option(strike=100, maturity=1.0, option_type=OptionType.CALL)
        self.option_put = Option(strike=100, maturity=1.0, option_type=OptionType.PUT)
        self.market = MarketEnvironment(spot=100, volatility=0.20, risk_free_rate=0.05)
    
    def test_put_call_parity(self):
        """Test that put-call parity holds: C - P = S - K*e^(-rT)"""
        call_price = self.model.price(self.option_call, self.market)
        put_price = self.model.price(self.option_put, self.market)
        
        S = self.market.spot
        K = self.option_call.strike
        r = self.market.risk_free_rate
        T = self.option_call.maturity
        
        lhs = call_price - put_price
        rhs = S - K * np.exp(-r * T)
        
        assert abs(lhs - rhs) < 1e-10, f"Put-call parity violated: {lhs} != {rhs}"
    
    def test_call_price_positive(self):
        """Test that call price is positive."""
        price = self.model.price(self.option_call, self.market)
        assert price > 0
    
    def test_put_price_positive(self):
        """Test that put price is positive."""
        price = self.model.price(self.option_put, self.market)
        assert price > 0
    
    def test_call_delta_between_zero_and_one(self):
        """Test that call delta is between 0 and 1."""
        greeks = self.model.greeks(self.option_call, self.market)
        assert 0 <= greeks.delta <= 1
    
    def test_put_delta_between_minus_one_and_zero(self):
        """Test that put delta is between -1 and 0."""
        greeks = self.model.greeks(self.option_put, self.market)
        assert -1 <= greeks.delta <= 0


class TestBinomialConvergence:
    """Tests for Binomial model convergence to Black-Scholes."""
    
    def setup_method(self):
        self.bs_model = BlackScholesModel()
        self.binomial_model = BinomialModel()
        self.option = Option(strike=100, maturity=1.0, option_type=OptionType.CALL)
        self.market = MarketEnvironment(spot=100, volatility=0.20, risk_free_rate=0.05)
    
    def test_convergence_to_black_scholes(self):
        """Test that binomial converges to BS as steps increase."""
        bs_price = self.bs_model.price(self.option, self.market)
        
        step_values = [50, 100, 200, 500]
        errors = []
        
        for steps in step_values:
            option_params = OptionParams(100, 1.0, OptionType.CALL)
            market_params = MarketParams(100, 0.20, 0.05)
            numerical = NumericalParams(num_steps=steps, num_paths=1, seed=42)
            inputs = PricingInputs(option_params, market_params, numerical)
            
            binomial_price = self.binomial_model.price(self.option, self.market, inputs)
            error = abs(binomial_price - bs_price)
            errors.append(error)
        
        # Error should decrease as steps increase
        for i in range(len(errors) - 1):
            assert errors[i+1] <= errors[i] * 1.5, "Error should generally decrease"
        
        # Final error should be small
        assert errors[-1] < 0.01, f"Error with 500 steps should be < 0.01, got {errors[-1]}"


class TestMonteCarloConvergence:
    """Tests for Monte Carlo convergence to Black-Scholes."""
    
    def setup_method(self):
        self.bs_model = BlackScholesModel()
        self.mc_model = MonteCarloModel()
        self.option = Option(strike=100, maturity=1.0, option_type=OptionType.CALL)
        self.market = MarketEnvironment(spot=100, volatility=0.20, risk_free_rate=0.05)
    
    def test_convergence_to_black_scholes(self):
        """Test that MC converges to BS as paths increase."""
        bs_price = self.bs_model.price(self.option, self.market)
        
        option_params = OptionParams(100, 1.0, OptionType.CALL)
        market_params = MarketParams(100, 0.20, 0.05)
        
        # Test with high path count
        numerical = NumericalParams(num_steps=1, num_paths=100000, seed=42)
        inputs = PricingInputs(option_params, market_params, numerical)
        
        mc_price = self.mc_model.price(self.option, self.market, inputs)
        error = abs(mc_price - bs_price)
        
        # Error should be within 1% for 100k paths
        relative_error = error / bs_price
        assert relative_error < 0.01, f"Relative error {relative_error:.2%} should be < 1%"
    
    def test_reproducibility_with_seed(self):
        """Test that same seed gives same result."""
        option_params = OptionParams(100, 1.0, OptionType.CALL)
        market_params = MarketParams(100, 0.20, 0.05)
        numerical = NumericalParams(num_steps=1, num_paths=10000, seed=42)
        inputs = PricingInputs(option_params, market_params, numerical)
        
        price1 = self.mc_model.price(self.option, self.market, inputs)
        
        # Create new model instance
        mc_model2 = MonteCarloModel()
        price2 = mc_model2.price(self.option, self.market, inputs)
        
        assert price1 == price2, "Same seed should give same price"


class TestFiniteDifferenceGreeks:
    """Tests for finite-difference Greeks convergence to analytical."""
    
    def setup_method(self):
        self.bs_model = BlackScholesModel()
        self.option = Option(strike=100, maturity=1.0, option_type=OptionType.CALL)
        self.market = MarketEnvironment(spot=100, volatility=0.20, risk_free_rate=0.05)
    
    def test_fd_delta_converges_to_analytical(self):
        """Test that FD delta converges to analytical delta."""
        analytical = calculate_all_greeks(self.option, self.market)
        
        # Use tight bumps for better accuracy
        config = FiniteDifferenceConfig(relative_bump_spot=0.001)
        fd_calc = FiniteDifferenceCalculator(self.bs_model, config)
        fd_greeks = fd_calc.calculate_all(self.option, self.market)
        
        error = abs(fd_greeks.delta - analytical.delta)
        assert error < 0.001, f"Delta error {error} should be < 0.001"
    
    def test_fd_gamma_converges_to_analytical(self):
        """Test that FD gamma converges to analytical gamma."""
        analytical = calculate_all_greeks(self.option, self.market)
        
        config = FiniteDifferenceConfig(relative_bump_spot=0.001)
        fd_calc = FiniteDifferenceCalculator(self.bs_model, config)
        fd_greeks = fd_calc.calculate_all(self.option, self.market)
        
        error = abs(fd_greeks.gamma - analytical.gamma)
        relative_error = error / analytical.gamma if analytical.gamma != 0 else error
        assert relative_error < 0.01, f"Gamma relative error {relative_error:.2%} should be < 1%"
    
    def test_fd_vega_converges_to_analytical(self):
        """Test that FD vega converges to analytical vega."""
        analytical = calculate_all_greeks(self.option, self.market)
        
        config = FiniteDifferenceConfig(relative_bump_vol=0.001)
        fd_calc = FiniteDifferenceCalculator(self.bs_model, config)
        fd_greeks = fd_calc.calculate_all(self.option, self.market)
        
        error = abs(fd_greeks.vega - analytical.vega)
        relative_error = error / analytical.vega if analytical.vega != 0 else error
        assert relative_error < 0.01, f"Vega relative error {relative_error:.2%} should be < 1%"


class TestInputValidation:
    """Tests for input validation."""
    
    def test_negative_strike_raises(self):
        """Test that negative strike raises ValueError."""
        with pytest.raises(ValueError):
            Option(strike=-100, maturity=1.0, option_type=OptionType.CALL)
    
    def test_negative_maturity_raises(self):
        """Test that negative maturity raises ValueError."""
        with pytest.raises(ValueError):
            Option(strike=100, maturity=-1.0, option_type=OptionType.CALL)
    
    def test_negative_spot_raises(self):
        """Test that negative spot raises ValueError."""
        with pytest.raises(ValueError):
            MarketEnvironment(spot=-100, volatility=0.20, risk_free_rate=0.05)
    
    def test_negative_volatility_raises(self):
        """Test that negative volatility raises ValueError."""
        with pytest.raises(ValueError):
            MarketEnvironment(spot=100, volatility=-0.20, risk_free_rate=0.05)
    
    def test_inputs_are_immutable(self):
        """Test that input objects are frozen."""
        option = Option(strike=100, maturity=1.0, option_type=OptionType.CALL)
        with pytest.raises(AttributeError):
            option.strike = 110


class TestInputSignatures:
    """Tests for reproducibility signatures."""
    
    def test_identical_inputs_same_signature(self):
        """Test that identical inputs produce the same signature."""
        params1 = OptionParams(100, 1.0, OptionType.CALL)
        params2 = OptionParams(100, 1.0, OptionType.CALL)
        
        assert params1.input_signature() == params2.input_signature()
    
    def test_different_inputs_different_signature(self):
        """Test that different inputs produce different signatures."""
        params1 = OptionParams(100, 1.0, OptionType.CALL)
        params2 = OptionParams(110, 1.0, OptionType.CALL)
        
        assert params1.input_signature() != params2.input_signature()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
