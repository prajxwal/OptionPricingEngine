"""
Input validation forms for the web interface.

Uses WTForms with hard limits to protect the server.
"""

from flask_wtf import FlaskForm
from wtforms import (
    FloatField, IntegerField, SelectField, BooleanField, SubmitField
)
from wtforms.validators import (
    DataRequired, NumberRange, Optional, ValidationError
)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import VALIDATION_LIMITS, NUMERICAL_DEFAULTS


class PricingForm(FlaskForm):
    """
    Main pricing input form.
    
    Includes server-side validation with hard limits.
    """
    
    # Option parameters
    option_type = SelectField(
        'Option Type',
        choices=[('call', 'Call'), ('put', 'Put')],
        default='call',
    )
    
    strike = FloatField(
        'Strike Price (K)',
        validators=[
            DataRequired(message="Strike price is required"),
            NumberRange(
                min=VALIDATION_LIMITS.min_strike,
                max=VALIDATION_LIMITS.max_strike,
                message=f"Strike must be between {VALIDATION_LIMITS.min_strike} and {VALIDATION_LIMITS.max_strike}"
            ),
        ],
        default=100.0,
    )
    
    maturity = FloatField(
        'Time to Maturity (years)',
        validators=[
            DataRequired(message="Maturity is required"),
            NumberRange(
                min=VALIDATION_LIMITS.min_maturity,
                max=VALIDATION_LIMITS.max_maturity,
                message=f"Maturity must be between {VALIDATION_LIMITS.min_maturity} and {VALIDATION_LIMITS.max_maturity} years"
            ),
        ],
        default=1.0,
    )
    
    # Market parameters
    spot = FloatField(
        'Spot Price (S)',
        validators=[
            DataRequired(message="Spot price is required"),
            NumberRange(
                min=VALIDATION_LIMITS.min_spot,
                max=VALIDATION_LIMITS.max_spot,
                message=f"Spot must be between {VALIDATION_LIMITS.min_spot} and {VALIDATION_LIMITS.max_spot}"
            ),
        ],
        default=100.0,
    )
    
    volatility = FloatField(
        'Volatility (σ) [decimal]',
        validators=[
            DataRequired(message="Volatility is required"),
            NumberRange(
                min=VALIDATION_LIMITS.min_volatility,
                max=VALIDATION_LIMITS.max_volatility,
                message=f"Volatility must be between {VALIDATION_LIMITS.min_volatility} and {VALIDATION_LIMITS.max_volatility}"
            ),
        ],
        default=0.20,
    )
    
    risk_free_rate = FloatField(
        'Risk-Free Rate (r) [decimal]',
        validators=[
            DataRequired(message="Risk-free rate is required"),
            NumberRange(
                min=VALIDATION_LIMITS.min_rate,
                max=VALIDATION_LIMITS.max_rate,
                message=f"Rate must be between {VALIDATION_LIMITS.min_rate} and {VALIDATION_LIMITS.max_rate}"
            ),
        ],
        default=0.05,
    )
    
    # Model selection
    use_black_scholes = BooleanField('Black-Scholes', default=True)
    use_binomial = BooleanField('Binomial (CRR)', default=True)
    use_monte_carlo = BooleanField('Monte Carlo', default=True)
    
    # Numerical parameters
    num_steps = IntegerField(
        'Binomial Steps',
        validators=[
            Optional(),
            NumberRange(
                min=VALIDATION_LIMITS.min_steps,
                max=VALIDATION_LIMITS.max_steps,
                message=f"Steps must be between {VALIDATION_LIMITS.min_steps} and {VALIDATION_LIMITS.max_steps}"
            ),
        ],
        default=NUMERICAL_DEFAULTS.default_steps,
    )
    
    num_paths = IntegerField(
        'Monte Carlo Paths',
        validators=[
            Optional(),
            NumberRange(
                min=VALIDATION_LIMITS.min_paths,
                max=VALIDATION_LIMITS.max_paths,
                message=f"Paths must be between {VALIDATION_LIMITS.min_paths} and {VALIDATION_LIMITS.max_paths}"
            ),
        ],
        default=NUMERICAL_DEFAULTS.default_paths,
    )
    
    seed = IntegerField(
        'Random Seed',
        validators=[Optional()],
        default=NUMERICAL_DEFAULTS.default_seed,
    )
    
    # Analysis options
    run_convergence = BooleanField('Run Convergence Analysis', default=False)
    
    submit = SubmitField('Price Option')
    
    def validate(self, extra_validators=None) -> bool:
        """Custom validation."""
        if not super().validate(extra_validators):
            return False
        
        # At least one model must be selected
        if not any([
            self.use_black_scholes.data,
            self.use_binomial.data,
            self.use_monte_carlo.data,
        ]):
            self.use_black_scholes.errors.append("Select at least one model")
            return False
        
        return True
