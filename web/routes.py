"""
HTTP routes for the Option Pricing Engine.

All endpoints are defined here. NO business logic - delegates to services.
"""

from flask import Blueprint, render_template, request, flash, redirect, url_for
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.forms import PricingForm
from web.services import PricingService, PricingRequest


main_bp = Blueprint('main', __name__)


@main_bp.route('/', methods=['GET'])
def index():
    """
    Display the main pricing input form.
    """
    form = PricingForm()
    return render_template('index.html', form=form)


@main_bp.route('/price', methods=['POST'])
def price():
    """
    Process pricing request and display results.
    
    This route:
    1. Validates form input
    2. Delegates to service layer
    3. Renders results template
    
    NO pricing logic here - all delegated to services.
    """
    form = PricingForm()
    
    if not form.validate_on_submit():
        # Re-render form with errors
        for field, errors in form.errors.items():
            for error in errors:
                flash(f'{field}: {error}', 'error')
        return render_template('index.html', form=form)
    
    # Build request from form
    pricing_request = PricingRequest(
        option_type=form.option_type.data,
        strike=form.strike.data,
        maturity=form.maturity.data,
        spot=form.spot.data,
        volatility=form.volatility.data,
        risk_free_rate=form.risk_free_rate.data,
        use_black_scholes=form.use_black_scholes.data,
        use_binomial=form.use_binomial.data,
        use_monte_carlo=form.use_monte_carlo.data,
        num_steps=form.num_steps.data or 100,
        num_paths=form.num_paths.data or 100000,
        seed=form.seed.data or 42,
        run_convergence=form.run_convergence.data,
    )
    
    # Process through service layer
    try:
        service = PricingService()
        response = service.process_request(pricing_request)
    except Exception as e:
        flash(f'Pricing error: {str(e)}', 'error')
        return render_template('index.html', form=form)
    
    # Render results
    return render_template(
        'results.html',
        form=form,
        response=response,
        results=response.results,
        comparison=response.comparison,
        summary=response.summary,
    )


@main_bp.route('/api/price', methods=['POST'])
def api_price():
    """
    JSON API endpoint for pricing.
    
    Returns structured JSON for programmatic access.
    """
    from flask import jsonify
    
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    try:
        pricing_request = PricingRequest(
            option_type=data.get('option_type', 'call'),
            strike=float(data.get('strike', 100)),
            maturity=float(data.get('maturity', 1.0)),
            spot=float(data.get('spot', 100)),
            volatility=float(data.get('volatility', 0.20)),
            risk_free_rate=float(data.get('risk_free_rate', 0.05)),
            use_black_scholes=data.get('use_black_scholes', True),
            use_binomial=data.get('use_binomial', True),
            use_monte_carlo=data.get('use_monte_carlo', True),
            num_steps=int(data.get('num_steps', 100)),
            num_paths=int(data.get('num_paths', 100000)),
            seed=int(data.get('seed', 42)),
            run_convergence=data.get('run_convergence', False),
        )
        
        service = PricingService()
        response = service.process_request(pricing_request)
        
        return jsonify({
            'success': True,
            'results': [r.to_dict() for r in response.results],
            'comparison': response.comparison.to_dict(),
            'input_signature': response.input_signature,
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
