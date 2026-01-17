# Option Pricing Engine

A modular, extensible European option pricing engine with Flask web interface for quantitative research and model comparison.

## Features

- **Three Pricing Models**: Black-Scholes (analytical), Binomial (CRR), Monte Carlo (GBM)
- **Greeks Calculation**: Analytical and finite-difference methods
- **Model Comparison**: Automatic benchmarking against Black-Scholes baseline
- **Convergence Analysis**: Visualize how numerical models converge
- **Web Interface**: Clean, responsive Flask application
- **Reproducibility**: Seed-controlled randomness & input signatures

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web application
python -m web.app

# Or run tests
pytest tests/ -v
```

Then open http://127.0.0.1:5000 in your browser.

## Project Structure

```
OptionPricingEngine/
├── core/                 # Domain objects and inputs
│   ├── enums.py          # OptionType, ModelType
│   ├── inputs.py         # Immutable, hashable input dataclasses
│   ├── option.py         # Option contract abstraction
│   └── market.py         # Market environment
│
├── models/               # Pricing models
│   ├── base.py           # PricingModel ABC, PricingResult contract
│   ├── black_scholes.py  # Analytical baseline
│   ├── binomial.py       # CRR tree model
│   └── monte_carlo.py    # GBM simulation
│
├── greeks/               # Sensitivity calculations
│   ├── analytical.py     # Closed-form Greeks
│   └── finite_difference.py  # Numerical Greeks
│
├── analysis/             # Research tools
│   ├── comparison.py     # Multi-model comparison
│   ├── convergence.py    # Convergence analysis
│   └── experiments.py    # Parameter sweeps
│
├── reporting/            # Output generation
│   ├── plots.py          # Centralized plotting
│   └── summary.py        # Structured reports
│
├── web/                  # Flask application
│   ├── app.py            # Application factory
│   ├── routes.py         # HTTP endpoints
│   ├── forms.py          # Input validation
│   ├── services.py       # Bridge layer
│   └── templates/        # HTML templates
│
├── tests/                # Unit tests
├── config.py             # Configuration
└── requirements.txt      # Dependencies
```

## Design Principles

1. **Separation of Concerns**: Pricing logic isolated from web layer
2. **Reproducibility**: Immutable inputs with deterministic signatures
3. **Numerical Rigor**: Cross-validation between analytical and numerical methods
4. **Transparency**: Explicit model assumptions and limitations

## Important Limitations

- Black-Scholes assumptions are unrealistic in practice
- Constant volatility is a simplification
- Monte Carlo results are stochastic
- No dividends, transaction costs, or market frictions
- **Educational/research use only - not for trading decisions**

## API Usage

```python
from core.option import Option
from core.market import MarketEnvironment
from core.enums import OptionType
from models.black_scholes import BlackScholesModel

# Create domain objects
option = Option(strike=100, maturity=1.0, option_type=OptionType.CALL)
market = MarketEnvironment(spot=100, volatility=0.20, risk_free_rate=0.05)

# Price with Black-Scholes
model = BlackScholesModel()
result = model.compute(option, market)

print(f"Price: {result.price:.4f}")
print(f"Delta: {result.greeks.delta:.4f}")
```

## License

Educational use only.
