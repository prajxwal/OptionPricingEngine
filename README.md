# Option Pricing Engine
*it works. (we're as surprised as you are.)*
## What Is This

A European option pricing engine with three models, a Greeks calculator, and a Flask web interface. Built instead of touching grass. The models disagree with each other. This is expected and has not been fixed.

## Features

- **Black-Scholes** — assumes volatility is constant. volatility has never been constant. the model is unaware and unbothered.
- **Binomial Tree** — builds a recombining tree of future stock prices. also built in 1979. has not changed its personality since.
- **Monte Carlo** — runs 10,000 simulations and averages them. confidently approximate.
- **Greeks** — delta, gamma, vega, theta, rho. named after greek letters because the math is old and wanted you to feel that way.
- **Model Comparison** — shows you how wrong each model is relative to the other models, which are also wrong.
- **Convergence Analysis** — watch numerical methods slowly inch toward the analytical answer like they're apologizing.
- **Web Interface** — flask app. runs on localhost. do not tell anyone about this in production.

> **fun fact**: the Black-Scholes model won the Nobel Prize in 1997. the hedge fund co-founded by one of its authors blew up two years later. the model remains on the curriculum.

## Quick Start
```bash
pip install -r requirements.txt
python -m web.app
```

open http://127.0.0.1:5000. feel like a quant. you are not a quant.

## Project Structure
```
OptionPricingEngine/
├── core/          # respectable
├── models/        # three ways to be approximately wrong
├── greeks/        # sensitivities that won't save you
├── analysis/      # for studying why the models fight
├── web/           # please don't deploy this
└── tests/         # they pass
```

## Limitations

- no dividends, transaction costs, market impact, liquidity, or any of the things that actually determine option prices
- monte carlo results are stochastic. the market doesn't care about your seed.
- the risk-free rate input assumes risk-free things exist
- **do not trade on this. please. we're asking nicely.**

## API
```python
option = Option(strike=100, maturity=1.0, option_type=OptionType.CALL)
market = MarketEnvironment(spot=100, volatility=0.20, risk_free_rate=0.05)

result = BlackScholesModel().compute(option, market)
print(result.price)  # precise-looking number based on assumptions that don't hold
                     # this is also how wall street works, so you're fine
```

## Why

a cs student wanted to understand options pricing. this was faster than reading hull.

the code is fine. the math is correct given that the math is a simplification of a system that ignores the math. 

## License

educational use only. if you use this to trade, that's between you and your broker.
