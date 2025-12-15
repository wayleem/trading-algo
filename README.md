# Marcus - 0DTE Options Trading System

A backtesting and paper trading system for 0DTE (zero days to expiration) SPY/SPXW options.

## Conclusion: Nearly All Strategies Are Unprofitable

After implementing proper backtesting with **look-ahead bias prevention** and **real historical option prices** via ThetaData, almost all strategies produced negative returns. Only RSI 30/70 with conservative thresholds showed marginal profitability (+$821).

**The main killer is stop-losses**, not EOD expiration. In the RSI strategy, stop-losses triggered 997 times for -$58,856 while profit targets hit 413 times for +$42,375.

### Strategy Backtest Results (Jan-Dec 2024)

| Strategy | Trades | Win Rate | Total P&L | Sharpe | Profit Factor |
|----------|--------|----------|-----------|--------|---------------|
| RSI 30/70 (conservative) | 215 | 38.1% | +$821 | 0.53 | 1.08 |
| RSI 44/60 (default) | 1,263 | 31.0% | -$15,816 | -4.17 | 0.72 |
| ORB (Opening Range Breakout) | 302 | 37.7% | -$2,298 | -5.17 | 0.56 |
| VWAP Reversion | 673 | 44.7% | -$6,071 | -14.64 | 0.36 |
| Momentum Scalp | 450 | 34.0% | -$4,124 | -13.69 | 0.23 |
| IV Rank | 366 | 42.3% | -$4,364 | -8.80 | 0.37 |
| Support/Resistance | 262 | 34.4% | -$2,594 | -7.12 | 0.43 |
| Lunch Scalp | 346 | 26.3% | -$2,963 | -16.24 | 0.15 |
| First Pullback | 165 | 33.3% | -$1,583 | -6.89 | 0.40 |
| Gap Fade | 96 | 21.9% | -$890 | -5.72 | 0.46 |
| VWAP Fade | 28 | 35.7% | -$727 | -8.59 | 0.27 |
| Power Hour | 63 | 28.6% | -$360 | -6.36 | 0.40 |
| Theta Dump | 41 | 29.3% | -$138 | -6.54 | 0.42 |
| Morning Fade | 22 | 63.6% | -$45 | -0.30 | 0.96 |

### Why Strategies Fail

1. **Look-ahead bias elimination**: Signals execute on the NEXT bar's open price, not the signal bar's close
2. **Real option prices**: Using ThetaData historical quotes instead of theoretical pricing
3. **Slippage modeling**: 3% entry slippage, 3% exit slippage, +2% extra on stop-losses
4. **0DTE settlement**: Options expire worthless if OTM at close (intrinsic value only)
5. **Bid-ask spread**: Wide spreads on 0DTE options eat into profits

## Project Structure

```
backend/
├── app/
│   ├── api/routes/          # FastAPI endpoints
│   ├── core/config.py       # Settings from .env
│   ├── models/schemas.py    # Pydantic models
│   └── services/
│       ├── backtest/        # Backtesting engine
│       ├── indicators.py    # RSI, MACD, Bollinger Bands
│       ├── signal_generator.py
│       ├── alpaca_client.py # Broker API
│       └── theta_data.py    # Option price API
├── strategies/              # Strategy implementations
│   ├── base/               # Abstract base class
│   ├── orb/                # Opening Range Breakout
│   ├── vwap_fade/          # VWAP mean reversion
│   └── ...                 # Other strategies
├── scripts/                # CLI tools
└── paper_trading/          # Live paper trading
```

## Scripts

### Backtesting

| Script | Purpose |
|--------|---------|
| `run_backtest.py` | RSI mean reversion strategy backtest |
| `run_strategy_backtest.py` | Run any strategy from strategies/ |
| `compare_strategies.py` | Compare multiple strategies |
| `compare_modes.py` | Compare RSI modes (single vs parallel) |

### Optimization

| Script | Purpose |
|--------|---------|
| `optimize_bayesian.py` | Bayesian hyperparameter optimization |
| `optimize_orb_optuna.py` | Optuna optimization for ORB |
| `grid_search_ptsl.py` | Grid search profit target/stop loss |
| `monte_carlo.py` | Monte Carlo simulation |

### Analysis

| Script | Purpose |
|--------|---------|
| `analyze_orb_by_vix.py` | ORB performance by VIX regime |
| `analyze_vwap_morning.py` | VWAP morning session analysis |
| `analyze_gap_fills.py` | Gap fill probability analysis |
| `diagnose_backtest.py` | Debug backtest issues |

### Trading

| Script | Purpose |
|--------|---------|
| `run_paper_trading.py` | Start paper trading bot |
| `run_strategy.py` | Run live strategy |

## Requirements

- Python 3.11+
- Alpaca API account (paper or live)
- ThetaData subscription + ThetaTerminal running locally

## Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# Start ThetaTerminal (required for backtesting)
java -jar ThetaTerminalv3.jar
```

## Usage

```bash
# Run RSI backtest
python scripts/run_backtest.py --start 2024-01-01 --end 2024-12-01

# Run specific strategy
python scripts/run_strategy_backtest.py --strategy orb --start 2024-01-01 --end 2024-12-01

# List available strategies
python scripts/run_strategy_backtest.py --list
```

## Backtest Realism

The backtester implements several features to prevent overfitting:

- **No look-ahead**: Entry at next bar open, not signal bar close
- **Real prices**: ThetaData historical option quotes (required)
- **Slippage**: Configurable entry/exit/stop slippage percentages
- **EOD settlement**: 0DTE options settle at intrinsic value
- **Entry cutoff**: No new trades after 1 PM ET (configurable)
