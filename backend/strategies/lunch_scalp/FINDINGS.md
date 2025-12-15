# Lunch Scalp Strategy - Findings

## Strategy Overview

Scalps during the lunch hour (11:30 AM - 1:00 PM ET) low volatility period.

**Signal Logic:**
- Trades range-bound conditions during lunch doldrums
- Mean reversion within tight ranges

## Backtest Results (Jan-Dec 2024)

```
Total Trades:     346
Win Rate:         26.3%
Total P&L:        -$2,963
Sharpe Ratio:     -16.24
Profit Factor:    0.15
```

## Key Findings

1. **Worst profit factor (0.15)**: Losses 6.7x larger than wins
2. **Worst Sharpe (-16.24)**: Extremely poor risk-adjusted returns
3. **Low volume = wide spreads**: Lunch hour liquidity is poor

## Why It Fails

- Low volatility = small moves
- Wide spreads during lunch hour
- Scalping doesn't work when spreads > expected move
- Options theta decay continues while waiting

## Conclusion

Lunch scalping is the WORST strategy tested. Low volatility + wide spreads = guaranteed losses on 0DTE options.
