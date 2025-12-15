# IV Rank Strategy - Findings

## Strategy Overview

Trades based on implied volatility rank, selling when IV is high and buying when IV is low.

**Signal Logic:**
- Trade when IV rank reaches extreme levels
- Direction based on mean reversion expectation

## Backtest Results (Jan-Dec 2024)

```
Total Trades:     366
Win Rate:         42.3%
Total P&L:        -$4,364
Sharpe Ratio:     -8.80
Profit Factor:    0.37
```

## Key Findings

1. **Decent win rate (42.3%)**: IV signals have some predictive value
2. **Profit factor 0.37**: Losses still dominate
3. **IV crush timing difficult**: IV changes don't always translate to price moves

## Why It Fails

- IV rank is better for selling premium, not buying
- 0DTE options have minimal vega exposure
- IV moves don't guarantee directional moves

## Conclusion

IV rank is NOT suitable for directional 0DTE trading. Consider for premium selling strategies with longer-dated options.
