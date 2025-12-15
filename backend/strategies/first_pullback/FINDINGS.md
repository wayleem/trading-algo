# First Pullback Strategy - Findings

## Strategy Overview

Trades the first pullback after a strong opening move.

**Signal Logic:**
- BUY CALL: First pullback after strong bullish open
- BUY PUT: First pullback after strong bearish open

## Backtest Results (Jan-Dec 2024)

```
Total Trades:     165
Win Rate:         33.3%
Total P&L:        -$1,583
Sharpe Ratio:     -6.89
Profit Factor:    0.40
```

## Key Findings

1. **1 in 3 win rate**: Pullbacks often become reversals
2. **Profit factor 0.40**: Losses 2.5x larger than wins
3. **Timing difficult**: Hard to identify "first" pullback accurately

## Why It Fails

- Pullbacks often continue into full reversals
- Morning volatility makes entries difficult
- Stop-losses trigger before continuation

## Conclusion

First pullback strategy does NOT work for 0DTE options. Pullback identification is too noisy.
