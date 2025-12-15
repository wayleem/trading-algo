# Theta Dump Strategy - Findings

## Strategy Overview

Attempts to profit from accelerated theta decay in the afternoon session.

**Signal Logic:**
- Trade direction based on trend, profit from theta decay
- Targets afternoon premium collection

## Backtest Results (Jan-Dec 2024)

```
Total Trades:     41
Win Rate:         29.3%
Total P&L:        -$138
Sharpe Ratio:     -6.54
Profit Factor:    0.42
```

## Key Findings

1. **Low trade count (41)**: Conservative entry criteria
2. **Small loss (-$138)**: Limited exposure
3. **Theta decay is real but unpredictable**: Direction still matters

## Why It Fails

- Buying options = paying theta, not collecting
- Strategy is backwards - should be selling premium
- 0DTE buyers lose to theta, they don't benefit from it

## Conclusion

Theta dump concept is INVERTED. To profit from theta, SELL options, don't buy them. This strategy should be redesigned as credit spreads.
