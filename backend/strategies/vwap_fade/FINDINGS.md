# VWAP Fade Strategy - Findings

## Strategy Overview

Fades (trades against) extended moves away from VWAP, expecting mean reversion.

**Signal Logic:**
- BUY CALL: Price drops significantly below VWAP
- BUY PUT: Price rises significantly above VWAP

## Backtest Results (Jan-Dec 2024)

```
Total Trades:     28
Win Rate:         35.7%
Total P&L:        -$727
Sharpe Ratio:     -8.59
Profit Factor:    0.27
```

## Key Findings

1. **Very few trades**: Only 28 signals generated
2. **Terrible profit factor (0.27)**: Losses 3.7x larger than wins
3. **Mean reversion too slow**: VWAP reversion doesn't happen fast enough for 0DTE

## Why It Fails

- VWAP reversion is a daily phenomenon
- 0DTE options expire before reversion completes
- Catching falling knives = large stop-loss hits

## Conclusion

VWAP fade is NOT viable for 0DTE options. Mean reversion timescales don't match 0DTE expiration.
